import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.dgcnn_seg import DGCNNSeg, Classifer
from dataloaders.loader import MyDataset
from utils.logger import init_logger
from utils.checkpoint_util import save_train_checkpoint, load_trained_checkpoint, save_classifer_checkpoint
from utils.Buffer import RandomBuffer
from utils.AnalyticLinear import RecursiveLinear

def metric_evaluate(predicted_label, gt_label, NUM_CLASS, TOTAL_CLASS, eval_mode, class_id, logger, dataset):
    '''Caluate the mIoU for Classes'''
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]
    if isinstance(class_id, int):
        class_id = list([class_id])

    for i in range(gt_label.size()[0]):
        pred_pc = predicted_label[i]
        gt_pc = gt_label[i]

        for j in range(gt_pc.shape[0]):
            gt_l = int(gt_pc[j])
            pred_l = int(pred_pc[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    OA = sum(true_positive_classes)/float(sum(positive_classes))

    if eval_mode == 'eval_Base':
        IoU_list = []
        for i in range(NUM_CLASS):
            iou_class = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
            IoU_list.append(iou_class)
        logger.cprint('background class IoU: %f' % (IoU_list[0]))
        for j in range(TOTAL_CLASS):
            if j not in class_id:
                logger.cprint('Class_%d IoU: X' % j)
            else:
                ind = class_id.index(j)
                logger.cprint('Class_%d IoU: %f' % (class_id[ind], IoU_list[ind+1]))
        mean_IoU = np.array(IoU_list[1:]).mean()  # Caluate the Mean IoU for classes exclude background
    elif eval_mode == 'eval_Incre':
        IoU_list = []
        for i in range(NUM_CLASS):
            if i in class_id:
                iou_class = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
                IoU_list.append(iou_class)
                logger.cprint('Class_%d IoU: %f' % (i, iou_class))
            else:
                IoU_list.append(0.0)
                logger.cprint('Class_%d IoU: X' % i)
        mean_IoU = np.sum(IoU_list)/len(class_id)
    else:
        return NotImplementedError('Unknown eval mode!')

    return OA, mean_IoU, IoU_list

def ACL(args):
    # 设置日志目录
    args.log_dir = os.path.join(args.save_path, 'log_acl_%s_cv%d' % (args.dataset, args.cvfold))

    # 初始化数据集、数据加载器和记录器
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }

    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.tasks, args.data_path)
        VALID_SET = 'Area_5'

        BASE_CLASSES = DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        NUM_BASE_CLASSES = len(BASE_CLASSES)
        NUM_ALL_CLASSES = DATASET.classes
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.tasks, args.data_path)
        VALID_SET = []
        lines = open(os.path.join(os.path.dirname(args.data_path.rstrip('/')), 'scannetv2_val.txt')).readlines()
        for line in lines:
            VALID_SET.append(line.strip('\n'))

        BASE_CLASSES = [0] + DATASET.base_classes
        INCRE_CLASSES = DATASET.incre_classes
        NUM_BASE_CLASSES = len(BASE_CLASSES)
        NUM_ALL_CLASSES = DATASET.classes
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    INCRE = int((args.tasks).split('-')[1]) # Incre x classes for 1 step
    STEP = int(len(INCRE_CLASSES) / INCRE) + 1 # Total steps
    CLASS2SCAN = DATASET.class2scans





    # 进行增量学习
    for step in range(0, STEP):
        # 获取当前任务的类
        CURRENT_CLASS = INCRE_CLASSES[(step - 1) * num_tasks: step * num_tasks]

        TRAIN_DATASET = MyDataset(args.data_path, BASE_CLASSES + CURRENT_CLASS, CURRENT_CLASS, DATASET.class2scans, step, mode='train', valid_set=VALID_SET,
                                  num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                  pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

        VALID_DATASET = MyDataset(args.data_path, BASE_CLASSES + CURRENT_CLASS, CURRENT_CLASS, DATASET.class2scans, step, mode='test', valid_set=VALID_SET,
                                  num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                  pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

        TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, drop_last=True)
        VALID_LOADER = DataLoader(VALID_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, drop_last=True)

        # 训练第一个任务
        if step == 0:
            logger = init_logger(args.log_dir + '/base_model', args)
            logger.cprint('*******************Training the Model on Base Classes: %d Classes | Step %d' % (NUM_BASE_CLASSES, step))
            # Init base model and optimizer
            model = DGCNNSeg(args)
            classifer = Classifer(NUM_BASE_CLASSES + 1)
            print(model)
            if torch.cuda.is_available():
                model.cuda()
                classifer.cuda()

            optimizer_base = optim.Adam([{'params': model.parameters(), 'lr': args.base_lr},
                                       {'params': classifer.parameters(), 'lr': args.base_lr}],
                                        weight_decay=args.base_weight_decay)

            # Set learning rate scheduler
            lr_scheduler_base = optim.lr_scheduler.StepLR(optimizer_base, step_size=args.base_decay_size, gamma=args.base_gamma)

            # train
            best_iou = 0
            global_iter = 0
            for epoch in range(args.n_epochs):
                for batch_idx, (ptclouds, labels) in enumerate(TRAIN_LOADER):
                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()

                    _, logits = model(ptclouds)
                    cls_logits = classifer(logits)

                    if args.dataset == 'scannet':
                        loss = F.cross_entropy(cls_logits, labels, ignore_index=1) # For scannet ignore labels 0 -> 1 : unannotated
                    else:
                        loss = F.cross_entropy(cls_logits, labels)

                    # Loss backwards and optimizer updates
                    optimizer_base.zero_grad()
                    loss.backward()
                    optimizer_base.step()

                    WRITER.add_scalar('Train/loss', loss, global_iter)
                    logger.cprint('=====[Train] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, batch_idx, loss.item()))
                    global_iter += 1

                lr_scheduler_base.step()

                if (epoch + 1) % args.eval_interval == 0:
                    pred_total = []
                    gt_total = []
                    with torch.no_grad():
                        for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                            gt_total.append(labels.detach())

                            if torch.cuda.is_available():
                                ptclouds = ptclouds.cuda()
                                labels = labels.cuda()
                            model.eval()
                            classifer.eval()

                            # 特征提取
                            _, logits = model(ptclouds)
                            logits_cls = classifer(logits)
                            loss = F.cross_entropy(logits_cls, labels)

                            # Compute predictions
                            _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                            pred_total.append(preds.cpu().detach())

                            WRITER.add_scalar('Valid/loss', loss, global_iter)
                            logger.cprint('=====[Valid] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, i, loss.item()))

                    pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
                    gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)
                    accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_BASE_CLASSES + 1, NUM_ALL_CLASSES, 'eval_Base', BASE_CLASSES, logger, args.dataset)
                    logger.cprint('===== EPOCH [%d]: Accuracy: %f | mIoU: %f =====\n' % (epoch, accuracy, mIoU))
                    WRITER.add_scalar('Valid/overall_accuracy', accuracy, global_iter)
                    WRITER.add_scalar('Valid/meanIoU', mIoU, global_iter)

                    if mIoU > best_iou:
                        best_iou = mIoU
                        logger.cprint('*******************Model Saved*******************')
                        save_train_checkpoint(model, args.log_dir, 'best_base_model')
                        save_classifer_checkpoint(classifer, args.log_dir, 'best_base_model')

            logger.cprint('*******************End of Training the Base Model*******************')
            logger.cprint('*******************Eval the Base Model*******************')
            pred_end_total = []
            gt_end_total = []
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                    gt_end_total.append(labels.detach())

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()
                    model.eval()
                    classifer.eval()

                    _, logits = model(ptclouds)
                    logits_cls = classifer(logits)
                    if args.dataset == 'scannet':
                        loss = F.cross_entropy(logits_cls, labels, ignore_index=1)  # For scannet ignore labels 0 -> 1
                    else:
                        loss = F.cross_entropy(logits_cls, labels)

                    # Compute predictions
                    _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                    pred_end_total.append(preds.cpu().detach())

                    WRITER.add_scalar('Valid/loss', loss, global_iter)
                    logger.cprint('=====[Valid] End | Iter: %d | Loss: %.4f =====' % (i, loss.item()))

            pred_total = torch.stack(pred_end_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_end_total, dim=0).view(-1, args.pc_npts)
            accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_BASE_CLASSES + 1, NUM_ALL_CLASSES,
                                                           'eval_Base', BASE_CLASSES, logger, args.dataset)
            logger.cprint('===== Accuracy: %f | mIoU: %f =====\n' % (accuracy, mIoU))
            logger.cprint('*******************Model Saved*******************')
            save_train_checkpoint(model, args.log_dir, 'end_base_model')
            save_classifer_checkpoint(classifer, args.log_dir, 'end_base_model')

            WRITER.close()

        # 在第一个任务完成后冻结backbone
        for param in model.parameters():
            param.requires_grad = False

        # 使用RandomBuffer升高维度
        buffer = RandomBuffer(in_features=model.encoder.out_features, out_features=model.encoder.out_features + 128)
        if torch.cuda.is_available():
            buffer.cuda()

        # 使用RecursiveLinear进行增量学习
        recursive_linear = RecursiveLinear(in_features=buffer.out_features)
        if torch.cuda.is_available():
            recursive_linear.cuda()

        # 进行后续任务的增量学习
        logger = init_logger(args.log_dir + '/incre_model', args)
        logger.cprint('=== Training on Incremental Classes: %d Classes ===' % len(CURRENT_CLASS))
        for epoch in range(args.n_epochs):
            for batch_idx, (ptclouds, labels) in enumerate(TRAIN_LOADER):
                if torch.cuda.is_available():
                    ptclouds = ptclouds.cuda()
                    labels = labels.cuda()

                # 特征提取
                features = model(ptclouds)
                features = buffer(features)

                # 生成混合标签
                with torch.no_grad():
                    pred_old = classifer(features)
                    labels_new = torch.where(labels == 0, pred_old.argmax(dim=1) - int(1.0), labels + int(NUM_BASE_CLASSES) - int(1.0))
                    index_probs, uncertain_old, uncertain_knn = model.uncertainty_estimation(ptclouds[:, :3, :].transpose(1, 2), pred_old.transpose(1, 2))
                    uncertain_old = (uncertain_old.reshape(-1, pred_old.shape[2], 1)).transpose(1, 2).squeeze()
                    uncertain_knn = uncertain_knn.reshape(uncertain_old.shape[0], pred_old.shape[2], -1)[:, :, 1:]

                    _, k_probs, _ = index_probs.shape
                    index_probs = (torch.argmax(index_probs[:, 1:, :], dim=-1)).reshape(-1, pred_old.shape[2], k_probs - 1)
                    index_probs = torch.where(uncertain_knn > args.uncertain_t, -int(1), index_probs)

                    labels_new_cls = labels_new
                    for k in range(index_probs.shape[2]):
                        if (labels_new_cls==-1).long().sum([0, 1]) >= 1:
                            if k == 0:
                                labels_new_cls = torch.where((labels_new_cls==-1) | (uncertain_old > args.uncertain_t),
                                                             index_probs[:, :, k], labels_new_cls)
                            else:
                                labels_new_cls = torch.where((labels_new_cls ==-1),
                                                                index_probs[:, :, k], labels_new_cls)
                        else:
                            break

                # 计算损失
                logits_new_cls = recursive_linear(features)
                loss = F.cross_entropy(logits_new_cls, labels_new_cls)

                # 调用fit和update方法
                recursive_linear.fit(features, labels_new_cls)


                # 记录损失
                logger.cprint('Epoch: %d, Batch: %d, Loss: %.4f' % (epoch, batch_idx, loss.item()))

    logger.cprint('ACL训练完成！')
