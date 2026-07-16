import os
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from models.dgcnn_seg import DGCNNSeg, Classifer
from dataloaders.loader import MyDataset
from utils.logger import init_logger
from dataloaders.loader import MyTestDataset
from utils.checkpoint_util import save_train_checkpoint, load_trained_checkpoint, save_classifer_checkpoint
from utils.AnalyticLinear import RecursiveLinear,GeneralizedARM
from utils.Buffer import RandomBuffer
from utils.result_io import write_acl_manifest, write_acl_result_summary

class AIR(torch.nn.Module):
    def __init__(
        self,
        backbone_output: int,
        backbone: torch.nn.Module = torch.nn.Flatten(),
        buffer_size: int = 10000,
        gamma: float = 1,
        device=None,
        dtype=torch.double,
        linear: type = GeneralizedARM,  # Changed from type[AnalyticLinear] to type
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.eval()

    @torch.no_grad()
    def feature_extraction(self, X: torch.Tensor) -> torch.Tensor:
        _, features = self.backbone(X)  # 解包两个返回值，只保留第二个
        return features

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.feature_extraction(X)
        B, C, N = X.shape
        X = X.permute(0, 2, 1).reshape(B * N, C)
        X=self.buffer(X)
        return self.analytic_linear(X)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor,ignore_index: int = None, *args, **kwargs) -> None:

        X = self.feature_extraction(X)
        # X的维度是(B,C,N),y的维度是(B,N)，现在要变为(B*N,C)和(B*N)
        B, C, N = X.shape
        X = X.permute(0, 2, 1).reshape(B * N, C)
        y = y.reshape(B * N)
        X=self.buffer(X)
        # 如果指定了ignore_index，过滤掉对应的样本
        if ignore_index is not None:
            mask = y != ignore_index
            if mask.sum() > 0:  # 确保有样本可用
                X = X[mask]
                y = y[mask]
        # print('X:', X.shape)
        # print('y:', y.shape)
        self.analytic_linear.fit(X, y)

    @torch.no_grad()
    def update(self) -> None:
        self.analytic_linear.update()

def metric_evaluate_test(predicted_label, gt_label, NUM_CLASS, class_id, logger, dataset, base_classes=None, incre_classes=None):
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

    IoU_list = []

    for i in range(NUM_CLASS):
        iou_class = true_positive_classes[i] / float(
            gt_classes[i] + positive_classes[i] - true_positive_classes[i])
        IoU_list.append(iou_class)
        logger.cprint('Class_%d IoU: %f' % (i, iou_class))

    if dataset == 'scannet':
        mean_IoU = np.sum(IoU_list[1:]) / (len(class_id) - 1) # Exclude the ignore labels
    else:
        mean_IoU = np.sum(IoU_list) / len(class_id)

    # 计算基类和增量类的平均mIoU
    base_mIoU, incre_mIoU = None, None

    # print('base_classes',base_classes)
    # print('incre_classes',incre_classes)

    if base_classes is not None and incre_classes is not None:
        # 计算基类的平均mIoU
        base_indices = [cls_idx for cls_idx in range(NUM_CLASS) if cls_idx in base_classes]
        if dataset == 'scannet' and 0 in base_indices:
            base_indices.remove(0)  # 对于scannet，排除忽略标签
        base_IoUs = [IoU_list[i] for i in base_indices]
        base_mIoU = np.mean(base_IoUs) if base_IoUs else 0.0
        logger.cprint(f'Base Classes mIoU: {base_mIoU:.4f}')

        # 计算增量类的平均mIoU
        incre_indices = [cls_idx for cls_idx in range(NUM_CLASS) if cls_idx in incre_classes]
        incre_IoUs = [IoU_list[i] for i in incre_indices]
        incre_mIoU = np.mean(incre_IoUs) if incre_IoUs else 0.0
        logger.cprint(f'Incremental Classes mIoU: {incre_mIoU:.4f}')

    return OA, mean_IoU, IoU_list, base_mIoU, incre_mIoU

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
    print('class_id:',class_id)
    if eval_mode == 'eval_Base': # Include the background classes for eval_base or eval_incre backgroud是0，unannotated是1
        IoU_list = []
        for i in range(NUM_CLASS):
            # 添加除零保护
            denominator = gt_classes[i] + positive_classes[i] - true_positive_classes[i]
            if denominator > 0:
                iou_class = true_positive_classes[i] / float(denominator)
            else:
                # 如果分母为零，设置IoU为0或其他默认值
                iou_class = 0.0
            IoU_list.append(iou_class)
        logger.cprint('background class IoU: %f' % (IoU_list[0]))
        for j in range(TOTAL_CLASS):
            if j not in class_id:
                logger.cprint('Class_%d IoU: X' % j)
            else:
                ind = class_id.index(j)
                logger.cprint('Class_%d IoU: %f' % (class_id[ind], IoU_list[ind+1]))
        if dataset == 'scannet':
            mean_IoU = np.array(IoU_list[2:]).mean()  # Caluate the Mean IoU for classes exclude background and unannotated
        else:
            mean_IoU = np.array(IoU_list[1:]).mean()  # Caluate the Mean IoU for classes exclude background
    elif eval_mode == 'eval_Incre':
        IoU_list = []
        for i in range(NUM_CLASS):
            if i in class_id:
                denominator = gt_classes[i] + positive_classes[i] - true_positive_classes[i]
                if denominator > 0:
                    iou_class = true_positive_classes[i] / float(denominator)
                else:
                    # 如果分母为零，设置IoU为0或其他默认值
                    iou_class = 0.0
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
    # Init datasets, dataloaders, and writer
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

    START_STEP = int(getattr(args, 'start_step', 0))
    if START_STEP not in (0, 1) or START_STEP >= STEP:
        raise ValueError('start_step must be 0 or 1 and smaller than total steps; got %d' % START_STEP)

    write_acl_manifest(args.log_dir, args, BASE_CLASSES, INCRE_CLASSES, STEP)

    # Set --start_step 1 if a trained base model already exists and only incremental steps are needed.
    for step in range(START_STEP, STEP): # 0 for base step, 1~STEP for incre step
        SAMPLE_CLASS = BASE_CLASSES.copy()  # intital as old class
        if step == 0: # Train the model for base classes
            CLASSES = BASE_CLASSES
            CURRENT_CLASS = BASE_CLASSES
            LOG_DIR = args.log_dir + '/base_model'
        else: # Train the model for incremental classes
            LOG_DIR = args.log_dir + '/incre_model'
            args.batch_size = 32
            if step==1 and int((args.tasks).split('-')[1]) == (NUM_ALL_CLASSES-NUM_BASE_CLASSES):
                CLASSES = INCRE_CLASSES
                CURRENT_CLASS = INCRE_CLASSES
                SAMPLE_CLASS.extend(CLASSES)
            else: # multi-steps increments
                SAMPLE_CLASS.extend(INCRE_CLASSES[:step*INCRE])
                CLASSES = INCRE_CLASSES[(step - 1) * INCRE:step * INCRE]
                CURRENT_CLASS = INCRE_CLASSES[(step - 1) * INCRE:step * INCRE]

        TRAIN_DATASET = MyDataset(args.data_path, CLASSES, CURRENT_CLASS, CLASS2SCAN, step, mode='train', valid_set=VALID_SET,
                                          num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                          pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

        VALID_DATASET = MyDataset(args.data_path, CLASSES, CURRENT_CLASS, CLASS2SCAN, step, mode='test', valid_set=VALID_SET,
                                          num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                          pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

        TRAIN_BASE_DATASET = MyDataset(args.data_path, BASE_CLASSES, BASE_CLASSES, CLASS2SCAN, step, mode='train', valid_set=VALID_SET,
                                          num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                          pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

        VALID_BASE_DATASET = MyDataset(args.data_path, BASE_CLASSES, BASE_CLASSES, CLASS2SCAN, step, mode='test', valid_set=VALID_SET,
                                          num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                          pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)
        logger = init_logger(LOG_DIR, args)
        logger.cprint('=== Train Dataset (classes: {0}) | Train: {1} blocks | Valid: {2} blocks ==='.format(
                                                         CLASSES, len(TRAIN_DATASET), len(VALID_DATASET)))



        TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True,
                                  drop_last=True)

        VALID_LOADER = DataLoader(VALID_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                  drop_last=True)

        TRAIN_BASE_LOADER = DataLoader(TRAIN_BASE_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True,
                                  drop_last=True)

        VALID_BASE_LOADER = DataLoader(VALID_BASE_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                  drop_last=True)


        WRITER = SummaryWriter(log_dir=LOG_DIR)

        if step == 0: # Train the base model on the base classes
            logger.cprint('*******************Training the Model on Base Classes: %d Classes | Step %d'
                          '*******************' % (NUM_BASE_CLASSES,step))
            # Init base model and optimizer
            model = DGCNNSeg(args)
            classifer = Classifer(NUM_BASE_CLASSES + 1)
            print(model)
            if torch.cuda.is_available():
                model.cuda()
                classifer.cuda()

            optimizer_base = optim.Adam([{'params': model.parameters(), 'lr': args.base_lr}, \
                                   {'params': classifer.parameters(), 'lr': args.base_lr}], \
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
                        #打印labels的唯一值
                        # print('labels:',labels.unique())

                    _, logits = model(ptclouds)
                    cls_logits = classifer(logits)

                    if args.dataset == 'scannet':
                        loss = F.cross_entropy(cls_logits, labels, ignore_index=1)  # For scannet ignore labels 0 -> 1 : unannotated
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

                if (epoch+1) % args.eval_interval == 0:
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

                            _, logits = model(ptclouds)
                            logits_cls = classifer(logits)

                            if args.dataset == 'scannet':
                                loss = F.cross_entropy(logits_cls, labels, ignore_index=1)  # For scannet ignore labels 0 -> 1 : unannotated
                            else:
                                loss = F.cross_entropy(logits_cls, labels)

                            # Compute predictions
                            _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                            pred_total.append(preds.cpu().detach())

                            WRITER.add_scalar('Valid/loss', loss, global_iter)
                            logger.cprint(
                                '=====[Valid] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, i, loss.item()))

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
                    # print('logits:', logits.shape)
                    logits_cls = classifer(logits)

                    if args.dataset == 'scannet':
                        loss = F.cross_entropy(logits_cls, labels, ignore_index=1)  # For scannet ignore labels 0 -> 1 : unannotated
                    else:
                        loss = F.cross_entropy(logits_cls, labels)

                    # 　Compute predictions
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

            logger.cprint('*******************End of evaluate the Base Model*******************')



            WRITER.close()
        elif step > 0 and STEP==2:
            logger.cprint('*******************Training the Model on Incremental Classes: %d Classes | Total 2 Tasks | Step %d'
                          '*******************' % (len(INCRE_CLASSES), step))
            # train the model for incremental classes: step > 0 and tasks == 2
            # Init the old model
            model_old = DGCNNSeg(args, option='uncertain')
            model_old = load_trained_checkpoint(model_old, args.base_model_checkpoint_path,
                                                'end_base_model_checkpoint.tar')  # Use the last base outputs
            classifer_old = Classifer(num_classes=NUM_BASE_CLASSES + 1)
            classifer_old = load_trained_checkpoint(classifer_old, args.base_model_checkpoint_path,
                                                'end_base_model_classifer_checkpoint.tar')
            # Freeze the old model and classifer
            for param in model_old.parameters():
                param.requires_grad = False
            for param in classifer_old.parameters():
                param.requires_grad = False

            if torch.cuda.is_available():
                model_old.cuda()

                classifer_old.cuda()


            logger.cprint('*******************Start Re-align Base *******************')
            # Re-align the base model
            AIRMODEL= AIR(backbone_output=128, backbone=model_old, buffer_size=5000, gamma=1, device=torch.device("cuda"), dtype=torch.double, linear=GeneralizedARM)
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(TRAIN_BASE_LOADER):
                    # print('ptclouds:', ptclouds.shape)
                    # print('labels:', labels.shape)

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()

                    if args.dataset == 'scannet':
                       AIRMODEL.fit(ptclouds, labels, ignore_index=1)
                    else:
                        AIRMODEL.fit(ptclouds, labels)
                    # if i>=2:
                    #     break
                AIRMODEL.update()
            logger.cprint('*******************End of Re-align Base *******************')
            logger.cprint('*******************Evaluate Re-align Base *******************')
            pred_end_total = []
            gt_end_total = []
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(VALID_BASE_LOADER):
                    gt_end_total.append(labels.detach())

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()


                    logits_cls=AIRMODEL(ptclouds)

                    # 　Compute predictions
                    _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                    pred_end_total.append(preds.cpu().detach())


            pred_total = torch.stack(pred_end_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_end_total, dim=0).view(-1, args.pc_npts)
            accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_BASE_CLASSES + 1, NUM_ALL_CLASSES,
                                                           'eval_Base', BASE_CLASSES, logger, args.dataset)
            logger.cprint('===== Accuracy: %f | mIoU: %f =====\n' % (accuracy, mIoU))
            logger.cprint('*******************End Evaluate Re-align Base *******************')
            logger.cprint('*******************Start of Training the Incre Model*******************')

            AIRMODEL_OLD = copy.deepcopy(AIRMODEL)
            all_uncertain_values = []
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(TRAIN_LOADER):
                    # print('ptclouds:', ptclouds.shape)
                    # print('labels:', labels.shape)
                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()
                    pred_old = AIRMODEL_OLD(ptclouds)
                    pred_old=F.softmax(pred_old, dim=1)
                    B,N=labels.shape
                    # print('labels:',labels.unique())
                    #把pred_old的维度由[B*N,C]变为[B,C,N]
                    pred_old = pred_old.reshape(B, -1, N)
                    # labels = torch.where(labels == 0, 0, labels + int(NUM_BASE_CLASSES))
                    # #打印labels的唯一值
                    # print('labels:',labels.unique())
                    #ptclouds B,D,N  pred_old B,C,N
                    #index_probs [B×N, k_probs, C] uncertain_old [B×N] uncertain_knn [B×N, k_probs, 1]
                    # Uncertainty-aware Pseudo-label Generation -> Mixed Labels
                    labels_new = torch.where(labels == 0, pred_old.argmax(dim=1), labels + int(NUM_BASE_CLASSES) + step - 1)
                    labels_pred_old = pred_old.argmax(dim=1)
                    labels = torch.where(labels == 0, 0, labels + int(NUM_BASE_CLASSES) + step - 1)

                    # Uncertainty estimation
                    index_probs, uncertain_old, uncertain_knn = model_old.uncertainty_estimation(
                        ptclouds[:, :3, :].transpose(1, 2), pred_old.transpose(1, 2)
                    )
                    uncertain_old = (uncertain_old.reshape(-1, pred_old.shape[2], 1)).transpose(1, 2).squeeze()

                    # Collect uncertain_old values
                    all_uncertain_values.append(uncertain_old.detach().cpu())

                    # Compute and log uncertain points ratio
                    uncertain_ratio = (uncertain_old > args.uncertain_t).float().mean().item()
                    logger.cprint(f'Batch {i}: Uncertain points ratio: {uncertain_ratio:.4f} (threshold: {args.uncertain_t})')

                    # Reshape uncertain_knn and index_probs
                    uncertain_knn = uncertain_knn.reshape(uncertain_old.shape[0], pred_old.shape[2], -1)[:, :, 1:]
                    _, k_probs, _ = index_probs.shape
                    index_probs = (torch.argmax(index_probs[:, 1:, :], dim=-1)).reshape(-1, pred_old.shape[2], k_probs - 1)
                    index_probs = torch.where(uncertain_knn > args.uncertain_t, 0, index_probs)

                    # New label update logic based on Equation 5
                    labels_new = labels.clone()  # Clone the original labels to avoid in-place modification

                    # Condition 1: labels == 0, predicted class != 0, and uncertainty <= threshold
                    condition1 = (labels == 0) & (labels_pred_old != 0) & (uncertain_old <= args.uncertain_t)
                    labels_new = torch.where(condition1, labels_pred_old, labels_new)

                    # Condition 2: labels == 0, and either predicted class == 0 or uncertainty > threshold
                    condition2 = (labels == 0) & ((labels_pred_old == 0) | (uncertain_old > args.uncertain_t))

                    # For condition 2, use kNN predictions (index_probs) if available
                    for k in range(index_probs.shape[2]):
                        # Check if the k-th kNN prediction is not background and uncertainty condition holds
                        kth_pred = index_probs[:, :, k]
                        kth_condition = condition2 & (kth_pred != 0) & (uncertain_knn[:,:,k] <= args.uncertain_t)
                        labels_new = torch.where(kth_condition, kth_pred, labels_new)


                    # Condition 3: If none of the above, labels_new remains unchanged (already handled by torch.where)

                    # Rest of the code can continue as is
                    labels_new_cls = labels_new

                    if args.dataset == 'scannet':
                        labels_new = torch.where(labels_new_cls==1, 0, labels_new_cls) # For scannet ignore labels 1 : unannotated
                        labels_new_cls = labels_new

                    print('labels_new_cls:',labels_new_cls.unique())
                    #fit时候忽略0
                    AIRMODEL.fit(ptclouds, labels_new_cls, ignore_index=0)
                    # if i>=2:
                    #     break

                AIRMODEL.update()
            logger.cprint('*******************End of Training the Incre Model*******************')
            logger.cprint('*******************Eval the Incre Model*******************')
            pred_end_total = []
            gt_end_total = []

            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(VALID_LOADER):
                    gt_end_total.append(labels.detach())

                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()

                    logits_new_cls=AIRMODEL(ptclouds)
                    logits_new_cls = F.softmax(logits_new_cls, dim=1)
                    #logits_new_cls 是B*N,C

                    BN, _,  = logits_new_cls.shape
                    logits_new_cls_eval = torch.zeros([BN, 1 + NUM_ALL_CLASSES - NUM_BASE_CLASSES]).cuda()
                    logits_new_cls_eval[:, 0] = torch.sum(logits_new_cls[:, :NUM_BASE_CLASSES+1], dim=1)
                    logits_new_cls_eval[:, 1:] = logits_new_cls[:, NUM_BASE_CLASSES+1:]


                    # Compute predictions
                    _, preds = torch.max(logits_new_cls_eval.detach(), dim=1, keepdim=False)
                    # print('preds:',preds.unique())
                    # print('labels:',labels.unique())
                    pred_end_total.append(preds.cpu().detach())



            pred_total = torch.stack(pred_end_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_end_total, dim=0).view(-1, args.pc_npts)
            _, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_ALL_CLASSES - NUM_BASE_CLASSES + 1,
                                                    NUM_ALL_CLASSES - NUM_BASE_CLASSES + 1, 'eval_Incre',
                                                    range(1, len(INCRE_CLASSES)+1), logger, args.dataset)
            logger.cprint('===== End Model | mIoU: %f =====\n' % (mIoU))
            logger.cprint('*******************Start of eval the overall Model*******************')
            if args.dataset == 's3dis':
                from dataloaders.s3dis import S3DISDataset
                DATASET = S3DISDataset(args.cvfold, args.tasks, args.data_path)
                TEST_SET = 'Area_5'
                BASE_CLASSES = DATASET.base_classes
                INCRE_CLASSES = DATASET.incre_classes
                TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES
                print('test classes:', TEST_CLASSES)
            elif args.dataset == 'scannet':
                from dataloaders.scannet import ScanNetDataset
                DATASET = ScanNetDataset(args.cvfold, args.tasks, args.data_path)
                TEST_SET = []
                lines = open(os.path.join(os.path.dirname(args.data_path.rstrip('/')), 'scannetv2_val.txt')).readlines()
                for line in lines:
                    TEST_SET.append(line.strip('\n'))

                BASE_CLASSES = [0] + DATASET.base_classes
                INCRE_CLASSES = DATASET.incre_classes
                TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES
                print('test classes:', TEST_CLASSES)
            else:
                raise NotImplementedError('Unknown dataset %s!' % args.dataset)
            INCRE = int((args.tasks).split('-')[1])
            TOTAL_STEP = int(len(INCRE_CLASSES) / INCRE) + 1

            TEST_DATASET = MyTestDataset(args.data_path, TEST_CLASSES, TOTAL_STEP, test_set=TEST_SET,
                                        num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                        pc_augm=False, pc_augm_config=None)

            TEST_LOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                    drop_last=True)
            #TEST_LOADER 没有背景类的标签而且是从1开始的 s3dis是1到13，scannet是1到21
            pred_total = []
            gt_total = []
            with torch.no_grad():
                for i, (_, ptclouds, labels) in enumerate(TEST_LOADER):
                    # print('labels:',labels.unique())
                    labels = labels-int(1)
                    # print('labels:',labels.unique())
                    gt_total.append(labels.detach())


                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()


                    logits_cls=AIRMODEL(ptclouds)
                    # print('logits_cls:',logits_cls.shape)
                    #logits_cls 是B*N,C 我要去掉第一个背景类

                    logits_cls=logits_cls[:,1:]

                    # 　Compute predictions
                    _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                    pred_total.append(preds.cpu().detach())


            pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
            gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)

            accuracy, mIoU, iou_perclass, base_mIoU, incre_mIoU = metric_evaluate_test(
                pred_total, gt_total, len(TEST_CLASSES), TEST_CLASSES, logger, args.dataset,
                base_classes=BASE_CLASSES, incre_classes=INCRE_CLASSES
            )

            logger.cprint('===== [Test]: Accuracy: %f | mIoU: %f | Base mIoU: %f | Incre mIoU: %f =====\n' %
                        (accuracy, mIoU, base_mIoU, incre_mIoU))
            write_acl_result_summary(
                args.log_dir, args, step, TEST_CLASSES, BASE_CLASSES, INCRE_CLASSES,
                accuracy, mIoU, iou_perclass, base_mIoU, incre_mIoU
            )
            logger.cprint('*******************End of eval the overall Model*******************')

            WRITER.close()
        else:
            # ********************************************************************************************
            # Train the model for incremental classes: step > 0 and tasks > 2 -> Multi-steps Increments
            if step==1:
                logger.cprint('*******************Training the Model on Incremental Classes: %d Classes | Total %d Tasks | Step %d'
                            '*******************' % (len(INCRE_CLASSES), STEP, step))
                # Init the old model
                model_old = DGCNNSeg(args, option='uncertain')
                model_old = load_trained_checkpoint(model_old, args.base_model_checkpoint_path,
                                                    'end_base_model_checkpoint.tar')  # Use the last base outputs
                # print('NUM_BASE_CLASSES + 1',NUM_BASE_CLASSES + 1)
                classifer_old = Classifer(num_classes=NUM_BASE_CLASSES + 1)
                classifer_old = load_trained_checkpoint(classifer_old, args.base_model_checkpoint_path,
                                                    'end_base_model_classifer_checkpoint.tar')
                # Freeze the old model and classifer
                for param in model_old.parameters():
                    param.requires_grad = False
                for param in classifer_old.parameters():
                    param.requires_grad = False

                if torch.cuda.is_available():
                    model_old.cuda()

                    classifer_old.cuda()


                logger.cprint('*******************Start Re-align Base *******************')
                # Re-align the base model
                AIRMODEL= AIR(backbone_output=128, backbone=model_old, buffer_size=5000, gamma=1, device=torch.device("cuda"), dtype=torch.double, linear=GeneralizedARM)
                with torch.no_grad():
                    for i, (ptclouds, labels) in enumerate(TRAIN_BASE_LOADER):
                        # print('ptclouds:', ptclouds.shape)
                        # print('labels:', labels.shape)

                        if torch.cuda.is_available():
                            ptclouds = ptclouds.cuda()
                            labels = labels.cuda()

                        if args.dataset == 'scannet':
                            AIRMODEL.fit(ptclouds, labels, ignore_index=1)
                        else:
                            AIRMODEL.fit(ptclouds, labels)
                        # if i>=2:
                        #     break
                    AIRMODEL.update()
                logger.cprint('*******************End of Re-align Base *******************')
                logger.cprint('*******************Evaluate Re-align Base *******************')
                pred_end_total = []
                gt_end_total = []
                with torch.no_grad():
                    for i, (ptclouds, labels) in enumerate(VALID_BASE_LOADER):
                        gt_end_total.append(labels.detach())

                        if torch.cuda.is_available():
                            ptclouds = ptclouds.cuda()
                            labels = labels.cuda()


                        logits_cls=AIRMODEL(ptclouds)

                        # 　Compute predictions
                        _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                        pred_end_total.append(preds.cpu().detach())


                pred_total = torch.stack(pred_end_total, dim=0).view(-1, args.pc_npts)
                gt_total = torch.stack(gt_end_total, dim=0).view(-1, args.pc_npts)
                accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_BASE_CLASSES + 1, NUM_ALL_CLASSES,
                                                            'eval_Base', BASE_CLASSES, logger, args.dataset)
                logger.cprint('===== Accuracy: %f | mIoU: %f =====\n' % (accuracy, mIoU))
                logger.cprint('*******************End Evaluate Re-align Base *******************')

            logger.cprint('*******************Start of Training the Incre Model*******************')
            AIRMODEL_OLD =AIRMODEL
            all_uncertain_values = []
            with torch.no_grad():
                for i, (ptclouds, labels) in enumerate(TRAIN_LOADER):
                    # print('ptclouds:', ptclouds.shape)
                    # print('labels:', labels.shape)
                    if torch.cuda.is_available():
                        ptclouds = ptclouds.cuda()
                        labels = labels.cuda()
                    pred_old = AIRMODEL_OLD(ptclouds)
                    pred_old=F.softmax(pred_old, dim=1)
                    B,N=labels.shape
                    # print('labels:',labels.unique())
                    #把pred_old的维度由[B*N,C]变为[B,C,N]
                    pred_old = pred_old.reshape(B, -1, N)
                    # labels = torch.where(labels == 0, 0, labels + int(NUM_BASE_CLASSES))
                    # #打印labels的唯一值
                    # print('labels:',labels.unique())
                    #ptclouds B,D,N  pred_old B,C,N
                    #index_probs [B×N, k_probs, C] uncertain_old [B×N] uncertain_knn [B×N, k_probs, 1]
                    # Uncertainty-aware Pseudo-label Generation -> Mixed Labels
                    # Existing code for context
                    labels_new = torch.where(labels == 0, pred_old.argmax(dim=1), labels + int(NUM_BASE_CLASSES) + step - 1)
                    labels_pred_old = pred_old.argmax(dim=1)
                    labels = torch.where(labels == 0, 0, labels + int(NUM_BASE_CLASSES) + step - 1)

                    # Uncertainty estimation
                    index_probs, uncertain_old, uncertain_knn = model_old.uncertainty_estimation(
                        ptclouds[:, :3, :].transpose(1, 2), pred_old.transpose(1, 2)
                    )
                    uncertain_old = (uncertain_old.reshape(-1, pred_old.shape[2], 1)).transpose(1, 2).squeeze()

                    # Collect uncertain_old values
                    all_uncertain_values.append(uncertain_old.detach().cpu())

                    # Compute and log uncertain points ratio
                    uncertain_ratio = (uncertain_old > args.uncertain_t).float().mean().item()
                    logger.cprint(f'Batch {i}: Uncertain points ratio: {uncertain_ratio:.4f} (threshold: {args.uncertain_t})')


                    # Reshape uncertain_knn and index_probs
                    uncertain_knn = uncertain_knn.reshape(uncertain_old.shape[0], pred_old.shape[2], -1)[:, :, 1:]
                    _, k_probs, _ = index_probs.shape
                    index_probs = (torch.argmax(index_probs[:, 1:, :], dim=-1)).reshape(-1, pred_old.shape[2], k_probs - 1)
                    index_probs = torch.where(uncertain_knn > args.uncertain_t, 0, index_probs)

                    # New label update logic based on Equation 5
                    labels_new = labels.clone()  # Clone the original labels to avoid in-place modification

                    # Condition 1: labels == 0, predicted class != 0, and uncertainty <= threshold
                    condition1 = (labels == 0) & (labels_pred_old != 0) & (uncertain_old <= args.uncertain_t)
                    labels_new = torch.where(condition1, labels_pred_old, labels_new)

                    # Condition 2: labels == 0, and either predicted class == 0 or uncertainty > threshold
                    condition2 = (labels == 0) & ((labels_pred_old == 0) | (uncertain_old > args.uncertain_t))

                    # For condition 2, use kNN predictions (index_probs) if available
                    # for k in range(index_probs.shape[2]):
                    #     # Check if the k-th kNN prediction is not background and uncertainty condition holds
                    #     kth_pred = index_probs[:, :, k]
                    #     kth_condition = condition2 & (kth_pred != 0) & (uncertain_knn[:,:,k] <= args.uncertain_t)
                    #     labels_new = torch.where(kth_condition, kth_pred, labels_new)

                    # Condition 3: If none of the above, labels_new remains unchanged (already handled by torch.where)

                    # Rest of the code can continue as is
                    labels_new_cls = labels_new


                    for k in range(index_probs.shape[2]):
                        if (labels_new_cls==0).long().sum([0, 1]) >= 1:
                            if k == 0:

                                labels_new_cls = torch.where((labels_new_cls==0) | (uncertain_old > args.uncertain_t),
                                                            index_probs[:, :, k], labels_new_cls)
                            else:
                                labels_new_cls = torch.where((labels_new_cls ==0),
                                                                index_probs[:, :, k], labels_new_cls)
                        else:
                            break

                    if args.dataset == 'scannet':
                        labels_new = torch.where(labels_new_cls==1, 0, labels_new_cls) # For scannet ignore labels 1 : unannotated
                        labels_new_cls = labels_new

                    print('labels_new_cls:',labels_new_cls.unique())
                    #fit时候忽略0
                    AIRMODEL.fit(ptclouds, labels_new_cls, ignore_index=0)
                    # if i>=2:
                    #     break

                AIRMODEL.update()

            logger.cprint('*******************End of Training the Incre Model*******************')

            if step==STEP-1:
                logger.cprint('*******************Start of eval the overall Model*******************')
                if args.dataset == 's3dis':
                    from dataloaders.s3dis import S3DISDataset
                    DATASET = S3DISDataset(args.cvfold, args.tasks, args.data_path)
                    TEST_SET = 'Area_5'
                    BASE_CLASSES = DATASET.base_classes
                    INCRE_CLASSES = DATASET.incre_classes
                    TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES
                    print('test classes:', TEST_CLASSES)
                elif args.dataset == 'scannet':
                    from dataloaders.scannet import ScanNetDataset
                    DATASET = ScanNetDataset(args.cvfold, args.tasks, args.data_path)
                    TEST_SET = []
                    lines = open(os.path.join(os.path.dirname(args.data_path.rstrip('/')), 'scannetv2_val.txt')).readlines()
                    for line in lines:
                        TEST_SET.append(line.strip('\n'))

                    BASE_CLASSES = [0] + DATASET.base_classes
                    INCRE_CLASSES = DATASET.incre_classes
                    TEST_CLASSES = BASE_CLASSES + INCRE_CLASSES
                    print('test classes:', TEST_CLASSES)
                else:
                    raise NotImplementedError('Unknown dataset %s!' % args.dataset)
                INCRE = int((args.tasks).split('-')[1])
                TOTAL_STEP = int(len(INCRE_CLASSES) / INCRE) + 1

                TEST_DATASET = MyTestDataset(args.data_path, TEST_CLASSES, TOTAL_STEP, test_set=TEST_SET,
                                            num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                            pc_augm=False, pc_augm_config=None)

                TEST_LOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                        drop_last=True)
                #TEST_LOADER 没有背景类的标签而且是从1开始的 s3dis是1到13，scannet是1到21
                pred_total = []
                gt_total = []
                with torch.no_grad():
                    for i, (_, ptclouds, labels) in enumerate(TEST_LOADER):
                        # print('labels:',labels.unique())
                        labels = labels-int(1)
                        # print('labels:',labels.unique())
                        gt_total.append(labels.detach())


                        if torch.cuda.is_available():
                            ptclouds = ptclouds.cuda()
                            labels = labels.cuda()


                        logits_cls=AIRMODEL(ptclouds)
                        # print('logits_cls:',logits_cls.shape)
                        #logits_cls 是B*N,C 我要去掉第一个背景类

                        logits_cls=logits_cls[:,1:]

                        # 　Compute predictions
                        _, preds = torch.max(logits_cls.detach(), dim=1, keepdim=False)
                        pred_total.append(preds.cpu().detach())


                pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
                gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)



                accuracy, mIoU, iou_perclass, base_mIoU, incre_mIoU = metric_evaluate_test(
                    pred_total, gt_total, len(TEST_CLASSES), TEST_CLASSES, logger, args.dataset,
                    base_classes=BASE_CLASSES, incre_classes=INCRE_CLASSES
                )

                logger.cprint('===== [Test]: Accuracy: %f | mIoU: %f | Base mIoU: %f | Incre mIoU: %f =====\n' %
                            (accuracy, mIoU, base_mIoU, incre_mIoU))
                write_acl_result_summary(
                    args.log_dir, args, step, TEST_CLASSES, BASE_CLASSES, INCRE_CLASSES,
                    accuracy, mIoU, iou_perclass, base_mIoU, incre_mIoU
                )
                logger.cprint('*******************End of eval the overall Model*******************')
                WRITER.close()



##评估的话可以直接在这里写，改那个step就可以了 如果有backbone的话
