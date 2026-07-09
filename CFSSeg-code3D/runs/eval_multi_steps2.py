""" Evaluate the joint model and 2-step incremental model of class-incremental 3D point cloud semantic segmentation """
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from models.dgcnn_seg import DGCNNSeg, Classifer
from models.dgcnn_seg_joint import DGCNNSeg as DGCNNSeg_Joint
from dataloaders.loader import MyTestDataset
from dataloaders.joint_loader import MyTestDataset as MyTestDataset_Joint
import torch.nn.functional as F
from utils.logger import init_logger
from utils.checkpoint_util import load_trained_checkpoint

def metric_evaluate(predicted_label, gt_label, NUM_CLASS, class_id, logger, dataset, base_classes=None, incre_classes=None):
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

    print('base_classes',base_classes)
    print('incre_classes',incre_classes)
    
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

def eval(args):
    logger = init_logger(args.log_dir, args)

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
    # TEST_CLASSES = TEST_CLASSES[0:10]
    TEST_DATASET = MyTestDataset(args.data_path, TEST_CLASSES, TOTAL_STEP, test_set=TEST_SET,
                                num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                pc_augm=False, pc_augm_config=None)

    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                            drop_last=True)      
    #TEST_LOADER 没有背景类的标签而且是从1开始的 s3dis是1到13，scannet是1到21
    model = DGCNNSeg(args)
    model = load_trained_checkpoint(model, args.model_checkpoint_path, name=f'end_incre_step_{TOTAL_STEP-1}_model_checkpoint.tar')
    # model=load_trained_checkpoint(model, args.model_checkpoint_path, name=f'end_base_model_checkpoint.tar')
    model.eval()
    if args.phase == "eval_finetuning_or_freeze_and_add_or_ewc":

        #创建一个列表存储base以及各个step的classifer
        classifer_list = []
        # print('TOTAL_STEP',TOTAL_STEP)
        for step in range(TOTAL_STEP):
            if step == 0:
                classifer_base = Classifer(num_classes = len(BASE_CLASSES) + 1)
                classifer_base = load_trained_checkpoint(classifer_base, args.model_checkpoint_path, name=f'end_base_model_classifer_checkpoint.tar')
                classifer_list.append(classifer_base)
            else:
                classifer_incre = Classifer(num_classes=INCRE + 1)
                classifer_incre = load_trained_checkpoint(classifer_incre, args.model_checkpoint_path, name=f'end_incre_step_{step}_model_classifer_checkpoint.tar')
                classifer_list.append(classifer_incre)
        #用enroll_weights将classifer_list中的classifer进行融合
     
        for i in range(0, len(classifer_list)):
            # classifer_list[i-1].classifer_weights = torch.empty_like(classifer_list[i-1].classifer_weights).copy_(classifer_list[i-1].classifer_weights.detach())
            if i==0:
                classifer_weights = classifer_list[i].classifer_weights[1:, :, :]
            else:
                classifer_weights = torch.cat((classifer_weights,classifer_list[i].classifer_weights[1:, :, :]),dim=0)
        classifer=Classifer(num_classes=len(BASE_CLASSES) + len(INCRE_CLASSES),initial_old_classifer_weights=classifer_weights,save_backgroudclass=True)
    elif args.phase == "eval_lwF":
        classifer=Classifer(num_classes=len(BASE_CLASSES) + len(INCRE_CLASSES))
        classifer=load_trained_checkpoint(classifer, args.model_checkpoint_path, name=f'end_incre_step_{TOTAL_STEP-1}_model_classifer_checkpoint.tar')
        # classifer=Classifer(num_classes=len(BASE_CLASSES)+1)
        # classifer=load_trained_checkpoint(classifer, args.model_checkpoint_path, name=f'end_base_model_classifer_checkpoint.tar')
        # classifer.classifer_weights=nn.Parameter(classifer.classifer_weights[1:,:,:],requires_grad=False)
   
        
        
       


    model.cuda()
    classifer.cuda()
    pred_total = []
    gt_total = []
    with torch.no_grad():
        for i, (_, ptclouds, labels) in enumerate(TEST_LOADER):
            labels = labels - int(1)
            
            # print('labels',labels.unique())
            gt_total.append(labels.detach())

            if torch.cuda.is_available():
                ptclouds = ptclouds.cuda()
                labels = labels.cuda()
            model.eval()
            _, logits = model(ptclouds)
            logits_new = classifer(logits)
            # print('logits_new',logits_new.shape)
            # print('labels',labels.unique())
            loss = F.cross_entropy(logits_new, labels)

            # Compute predictions
            _, preds = torch.max(logits_new.detach(), dim=1, keepdim=False)
            pred_total.append(preds.cpu().detach())

            logger.cprint(
                '=====[Test] Iter: %d | Loss: %.4f =====' % (i, loss.item()))

    pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
    gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)
    # 修改调用部分，在eval函数中:
    accuracy, mIoU, iou_perclass, base_mIoU, incre_mIoU = metric_evaluate(
        pred_total, gt_total, len(TEST_CLASSES), TEST_CLASSES, logger, args.dataset, 
        base_classes=BASE_CLASSES, incre_classes=INCRE_CLASSES
    )

    logger.cprint('===== [Test]: Accuracy: %f | mIoU: %f | Base mIoU: %f | Incre mIoU: %f =====\n' % 
                (accuracy, mIoU, base_mIoU, incre_mIoU))
