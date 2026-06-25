import os
import time
import json
import copy
import numpy as np
from datetime import datetime

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from metrics import *
from network import *
from utils import *
from datasets import *
from utils.run_manifest import write_run_manifest

class AIR(nn.Module):
    def __init__(
        self,
        backbone_output,
        backbone,
        buffer_size,
        gamma,
        feature_source="decoder",
        device=None,
        dtype=torch.double,
        linear=RecursiveLinear,
        learned_classes=None,
        rhl_norm="none",
        rhl_norm_eps=1e-6,
        rhl_seed=-1,
        rhl_stats=False,
    ):
        super(AIR, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.feature_source = feature_source
        # RandomBuffer 是 CFSSeg/RHL 的固定随机特征映射：
        # - rhl_norm 只改变送入 RecursiveLinear 前的特征尺度；
        # - rhl_seed 只改变这一个随机映射的初始化，用于 RHL-SE 多成员构造；
        # - 二者都不引入可训练参数，也不改 C-RLS 的闭式递推公式。
        self.buffer = RandomBuffer(
            backbone_output,
            buffer_size,
            rhl_norm=rhl_norm,
            rhl_norm_eps=rhl_norm_eps,
            rhl_seed=rhl_seed,
            **factory_kwargs,
        )
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.H = 0
        self.W = 0
        self.channle = 0
        self.B = 0
        self.rhl_stats = rhl_stats
        self.rhl_stats_count = 0
        self.rhl_stats_max_batches = 3
        self.eval()
    
    @torch.no_grad()
    def feature_expansion(self, X: torch.Tensor):
        feature_source = getattr(self, "feature_source", "decoder")
        X = self.backbone.forward_air_features(X, feature_source)
        self.B, self.channle, self.H, self.W = X.shape
        # 16 256 33 33
        X = X.view(self.B,self.channle,-1).permute(0,2,1) # B, H*W, c
        X = self.buffer(X) # B, H*W, C -> B, H*W, buffer_size
        if self.rhl_stats and self.rhl_stats_count < self.rhl_stats_max_batches:
            row_norm = X.norm(dim=-1)
            print(
                "[RHL stats] "
                f"mode={getattr(self.buffer, 'rhl_norm', 'none')} "
                f"eps={getattr(self.buffer, 'rhl_norm_eps', 1e-6)} "
                f"mean={row_norm.mean().item():.6f} "
                f"std={row_norm.std().item():.6f} "
                f"min={row_norm.min().item():.6f} "
                f"max={row_norm.max().item():.6f} "
                f"nan={torch.isnan(X).any().item()} "
                f"inf={torch.isinf(X).any().item()}"
            )
            self.rhl_stats_count += 1
        return X
    
    @torch.no_grad()
    def forward(self, X):
        return self.analytic_linear(self.feature_expansion(X))
    
    @torch.no_grad()
    def fit(self, X, y, *args, **kwargs):
        X = self.feature_expansion(X)   # X: B, H*W, buffer_size
        y = y.unsqueeze(1).float()  # y: B, 1, h, w
        y = F.interpolate(y, size=(self.H, self.W), mode='nearest').long()
        self.analytic_linear.fit(X, y)
    
    @torch.no_grad()
    def update(self):
        self.analytic_linear.update()

class Trainer(object):
    @staticmethod
    def make_step0_loader_opts(opts):
        step0_opts = copy.deepcopy(opts)
        step0_opts.curr_step = 0
        return step0_opts

    @staticmethod
    def resolve_resumed_air_feature_source(model, requested_source):
        checkpoint_source = getattr(model, "feature_source", "decoder")
        if requested_source != "auto" and requested_source != checkpoint_source:
            raise ValueError(
                f"Requested AIR feature source '{requested_source}', but the "
                f"checkpoint uses '{checkpoint_source}'"
            )
        return checkpoint_source

    def __init__(self, opts: Config) -> None:
        super(Trainer, self).__init__()

        self.opts = opts
        self.local_rank = 0
        self.curr_idx = [
            sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
            sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
        ]

        self.device = 'cuda:0' if opts.gpu_id != None else 'cpu'
        self.root_path = f"checkpoints/{opts.subpath}/{opts.dataset}/{self.opts.task}/{opts.setting}/step{opts.curr_step}/"
        self.ckpt_str = f"{self.root_path}%s_%s_%s_step_%d_{opts.setting}.pth"

        prev_subpath = opts.base_subpath if opts.base_subpath and opts.curr_step == 1 else opts.subpath
        self.root_path_prev = f"checkpoints/{prev_subpath}/{opts.dataset}/{self.opts.task}/{opts.setting}/step{opts.curr_step-1}/"
        self.ckpt_str_prev = f"{self.root_path_prev}%s_%s_%s_step_%d_{opts.setting}.pth"
        mkdir(self.root_path)

        self.train_loader, self.val_loader, self.test_loader = init_dataloader(opts)
        self.total_itrs = self.opts.train_epoch * len(self.train_loader)
        self.val_interval = max(100, self.total_itrs // 100)
        print(f"train epoch : {self.opts.train_epoch} , iterations : {self.total_itrs} , val_interval : {self.val_interval}")
        
        # init model
        if opts.curr_step == 0:
            self.model_factory = DeepLabModelFactory()
            self.init_models()
            self.init_ckpt()
            self.optimizer = self.init_optimizer()
            self.scheduler = build_scheduler(opts, self.optimizer, self.total_itrs)
            self.criterion = build_criterion(opts)
            write_run_manifest(
                output_dir=self.root_path,
                opts=self.opts,
                requested_air_feature_source=self.opts.air_feature_source,
                resolved_air_feature_source=None,
                base_checkpoint_path=self.opts.ckpt,
            )

        
        # previous step checkpoint
        if self.opts.curr_step <= 1:  
            self.ckpt = self.ckpt_str_prev % (self.opts.model, self.opts.dataset, self.opts.task, self.opts.curr_step - 1)
        elif self.opts.curr_step > 1:
            self.ckpt = self.root_path_prev + "final.pth"
        if self.opts.curr_step == 1 and self.opts.base_subpath:
            print(f"Loading step0 checkpoint from base_subpath '{self.opts.base_subpath}': {self.ckpt}")


        self.best_score = -1

        # Set up metrics
        self.metrics = StreamSegMetrics(opts.num_classes, dataset=opts.dataset)
        self.avg_loss = AverageMeter()
        self.avg_time = AverageMeter()
        self.avg_loss_std = AverageMeter()

        self.logger = Logger(self.root_path)

        if opts.use_pseudo_label:
            print("use Pseudo Labeling")

    def init_models(self):
        # Set up model
        model_map = self.model_factory.model_map
        print(f"Category components: {self.opts.num_classes}")
        model_constructor = model_map.get(self.opts.model)
        if not model_constructor:
            raise ValueError(f"model named '{self.opts.model}' is not exist!")

        self.model = model_constructor(
            num_classes=self.opts.num_classes,
            output_stride=self.opts.output_stride,
            pretrained_backbone=self.opts.pretrained_backbone,
            bn_freeze=self.opts.bn_freeze
        )

        if self.opts.separable_conv and 'plus' in self.opts.model:
            convert_to_separable_conv(self.model.classifier)

        set_bn_momentum(self.model.backbone, momentum=0.01)

        self.model = self.model.to(self.device)
        

    def init_ckpt(self):
        if self.opts.curr_step > 0:  # previous step checkpoint
            self.ckpt = self.ckpt_str_prev % (self.opts.model, self.opts.dataset, self.opts.task, self.opts.curr_step - 1)
        elif self.opts.curr_step > 1:
            self.ckpt = self.root_path_prev + "final.pth"
        else:
            return
        # check the .pth file
        if not os.path.isfile(self.ckpt):
            raise FileNotFoundError(f"Checkpoint file '{self.ckpt}' does not exist.")
        try:
            # load checkpoint
            checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))
            model_state = checkpoint.get("model_state")
            if model_state is None:
                raise KeyError("The checkpoint does not contain 'model_state' key.")

            model_state_filtered = {k: v for k, v in model_state.items() if not k.startswith('classifier.head')}
            self.model.load_state_dict(model_state_filtered, strict=False)

            print(f"Model restored from {self.ckpt}")
        except (KeyError) as e:
            raise RuntimeError(f"Error loading checkpoint from '{self.ckpt}': {e}")


    def init_optimizer(self):
        training_params = [{'params': self.model.backbone.parameters(), 'lr': 0.001},
                        {'params': self.model.classifier.parameters(), 'lr': 0.01}]
        optimizer = torch.optim.SGD(params=training_params, 
                                    lr=self.opts.lr, 
                                    momentum=0.9, 
                                    weight_decay=self.opts.weight_decay,
                                    nesterov=True)
        return optimizer
    

    def train(self):
        # =====  Train  =====
        if self.opts.curr_step == 0:
            for epoch in range(self.opts.train_epoch):
                self.model.train()
                for seq, (images, labels, _) in enumerate(self.train_loader):
                    images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                    labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    end_time = time.time()
                    
                    outputs, _ = self.model(images)

                    outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)
                    loss = self.criterion(outputs, labels)
                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()

                    self.avg_loss.update(loss.item())
                    self.avg_time.update(time.time() - end_time)
                    self.avg_loss_std.update(loss.item())
                    
                    if seq % 10 == 0:
                        print(f"[{self.opts.task} / step {self.opts.curr_step}] "
                            f"Epoch {epoch}, Itrs {seq}/{len(self.train_loader)}, "
                            f"Loss={self.avg_loss.avg:.4f}, StdLoss={self.avg_loss_std.avg:.4f}, "
                            f"Time={self.avg_time.avg*1000:.2f} ms, "
                            f"LR={self.optimizer.param_groups[0]['lr']:.8f}")
                        
                        self.logger.write_loss(self.avg_loss.avg, epoch * len(self.train_loader) + seq + 1)

                if len(self.train_loader) > 100 or epoch % 5 == 4:
                    print("[Validation]")
                    val_score = self.validate()
                    print(self.metrics.to_str_val(val_score))
                    
                    class_iou = list(val_score['Class IoU'].values())
                    val_score_mean = np.mean([class_iou[i] for i in range(self.curr_idx[0], self.curr_idx[1])] + [class_iou[0]])
                    curr_score = np.mean([class_iou[i] for i in range(self.curr_idx[0], self.curr_idx[1])])
                    print(f"curr_val_score : {curr_score:.4f}\n")
                    self.logger.write_score(curr_score, epoch)
                    
                    if curr_score > self.best_score:
                        print(f"... save best ckpt : {curr_score}")
                        self.best_score = curr_score
                        save_ckpt(self.ckpt_str % (self.opts.model, self.opts.dataset, self.opts.task, self.opts.curr_step), 
                                self.model, self.optimizer, self.best_score)
                save_ckpt(self.root_path + "final.pth", self.model, self.optimizer, curr_score)
        elif self.opts.curr_step == 1:
            step0_opts = self.make_step0_loader_opts(self.opts)
            self.train_loader0, self.val_loader0, self.test_loader0 = init_dataloader(step0_opts)
            self.root_path0 = f"checkpoints/{self.opts.subpath}/{self.opts.dataset}/{self.opts.task}/{self.opts.setting}/step0/"
            mkdir(self.root_path0)
            self.model = load_ckpt(self.ckpt)[0]
            self.model = self.model.to(self.device)
            print("make new model!")
            backbone = self.model
            requested_feature_source = self.opts.air_feature_source
            resolved_feature_source = backbone.resolve_air_feature_source(
                requested_feature_source
            )
            print(
                "AIR feature source: "
                f"requested={requested_feature_source}, resolved={resolved_feature_source}"
            )
            write_run_manifest(
                output_dir=self.root_path,
                opts=self.opts,
                requested_air_feature_source=requested_feature_source,
                resolved_air_feature_source=resolved_feature_source,
                base_checkpoint_path=self.ckpt,
            )
            # Overwrite self.model with ACIL model
            # step1 构造 AIR 时把命令行中的 RHL 控制项接进来。
            # 这里是方案一真正进入训练逻辑的位置：backbone 和全局随机种子保持不变，
            # 只有 RandomBuffer 的固定随机映射会根据 opts.rhl_seed 发生变化。
            self.model = AIR(
                backbone_output=256,
                backbone = backbone,
                buffer_size=self.opts.buffer,
                gamma=self.opts.gamma,
                feature_source=resolved_feature_source,
                device=self.device,
                dtype=torch.double,
                linear=RecursiveLinear,
                rhl_norm=self.opts.rhl_norm,
                rhl_norm_eps=self.opts.rhl_norm_eps,
                rhl_seed=self.opts.rhl_seed,
                rhl_stats=self.opts.rhl_stats,
            ).to(self.device).eval()
            for seq, (X, y, _) in enumerate(self.train_loader0):
                X, y = X.to(self.device), y.to(self.device)
                self.model.fit(X, y)
            self.model.update()
            print("start test!")
            save_ckpt(self.root_path0 + "final.pth", self.model, None, None)
            del self.model
            self.do_evaluate_after_realign(mode='test')

            print("start training")
            torch.cuda.empty_cache()

            for _, (X, y, _) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                if (self.opts.use_pseudo_label and self.opts.curr_step > 1 and self.opts.setting!='sequential'):
                    y=self.get_pseudo_labels(X, y)
                self.model.fit(X, y)
            self.model.update()
            save_ckpt(self.root_path + "final.pth", self.model, None, None)
            self.do_evaluate(mode='test')
        else:
            self.model = load_ckpt(self.ckpt)[0].to(self.device).eval()
            self.model_prev = load_ckpt(self.ckpt)[0].to(self.device).eval()
            resolved_feature_source = self.resolve_resumed_air_feature_source(
                self.model,
                self.opts.air_feature_source,
            )
            write_run_manifest(
                output_dir=self.root_path,
                opts=self.opts,
                requested_air_feature_source=self.opts.air_feature_source,
                resolved_air_feature_source=resolved_feature_source,
                base_checkpoint_path=self.ckpt,
            )

            for _, (X, y, _) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                if (self.opts.use_pseudo_label and self.opts.curr_step > 1 and self.opts.setting!='sequential'):
                        y=self.get_pseudo_labels(X, y)
                self.model.fit(X, y)
            self.model.update()

            save_ckpt(self.root_path + "final.pth", self.model, None, None)
            self.do_evaluate(mode='test')

    def do_evaluate(self, mode='val'):
        print("[Testing Best Model]")
        best_ckpt = self.root_path+"final.pth"
        del self.model
        self.model=load_ckpt(best_ckpt)[0].to(self.device).eval()
        """Do validation and return specified samples"""
        self.metrics.reset()
        with torch.no_grad():
            for i, (images, labels, _ ) in enumerate(tqdm(self.val_loader if mode=='val' else self.test_loader)):
                
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs = self.model(images)

                if self.opts.loss_type == 'bce_loss':
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=1)
                outputs = outputs.permute(0,3,1,2)
                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')

                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                
                self.metrics.update(targets, preds)
                    
            test_score = self.metrics.get_results()
            
        print(self.metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(self.opts.dataset, self.opts.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))

        test_score[f'0 to {first_cls-1} mIoU'] = np.mean(class_iou[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mIoU'] = np.mean(class_iou[first_cls:])
        test_score[f'0 to {first_cls-1} mAcc'] = np.mean(class_acc[:first_cls])
        test_score[f'{first_cls} to {len(class_iou)-1} mAcc'] = np.mean(class_acc[first_cls:])

        # save results as json
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{self.root_path}/test_results_{current_time}.json", 'w') as f:
            f.write(json.dumps(test_score, indent=4))
            f.close()


    def do_evaluate_after_realign(self, mode='val'):
        print("[Testing Best Model]")
        best_ckpt = self.root_path0+"final.pth"
        
        self.model = load_ckpt(best_ckpt)[0].to(self.device).eval()
        if self.opts.use_pseudo_label:
            self.model_prev = load_ckpt(self.ckpt)[0].to(self.device).eval()
        
        """Do validation and return specified samples"""
        self.metrics.reset()
        with torch.no_grad():
            for i, (images, labels, _ ) in enumerate(tqdm(self.val_loader if mode=='val' else self.test_loader)):
                
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs = self.model(images)

                if self.opts.loss_type == 'bce_loss':
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=1)
                outputs=outputs.permute(0,3,1,2)
                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')

                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                
                self.metrics.update(targets, preds)
            test_score = self.metrics.get_results()

        print(self.metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(self.opts.dataset, self.opts.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        test_score[f'0 to {first_cls-1} mIoU'] = np.mean(class_iou[:first_cls])
        test_score[f'0 to {first_cls-1} mAcc'] = np.mean(class_acc[:first_cls])

        # save results as json
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{self.root_path0}/test_results_{current_time}.json", 'w') as f:
            f.write(json.dumps(test_score, indent=4))
            f.close()



    def validate(self, mode='val'):
        """Do validation and return specified samples"""
        self.metrics.reset()
        self.model.eval()

        with torch.no_grad():
            for i, (images, labels, _) in enumerate(tqdm(self.val_loader if mode=='val' else self.test_loader)):
                
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                
                outputs, _ = self.model(images)
                
                if self.opts.loss_type == 'bce_loss':
                    outputs = torch.sigmoid(outputs)
                else:
                    outputs = torch.softmax(outputs, dim=1)

                outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')
                
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                self.metrics.update(targets, preds)
                    
            score = self.metrics.get_results()
        return score

    def get_pseudo_labels(self, images, labels):
        with torch.no_grad():
            
            images = images.to(self.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
            
            outputs= self.model_prev(images)

            if self.opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            outputs=outputs.permute(0,3,1,2)
            outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)

            pred_scores, pred_labels = torch.max(outputs, dim=1)
            pseudo_labels= torch.where(
                (labels==0) & (pred_labels>0) & (pred_scores >= self.opts.pseudo_label_confidence), 
                pred_labels, 
                labels)
            return pseudo_labels
