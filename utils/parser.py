import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Dataset Options
    data_root: str = 'datasets/data/voc/VOC2012'
    dataset: str = 'voc'
    num_classes: Optional[int] = None

    # Deeplab Options
    model: Optional[str] = None
    separable_conv: bool = False
    output_stride: int = 16
    pretrained_backbone: bool = False

    # Train Options
    test_only: bool = False
    train_epoch: int = 50
    curr_itrs: int = 0
    lr: float = 0.01
    lr_policy: str = 'warm_poly'
    step_size: int = 10000
    crop_val: bool = True
    batch_size: int = 32
    val_batch_size: int = 1
    crop_size: int = 513

    ckpt: Optional[str] = None
    unknown: int = 0

    loss_type: str = 'bce_loss'
    
    weight_decay: float = 1e-4
    random_seed: int = 1
    print_interval: int = 10
    val_interval: int = 100

    subpath: Optional[str] = None
    
    # GPU
    gpu_id: list[int] = None
    local_rank: int = 0

    # CIL Options
    task: str = '15-1'
    curr_step: int = 0
    overlap: bool = False
    setting: str = 'overlap'
    bn_freeze: bool = False
    cil_step: int = 0
    initial: bool = False
    target_cls: dict = None
    
    buffer: int = 16384
    gamma: float = 10.0
    method: str = 'None'

    use_pseudo_label: bool = False
    pseudo_label_confidence: float = 0.7


def get_argparser() -> Config:
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default=Config.data_root, help="path to Dataset")
    parser.add_argument("--dataset", type=str, default=Config.dataset, choices=['voc', 'ade', 'cityscapes_domain'], help="Name of dataset")
    parser.add_argument("--num_classes", type=int, default=Config.num_classes, help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=Config.separable_conv, help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=Config.output_stride, choices=[8, 16])
    parser.add_argument("--pretrained_backbone", action='store_true', default=Config.pretrained_backbone)

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=Config.test_only)
    parser.add_argument("--train_epoch", type=int, default=Config.train_epoch, help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=Config.curr_itrs)
    parser.add_argument("--lr", type=float, default=Config.lr, help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default=Config.lr_policy, choices=['poly', 'step', 'warm_poly'], help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=Config.step_size)
    parser.add_argument("--crop_val", action='store_true', default=Config.crop_val, help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=Config.val_batch_size, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=Config.crop_size)

    parser.add_argument("--ckpt", type=str, default=Config.ckpt, help="restore from checkpoint")
    parser.add_argument("--unknown", type=int, default=Config.unknown, help='unknown 属性的默认值')

    parser.add_argument("--loss_type", type=str, default=Config.loss_type, choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")

    parser.add_argument(
        "--gpu_id",
        type=int,
        nargs='+',
        default=[0],
        help="GPU ID"
    )

    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay, help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=Config.random_seed, help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=Config.print_interval, help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=Config.val_interval, help="epoch interval for eval (default: 100)")

    parser.add_argument("--subpath", type=str, default=Config.subpath, help="subpath for saving ckpt and log")

    # CIL Options
    parser.add_argument("--task", type=str, default=Config.task, help="cil task")
    parser.add_argument("--curr_step", type=int, default=Config.curr_step)
    parser.add_argument("--overlap", action='store_true', default=Config.overlap, help="overlap setup (True), disjoint setup (False)")
    parser.add_argument("--setting", type=str, default=Config.setting, choices=['sequential', 'disjoint', 'overlap'], help="continual learning setting")
    parser.add_argument("--bn_freeze", action='store_true', default=Config.bn_freeze, help="enable batchnorm freezing")
    parser.add_argument("--cil_step", type=int, default=Config.cil_step, help="cil step")
    parser.add_argument("--initial", action='store_true', default=Config.initial, help="initial training")

    parser.add_argument("--buffer", type=int, default=Config.buffer, help="buffer size")
    parser.add_argument("--gamma", type=float, default=Config.gamma, help="gamma value")
    parser.add_argument("--method", type=str, default=Config.method, help="method value")

    parser.add_argument("--use_pseudo_label", action='store_true', default=Config.use_pseudo_label, help="is use pseudo label")
    parser.add_argument("--pseudo_label_confidence", type=float, default=Config.pseudo_label_confidence, help="use the label confidence")
    args = parser.parse_args()
    return Config(**vars(args))

def main():
    config = get_argparser()
    print(config)

if __name__ == "__main__":
    main()