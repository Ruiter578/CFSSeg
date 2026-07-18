from torch.utils.data import DataLoader
from datasets import VOCSegmentation, ADESegmentation, CityscapesSegmentationIncrementalDomain

from utils import ext_transforms as et
from utils.parser import Config

from torch.utils.data.distributed import DistributedSampler


def read_image_id_set(path, description):
    with open(path, 'r') as handle:
        image_ids = [line.split()[0] for line in handle if line.split()]
    if not image_ids:
        raise ValueError(f"{description} list is empty: {path}")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{description} list contains duplicate image ids: {path}")
    return set(image_ids)


def validate_clean_validation_lists(opts):
    if not opts.validation_list:
        return
    if not opts.train_exclude_list:
        raise ValueError(
            "--validation_list requires --train_exclude_list so held-out samples cannot train the model"
        )
    validation_ids = read_image_id_set(opts.validation_list, 'Independent validation')
    excluded_ids = read_image_id_set(opts.train_exclude_list, 'Training exclusion')
    missing = validation_ids - excluded_ids
    if missing:
        raise ValueError(
            "Training exclusion does not cover every independent validation image "
            f"(missing {len(missing)} ids)"
        )



def get_dataset(opts : Config):
    """ Dataset And Augmentation
    """
    validate_clean_validation_lists(opts)

    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    elif opts.dataset == 'cityscapes_domain':
        dataset = CityscapesSegmentationIncrementalDomain
    else:
        raise NotImplementedError
        
    dataset_dict = {}

    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    
    val_image_set = 'tuning_val' if opts.validation_list else 'val'
    dataset_dict['val'] = dataset(opts=opts,image_set=val_image_set, transform=val_transform, cil_step=opts.curr_step)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    return dataset_dict


def init_dataloader(opts : Config):
    # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    dataset_dict = get_dataset(opts)

    train_loader = DataLoader(
        dataset_dict['train'], shuffle=True, batch_size=opts.batch_size, num_workers=4, pin_memory=True, drop_last=True)
    
    if opts.local_rank == 0:
        val_loader = DataLoader(
            dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(
            dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader, test_loader = None, None
    
    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
        (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    return train_loader, val_loader, test_loader
