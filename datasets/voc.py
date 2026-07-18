import os
import torch.utils.data as data
import numpy as np

from PIL import Image

from utils.tasks import get_dataset_list, get_tasks
from utils.parser import Config


def read_image_id_list(path, description):
    if not path:
        raise ValueError(f"{description} requires a list path.")
    with open(path, 'r') as handle:
        image_ids = [line.split()[0] for line in handle if line.split()]
    if not image_ids:
        raise ValueError(f"{description} list is empty: {path}")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{description} list contains duplicate image ids: {path}")
    return image_ids


def exclude_image_ids(file_names, excluded_ids):
    excluded = set(excluded_ids)
    return [image_id for image_id in file_names if image_id not in excluded]

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(0, N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap[254] = np.array([0, 0, 0])       # bg
    cmap[255] = np.array([0, 0, 0])       # bg
    cmap[200] = np.array([192, 192, 192])       # unknown
    cmap = cmap/255 if normalized else cmap
    return cmap


class VOCSegmentation(data.Dataset):
    cmap = voc_cmap()
    
    def __init__(self,
                 opts : Config,
                 image_set='train',
                 transform=None,
                 cil_step=0):

        self.root=opts.data_root        
        self.task=opts.task
        self.overlap=opts.overlap
        self.unknown=opts.unknown
        self.setting = opts.setting
        self.image_set = image_set
        self.transform = transform
        
        image_dir = os.path.join(self.root, 'JPEGImages')

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or corrupted.')
        
        mask_dir = os.path.join(self.root, 'SegmentationClassAug')

        assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            
        self.target_cls = get_tasks('voc', self.task, cil_step)
        self.target_cls += [255] # including ignore index (255)
        
        if image_set=='test':
            file_names = open(os.path.join(self.root, 'ImageSets/Segmentation', 'val.txt'), 'r')
            file_names = file_names.read().splitlines()
        elif image_set == 'tuning_val':
            file_names = read_image_id_list(
                opts.validation_list,
                'Independent validation',
            )
        else:
            file_names = get_dataset_list('voc', self.task, cil_step, image_set, self.setting)
            if image_set == 'train' and opts.train_exclude_list:
                before_count = len(file_names)
                file_names = exclude_image_ids(
                    file_names,
                    read_image_id_list(opts.train_exclude_list, 'Training exclusion'),
                )
                print(
                    f"VOC train exclusion: removed {before_count - len(file_names)} "
                    f"of {before_count} task-filtered samples"
                )
                if not file_names:
                    raise ValueError("Training exclusion removed all task-filtered VOC samples")
            
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        self.file_names = file_names
        
        # class re-ordering
        all_steps = get_tasks('voc', self.task)
        all_classes = []
        for i in range(len(all_steps)):
            all_classes += all_steps[i]
            
        self.ordering_map = np.zeros(256, dtype=np.uint8) + 255 # [255, 255, ..., 255]
        self.ordering_map[:len(all_classes)] = [all_classes.index(x) for x in range(len(all_classes))] # [0, 1, 2, ..., 255]

        assert (len(self.images) == len(self.masks))

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        file_name = self.file_names[index]
       
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        
        # re-define target label according to the CIL case
        target = self.gt_label_mapping(target)
        
        if self.transform is not None:
            img, target = self.transform(img, target)
            
        return img, target.long(), file_name


    def __len__(self):
        return len(self.images)

    def gt_label_mapping(self, gt):
        gt = np.array(gt, dtype=np.uint8)
        if self.image_set != 'test':
            if self.setting == 'sequential':
                pass
            else:
                gt = np.where(np.isin(gt, self.target_cls), gt, 0)

        gt = self.ordering_map[gt]
        gt = Image.fromarray(gt)
        
        return gt
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
