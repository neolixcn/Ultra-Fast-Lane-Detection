import torch, os
import numpy as np

import torchvision.transforms as transforms
# Pyten-20210126-AddAlbumTransform
import albumentations as A
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor
from data.dataset_bdd import BddLaneClsDataset, BddLaneTestDataset
from data.dataset import LaneClsDataset, LaneTestDataset

def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed, num_lanes, cfg):
    target_transform = transforms.Compose([
        # Pyten-20200128-ChangeInputSize
        # mytransforms.FreeScaleMask((288, 800)),
        mytransforms.FreeScaleMask((cfg.height, cfg.width)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        # Pyten-20200128-ChangeInputSize
        # mytransforms.FreeScaleMask((36, 100)),
        mytransforms.FreeScaleMask((cfg.height//8, cfg.width//8)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        # Pyten-20200128-ChangeInputSize
        # transforms.Resize((288, 800)),
        transforms.Resize((cfg.height, cfg.width)),
        transforms.ToTensor(),
        # Pyten-20210126-Addnewtransform
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Pyten-20210203-AddAlbumTransform
    albumtransforms = A.Compose([
                                                                    A.HorizontalFlip(p=0.5),
                                                                    A.Blur(blur_limit=3, p=0.5),
                                                                    # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                                                                    # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                                                                    # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                                                                    ])

    simu_transform = mytransforms.Compose2([
        # mytransforms.AlbumAug(albumtransforms),
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    if dataset == 'CULane':
        if "anchors" not in cfg:
            cfg.anchors = culane_row_anchor
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform,
                                           row_anchor = cfg.anchors,
                                           griding_num=griding_num, use_aux=use_aux,num_lanes = num_lanes)
        cls_num_per_lane = 18
    elif dataset == 'Bdd100k':
        if "anchors" not in cfg:
            cfg.anchors = tusimple_row_anchor # culane_row_anchor
        train_dataset = BddLaneClsDataset(data_root,
                                           os.path.join(data_root, 'train.txt'), #'new_train.txt ' #'train.txt' 2000
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num,
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform, use_aux=use_aux, num_lanes = num_lanes, mode = "train")
        cls_num_per_lane = 56 # 18

    elif dataset == 'neolix':
        if "anchors" not in cfg:
            cfg.anchors = tusimple_row_anchor
        train_dataset = LaneClsDataset(data_root,
                                        os.path.join(data_root, 'train.txt'),
                                        img_transform=img_transform, target_transform=target_transform,
                                        simu_transform = simu_transform,
                                        segment_transform=segment_transform,
                                        row_anchor = cfg.anchors,
                                        griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes, extend=False)
        cls_num_per_lane = 56
    
    elif dataset == 'Tusimple':
        if "anchors" not in cfg:
            cfg.anchors = tusimple_row_anchor
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num, 
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform,use_aux=use_aux, num_lanes = num_lanes)
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)

    return train_loader, cls_num_per_lane

# Pyten-20210106-Add_val_loader
def get_val_loader(batch_size, data_root, griding_num, dataset, use_aux, distributed, num_lanes, cfg):
    target_transform = transforms.Compose([
        # Pyten-20200128-ChangeInputSize
        # mytransforms.FreeScaleMask((288, 800)),
        mytransforms.FreeScaleMask((cfg.height, cfg.width)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        # Pyten-20200129-ChangeInputSize
        # transforms.Resize((288, 800)),
        transforms.Resize((cfg.height, cfg.width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    segment_transform = transforms.Compose([
        # Pyten-20200128-ChangeInputSize
        # mytransforms.FreeScaleMask((36, 100)),
        mytransforms.FreeScaleMask((cfg.height//8, cfg.width//8)),
        mytransforms.MaskToTensor(),
    ])
    
    if dataset == 'CULane':
        val_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/val_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           segment_transform=segment_transform,
                                           row_anchor = cfg.anchors,
                                           griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes)

    elif dataset == 'Bdd100k':
        val_dataset = BddLaneClsDataset(data_root,
                                           os.path.join(data_root, 'val.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           griding_num=griding_num,
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform,use_aux=use_aux, num_lanes = num_lanes, mode = "val")
    
    elif dataset == 'neolix':
        val_dataset = LaneClsDataset(data_root,
                                        os.path.join(data_root, 'val.txt'),
                                        img_transform=img_transform, target_transform=target_transform,
                                        simu_transform = None,
                                        segment_transform=segment_transform,
                                        row_anchor = cfg.anchors,
                                        griding_num=griding_num, use_aux=use_aux, num_lanes = num_lanes, extend=False)
    
    elif dataset == 'Tusimple':
        val_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           griding_num=griding_num, 
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform,use_aux=use_aux, num_lanes = num_lanes)
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)

    return val_loader

def get_test_loader(batch_size, data_root,dataset, distributed):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/test.txt'),img_transform = img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'test.txt'), img_transform = img_transforms)
        cls_num_per_lane = 56
    elif dataset == 'Neolix':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'test.txt'), img_transform = img_transforms)
        cls_num_per_lane = 56

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle = False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)
    return loader


class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size


        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)