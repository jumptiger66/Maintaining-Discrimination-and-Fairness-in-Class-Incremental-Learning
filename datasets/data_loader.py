from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100 as TorchVisionCIFAR100

from . import memory_dataset as memd
from .dataset_config import dataset_config


def get_loaders(datasets, num_tasks,batch_size, num_workers):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load,tst_load = [],[]
    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # transformations
        trn_transform, tst_transform = get_transforms(resize=dc['resize'],
                                                      pad=dc['pad'],
                                                      crop=dc['crop'],
                                                      flip=dc['flip'],
                                                      normalize=dc['normalize'],
                                                      extend_channel=dc['extend_channel'])

        # datasets
        trn_dset, tst_dset, taskcla = get_datasets(cur_dataset, dc['path'], num_tasks,trn_transform=trn_transform,
                                                                tst_transform=tst_transform,class_order=dc['class_order'])

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers))
    return trn_load, tst_load, taskcla


def get_datasets(dataset, path, num_tasks,trn_transform, tst_transform, class_order=None):
    """Extract datasets and create Dataset class"""

    trn_dset,tst_dset = [], []

    if 'cifar100' in dataset:
        tvcifar_trn = TorchVisionCIFAR100(path, train=True, download=True) #50000个训练和测试数据
        tvcifar_tst = TorchVisionCIFAR100(path, train=False, download=True) #10000个训练和测试数据
        trn_data = {'x': tvcifar_trn.data, 'y': tvcifar_trn.targets} # trn_data: {'x': (50000,32,32,3)维的数据 ; 'y': (50000,)维的标签}
        tst_data = {'x': tvcifar_tst.data, 'y': tvcifar_tst.targets} # tst_data: {'x': (10000,32,32,3)维的数据 ; 'y': (10000,)维的标签}
        # compute splits
        all_data, taskcla, class_indices = memd.get_data(trn_data, tst_data,num_tasks=num_tasks,
                                                         shuffle_classes=class_order is None, class_order=class_order)
        #all_data: {dict:11}

        # set dataset type
        Dataset = memd.MemoryDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], trn_transform, class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], tst_transform, class_indices))
        offset += taskcla[task][1]

    return trn_dset,tst_dset, taskcla


def get_transforms(resize, pad, crop, flip, normalize, extend_channel):
    """Unpack transformations and apply to train or test splits"""

    trn_transform_list = []
    tst_transform_list = []

    # resize
    if resize is not None:
        trn_transform_list.append(transforms.Resize(resize))
        tst_transform_list.append(transforms.Resize(resize))

    # padding
    if pad is not None:
        trn_transform_list.append(transforms.Pad(pad))
        tst_transform_list.append(transforms.Pad(pad))

    # crop
    if crop is not None:
        trn_transform_list.append(transforms.RandomResizedCrop(crop))
        tst_transform_list.append(transforms.CenterCrop(crop))

    # flips
    if flip:
        trn_transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    trn_transform_list.append(transforms.ToTensor())
    tst_transform_list.append(transforms.ToTensor())

    # normalization
    if normalize is not None:
        trn_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        tst_transform_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))

    # gray to rgb
    if extend_channel is not None:
        trn_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))
        tst_transform_list.append(transforms.Lambda(lambda x: x.repeat(extend_channel, 1, 1)))

    return transforms.Compose(trn_transform_list), \
           transforms.Compose(tst_transform_list)
