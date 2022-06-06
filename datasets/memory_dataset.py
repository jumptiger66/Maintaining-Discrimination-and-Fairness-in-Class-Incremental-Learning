import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""
    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.fromarray(self.images[index])
        x = self.transform(x)
        y = self.labels[index]
        return x, y


"""
get_data():
    该函数主要作用就是以class_order的数据集划分为num_tasks个子集，
    划分为 train、val、test 三个数据集。
    以cifar100-bic为例：
        train:10个4500
        val:10个500
        test:10个1000
data:
    data-->dict ,data[0]-->dict
    data[0]['xxx']['x'] --> image
    data[0]['xxx']['y'] --> label

    data[0]['xxx'], data[1]['xxx'], data[2]['xxx'] ... 每个阶段的数据，包括 trn,val,tst
    。。。
    data['ncla'] = 100
taskcla:
    [(0,10), (1,10), (2,10) ...]
class_order:
    [23,12,45,0,3 ...]
"""
# Prepare data: dataset splits, task partition, class order
def get_data(trn_data, tst_data, num_tasks, shuffle_classes, class_order=None):
    """
    :param trn_data: train data
    :param tst_data: test data
    :param num_tasks: total experiences
    :param shuffle_classes: shuffle class order
    :param class_order: fix class order ,dont use when shuffle==True
    :return: data, taskcla, class_order
    """

    data = {}
    taskcla = []
    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order) #total classes-->100(cifar100)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    cpertask = np.array([num_classes // num_tasks] * num_tasks) #array([10,10,10,10,10,10,10,10,10,10])
    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"

    cpertask_cumsum = np.cumsum(cpertask) #array([10,20,30,40,50,60,70,80,90,100])
    init_class = np.concatenate(([0], cpertask_cumsum[:-1])) #array([0,10,20,30,40,50,60,70,80,90])

    # initialize data structure
    for tt in range(num_tasks): #num_tasks:10
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    filtering = np.isin(trn_data['y'], class_order)
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]

    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label) #给定该数据的label在class_order中的顺序
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum() #该数据分配在第几个experience中
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task]) #！！！每一个任务内的标签都是0-10，除非label也分配到具体任务内，否则如何做损失？

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]

    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y'])) #每个experience的类别数
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # convert them to numpy arrays
    for tt in data.keys():
        for split in ['trn','tst']:
            data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n #100

    return data, taskcla, class_order
