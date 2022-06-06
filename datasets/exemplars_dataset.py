import importlib
from argparse import ArgumentParser

from .memory_dataset import MemoryDataset

"""
该模块
    1. 有用的两个参数 self.max_num_exemplars（样例集数据量） 和 self.exemplars_selector（采样策略）
    2. 两个额外参数：保留数量-> num-exemplars和 采样保留集策略-> exemplar-selection
    3. 一个方法：collect_exemplars: 输入通过选择的采样策略返回采样数据（images,labels）
"""
class ExemplarsDataset(MemoryDataset):
    """Exemplar storage for approaches with an interface of Dataset interface：接口"""

    def __init__(self, transform, class_indices,num_exemplars=2000, exemplar_selection='random'):
        super().__init__({'x': [], 'y': []}, transform, class_indices=class_indices)
        self.max_num_exemplars = num_exemplars # fixed memory

        cls_name = "{}ExemplarsSelector".format(exemplar_selection.capitalize()) #'RandomExemplarsSelector'
        selector_cls = getattr(importlib.import_module(name='datasets.exemplars_selection'), cls_name) #调用name模块下的cls_name函数

        self.exemplars_selector = selector_cls(self)

    # Returns a parser containing the approach specific p arameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser("Exemplars Management Parameters")
        parser.add_argument('--num-exemplars', default=2000, type=int, required=False,
                            help='Fixed memory, total number of exemplars (default=%(default)s)')
        parser.add_argument('--exemplar-selection', default='random', type=str,
                            choices=['herding', 'random', 'entropy', 'distance'],
                            required=False, help='Exemplar selection strategy (default=%(default)s)')
        return parser.parse_known_args(args)

    def collect_exemplars(self, model, trn_loader, selection_transform):
        if self.max_num_exemplars>0:
            self.images, self.labels = self.exemplars_selector(model, trn_loader, selection_transform)
