import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce

import utils
import approach_wa

from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config

def main():
    tstart = time.time()
    # Arguments

    parser = argparse.ArgumentParser()

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,help='GPU (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=2222,help='Random seed (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=0, type=int, required=False,help='Number of subprocesses to use for dataloader')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=10, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')

    # model args
    parser.add_argument('--network', default='ResNet', type=str,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='WA', type=str,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=100, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0002, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument("--milestone",default=[42,60,80],type=list,help="reduce lr by 10 when meet milestone (default=%(default)s)")

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args()
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr,momentum=args.momentum,wd=args.weight_decay,milestone=args.milestone)

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'

    # Args -- Network
    from networks.network import LLL_Net
    net = getattr(importlib.import_module(name='networks.'+args.network), 'resnet18_cbam')
    init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from approach_wa import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach_wa'), args.approach)
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks,args.batch_size, num_workers=args.num_workers)

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices

    appr_kwargs = {**base_kwargs,**dict(**appr_args.__dict__)}

    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices)
    utils.seed_everything(seed=args.seed)

    appr = Appr(net, device, **appr_kwargs)

    # Loop tasks
    print(taskcla)
    max_task = args.num_tasks #10
    accuary = np.zeros((max_task, max_task))

    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        # Train
        appr.train(t, trn_loader[t], tst_loader[t])
        print('-' * 108)

        # Test
        print("test ...")
        for u in range(t + 1):
            accuary[t, u] = appr.eval(u, tst_loader[u])
        print("experience: %d average acc: %.4f"%(t,accuary[t].sum().item()/(t+1)))
        print("******")
        print(accuary)

    #print final results
    final_acc_list = []
    for i in range(args.num_tasks):
        final_acc_list.append(accuary[i].sum().item()/(i+1))
    print("final_acc : ",final_acc_list)
    print("cost time: %.2fh"%((time.time()-tstart)/(60*60)))

if __name__ == '__main__':
    main()
