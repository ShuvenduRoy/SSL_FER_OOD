import warnings
import torch

from datasets.data_utils import get_data_loader
from datasets.ssl_dataset import SSL_Dataset, ImageDatasetLoader


def get_dataset_and_loader(args, ulb_workers_multiplier=1, extra_data=False):
    # Construct Dataset & DataLoader
    if args.dataset in ['AffectNet']:
        image_loader = ImageDatasetLoader(root_path=args.data_dir, num_labels=args.num_labels if args.dataset != 'SemiInat' else -1, dataset=args.dataset, crop_size=args.crop_size if 'crop_size' in args else -1, num_class=args.num_classes, algo=args.alg, args=args)
        lb_dset = image_loader.get_lb_train_data()
        ulb_dset = image_loader.get_ulb_train_data()
    else:  # CIFAR10, CIFAR100, SVHN, STL10, ImageNet100, KDEF, RAF_DB, AffectNet, CelebA
        train_dset = SSL_Dataset(args, alg=args.alg, name=args.dataset, train=True, crop_size=args.crop_size if 'crop_size' in args else -1,
                                 num_classes=args.num_classes, data_dir=args.data_dir)

        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)
        _eval_dset = SSL_Dataset(args, alg=args.alg, name=args.dataset, train=False, crop_size=args.crop_size if 'crop_size' in args else -1,
                                 num_classes=args.num_classes, data_dir=args.data_dir)
        eval_dset = _eval_dset.get_dset()

    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    if 'supervision_aug' in args:
        if 'RandAug' in args.supervision_aug:
            lb_dset.transform = ulb_dset.strong_transform

    if 'ubl_dataset' in args and args.dataset != 'Cifar10_OpenSet':
        print('loading Unconstrained Unlabeled dataset ...', args.ubl_dataset)
        if args.ubl_dataset in ["imagenet", 'SemiInat', 'CelebA', 'AffectNet']:
            print('Loading folder dataset ...')
            image_loader = ImageDatasetLoader(root_path=args.ubl_data_dir, num_labels=args.num_labels,
                                              crop_size=args.crop_size if 'crop_size' in args else -1,
                                              dataset=args.dataset, num_class=args.num_classes, algo=args.alg)




        else:  # all numpy datasets
            print('Loading PyTroch dataset ...')
            if not torch.cuda.is_available():
                warnings.warn('DATA LOCATION OVERRIDE FOR DEBUGGING')
                args.ubl_data_dir = '/Users/shuvenduroy/Documents/dataset/imagenet100'
            train_dset = SSL_Dataset(args, alg=args.alg, name=args.ubl_dataset, train=True, num_classes=100,
                                     crop_size=args.crop_size if 'crop_size' in args else 96 if args.dataset.upper() == 'STL10' else -1,
                                     data_dir=args.ubl_data_dir, extra_data=dset_dict['train_lb'] if extra_data else None)
            lb_dset, ulb_dset = train_dset.get_ssl_dset(100)
        dset_dict['train_ulb'] = ulb_dset
    print(dset_dict)

    if args.rank == 0 and args.distributed:
        torch.distributed.barrier()

    loader_dict = {}

    num_worker_multiplier_ulb = 4 * ulb_workers_multiplier
    num_worker_multiplier_lb = 1
    if 'ubl_dataset' in args:
        if args.ubl_dataset in ['CelebA', 'AffectNet']:
            num_worker_multiplier_ulb = 20
            num_worker_multiplier_lb = 2
            print('increasing number of workers for unlabelled loader to {}!!'.format(num_worker_multiplier_ulb))
    elif args.dataset in ["imagenet", 'SemiInat', 'CelebA', 'AffectNet']:
        num_worker_multiplier_ulb = 24
        num_worker_multiplier_lb = 4
        print('increasing number of workers for unlabelled loader to {}!!'.format(num_worker_multiplier_ulb))
        print('increasing number of workers for labelled loader to {}!!'.format(num_worker_multiplier_lb))

    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler=args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=num_worker_multiplier_lb * args.num_workers,
                                              distributed=args.distributed)

    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size * args.uratio,
                                               data_sampler=args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=num_worker_multiplier_ulb * args.num_workers,
                                               distributed=args.distributed)

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.batch_size * args.uratio,
                                          num_workers=num_worker_multiplier_lb * args.num_workers,
                                          drop_last=False)

    return dset_dict, loader_dict
