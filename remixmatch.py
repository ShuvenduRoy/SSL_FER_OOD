# import needed library
import contextlib
import json
import logging
import os
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn.parallel
from sklearn.metrics import *
from torch.cuda.amp import autocast, GradScaler

from datasets.data_utils import get_data_loader
from datasets.dataset_helper import get_dataset_and_loader
from datasets.ssl_dataset import SSL_Dataset, ImageDatasetLoader
from train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup
from train_utils import ce_loss, EMA, Bn_Controller
from utils import net_builder, get_logger, count_parameters, over_write_args_from_file, print_args
from models.remixmatch.remixmatch_utils import Get_Scalar, one_hot, mixup_one_target


class ReMixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, lambda_u, \
                 w_match,
                 t_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(ReMixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        self.model = net_builder(num_classes=num_classes)
        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.w_match = w_match  # weight of distribution matching
        self.lambda_u = lambda_u
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.best_eval_acc = 0.0
        self.best_it = 0
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1  # to avoid div by 0 error

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume:
            self.ema.load(self.ema_model)

        # for gpu profiling
        if torch.cuda.is_available():
            start_batch = torch.cuda.Event(enable_timing=True)
            end_batch = torch.cuda.Event(enable_timing=True)
            start_run = torch.cuda.Event(enable_timing=True)
            end_run = torch.cuda.Event(enable_timing=True)
            start_batch.record()

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + args.dataset + '_' + str(args.num_labels) + '.json'
        if not os.path.exists(dist_file_name):
            p_target = torch.ones(args.num_classes) / args.num_classes
        else:
            with open(dist_file_name, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
        if torch.cuda.is_available():
            p_target = p_target.cuda(args.gpu)
        print('p_target:', p_target)

        p_model = None

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        # x_ulb_s1_rot: rotated data, rot_v: rot angles
        for (_, x_lb, y_lb), (_, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v) in zip(self.loader_dict['train_lb'],
                                                                                          self.loader_dict[
                                                                                              'train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            if torch.cuda.is_available():
                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            num_rot = x_ulb_s1_rot.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]

            if torch.cuda.is_available():
                x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(
                    args.gpu), x_ulb_s2.cuda(args.gpu)
                x_ulb_s1_rot = x_ulb_s1_rot.cuda(args.gpu)  # rot_image
                rot_v = rot_v.cuda(args.gpu)  # rot_label
                y_lb = y_lb.cuda(args.gpu)

            # inference and calculate sup/unsup losses
            with amp_cm():
                with torch.no_grad():
                    self.bn_controller.freeze_bn(self.model)
                    # logits_x_lb = self.model(x_lb)[0]
                    logits_x_ulb_w = self.model(x_ulb_w)[0]
                    # logits_x_ulb_s1 = self.model(x_ulb_s1)[0]
                    # logits_x_ulb_s2 = self.model(x_ulb_s2)[0]
                    self.bn_controller.unfreeze_bn(self.model)

                    # hyper-params for update
                    T = self.t_fn(self.it)

                    prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)

                    # p^~_(y): moving average of p(y)
                    if p_model == None:
                        p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                    else:
                        p_model = p_model * 0.999 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.001

                    # Distribution alignment
                    if args.dist_alignment_factor > 0:
                        prob_x_ulb = prob_x_ulb * p_target / p_model
                    prob_x_ulb = (prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True))

                    sharpen_prob_x_ulb = prob_x_ulb ** (1 / T)
                    sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

                    # mix up
                    mixed_inputs = torch.cat((x_lb, x_ulb_s1, x_ulb_s2, x_ulb_w))
                    input_labels = torch.cat(
                        [one_hot(y_lb, args.num_classes, args.gpu), sharpen_prob_x_ulb, sharpen_prob_x_ulb,
                         sharpen_prob_x_ulb], dim=0)

                    mixed_x, mixed_y, _ = mixup_one_target(mixed_inputs, input_labels,
                                                           args.gpu,
                                                           args.alpha,
                                                           is_bias=True)

                    # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                    mixed_x = list(torch.split(mixed_x, num_lb))
                    mixed_x = self.interleave(mixed_x, num_lb)

                    # inter_inputs = torch.cat([mixed_x, x_ulb_s1], dim=0)
                    # inter_inputs = list(torch.split(inter_inputs, num_lb))
                    # inter_inputs = self.interleave(inter_inputs, num_lb)

                    # calculate BN only for the first batch
                logits = [self.model(mixed_x[0])[0]]

                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt)[0])

                u1_logits = self.model(x_ulb_s1)[0]
                logits_rot = self.model(x_ulb_s1_rot)[1]
                logits = self.interleave(logits, num_lb)
                self.bn_controller.unfreeze_bn(self.model)

                logits_x = logits[0]
                logits_u = torch.cat(logits[1:])

                # calculate rot loss with w_rot
                rot_loss = ce_loss(logits_rot, rot_v, reduction='mean')
                rot_loss = rot_loss.mean()
                # sup loss
                sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
                sup_loss = sup_loss.mean()
                # unsup_loss
                unsup_loss = ce_loss(logits_u, mixed_y[num_lb:], use_hard_labels=False)
                unsup_loss = unsup_loss.mean()
                # loss U1
                u1_loss = ce_loss(u1_logits, sharpen_prob_x_ulb, use_hard_labels=False)
                u1_loss = u1_loss.mean()
                # ramp for w_match
                w_match = args.w_match * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))
                w_kl = args.w_kl * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))

                total_loss = sup_loss + args.w_rot * rot_loss + w_kl * u1_loss + w_match * unsup_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            if torch.cuda.is_available():
                end_run.record()
                torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            if torch.cuda.is_available():
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if 'terminal_iter' in args:
                if self.it >= args.terminal_iter:
                    break

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > self.best_eval_acc:
                    self.best_eval_acc = tb_dict['eval/top-1-acc']
                    self.best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_it} iters")
                total_time = 0

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == self.best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        try:
            os.makedirs(args.save_dir + '/eval_acc', exist_ok=True)
            with open(os.path.join(args.save_dir + '/eval_acc', args.save_name[:-2] + '.txt'), 'a') as f:
                f.write(args.save_name + ' ' + str(round(self.best_eval_acc * 100, 2)) + '\n')
        except:
            pass
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            if torch.cuda.is_available():
                x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        cf_mat = confusion_matrix(y_true, y_pred)
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = deepcopy(self.model)
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'best_eval_acc': self.best_eval_acc,
                    'best_it': self.best_it,
                    'ema_model': ema_model.state_dict()},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.print_fn('Loading save model from: ' + str(load_path))
        self.ema_model = deepcopy(self.model)

        try:
            self.model.load_state_dict(checkpoint['model'])
            self.ema_model.load_state_dict(checkpoint['ema_model'])
            self.print_fn('model loaded')
        except Exception as e:
            self.print_fn('SAVED CHECKPOINT IS NOT COMPATIBLE WITH CURRENT MODEL! ATTEMPTING TO LOAD MANUALLY')

            # Scenario 1: current single gpu & loading multi-gpu
            if type(self.model) != torch.nn.parallel.distributed.DistributedDataParallel:
                state_dict = checkpoint['model']
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict
                self.model.load_state_dict(state_dict)

                state_dict_ema = checkpoint['ema_model']
                new_state_dict_ema = {}
                for k, v in state_dict_ema.items():
                    k = k.replace("module.", "")
                    new_state_dict_ema[k] = v
                state_dict_ema = new_state_dict_ema
                self.ema_model.load_state_dict(state_dict_ema)

                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.it = checkpoint['it']
                self.print_fn('model loaded from multi-gpu checkpoint to single gpu')

            # Scenario 2: saved multi-gpu loading single-gpu
            else:
                state_dict = checkpoint['model']
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = "module." + k
                    new_state_dict[k] = v
                state_dict = new_state_dict
                self.model.load_state_dict(state_dict)

                state_dict_ema = checkpoint['ema_model']
                new_state_dict_ema = {}
                for k, v in state_dict_ema.items():
                    k = "module." + k
                    new_state_dict_ema[k] = v
                state_dict_ema = new_state_dict_ema
                self.ema_model.load_state_dict(state_dict_ema)

                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.it = checkpoint['it']
                self.print_fn('model loaded from single-gpu checkpoint to multi gpu')
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        try:
            self.best_eval_acc = checkpoint['best_eval_acc']
            self.best_it = checkpoint['best_it']
        except:
            print('no best eval acc found')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    save_path = os.path.join(args.save_dir, args.save_name)

    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed and torch.cuda.is_available():
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    if not torch.cuda.is_available():
        args.distributed = False

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"
    else:
        if args.rank % ngpus_per_node == 0:
            tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
            logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    # SET RemixMatch: class RemixMatch in models.remixmatch
    args.bn_momentum = 1.0 - 0.999

    if 'imagenet' in args.dataset.lower():
        _net_builder = net_builder(args.net, False, None, is_remix=True)
    else:
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'first_stride': 2 if 'stl' in args.dataset else 1,
                                    'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'bn_momentum': args.bn_momentum,
                                    'dropRate': args.dropout,
                                    'use_embed': False,
                                    'is_remix': True},
                                   )

    model = ReMixMatch(_net_builder,
                       args.num_classes,
                       args.ema_m,
                       args.T,
                       args.ulb_loss_ratio,
                       num_eval_iter=args.num_eval_iter,
                       tb_log=tb_log,
                       logger=logger,
                       w_match=args.w_match,
                       )

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Optimizer & LR Scheduler
    # construct SGD and cosine lr scheduler
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter * 0)
    # set SGD and cosine lr on RemixMatch
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        pass
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.model.cuda(args.gpu)
            model.model = torch.nn.parallel.DistributedDataParallel(model.model,
                                                                    device_ids=[args.gpu],
                                                                    broadcast_buffers=False,
                                                                    find_unused_parameters=True)

        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.model = model.model.cuda(args.gpu)

    else:
        model.model = torch.nn.DataParallel(model.model).cuda()

    import copy
    model.ema_model = copy.deepcopy(model.model)

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")
    if args.rank == 0:
        print_args(args)

    cudnn.benchmark = True
    if args.rank != 0 and args.distributed:
        torch.distributed.barrier()

    dset_dict, loader_dict = get_dataset_and_loader(args)

    # set DataLoader on RemixMatch
    model.set_data_loader(loader_dict)

    # If args.resume, load checkpoints from args.load_path
    if os.path.exists(os.path.join(save_path, 'latest_model.pth')):
        print('Attempting auto-resume!!')
        model.load_model(os.path.join(save_path, 'latest_model.pth'))

    # START TRAINING of RemixMatch
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args, logger=logger)

    if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {args.rank} training is FINISHED")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='remixmatch')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of RemixMatch
    '''
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=1000,
                        help='evaluation frequency')
    parser.add_argument('--num_labels', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--alpha', type=float, default=0.75, help='param for Beta distribution of Mix Up')
    parser.add_argument('--T', type=float, default=0.5, help='Temperature Sharpening')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0, help='weight of Unsupervised Loss')
    parser.add_argument('--w_kl', type=float, default=0.5, help='weight for KL loss')
    parser.add_argument('--w_match', type=float, default=1.5, help='weight for distribution matching loss.')
    parser.add_argument('--w_rot', type=float, default=0.5, help='weight for rot loss')
    parser.add_argument('--use_dm', type=str2bool, default=True, help='Whether to use distribution matching')
    parser.add_argument('--use_xe', type=str2bool, default=True, help='Whether to use cross-entropy or Brier')
    parser.add_argument('--warm_up', type=float, default=1 / 64)
    parser.add_argument('--dist_alignment_factor', type=float, default=1.0, help='weight for distribution alignment loss (0 to disable).')

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=True, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--use_extra_data', type=str2bool, default=False,
                        help='whether or not use labeled data as extra data to unconstrained data')
    '''
    multi-GPUs & Distrbitued Training
    '''
    # args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    main(args)

    print("Finish Training. Canceling job...")
    os.system('scancel %s' % os.environ["SLURM_ARRAY_JOB_ID"])
