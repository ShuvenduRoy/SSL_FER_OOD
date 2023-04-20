# import needed library
import logging
import os
import contextlib
import random
import warnings
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from sklearn.metrics import *
from torch.cuda.amp import autocast, GradScaler

from datasets.dataset_helper import get_dataset_and_loader
from models.fixmatch.fixmatch import FixMatch
from models.fixmatch.fixmatch_utils import consistency_loss, Get_Scalar
from train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup, ce_loss, EMA, Bn_Controller
from utils import net_builder, get_logger, count_parameters, over_write_args_from_file, print_args


class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, max_probs, labels=None, mask=None, reduction="mean", select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and select_matrix is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs, max_probs.T)
            # Some may find that the line 59 is different with eq(6)
            # Acutuall the final mask will set weight=0 when i=j, following Eq(8) in paper
            # For more details, please see issue 9
            # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
            mask = mask.mul(score_mask) * select_matrix

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # max_probs = max_probs.reshape((batch_size,1))
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs, max_probs.T)
            mask = mask.mul(score_mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if reduction == "mean":
            loss = loss.mean()

        return loss


class CCSSL:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None, args=None):
        """
        class CCSSL contains setter of data_loader, optimizer, and model update methods.
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

        super(CCSSL, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.contrast_with_labeled = args.contrast_with_labeled
        self.contrast_with_threshold = args.contrast_with_threshold
        self.contrast_with_softlabel = args.contrast_with_softlabel
        self.contrast_left_out = args.contrast_left_out
        self.contrast_with_thresh = args.contrast_with_thresh
        self.loss_contrast = SoftSupConLoss(temperature=args.temperature)

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.lambda_contrast = args.lambda_contrast
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.best_eval_acc = 0.0
        self.best_it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _contrast_left_out(self, max_probs):
        """contrast_left_out

        If contrast_left_out, will select positive pairs based on
            max_probs > contrast_with_thresh, others will set to 0
            later max_probs will be used to re-weight the contrastive loss

        Args:
            max_probs (torch Tensor): prediction probabilities

        Returns:
            select_matrix: select_matrix with probs < contrast_with_thresh set
                to 0
        """
        contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
        contrast_mask2 = torch.clone(contrast_mask)
        contrast_mask2[contrast_mask == 0] = -1
        select_elements = torch.eq(contrast_mask2.reshape([-1, 1]),
                                   contrast_mask.reshape([-1, 1]).T).float()
        select_elements += torch.eye(contrast_mask.shape[0]).to(self.device)
        select_elements[select_elements > 1] = 1
        select_matrix = torch.ones(contrast_mask.shape[0]).to(
            self.device) * select_elements
        return select_matrix

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # EMA Init
        debug_print = True
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
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

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label
        classwise_acc = torch.zeros((args.num_classes,))

        if torch.cuda.is_available():
            selected_label = selected_label.cuda(args.gpu)
            classwise_acc = classwise_acc.cuda(args.gpu)

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s1, x_ulb_s2) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            if torch.cuda.is_available():
                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]

            if torch.cuda.is_available():
                x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(args.gpu), x_ulb_s2.cuda(args.gpu)
                y_lb = y_lb.long().cuda(args.gpu)

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2))
            if debug_print:
                debug_print = False
                print(f'inputs shape: {inputs.shape}')
                print(f'x_lb shape: {x_lb.shape}')
                print(f'x_ulb_w shape: {x_ulb_w.shape}')
                print(f'x_ulb_s shape: {x_ulb_s1.shape}')
                print(f'y_lb shape: {y_lb.shape}')

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, features = self.model(inputs, return_projection=True)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s, _ = logits[num_lb:].chunk(3)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                _, f_u_s1, f_u_s2 = features[num_lb:].chunk(3)

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s,
                                                                       logits_x_ulb_w,
                                                                       'ce', T, p_cutoff,
                                                                       use_hard_labels=args.hard_label)

                probs_u_w = torch.softmax(logits_x_ulb_w.detach() / 1.0, dim=-1)  # hard coded temperature from CCSSL default of 1
                max_probs, labels = torch.max(probs_u_w, dim=-1)
                features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)

                # In case of early training stage, pseudo labels have low scores
                if labels.shape[0] != 0:
                    if self.contrast_with_softlabel:
                        select_matrix = None
                        if self.contrast_left_out:
                            with torch.no_grad():
                                select_matrix = self._contrast_left_out(max_probs)
                            Lcontrast = self.loss_contrast(features,
                                                           max_probs,
                                                           labels,
                                                           select_matrix=select_matrix)

                        elif self.contrast_with_threshold:
                            contrast_mask = max_probs.ge(
                                self.contrast_with_thresh).float()
                            Lcontrast = self.loss_contrast(features,
                                                           max_probs,
                                                           labels,
                                                           reduction=None)
                            Lcontrast = (Lcontrast * contrast_mask).mean()

                        else:
                            Lcontrast = self.loss_contrast(features, max_probs, labels)
                    else:
                        if self.contrast_left_out:
                            with torch.no_grad():
                                select_matrix = self.contrast_left_out(max_probs)
                            Lcontrast = self.loss_contrast(features,
                                                           labels,
                                                           select_matrix=select_matrix)
                        else:
                            Lcontrast = self.loss_contrast(features, labels)

                else:
                    Lcontrast = sum(features.view(-1, 1)) * 0

                total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_contrast * Lcontrast

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
            tb_dict['train/contrast_loss'] = Lcontrast.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
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

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == self.best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            if torch.cuda.is_available():
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
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')

        cf_mat = confusion_matrix(y_true, y_pred)  # , normalize='true'
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'best_eval_acc': self.best_eval_acc,
                    'best_it': self.best_it,
                    'ema_model': ema_model},
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
    """
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    """

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
    args.distributed = (args.world_size > 1 or args.multiprocessing_distributed) and torch.cuda.is_available()
    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if args.multiprocessing_distributed and torch.cuda.is_available():
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    """
    main_worker is conducted on each GPU.
    """

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

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
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    # SET FixMatch: class FixMatch in models.fixmatch
    args.bn_momentum = 1.0 - 0.999
    if 'imagenet' in args.dataset.lower():
        _net_builder = net_builder(args.net, False, None, is_remix=False)
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
                                    'projection_head': 'mlp',
                                    'dim_in': 512 if args.dataset in ['cifar100'] else 2048 if args.net in ['ResNet50', 'ResNet18'] else 256 if args.dataset == 'stl10' else 128,
                                    'feat_dim': 64,
                                    'is_remix': False},
                                   )

    model = CCSSL(_net_builder,
                  args.num_classes,
                  args.ema_m,
                  args.T,
                  args.p_cutoff,
                  args.ulb_loss_ratio,
                  args.hard_label,
                  num_eval_iter=args.num_eval_iter,
                  tb_log=tb_log,
                  logger=logger,
                  args=args)

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Optimizer & LR Scheduler
    # construct SGD and cosine lr scheduler
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter * 0)
    # set SGD and cosine lr on FixMatch
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        print('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

            '''
            batch_size: batch_size per node -> batch_size per gpu
            workers: workers per node -> workers per gpu
            '''
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model.model.cuda(args.gpu)
            model.model = nn.SyncBatchNorm.convert_sync_batchnorm(model.model)
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

    # set DataLoader on FixMatch
    model.set_data_loader(loader_dict)
    model.set_dset(dset_dict['train_ulb'])
    # If args.resume, load checkpoints from args.load_path
    if os.path.exists(os.path.join(save_path, 'latest_model.pth')):
        print('Attempting auto-resume!!')
        model.load_model(os.path.join(save_path, 'latest_model.pth'))

    # START TRAINING of FixMatch
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
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5000,
                        help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--fps', type=float, default=16.0)
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
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
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    CCSSL Specific configurations
    '''
    parser.add_argument('--contrast_with_labeled', type=str2bool, default=False)
    parser.add_argument('--contrast_with_threshold', type=str2bool, default=False)
    parser.add_argument('--contrast_with_softlabel', type=str2bool, default=True)
    parser.add_argument('--contrast_left_out', type=str2bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--lambda_contrast', type=float, default=1.0)
    parser.add_argument('--contrast_with_thresh', type=float, default=0)

    '''
    multi-GPUs & Distrbitued Training
    '''

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
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
