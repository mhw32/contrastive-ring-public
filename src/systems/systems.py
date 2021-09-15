import os
import numpy as np
from dotmap import DotMap
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets import datasets
from src.models.resnet import resnet18
from src.models.resnet2 import CustomResNet
from src.objectives.memory import MemoryBank
from src.models.logreg import LogisticRegression
from src.objectives.instdisc import NCE, NCEBall, NCERing
from src.objectives.moco import MoCo, MoCoBall, MoCoRing
from src.utils import utils
from src.utils.policy import (
    AdaptiveThresholdPolicy,
    ConstantThresholdPolicy,
    ExponentialDecayThresholdPolicy,
    LinearThresholdPolicy,
    StepThresholdPolicy,
)

import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset, 
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle, 
        pin_memory=True,
        drop_last=shuffle,
        num_workers=config.data_loader_workers,
    )
    return loader


class PretrainNCESystem(pl.LightningModule):
    """System for doing Instance Discrimination pretraining (with or without Ring)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.data_params.dataset,
            mocov2_transforms=config.data_params.mocov2_transforms or False,
            mocov2_32x32_transforms=config.data_params.mocov2_32x32_transforms or False,
            mocov2_64x64_transforms=config.data_params.mocov2_64x64_transforms or False,
            mocov2_96x96_transforms=config.data_params.mocov2_96x96_transforms or False,
        )
        if config.data_params.mocov2_32x32_transforms:
            self.input_size = 32
        elif config.data_params.mocov2_64x64_transforms:
            self.input_size = 64
        elif config.data_params.mocov2_96x96_transforms:
            self.input_size = 96
        else:
            self.input_size = 224

        # store all labels for metric computation (not used in model)
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)

        # initialize encoder
        self.model = self.create_encoder()

        # memory bank for storing representations
        self.memory_bank = MemoryBank(
            len(self.train_dataset), 
            self.config.model_params.out_dim,
        )

        self.create_negative_annealing_schedule()

    def create_negative_annealing_schedule(self):
        len_loader = int(len(self.train_dataset) / self.config.optim_params.batch_size)
        num_iters_total = self.config.num_epochs * len_loader

        # different policies for choosing the outer threshold, which 
        # is initialized to the full circle
        if self.config.loss_params.adaptive_anneal_on_loss:
            self.thres_outer_policy = AdaptiveThresholdPolicy(
                num_iters_total,
                lower_is_better=True,
                init_thres=1.0, 
                min_thres=0.1, 
                max_thres=1.0, 
                delta=0.01,
                window=10,
            )
        elif self.config.loss_params.adaptive_anneal_on_acc:
            self.thres_outer_policy = AdaptiveThresholdPolicy(
                num_iters_total,
                lower_is_better=False,
                init_thres=1.0, 
                min_thres=0.1, 
                max_thres=1.0, 
                delta=0.05,
                window=5,
            )
        elif self.config.loss_params.linear_anneal:
            self.thres_outer_policy = LinearThresholdPolicy(
                num_iters_total,
                int(self.config.loss_params.max_anneal_epoch * len_loader),
                init_thres=1.0, 
                min_thres=0.1, 
            )
        elif self.config.loss_params.step_anneal:
            self.thres_outer_policy = StepThresholdPolicy(
                num_iters_total,
                int(self.config.loss_params.max_anneal_epoch * len_loader),
                init_thres=1.0, 
                min_thres=0.1, 
            )
        elif self.config.loss_params.exponential_decay_anneal:
            self.thres_outer_policy = ExponentialDecayThresholdPolicy(
                num_iters_total,
                int(self.config.loss_params.max_anneal_epoch * len_loader),
                init_thres=1.0, 
                min_thres=0.1,
                decay=0.1,
            )
        else:
            self.thres_outer_policy = ConstantThresholdPolicy(
                init_thres=self.config.loss_params.thres_outer,
            )

        # everyone uses constant inner threshold!
        self.thres_inner_policy = ConstantThresholdPolicy(
            init_thres=self.config.loss_params.thres_inner,
        )

    def create_encoder(self):
        version = self.config.model_params.resnet_version or 'resnet18'
        return CustomResNet(version, self.config.model_params.out_dim,
                            conv3x3=self.config.model_params.conv3x3 or False)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, 
            T_max=int(len(self.train_dataset) / self.config.optim_params.batch_size),
            eta_min=0,
            last_epoch=-1,
        )
        """
        return [optim], []

    def forward(self, img):
        return self.model(img)[1]

    def get_losses_for_batch(self, batch, train=True):
        indices, img, _, = batch
        outputs = self.forward(img)
        loss_class = globals()[self.config.loss_params.loss]
        thres_outer = self.thres_outer_policy.get_threshold(self.global_step)
        thres_inner = self.thres_inner_policy.get_threshold(self.global_step)
        loss_fn = loss_class(
            indices, 
            outputs, 
            self.memory_bank,
            k=self.config.loss_params.k,
            t=self.config.loss_params.t,
            m=self.config.loss_params.m,
            thres_outer=thres_outer,
            thres_inner=thres_inner, 
            dynamic_inner=self.config.loss_params.dynamic_inner or True,
            neighbor_views=self.config.loss_params.neighbor_views or False,
            logsumexp_inner=self.config.loss_params.logsumexp_inner or True,
            self_negative=self.config.loss_params.self_negative or True,
        )
        loss = loss_fn.get_loss()

        if train:
            with torch.no_grad():
                new_data_memory = loss_fn.updated_new_data_memory()
                self.memory_bank.update(indices, new_data_memory)

            if self.config.loss_params.adaptive_anneal_on_loss:
                self.thres_outer_policy.record(loss.item())

        return loss, (thres_outer, thres_inner)

    def get_nearest_neighbor_label(self, batch):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.

        For each image in validation, find the nearest image in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        img, label = batch[1], batch[-1]
        outputs = self.forward(img)

        all_dps = self.memory_bank.get_all_dot_products(outputs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        neighbor_idxs = neighbor_idxs.cpu().numpy()

        neighbor_labels = self.train_ordered_labels[neighbor_idxs]
        neighbor_labels = torch.from_numpy(neighbor_labels).long()

        num_correct = torch.sum(neighbor_labels.cpu() == label.cpu()).item()
        return num_correct, img.size(0)

    def training_step(self, batch, batch_idx):
        loss, (thres_outer, thres_inner) = self.get_losses_for_batch(batch, train=True)
        metrics = {
            'loss': loss, 
            'thres_outer': thres_outer or 0, 
            'thres_inner': thres_inner or 0,
        }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss, _ = self.get_losses_for_batch(batch, train=False)
        num_correct, batch_size = self.get_nearest_neighbor_label(batch)
        output = OrderedDict({
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': batch_size,
        })
        return output
    
    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        if self.config.loss_params.adaptive_anneal_on_acc:
            self.thres_outer_policy.record(val_acc)
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc}

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)


class PretrainMoCoSystem(PretrainNCESystem):
    """System for doing MoCo-v2 pretraining (with or without Ring)."""

    def __init__(self, config):
        super().__init__(config)

        # duplicate of self.model
        self.model_k = self.create_encoder()

        # copy the parameters over
        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # do not update

        # create queue (k x out_dim)
        queue = torch.randn(
            self.config.loss_params.k,
            self.config.model_params.out_dim, 
        )
        self.register_buffer("queue", queue)
        self.queue = utils.l2_normalize(queue, dim=1)
        self.register_buffer(
            "queue_ptr", 
            torch.zeros(1, dtype=torch.long),
        )

    def get_losses_for_batch(self, batch, train=True):
        indices, img1, img2, _, = batch
        outputs1 = self.forward(img1)

        # this code is largely adapted from Pytorch Lightning Bolts
        # https://github.com/PyTorchLightning/lightning-bolts
        with torch.no_grad():
            self._momentum_update_key_encoder()

            if self.use_ddp or self.use_ddp2:
                img2, idx_unshuffle = self._batch_shuffle_ddp(img2)
            elif self.config.shuffle_bn:
                idx = torch.randperm(img2.size(0))
                img2 = img2[idx]

            _, outputs2 = self.model_k(img2)
            
            if self.use_ddp or self.use_ddp2:
                outputs2 = self._batch_unshuffle_ddp(outputs2, idx_unshuffle)
            elif self.config.shuffle_bn:
                outputs2_tmp = torch.zeros_like(outputs2)
                for i, j in enumerate(idx):
                    outputs2_tmp[j] = outputs2[i]
                outputs2 = outputs2_tmp

        # get thresholds to define the ring for negative samples
        thres_outer = self.thres_outer_policy.get_threshold(self.global_step)
        thres_inner = self.thres_inner_policy.get_threshold(self.global_step)

        loss_class = globals()[self.config.loss_params.loss]
        loss_fn = loss_class(
            outputs1, 
            outputs2,
            self.queue.clone().detach(),
            t=self.config.loss_params.t,
            thres_outer=thres_outer,
            thres_inner=thres_inner, 
            # if False, keep the inner ring defn fixed
            dynamic_inner=self.config.loss_params.dynamic_inner or True,
            # if True, use neighboring points for views (e.g. local aggregation)
            neighbor_views=self.config.loss_params.neighbor_views or False,
            # if True, use logsumexp instead of sum in NCE numerator
            logsumexp_inner=self.config.loss_params.logsumexp_inner or True,
            # if True, include positive view as a negative sample
            self_negative=self.config.loss_params.self_negative or True,
        )
        loss = loss_fn.get_loss()

        if train:
            # the queue stores l2 normalized examples
            outputs2 = utils.l2_normalize(outputs2, dim=1)
            self._dequeue_and_enqueue(outputs2)

            # update the memory bank, which we use either in the contrastive
            # algorithm itself, or for statistics
            with torch.no_grad():
                new_data_memory = utils.l2_normalize(outputs1, dim=1)
                self.memory_bank.update(indices, new_data_memory)

            # update outer ring threshold
            if self.config.loss_params.adaptive_anneal_on_loss:
                self.thres_outer_policy.record(loss.item())

        return loss, (thres_outer, thres_inner)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self.config.loss_params.m
        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = utils.concat_all_gather(keys)

        batch_size = keys.size(0)

        k = self.config.loss_params.k
        ptr = int(self.queue_ptr)
        assert k % batch_size == 0  # why?

        # replace keys at ptr
        self.queue[ptr:ptr+batch_size] = keys
        # move config by full batch size even if current batch is smaller
        ptr = (ptr + batch_size) % k

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no-cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = utils.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no-cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = utils.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


class TransferSystem(pl.LightningModule):
    """System for doing linear evaluation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder, self.pretrain_config = self.load_pretrained_model()
        utils.frozen_params(self.encoder)
        self.model = self.create_model()
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.data_params.dataset,
            mocov2_transforms=self.pretrain_config.data_params.mocov2_transforms or False,
            mocov2_32x32_transforms=self.pretrain_config.data_params.mocov2_32x32_transforms or False,
            mocov2_64x64_transforms=self.pretrain_config.data_params.mocov2_64x64_transforms or False,
            mocov2_96x96_transforms=self.pretrain_config.data_params.mocov2_96x96_transforms or False,
        )

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        config.data_params.dataset = self.config.pretrain_model.dataset

        # overwrite GPU to load on same as current agent
        config.gpu_device = self.config.gpu_device

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        system.load_state_dict(checkpoint['state_dict'])

        encoder = system.model.eval()
        return encoder, config

    def create_model(self):
        dataset_name = self.config.data_params.dataset
        if 'meta' in dataset_name:
            # for meta_dataset applications
            num_class = self.train_dataset.NUM_CLASSES
        else:
            NUM_CLASS_DICT = {
                'cifar10': 10, 
                'stl10': 10,
                'cifar100': 100, 
                'imagenet': 1000,
            }
            num_class = NUM_CLASS_DICT[dataset_name]

        # There are different choices for which embedding layer to 
        # use as the representation for transfer.
        if self.config.model_params.layer == 5:
            if self.pretrain_config.data_params.mocov2_32x32_transforms: 
                input_dim = 512*4*4
            elif self.pretrain_config.data_params.mocov2_64x64_transforms:
                input_dim = 512*8*8
            else:
                input_dim = 512*7*7
        else:
            if self.pretrain_config.model_params.resnet_version == 'resnet50':
                input_dim = 2048
            else:
                input_dim = 512
        model = LogisticRegression(input_dim, num_class)
        return model

    def forward(self, img):
        batch_size = img.size(0)
        embs, _ = self.encoder(img)
        embs = embs.view(batch_size, -1)
        return self.model(embs)

    def get_losses_for_batch(self, batch):
        _, img, label = batch
        logits = self.forward(img)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch):
        _, img, label = batch
        logits = self.forward(img)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = img.size(0)
        return num_correct, num_total

    def training_step(self, batch, _):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct': num_correct,
                'train_num_total': num_total,
                'train_acc': num_correct / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, _):
        loss = self.get_losses_for_batch(batch)
        num_correct, num_total = self.get_accuracies_for_batch(batch)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': num_total,
            'val_acc': num_correct / float(num_total)
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc}

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        """
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=10e-6,
        )
        """
        return [optim], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)
