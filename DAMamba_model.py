import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
from DAMamba_basenet import MambaFeature
import criterion_factory as cf
import torch.nn.functional as F
from Clustering_loss import ClusterLoss
from collections import deque
import random
from losses import loss_unl
from torch.autograd import Variable
import numpy as np
msc_config = {
    'k': 2,
    'm': 1,
    'mu': 50,
}#3 2 70/60
msc_module = cf.MSCLoss(msc_config)
msc_module = msc_module.cuda()

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

def kl_div_with_logit(q_logit, p_logit):
    ### return a matrix without mean over samples.
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1)
    qlogp = ( q *logp).sum(dim=1)

    return qlogq - qlogp
def consistency_loss(logits_w, logits_s, T=1.0, p_cutoff=0.95):
    logits_w = logits_w.detach()
    logits_w = logits_w / T
    logits_s = logits_s / T

    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    # print('max score is: %3f, mean score is: %3f' % (max(max_probs).item(), max_probs.mean().item()))
    mask_binary = max_probs.ge(p_cutoff)
    mask = mask_binary.float()

    masked_loss = kl_div_with_logit(logits_w, logits_s) * mask

    return masked_loss.mean()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class TransferNet(nn.Module):
    def __init__(self, n_bands, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True,
                 bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()

        self.num_class = num_class
        self.base_network = MambaFeature(n_bands, patch_size=12)
        # self.base_network = backbones.get_backbone(base_net)
        # self.base_network.conv1 = nn.Conv2d(n_bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                # nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.Linear(4608, bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        loss_device = torch.device("cuda")
        self.cluster_loss = ClusterLoss(self.num_class, 1.0, loss_device)
        self.ca = ChannelAttention(4608)
        self.sa = SpatialAttention()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, bbb, proto_t, source, target_s, target_w, source_label, epoch, start_msc, args):
        source = self.base_network(source)
        target_s = self.base_network(target_s)
        target_w = self.base_network(target_w)

        source = self.ca(source) * source
        target_s = self.ca(target_s) * target_s
        target_w = self.ca(target_w) * target_w

        target_s = self.sa(target_s) * target_s
        target_w = self.sa(target_w) * target_w
        source = self.sa(source) * source

        source = self.avgpool(source)
        target_s = self.avgpool(target_s)
        target_w = self.avgpool(target_w)
        source = source.view(source.size(0), -1)
        target_s = target_s.view(target_s.size(0), -1)
        target_w = target_w.view(target_w.size(0), -1)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target_s = self.bottleneck_layer(target_s)
            target_w = self.bottleneck_layer(target_w)


        # classification
        source_clf = self.classifier_layer(source)
        target_s_clf = self.classifier_layer(target_s)
        target_w_clf = self.classifier_layer(target_w)
        L_base = self.criterion(source_clf, source_label)
        # transfer
        # kwargs = {}
        # if self.transfer_loss == "lmmd":
        #     kwargs['source_label'] = source_label
        #     target_clf = self.classifier_layer(target)
        #     kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        # elif self.transfer_loss == "daan":
        #     source_clf = self.classifier_layer(source)
        #     kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
        #     target_clf = self.classifier_layer(target)
        #     kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        # elif self.transfer_loss == 'bnm':
        #     tar_clf = self.classifier_layer(target)
        #     target = nn.Softmax(dim=1)(tar_clf)
        #
        # transfer_loss = self.adapt_loss(source, target, **kwargs)##########
        #
        # msc_loss, ind, tgt_label = msc_module(source, source_label, target)

        # L_intra, L_inter, L_fixmatch, L_batch, unl_mask, unl_pseudo_label, feat_tu_w, feat_tu_s = loss_unl(source, source_label,
        #                                                                                         target_s, target_s_clf,
        #                                                                                         target_w, target_w_clf,
        #                                                                                         proto_t, bbb, args)

        L_intra, L_inter, L_fixmatch, L_batch, unl_mask, unl_pseudo_label, feat_tu_w, feat_tu_s = loss_unl(source, source_label,
                                                                                                target_s, target_s_clf,
                                                                                                target_w, target_w_clf,
                                                                                                proto_t, bbb, args)


        # if epoch > start_msc:
        #     tgt_loss = self.criterion(target_clf[ind], tgt_label)
        #     clf_loss = clf_loss + tgt_loss #+ cluster_loss
        #
        #     return clf_loss, transfer_loss, msc_loss
        # else:
        #     return clf_loss, transfer_loss, msc_loss
        return L_base, L_batch, L_fixmatch, L_inter, L_intra, unl_mask, unl_pseudo_label, feat_tu_w, feat_tu_s

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)

        features = self.ca(features) * features
        features = self.sa(features) * features
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)

        if self.use_bottleneck:
            x = self.bottleneck_layer(features)
        else:
            x = features
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass