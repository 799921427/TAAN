from __future__ import print_function, absolute_import
import time
import random

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from reid.loss.CrossTriplet import CrossTriplet
from torch import nn
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model_t, model_rgb, model_ir, criterion_z, criterion_I, criterion_att, criterion_t, trainvallabel, a, b, c, d, k):
        super(BaseTrainer, self).__init__()
        self.model_t = model_t
        self.model_rgb = model_rgb
        self.model_ir = model_ir
        self.criterion_z = criterion_z
        self.criterion_I = criterion_I
        self.criterion_att = criterion_att
        self.criterion_t = criterion_t
        self.trainvallabel = trainvallabel
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k

    def train(self, epoch, data_loader, optimizer_t, print_freq=1):
        self.model_t.train()
        self.model_rgb.train()
        self.model_ir.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_generator = AverageMeter()
        losses_triple = AverageMeter()
        losses_idloss = AverageMeter()
        losses_rank = AverageMeter()
        losses_idloss_s = AverageMeter()
        losses_att_rgb = AverageMeter()
        losses_att_ir = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, sub, label = self._parse_data(inputs)

            # Calc the loss
            #loss_rank, loss_id = self._forward(inputs, label, sub)
            #L = self.a * loss_rank + self.b * loss_id + loss_id_s
            loss_rank, loss_id, loss_id_s, loss_t, loss_att_rgb, loss_att_ir = self._forward(inputs, label, sub)
            L = self.a * loss_rank + self.b * loss_id + loss_t +  loss_id_s + self.c * loss_att_rgb + self.d * loss_att_ir

            # neg_L = - self.u * L

            # if ((epoch * len(data_loader) + i) % self.k == 0):
            #    optimizer_discriminator.zero_grad()
            #    neg_L.backward()
            #    optimizer_discriminator.step()
            #else:
            optimizer_t.zero_grad()
            L.backward()
            optimizer_t.step()

            losses_generator.update(L.data.item(), label.size(0))
            losses_idloss.update(loss_id.item(), label.size(0))
            losses_triple.update(loss_t.item(), label.size(0))
            losses_rank.update(loss_rank.item(), label.size(0))
            losses_idloss_s.update(loss_id_s.item(), label.size(0))
            losses_att_rgb.update(loss_att_rgb.item(), label.size(0))
            losses_att_ir.update(loss_att_ir.item(), label.size(0))

            #losses_discriminator.update(loss_discriminator.item(), label.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            print(optimizer_t.state_dict()['param_groups'][0]['lr'])
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Total Loss {:.3f} ({:.3f})\t'
                      'IDE Loss {:.3f} ({:.3f})\t'
                      'Single IDE Loss {:.3f} ({:.3f}) \t'
                      'Rank Loss {:.3f} ({:.3f}) \t'
                      'Single Triplet Loss {:.3f} ({:.3f}) \t'
                      'RGB Attention Loss {:.3f} ({:.3f}) \t'
                      'IR Attention Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_generator.val, losses_generator.avg,
                              losses_idloss.val, losses_idloss.avg,
                              losses_idloss_s.val, losses_idloss_s.avg,
                              losses_rank.val, losses_rank.avg,
                              losses_triple.val, losses_triple.avg,
                              losses_att_rgb.val, losses_att_rgb.avg,
                              losses_att_ir.val, losses_att_ir.avg))
        return losses_rank.avg, losses_generator.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, cams = inputs
        inputs = imgs.cuda()
        pids = pids.cuda()
        sub = ((cams == 2).long() + (cams == 5).long()).cuda()
        label = torch.cuda.LongTensor(range(pids.size(0)))
        for i in range(pids.size(0)):
            label[i] = self.trainvallabel[pids[i].item()]
        return inputs, sub, label

    def _forward(self, inputs, label, sub):

        n = inputs.size(0)
        rgb_inputs = inputs[0:n:2,:,:,:]
        ir_inputs = inputs[1:n:2,:,:,:]
        att_label = label[::2]
        att_sub = sub[1::2]
        #print(att_sub)
        outputs, outputs_pool, _, att_feats, att_g = self.model_t(inputs)
        loss_rank , _ = self.criterion_I(outputs_pool, label, sub)
        loss_id = self.criterion_z(outputs, label)

        # rgb branch
        outputs_rgb, _, outputs_pool_rgb, att_feats_rgb, _ = self.model_rgb(rgb_inputs)
        #loss_t_s, _ = self.criterion_t(outputs_pool_rgb, att_label)
        loss_t_s, _ = self.criterion_I(outputs_pool_rgb, att_label, att_sub)
        loss_id_s = self.criterion_z(outputs_rgb, att_label)

        # ir branch
        outputs_ir, _, outputs_pool_ir, att_feats_ir, _ = self.model_ir(ir_inputs)
        #loss_t_ir, _ = self.criterion_t(outputs_pool_ir, att_label)
        loss_t_ir, _ = self.criterion_I(outputs_pool_ir, att_label, att_sub)
        loss_id_ir = self.criterion_z(outputs_ir, att_label)

        loss_t_s += loss_t_ir
        loss_id_s += loss_id_ir

        loss_att_rgb = 0
        loss_att_ir = 0
        for i, fea in enumerate(att_feats):
            fea_rgb = fea[0:n:2, :]
            fea_ir = fea[1:n:2, :]
            # rgb attention loss
            fea_rgb = torch.nn.functional.normalize(fea_rgb, dim=1, p=2)
            att_feat_rgb = torch.nn.functional.normalize(att_feats_rgb[i], dim=1, p=2)
            loss_a = self.criterion_att(fea_rgb, att_feat_rgb)
            loss_att_rgb += loss_a

            # ir attention loss
            fea_ir = torch.nn.functional.normalize(fea_ir, dim=1, p=2)
            att_feat_ir = torch.nn.functional.normalize(att_feats_ir[i], dim=1, p=2)
            loss_a = self.criterion_att(fea_ir, att_feat_ir)
            loss_att_ir += loss_a


        #outputs_discriminator = self.model_discriminator(outputs_pool)
        #loss_t_s = 0
        #loss_discriminator = self.criterion_D(outputs_discriminator, sub)
        #return loss_rank, loss_id
        return  loss_rank, loss_id, loss_id_s, loss_t_s, loss_att_rgb, loss_att_ir
