
import math
import random
from bisect import bisect_right
import torch
from torch import nn
import time
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.0, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img




class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



class Rank_loss(nn.Module):

    ## Basic idea for cross_modality rank_loss 8

    def __init__(self, margin_1=1.0, margin_2=1.5, alpha_1=2.4, alpha_2=2.2, tval=1.0):
        super(Rank_loss, self).__init__()
        self.margin_1 = margin_1 # for same modality
        self.margin_2 = margin_2 # for different modalities
        self.alpha_1 = alpha_1 # for same modality
        self.alpha_2 = alpha_2 # for different modalities
        self.tval = tval

    def forward(self, x, targets, sub, norm = True):
        if norm:
            #x = self.normalize(x)
            x = torch.nn.functional.normalize(x, dim=1, p=2)

        dist_mat = self.euclidean_dist(x, x) # compute the distance


        loss = self.rank_loss(dist_mat, targets, sub)

        return loss,dist_mat

    def rank_loss(self, dist, targets, sub):
        loss = 0.0
        for i in range(dist.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 0
            is_neg = targets.ne(targets[i])


            intra_modality = sub.eq(sub[i])
            cross_modality = 1- intra_modality

            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality


            ap_pos_intra = torch.clamp(torch.add(dist[i][mask_pos_intra], self.margin_1-self.alpha_1),0)
            ap_pos_cross = torch.clamp(torch.add(dist[i][mask_pos_cross], self.margin_2-self.alpha_2),0)

            loss_ap = torch.div(torch.sum(ap_pos_intra), ap_pos_intra.size(0)+1e-5)
            loss_ap += torch.div(torch.sum(ap_pos_cross), ap_pos_cross.size(0)+1e-5)

            dist_an_intra = dist[i][mask_neg_intra]
            dist_an_cross = dist[i][mask_neg_cross]

            an_less_intra = dist_an_intra[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]
            an_less_cross = dist_an_cross[torch.lt(dist[i][mask_neg_cross], self.alpha_2)]

            an_weight_intra = torch.exp(self.tval*(-1* an_less_intra +self.alpha_1))
            an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5
            an_weight_cross = torch.exp(self.tval*(-1* an_less_cross +self.alpha_2))
            an_weight_cross_sum = torch.sum(an_weight_cross)+1e-5
            an_sum_intra = torch.sum(torch.mul(self.alpha_1-an_less_intra,an_weight_intra))
            an_sum_cross = torch.sum(torch.mul(self.alpha_2-an_less_cross,an_weight_cross))

            loss_an =torch.div(an_sum_intra,an_weight_intra_sum ) +torch.div(an_sum_cross,an_weight_cross_sum )
            #loss_an = torch.div(an_sum_cross,an_weight_cross_sum )
            loss += loss_ap + loss_an
            #loss += loss_an


        return loss * 1.0/ dist.size(0)

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist

    def Circle_dist(self, dist_mat, i, js, is_pos):
        circle_dist = -dist_mat[i][j] # original distance

        # for postive pair
        if is_pos:
            circle_dist += torch.sum(dist[i][mask_pos_cross]+torch.sum(dist[j][mask_pos_cross]))- dist[j][j]## since ij is iq

        # for negative pair
        else:
            neg1 = targets.ne(targets[i])
            neg2 = targets.ne(targets[j])
            neg = neg1 * neg2

            ## find hard-samples

            if dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()] < dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()] :
                circle_dist += dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()]

            else:
                circle_dist += dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()]


        return circle_dist


class TransitionLoss(nn.Module):
    def __init__(self, margin_1=0.3, alpha_1=0.5):
        super(TransitionLoss, self).__init__()
        self.margin_1 = margin_1
        self.alpha_1 = alpha_1

    def forward(self, feature, sub, targets):
        ## normalize each feature
        x  = self.normalize(feature)
        ## \
        loss = 0
        for i in range(feature.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 1
            is_neg = targets.ne(targets[i])

            intra_modality = sub.eq(sub[i])
            cross_modality = 1- intra_modality

            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality

            trans_pos, trans_neg = self.trans_dist(feature, mask_pos_intra, mask_pos_cross, \
                mask_neg_intra, mask_neg_cross)



            ap_pos_intra = torch.clamp(torch.add(trans_neg, self.margin_1-self.alpha_1),0)

            # print(ap_pos_intra)

            loss_ap = ap_pos_intra.mean().mean()



            an_neg_intra = torch.clamp(self.alpha_1-trans_pos,0)

            loss_an = an_neg_intra.mean().mean()

            loss += loss_ap + loss_an




            ## the similarity of cross-modality samples

            #circle_distmat = torch.mm(match_cross, match_cross.t())

            # intra_pos_distance =
            # feature[i][:]
        loss = loss/feature.size(0)

        return loss

    def trans_dist(self, feature, mask_pos_intra, mask_pos_cross, mask_neg_intra, mask_neg_cross):

        match_pos_cross = torch.mm(feature[mask_pos_intra],feature[mask_pos_cross].t())
        match_neg_cross = torch.mm(feature[mask_neg_intra],feature[mask_neg_cross].t())
        ab_pos_prob = F.softmax(match_pos_cross, dim=-1)
        ba_pos_prob = F.softmax(match_pos_cross.t(), dim=-1)
        ab_neg_prob = F.softmax(match_neg_cross, dim=-1)
        ba_neg_prob = F.softmax(match_neg_cross.t(), dim=-1)
        trans_pos = torch.mm(ab_pos_prob, ba_pos_prob)
        trans_neg = torch.mm(ab_neg_prob, ba_neg_prob)
        # trans_pos = torch.mm(match_pos_cross, match_pos_cross.t())
        # trans_neg = torch.mm(match_neg_cross, match_neg_cross.t())

        return torch.clamp(trans_pos,1e-12).sqrt(), torch.clamp(trans_neg,1e-12).sqrt()



    def visit_los(self, p, weight = 1.0):
        pass


    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x






class Circle_Rank_loss(nn.Module):

    ## Basic idea for cross_modality rank_loss 8

    def __init__(self, margin_1=1.0, margin_2=1.5, alpha_1=2.4, alpha_2=2.2, tval=1.0):
        super(Circle_Rank_loss, self).__init__()
        self.margin_1 = margin_1 # for same modality
        self.margin_2 = margin_2 # for different modalities
        self.alpha_1 = alpha_1 # for same modality
        self.alpha_2 = alpha_2 # for different modalities
        self.tval = tval

    def forward(self, x, targets, sub, norm = True):
        if norm:
            #x = self.normalize(x)
            x = torch.nn.functional.normalize(x, dim=1, p=2)

        dist_mat = self.euclidean_dist(x, x) # compute the distance


        loss = self.rank_loss(dist_mat, targets, sub)

        return loss, dist_mat

    def rank_loss(self, dist, targets, sub):
        loss = 0.0
        for i in range(dist.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 0
            is_neg = targets.ne(targets[i])


            intra_modality = sub.eq(sub[i])
            cross_modality = 1- intra_modality

            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality

            circ_pos = self.Circle_dist( dist, i, mask_pos_intra, mask_pos_cross, targets)
            circ_neg = self.Circle_dist( dist, i, mask_neg_intra, mask_neg_cross, targets, False)
            # print(circ_pos.max())
            # print(circ_neg.max())
            # print(dist[i][mask_pos_cross].max())
            # print(dist[i][mask_neg_cross].min())
            # time.sleep(10)

            #ap_pos_intra = torch.clamp(torch.add(dist[i][mask_pos_intra], self.margin_1-self.alpha_1),0)
            ap_pos_intra = torch.clamp(torch.add(circ_pos, self.margin_1-self.alpha_1),0)

            ap_pos_cross = torch.clamp(torch.add(dist[i][mask_pos_cross], self.margin_2-self.alpha_2),0)

            loss_ap = torch.div(torch.sum(ap_pos_intra), ap_pos_intra.size(0)+1e-5)
            loss_ap += torch.div(torch.sum(ap_pos_cross), ap_pos_cross.size(0)+1e-5)


            dist_an_intra = dist[i][mask_neg_intra]
            dist_an_cross = dist[i][mask_neg_cross]

            #an_less_intra = dist_an_intra[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]

            an_less_intra = circ_neg[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]


            an_less_cross = dist_an_cross[torch.lt(dist[i][mask_neg_cross], self.alpha_2)]

            an_weight_intra = torch.exp(self.tval*(-1* an_less_intra +self.alpha_1))
            an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5

            # an_weight_intra = torch.exp(self.tval *(-1*circ_neg+ self.alpha_1))
            # an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5

            an_weight_cross = torch.exp(self.tval*(-1* an_less_cross +self.alpha_2))
            an_weight_cross_sum = torch.sum(an_weight_cross)+1e-5
            an_sum_intra = torch.sum(torch.clamp(torch.mul(self.alpha_1-circ_neg,an_weight_intra),0))
            an_sum_cross = torch.sum(torch.mul(self.alpha_2-an_less_cross,an_weight_cross))
            # an_sum_intra = torch.clamp(self.alpha_1-circ_neg, 0).sum()

            loss_an =torch.div(an_sum_intra,an_weight_intra_sum ) +torch.div(an_sum_cross,an_weight_cross_sum )
            #loss_an = torch.div(an_sum_intra, circ_neg.size(0)+1e-5) +torch.div(an_sum_cross,an_weight_cross_sum )

            #loss_an = torch.div(an_sum_cross,an_weight_cross_sum )
            loss += loss_ap + loss_an


            #loss += loss_an


        return loss * 1.0/ dist.size(0)

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist



    def Circle_dist(self, dist_mat, i, js, mask_cross,  targets, is_pos=True):

        #circle_dist = -dist_mat[i][js] # original distance


        pos_dist_i = torch.sum(dist_mat[i][mask_cross])

        ## we will explore this after

        # print( dist_mat[i][mask_pos_cross].shape)
        # similar = dist_mat[js][mask_pos_cross].view(-1,1).mm(dist_mat[js][mask_pos_cross].view(-1,1).t())
        # print(similar.shape)

        neg1 = targets.ne(targets[i])

        circle_dist=[]


        # for postive pair
        if is_pos:
            tmp = 0
            for j in torch.nonzero(js):

                #print(torch.sum(dist_mat[j][mask_pos_cross], dim=1))
                tmp = pos_dist_i+torch.sum(dist_mat[j.item()][mask_cross])

                circle_dist.append(tmp/8.0) # we think

        # for negative pair
        else:
            neg1 = neg1*mask_cross
            for jt in torch.nonzero(js):
                j = jt.item()
                neg2 = targets.ne(targets[j])

                neg = neg2*neg1

            ## find hard-samples
                tmp =  dist_mat[i][dist_mat[i][neg].argmin()]+dist_mat[j][dist_mat[i][neg].argmin()]+ \
                        dist_mat[i][dist_mat[j][neg].argmin()]+dist_mat[j][dist_mat[j][neg].argmin()]
                # if dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()] < dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()] :
                #     tmp= dist[i][dist_mat[i][neg].argmin()]+dist[j][dist_mat[i][neg].argmin()]
                # else:
                #     circle_dist += dist[i][dist_mat[j][neg].argmin()]+dist[j][dist_mat[j][neg].argmin()]
                circle_dist.append(tmp/4.0)

        circle_dist = torch.stack(circle_dist, dim=0)
        circle_dist += dist_mat[i][js]
        # print(circle_dist)

        return circle_dist/2.0



class ASS_loss(nn.Module):
    def __init__(self, walker_loss=1.0, visit_loss=1.0):
        super(ASS_loss, self).__init__()
        self.walker_loss = walker_loss
        self.visit_loss = visit_loss
        self.ce = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, feature, sub, targets):
        ## normalize
        feature = torch.nn.functional.normalize(feature, dim=1, p=2)
        loss = 0.0
        for i in range(feature.size(0)):
            cross_modality = sub.ne(sub[i])

            # is_pos = targets.eq(targets[i])
            # is_neg = targets.ne(targets[i])
            p_logit_ab, v_loss_ab = self.probablity(feature, cross_modality,  targets)
            p_logit_ba, v_loss_ba = self.probablity(feature, ~cross_modality, targets)
            n1 = targets[cross_modality].size(0)
            n2 = targets[~cross_modality].size(0)

            is_pos_ab = targets[cross_modality].expand(n1,n1).eq(targets[cross_modality].expand(n1,n1).t())

            p_target_ab = is_pos_ab.float()/torch.sum(is_pos_ab, dim=1).float().expand_as(is_pos_ab)

            is_pos_ba = targets[~cross_modality].expand(n2,n2).eq(targets[cross_modality].expand(n2,n2).t())
            p_target_ba = is_pos_ba.float()/torch.sum(is_pos_ba, dim=1).float().expand_as(is_pos_ba)


            p_logit_ab = self.logsoftmax(p_logit_ab)
            p_logit_ba = self.logsoftmax(p_logit_ba)


            #loss += self.ce(p_logit_ab, p_target_ab) + self.ce(p_logit_ab,p_target_ba )
            loss += (- p_target_ab * p_logit_ab).mean(0).sum()+ (- p_target_ba * p_logit_ba).mean(0).sum()

            loss += 1.0*(v_loss_ab+v_loss_ba)

        return loss/feature.size(0)/4

    def probablity(self, feature, cross_modality, target):
        a = feature[cross_modality]
        b = feature[~cross_modality]

        match_ab = a.mm(b.t())


        p_ab = F.softmax(match_ab, dim=-1)
        p_ba = F.softmax(match_ab, dim=-1)
        p_aba = torch.log(1e-8+p_ab.mm(p_ba))


        #visit_loss = self.visit(p_ab)
        visit_loss = self.new_visit(p_ab, target, cross_modality)

        return p_aba, visit_loss

    def visit(self, p_ab):
        p_ab = torch.log(1e-8 +p_ab)
        visit_probability = p_ab.mean(dim=0).expand_as(p_ab)

        p_target_ab = torch.zeros_like(p_ab).fill_(1).div(p_ab.size(0))

        loss = (- p_target_ab * visit_probability).mean(0).sum()

        return loss

    def new_visit(self, p_ab, target, cross_modality):
        p_ab = torch.log(1e-8 +p_ab)
        visit_probability = p_ab.mean(dim=0).expand_as(p_ab)
        n1 = target[cross_modality].size(0)
        n2 = target[~cross_modality].size(0)
        p_target_ab = target[cross_modality].expand(n1,n1).eq(target[~cross_modality].expand(n2,n2))
        p_target_ab = p_target_ab.float()/torch.sum(p_target_ab, dim=1).float().expand_as(p_target_ab)
        loss = (- p_target_ab * visit_probability).mean(0).sum()
        return loss

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x
