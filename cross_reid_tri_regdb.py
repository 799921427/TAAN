from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from ranger import Ranger
from torch.optim import lr_scheduler
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss.CrossTriplet import CrossTriplet as TripletLoss
from reid.loss.triplet import TripletLoss as Triplet
from reid.tri_trainers import Trainer
from reid.evaluators_regdb import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import CamRandomIdentitySampler as RandomIdentitySampler
from reid.utils.data.sampler import CamSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, save_checkpoint_s, save_checkpoint_ir
from utlis import RandomErasing, WarmupMultiStepLR, CrossEntropyLabelSmooth, Rank_loss, Circle_Rank_loss



def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval, flip_prob, padding, re_prob, ii):
    root = osp.join(data_dir, name)
    print(root)

    dataset = datasets.create(name, root, split_id=split_id, ii=ii)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    trainvallabel = dataset.trainvallabel
    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)


    train_transformer = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=flip_prob),
        T.Pad(padding),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        RandomErasing(probability=re_prob, mean=[0.485, 0.456, 0.406])
        ])

    # train_transformer = T.Compose([
    #     T.RandomSizedRectCrop(height, width),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     normalizer,
    # ])

    test_transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=32, num_workers=workers,
        shuffle=False, pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(list(set(dataset.query)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.query)), [2]),
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(list(set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.gallery)), [0], 1),
        shuffle=False, pin_memory=True)

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
            transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    return dataset, num_classes, train_loader, trainvallabel, val_loader, query_loader, gallery_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    for ii in range(0,9):
        #if ii == 5 or ii == 6: ii = ii - 4
        #print(ii)
        #continue
        if not osp.exists(args.logs_dir+'/{}'.format(ii)):
            os.mkdir(args.logs_dir+'/{}'.format(ii))
        sys.stdout = Logger(osp.join(args.logs_dir+'/{}/log'.format(ii)))
        dataset, num_classes, train_loader, trainvallabel, val_loader, query_loader, gallery_loader = \
            get_data(args.dataset, args.split, args.data_dir, args.height,
            args.width, args.batch_size, args.num_instances, args.workers,
            args.combine_trainval, args.flip_prob, args.padding, args.re_prob,ii+1)
        if not args.evaluate:
            sys.stdout = Logger(osp.join(args.logs_dir+'/log'))

        if args.height is None or args.width is None:
            args.height, args.width = (144, 56) if args.arch == 'inception' else (256, 128)
        model_t = models.create(args.arch, num_classes=num_classes, num_features=args.features, attention_mode=args.att_mode)
        model_s = models.create(args.arch, num_classes=num_classes, num_features=args.features, attention_mode=args.att_mode)
        model_ir = models.create(args.arch, num_classes=num_classes, num_features=args.features, attention_mode=args.att_mode)
    #    print(model)
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        model_t = nn.DataParallel(model_t, device_ids=[0,1,2])
        model_t.to(device)
        model_s = nn.DataParallel(model_s, device_ids=[0,1,2])
        model_s.to(device)
        model_ir = nn.DataParallel(model_ir, device_ids=[0,1,2])
        model_ir.to(device)
        print(num_classes)
        #model = model.cuda()
        #model_discriminator = model_discriminator.cuda()
        #model_discriminator = nn.DataParallel(model_discriminator, device_ids=[0,1,2])
        #model_discriminator.to(device)

        evaluator = Evaluator(model_t)
        metric = DistanceMetric(algorithm=args.dist_metric)
        evaluator_s = Evaluator(model_s)
        metric_s = DistanceMetric(algorithm=args.dist_metric)
        evaluator_ir = Evaluator(model_ir)
        metric_ir = DistanceMetric(algorithm=args.dist_metric)

        start_epoch = 0
        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            model.load_state_dict(checkpoint['model'])
            model_discriminator.load_state_dict(checkpoint['model_discriminator'])
            start_epoch = checkpoint['epoch']
            print("=> Start epoch {}".format(start_epoch))

        if args.evaluate:
            metric.train(model, train_loader)
            evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
            exit()

        current_margin = args.margin
        #criterion_z = nn.CrossEntropyLoss().cuda()

        criterion_z = CrossEntropyLabelSmooth(num_classes= num_classes, epsilon=0.3).cuda()
        criterion_att = nn.MSELoss().cuda()
        #criterion_I = TripletLoss(margin= current_margin).cuda()
        #criterion_I = Circle_Rank_loss(margin_1=args.margin_1, margin_2=args.margin_2, alpha_1=args.alpha_1, alpha_2=args.alpha_2).cuda()
        criterion_I = Rank_loss(margin_1= args.margin_1, margin_2 =args.margin_2, alpha_1 =args.alpha_1, alpha_2= args.alpha_2).cuda()
        criterion_t = Triplet(margin=current_margin).cuda()

        print(args)

        if args.arch == 'ide':
            ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
        else:
            ignored_params = list(map(id, model_t.module.classifier.parameters())) + list(map(id, model_t.module.attention_module.parameters()))
            ignored_params_s = list(map(id, model_s.module.classifier.parameters())) + list(map(id, model_s.module.attention_module.parameters()))
            ignored_params_ir = list(map(id, model_ir.module.classifier.parameters())) + list(map(id, model_ir.module.attention_module.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model_t.parameters())
        base_params_s = filter(lambda p: id(p) not in ignored_params_s, model_s.parameters())
        base_params_ir = filter(lambda p: id(p) not in ignored_params_ir, model_ir.parameters())


        if args.use_adam:
            optimizer_ft = torch.optim.Adam([
            #print("Ranger")
            #optimizer_ft = Ranger([
                {'params': filter(lambda p: p.requires_grad,base_params), 'lr': args.lr},
                {'params': filter(lambda p: p.requires_grad,base_params_s), 'lr':args.lr},
                {'params': filter(lambda p: p.requires_grad,base_params_ir), 'lr':args.lr},
                {'params': model_t.module.classifier.parameters(), 'lr': args.lr},
                {'params': model_t.module.attention_module.parameters(), 'lr': args.lr},
                {'params': model_s.module.classifier.parameters(), 'lr': args.lr},
                {'params': model_s.module.attention_module.parameters(), 'lr': args.lr},
                {'params': model_ir.module.classifier.parameters(), 'lr': args.lr},
                {'params': model_ir.module.attention_module.parameters(), 'lr': args.lr},
                ],
                weight_decay=5e-4)

            #optimizer_discriminator = torch.optim.Adam([
            #    {'params': model_discriminator.module.model.parameters(), 'lr': args.lr},
            #    {'params': model_discriminator.module.classifier.parameters(), 'lr': args.lr}
            #    ],
            #    weight_decay=5e-4)


        else:
            optimizer_ft = torch.optim.SGD([
                {'params': filter(lambda p: p.requires_grad,base_params), 'lr': args.lr},
                {'params': model.classifier.parameters(), 'lr': args.lr},
                ],
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True)
            optimizer_discriminator = torch.optim.SGD([
                {'params': model_discriminator.model.parameters(), 'lr': args.lr},
                {'params': model_discriminator.classifier.parameters(), 'lr': args.lr},
                ],
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True)

        scheduler = WarmupMultiStepLR(optimizer_ft, args.mile_stone, args.gamma, args.warmup_factor,
                                            args.warmup_iters, args.warmup_methods)

        trainer = Trainer(model_t, model_s, model_ir, criterion_z, criterion_I, criterion_att, criterion_t, trainvallabel, 1, 1 ,args.rgb_w, args.ir_w, 1000)

        flag = 1
        best_top1 = -1
        # best_top1_s = -1
        # best_top1_ir = -1
        # Start training
        for epoch in range(start_epoch, args.epochs):
            scheduler.step()
            triple_loss, tot_loss = trainer.train(epoch, train_loader, optimizer_ft)

            save_checkpoint({
                'model': model_t.module.state_dict(),
                #'model_discriminator': model_discriminator.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, False, epoch, (args.logs_dir+'/{}'.format(ii)), fpath='checkpoint.pth.tar')

            if epoch < 1:
                continue
            if not epoch % 10 ==0:
                continue


            top1, cmc, mAP = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)

            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint({
                'model': model_t.module.state_dict(),
                #'model_discriminator': model_discriminator.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, epoch, (args.logs_dir+'/{}'.format(ii)), fpath='checkpoint.pth.tar')
       # print('Test with best model:')
       # print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
        #        format(epoch, top1, best_top1, ' *' if is_best else ''))

        #checkpoint = load_checkpoint(osp.join((args.logs_dir+'/{}'.format(ii)),'model_best.pth.tar'))
       # model_t.load_state_dict(checkpoint['model'])
       # metric.train(model, train_loader)
       # _, best_cmc, best_mAP= evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)
       # if ii == 0:
       #     all_cmc = best_cmc
       #     all_mAP = best_mAP

        #else:
        #    all_cmc = all_cmc + best_cmc
        #    all_mAP = all_mAP + best_mAP
        #del model, metric, evaluator, scheduler, criterion_z, criterion_I, criterion_D
   # print('------------------Final-Results------------------')
   # print('Mean AP: {:4.2%}'.format(all_mAP/10.0))
   # print('CMC Scores{:>12}'.format('RegDB')
   # )
   # for k in [1,10,20]:
   #     print('  top-{:<4}{:12.2%}'
   #           .format(k, all_cmc[k -1]/10.0)
   #           )

    #print('Test with best model:')
    #print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
    #          format(epoch, top1, best_top1, ' *' if is_best else ''))

    #checkpoint = load_checkpoint(osp.join(args.logs_dir,'model_best.pth.tar'))
    #model.load_state_dict(checkpoint['model'])
    #metric.train(model, train_loader)
    #evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross_modality for Person Re-identification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)


    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default= 128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    # transformer
    parser.add_argument('--flip_prob', type=float, default=0.5)
    parser.add_argument('--re_prob', type=float, default=0.0)
    parser.add_argument('--padding', type=int, default=0)

    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--margin_1', type=float, default=0.9,
                        help="margin_1 of the triplet loss, default: 0.5")
    parser.add_argument('--margin_2', type=float, default=1.5,
                        help="margin_1 of the triplet loss, default: 0.5")
    parser.add_argument('--alpha_1', type=float, default=2.4,
            help="alpha_1 of the triplet loss, default: 0.5")
    parser.add_argument('--alpha_2', type=float, default=2.2,
                        help="alpha_2 of the triplet loss, default: 0.5")

    # attention mode
    parser.add_argument('--att_mode', type=int, default=1)

    # weight of rgb and ir
    parser.add_argument('--rgb_w', type=float, default=0.5,
                        help='weight of rgb branch loss')
    parser.add_argument('--ir_w', type=float, default=0.9,
                        help='weight of ir branch loss')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4,
                         help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--use_adam', action='store_true',
                    help="use Adam as the optimizer, elsewise SGD ")
    parser.add_argument('--gamma', type=float, default = 0.1,
                    help="gamma for learning rate decay")

    parser.add_argument('--mile_stone', type=list, default=[120, 1000])

    parser.add_argument('--warmup_iters', type=int, default = 5)
    parser.add_argument('--warmup_methods', type=str, default = 'linear', choices=('linear', 'constant'))
    parser.add_argument('--warmup_factor', type=float, default = 0.01 )

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))


    main(parser.parse_args())
