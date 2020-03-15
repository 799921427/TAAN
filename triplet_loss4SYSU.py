from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss.CrossTriplet import CrossTriplet as TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import CamRandomIdentitySampler as RandomIdentitySampler
from reid.utils.data.sampler import CamSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from utlis import CrossEntropyLabelSmooth
def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval):
    root = osp.join(data_dir, name)
    print(root)
    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    trainvallabel = dataset.trainvallabel
    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])
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
        sampler=CamSampler(list(set(dataset.query)), [2,5]),
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(list(set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.gallery)), [0,1,3,4], 4),
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

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (224, 224)
    dataset, num_classes, train_loader, trainvallabel, val_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)
    # Create model
    model, model_discriminator = models.create(args.arch, num_classes=num_classes, num_features=args.features)
    model = model.cuda()
    model_discriminator = model_discriminator.cuda()
    evaluator = Evaluator(model)

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model'])
        model_discriminator.load_state_dict(checkpoint['model_discriminator'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}  "
          .format(start_epoch))
    #model = nn.DataParallel(model).cuda()

    if args.evaluate:
        metric.train(model, train_loader)
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)
        exit()

    # Init
    current_margin = args.margin
    criterion_z = CrossEntropyLabelSmooth(num_classes=num_classes).cuda()#nn.CrossEntropyLoss().cuda()
    criterion_I = TripletLoss(margin= current_margin).cuda()
    criterion_D = nn.CrossEntropyLoss().cuda()
    print(args)

    # Observe that all parameters are being optimized
    if args.arch == 'ide':
        ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
    else:
        ignored_params = list(map(id, model.classifier.parameters())) + list(map(id, model.base.fc.parameters() )) 
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    for k,v in model.named_parameters():
    	print(k)
    # optimizer_ft = optim.Adam([
    #          {'params': base_params, 'lr': 0.0001},
    #          {'params': model.classifier.parameters(), 'lr': 0.0001},
    #      ], weight_decay=5e-4)

    # optimizer_ft = torch.optim.SGD([
    #          {'params': filter(lambda p: p.requires_grad,base_params), 'lr': 0.0001},
    #          {'params': model.classifier.parameters(), 'lr': 0.0001},
    #     ],
    #     momentum=0.9,
    #     weight_decay=5e-4,
    #     nesterov=True)


    optimizer_ft = torch.optim.Adam([
             {'params': filter(lambda p: p.requires_grad,base_params), 'lr': 1e-4},
             {'params': model.classifier.parameters(), 'lr': 1e-4},
        ],
        weight_decay=5e-4)

    optimizer_discriminator = torch.optim.Adam([
             {'params': model_discriminator.model.parameters(), 'lr': 1e-4},
             {'params': model_discriminator.classifier.parameters(), 'lr': 1e-4}
        ],
        weight_decay=5e-4)

    # optimizer_discriminator = optim.Adam([
    #          {'params': model_discriminator.model.parameters(), 'lr': 0.0001},
    #          {'params': model_discriminator.classifier.parameters(), 'lr': 0.0001}
    #      ], weight_decay=5e-4)

    # Trainer
    trainer = Trainer(model, model_discriminator, criterion_z, criterion_I, criterion_D, trainvallabel, 1, 1 ,0.15 , 0.05, 5) # c: 0.15, u: 0.05

    flag = 1
    best_top1 = -1
    # Start training
    for epoch in range(start_epoch, args.epochs):
        triple_loss, tot_loss = trainer.train(epoch, train_loader, optimizer_ft, optimizer_discriminator)
        '''
        if (flag == 1 and triple_loss < 0.1):
            for g in optimizer_ft.param_groups:
                g['lr'] = 0.001
            flag = 0
        if (flag == 0 and triple_loss > 0.1):
            for g in optimizer_ft.param_groups:
                g['lr'] = 0.0001
            flag = 1
        '''
        save_checkpoint({
            'model': model.state_dict(),
            'model_discriminator': model_discriminator.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, False, epoch, args.logs_dir, fpath='checkpoint.pth.tar')
        print(epoch)
        if epoch < 200:
            continue
        if not epoch % 10 ==0:
            continue
        top1 = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'model': model.state_dict(),
            'model_discriminator': model_discriminator.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, epoch, args.logs_dir, fpath='checkpoint.pth.tar')

    print('Test with best model:')
    print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    checkpoint = load_checkpoint(osp.join(args.logs_dir,'model_best.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    metric.train(model, train_loader)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, metric)
    print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
            choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
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
    # optimizer
    # parser.add_argument('--lr', type=float, default=0.0002,
    #                     help="learning rate of all parameters")
    # parser.add_argument('--weight-decay', type=float, default=5e-4)
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
