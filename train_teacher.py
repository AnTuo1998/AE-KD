from __future__ import print_function

import argparse
import os
import random
import socket
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from helper.loops import train_vanilla as train
from helper.loops import validate
from helper.util import adjust_learning_rate
from models import model_dict


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    # distribution
    parser.add_argument('--master_port', default=29501,
                        type=int, help='master port for distributed')
    parser.add_argument('--distribution', action="store_true", default=False,
                        help='distribution')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='pytorch dist local rank')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='gpu_num')

    parser.add_argument('--print_freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int,
                        default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int,
                        default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--warm_up_epoch', type=int,
                        default=20, help='weight decay')

    # dataset
    parser.add_argument('--model', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
                                 'wrn_cifar10', 'resnet_feat_at_110', 'resnet_feat_at_20', 'resnet_feat_at_14'])
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10'], help='dataset')
    parser.add_argument('--description', type=str,
                        default='None', help='description')
    parser.add_argument('--nesterov', action='store_true',
                        help='if use nesterov')

    parser.add_argument('-t', '--trial', type=int,
                        default=0, help='the experiment id')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/teacher_models/{}'.format(opt.dataset)

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}_nesterov_{}_step_{}_bs_{}_seed_{}_ep_{}'.format(opt.model, opt.dataset,
                                                                                                    opt.learning_rate,
                                                                                                    opt.weight_decay, opt.trial,
                                                                                                    opt.nesterov,
                                                                                                    opt.lr_decay_epochs.replace(
                                                                                                        ' ', '').replace(',', '-'),
                                                                                                    opt.batch_size, opt.seed,
                                                                                                    opt.epochs)
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.distribution:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(opt.master_port)
        opt.gpu_num = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend="nccl")

        opt.batch_size = int(opt.batch_size / opt.gpu_num)

    opt.tb_folder = os.path.join(opt.model_path, opt.model_name, 'tb')
    if not opt.distribution or opt.local_rank == 0:
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not opt.distribution or opt.local_rank == 0:
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
    print(opt.save_folder)

    return opt


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)


def main():
    opt = parse_option()
    seed = opt.seed
    if opt.distribution:
        seed += opt.local_rank
        torch.cuda.set_device(opt.local_rank)
    set_seed(seed)
    best_acc = 0
    best_acc_top5 = 0

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                            num_workers=opt.num_workers,
                                                            distribution=opt.distribution)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                           num_workers=opt.num_workers,
                                                           distribution=opt.distribution)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # model
    set_seed(opt.seed)
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          nesterov=opt.nesterov)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = False
        if opt.distribution:
            model = DDP(model,
                        device_ids=[opt.local_rank],
                        output_device=opt.local_rank)

    # tensorboard
    if opt.distribution:
        dist.barrier()
    if not opt.distribution or opt.local_rank == 0:
        writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(
            epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()

        test_acc, test_acc_top5, test_loss = validate(
            val_loader, model, criterion, opt)

        if not opt.distribution or opt.local_rank == 0:
            print('epoch {}, total time {:.2f}'.format(
                epoch, time2 - time1))
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            writer.add_scalar('test_acc_top5', test_acc_top5, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_top5 = test_acc_top5
            if not opt.distribution or opt.local_rank == 0:
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                    'best_acc_top5': best_acc_top5,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(
                    opt.save_folder, '{}_best.pth'.format(opt.model))
                print('saving the best model!')
                torch.save(state, save_file)

        # regular saving
        if not opt.distribution or opt.local_rank == 0:
            if epoch % opt.save_freq == 0:
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'accuracy': test_acc,
                    'accuracy_top5': test_acc_top5,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

            print('best accuracy:', best_acc, best_acc_top5)
            f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')
            f.write(f'best accuracy:  {best_acc}, {best_acc_top5}')
            f.close()

    # save model
    if not opt.distribution or opt.local_rank == 0:
        state = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(
            opt.save_folder, '{}_last.pth'.format(opt.model))
        torch.save(state, save_file)


if __name__ == '__main__':
    main()
