"""
the general training framework
"""

from __future__ import print_function

import argparse
import os
import random
import shutil
import socket
import sys
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
from distiller_zoo import DistillKL, HintLoss
from helper.loops import train_distill_Multi_teacher as train
from helper.util import adjust_learning_rate
from helper.valid_loop import validate, validate_multi
from models import model_dict
from models.util import ConvReg, LinearEmbed
from setting import (cifar10_teacher_model_name, cifar100_teacher_model_name,
                     teacher_model_path_dict)


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
    parser.add_argument('--init_epochs', type=int, default=30,
                        help='init training for two-stage methods')

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
    parser.add_argument('--distill_decay', action='store_true', default=False,
                        help='distillation decay')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'ResNet18', 'ResNet34',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
                                 'wrn_cifar10', 'resnet_feat_at_110', 'resnet_feat_at_20', 'resnet_feat_at_14'])
    # model
    parser.add_argument('--model_t', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'ResNet18', 'ResNet34',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
                                 'wrn_cifar10', 'resnet_feat_at_110', 'resnet_feat_at_20', 'resnet_feat_at_14'])

    # distillation
    parser.add_argument('--distill', type=str, default='kd',
                        choices=['kd', 'hint', 'l2', ])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float,
                        default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float,
                        default=0.9, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float,
                        default=0, help='weight balance for other losses')
    parser.add_argument('--show_teacher_acc', type=bool, default=False)

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')

    parser.add_argument('--nesterov', action='store_true',
                        help='if use nesterov')
    parser.add_argument('--preact', action='store_true',
                        help='preact features')
    parser.add_argument("--teacher_num", type=int, default=5,
                        help='use multiple teacher')
    parser.add_argument('--description', default='None',
                        type=str, help='description of the experiment')
    # hint layer
    parser.add_argument('--hint_layer', default=2,
                        type=int, choices=[0, 1, 2, 3, 4])

    parser.add_argument("--ensemble_method", default="AEKD",
                        type=str, choices=["AEKD", "AVERAGE_LOSS"])
    parser.add_argument('-C', type=float,
                        default=0.6, help='torelance for disagreement among teachers')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--svm_norm', default=False, action="store_true",
                        help='if use norm when compute with svm')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    # if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    #     opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = f'./save/student_models/{opt.dataset}'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.dataset == 'cifar100':
        opt.teacher_model_name = cifar100_teacher_model_name

    elif opt.dataset == 'cifar10':
        opt.teacher_model_name = cifar10_teacher_model_name

    opt.teacher_num = min(opt.teacher_num, len(opt.teacher_model_name))
    opt.model_t_name = [name.split("-")[1]
                        for name in opt.teacher_model_name[:opt.teacher_num]]
    opt.teacher_name_str = "_".join(list(set(opt.model_t_name)))

    opt.model_name = f'S-{opt.model_s}_{opt.dataset}_{opt.distill}_' \
        f'r-{opt.gamma}_a-{opt.alpha}_b-{opt.beta}_{opt.trial}_seed_{opt.seed}'

    opt.model_name = '_'.join([opt.model_name, str(
        opt.epochs), '-'.join([str(i) for i in opt.lr_decay_epochs])])

    if opt.teacher_num > 1:
        opt.model_name = opt.model_name + '_' + str(opt.teacher_num) + '_' + \
            opt.teacher_name_str + "_" + opt.ensemble_method
    else:
        opt.ensemble_method = "AVERAGE_LOSS"
        opt.model_name = opt.model_name + '_' + opt.teacher_name_str

    if opt.ensemble_method == 'AEKD':
        opt.nu = 1 / (opt.C * opt.teacher_num)
        opt.model_name = opt.model_name + "_" + str(opt.C)
        if opt.svm_norm:
            opt.model_name = opt.model_name + "_norm"
    if opt.description != 'None':
        opt.model_name = opt.model_name + '_' + opt.description

    opt.save_folder = os.path.join(
        opt.model_path, opt.model_s, opt.distill, opt.model_name)
    if os.path.exists(f'{opt.save_folder}/ckpt_epoch_{opt.epochs}.pth'):
        print(f'Already have {opt.save_folder}/ckpt_epoch_{opt.epochs}.pth!!!')
        exit()

    if opt.distribution:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(opt.master_port)
        opt.gpu_num = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend="nccl")

        opt.batch_size = int(opt.batch_size / opt.gpu_num)

    if not opt.distribution or opt.local_rank == 0:
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)

    opt.tb_folder = os.path.join(
        opt.model_path, opt.model_s, opt.distill, opt.model_name, 'tb')
    if not opt.distribution or opt.local_rank == 0:
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

    print(opt.save_folder)
    return opt


def load_teacher(model_path, n_cls, model_t):
    print('==> loading teacher model')
    model = model_dict[model_t](num_classes=n_cls)
    print(model_path)
    checkpoint = torch.load(model_path)
    if 'state_dict' in checkpoint.keys():
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['model'])

    print('==> done')
    return model


def load_teacher_list(n_cls, teacher_model_name, model_t_name):
    print('==> loading teacher model list')
    teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], n_cls, model_t)
                          for (model_name, model_t) in zip(teacher_model_name, model_t_name)]
    print('==> done')
    return teacher_model_list


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
    best_epoch = 0

    # tensorboard logger
    if opt.distribution:
        dist.barrier()
    if not opt.distribution or opt.local_rank == 0:
        writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

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
    set_seed(seed)
    model_t_list = load_teacher_list(
        n_cls, opt.teacher_model_name, opt.model_t_name)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    # To get feature map's shape
    data = torch.randn(2, 3, 32, 32)
    feat_t_list = []
    model_s.eval()
    for model_t in model_t_list:
        model_t.eval()
    for model_t in model_t_list:
        feat_t, _ = model_t(data, is_feat=True)
        feat_t_list.append(feat_t)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_KLdiv = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        for i, feat_t in enumerate(feat_t_list):
            regress_s = ConvReg(feat_s[opt.hint_layer].shape,
                                feat_t[opt.hint_layer].shape)
            module_list.append(regress_s)
            trainable_list.append(regress_s)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_KLdiv)
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          nesterov=opt.nesterov)

    # append teacher after optimizer to avoid weight_decay
    module_list.extend(model_t_list)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = False
        if opt.distribution:
            module_list = [DDP(model,
                               device_ids=[opt.local_rank],
                               output_device=opt.local_rank)
                           for model in module_list]

    # validate teacher accuracy
    if opt.show_teacher_acc:
        if opt.teacher_num > 1:
            teacher_acc, teacher_acc_top5, _, teacher_acc_list = validate_multi(
                val_loader, model_t_list, criterion_cls, opt)
        else:
            model_t = model_t_list[0]
            teacher_acc, teacher_acc_top5, _ = validate(
                val_loader, model_t, criterion_cls, opt)
        print('teacher accuracy: ', teacher_acc, teacher_acc_top5)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, test_acc_top5, test_loss = validate(
            val_loader, model_s, criterion_cls, opt)

        # tensorboard
        if not opt.distribution or opt.local_rank == 0:
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc_top5', test_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_top5 = test_acc_top5
            best_epoch = epoch
            if not opt.distribution or opt.local_rank == 0:
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                    'best_acc_top5': best_acc_top5,
                }
                save_file = os.path.join(
                    opt.save_folder, '{}_best.pth'.format(opt.model_s))
                print('saving the best model!')
                torch.save(state, save_file)

        # regular saving
        if not opt.distribution or opt.local_rank == 0:
            if epoch % opt.save_freq == 0:
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'accuracy': test_acc,
                    'accuracy_top5': test_acc_top5,
                }
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

            print('best accuracy:', best_acc, best_acc_top5)
            f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')
            f.write(f'best accuracy:  {best_acc}, {best_acc_top5}')
            f.close()

    if not opt.distribution or opt.local_rank == 0:
        if opt.show_teacher_acc:
            print('teacher accuracy: ', teacher_acc, teacher_acc_top5)
        print('best accuracy:', best_acc, best_acc_top5, "at epoch", best_epoch)
        f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')
        f.write(' * best Acc@1 {top1:.3f} Acc@5 {top5:.3f} at epoch {best_epoch}\n'
                .format(top1=best_acc, top5=best_acc_top5, best_epoch=best_epoch))
        f.close()

        # save model
        state = {
            'opt': opt,
            'model': model_s.state_dict(),
        }
        save_file = os.path.join(
            opt.save_folder, '{}_last.pth'.format(opt.model_s))
        torch.save(state, save_file)


if __name__ == '__main__':
    print(sys.version)
    main()
