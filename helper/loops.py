from __future__ import division, print_function

import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .optimization import find_optimal_svm
from .util import AverageMeter, accuracy, reduce_tensor


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
            f.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    f.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
            .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')

    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target = data
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        preact = opt.preact
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.zeros(1).float().cuda()
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'l2':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        else:
            raise NotImplementedError(opt.distill)
        if epoch == 1:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div
        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_norm=20)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  '{gamma}*loss_cls ({cls:.4f}), {alpha}*loss_div ({div:.4f}), {beta}*{distill} ({kd:.4f})'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5,
                      gamma=opt.gamma, cls=loss_cls.item(), alpha=opt.alpha,
                      div=loss_div.item(), beta=opt.beta, distill=opt.distill, kd=loss_kd.item()))
            f.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    '{gamma}*loss_cls ({cls:.4f}), {alpha}*loss_div ({div:.4f}), {beta}*{distill} ({kd:.4f})\n'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5,
                        gamma=opt.gamma, cls=loss_cls.item(), alpha=opt.alpha, distill=opt.distill,
                        div=loss_div.item(), beta=opt.beta, kd=loss_kd.item()))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    f.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
            .format(top1=top1, top5=top5))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          idx, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))
                f.write('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                            idx, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        f.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def train_distill_Multi_teacher(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation with multiple teacher"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    [model_t.eval() for model_t in module_list[-opt.teacher_num:]]

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    # model_t = module_list[-1]
    model_t_list = module_list[-opt.teacher_num:]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')

    end = time.time()
    for idx, data in enumerate(train_loader):
        # ================prepare data==================
        input, target = data
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        preact = opt.preact
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        feat_t_list = []
        logit_t_list = []
        with torch.no_grad():
            for model_t in model_t_list:
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]
                feat_t_list.append(feat_t)
                logit_t_list.append(logit_t)

        # ================compute loss====================
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        # kl div
        if opt.ensemble_method == "AVERAGE_LOSS":
            loss_div_list = [criterion_div(logit_s, logit_t)
                             for logit_t in logit_t_list]
            loss_div = torch.stack(loss_div_list).mean(0)
        elif opt.ensemble_method == "AEKD":
            loss_div_list = []
            grads = []
            logit_s.register_hook(lambda grad: grads.append(
                Variable(grad.data.clone(), requires_grad=False)))
            for logit_t in logit_t_list:
                optimizer.zero_grad()
                loss_s = criterion_div(logit_s, logit_t)
                loss_s.backward(retain_graph=True)
                loss_div_list.append(loss_s)

            scale = find_optimal_svm(torch.stack(grads),
                                     nu=opt.nu,
                                     gpu_id=opt.local_rank,
                                     is_norm=opt.svm_norm)
            losses_div_tensor = torch.stack(loss_div_list)
            if torch.cuda.is_available():
                scale = scale.cuda()
                losses_div_tensor.cuda()
            loss_div = torch.dot(scale, losses_div_tensor)
        else:
            raise NotImplementedError

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.zeros(1).float().cuda()
        elif opt.distill == 'hint':
            loss_hint_list = []
            f_s_list = [regress_s(feat_s[opt.hint_layer])
                        for regress_s in module_list[1:1+opt.teacher_num]]
            f_t_list = [f_t[opt.hint_layer] for f_t in feat_t_list]
            if opt.ensemble_method == "AVERAGE_LOSS" or \
                    opt.ensemble_method == "AVERAGE_SOFTOUT":
                for f_s, f_t in zip(f_s_list, f_t_list):
                    loss_hint = criterion_kd(f_s, f_t)
                    loss_hint_list.append(loss_hint)
                loss_kd = torch.mean(torch.tensor(loss_hint_list))
                if torch.cuda.is_available():
                    loss_kd = loss_kd.cuda()
            elif opt.ensemble_method == "AEKD":
                grads = []
                feat_s[opt.hint_layer].register_hook(lambda grad: grads.append(
                    Variable(grad.data.clone(), requires_grad=False)))
                for f_s, f_t in zip(f_s_list, f_t_list):
                    optimizer.zero_grad()
                    loss_hint = criterion_kd(f_s, f_t)
                    loss_hint.backward(retain_graph=True)
                    loss_hint_list.append(loss_hint)

                scale = find_optimal_svm(torch.stack(grads),
                                         nu=opt.nu,
                                         gpu_id=opt.local_rank)

                loss_hint_tensor = torch.stack(loss_hint_list)
                if torch.cuda.is_available():
                    scale = scale.cuda()
                    loss_hint_tensor = loss_hint_tensor.cuda()
                loss_kd = torch.dot(scale, loss_hint_tensor)
        else:
            raise NotImplementedError(opt.distill)

        if opt.distill_decay and epoch > (opt.epochs / 2):
            new_alpha = int(opt.epochs - epoch) / \
                int(opt.epochs / 2) * opt.alpha
            new_gamma = 1 - new_alpha
        else:
            new_alpha = opt.alpha
            new_gamma = opt.gamma

        if epoch == 1:
            loss = new_gamma * loss_cls + new_alpha * loss_div
        else:
            loss = new_gamma * loss_cls + new_alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (not opt.distribution or opt.local_rank == 0) and \
                idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  '{gamma}*loss_cls ({cls:.4f}), {alpha}*loss_div ({div:.4f}), {beta}*{distill} ({kd:.4f})'.format(
                      epoch, idx, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5,
                      gamma=new_gamma, cls=loss_cls.item(), alpha=new_alpha,
                      div=loss_div.item(), beta=opt.beta, distill=opt.distill, kd=loss_kd.item()))
            f.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    '{gamma}*loss_cls ({cls:.4f}), {alpha}*loss_div ({div:.4f}), {beta}*{distill} ({kd:.4f})\n'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5,
                        gamma=new_gamma, cls=loss_cls.item(), alpha=new_alpha,
                        distill=opt.distill, div=loss_div.item(), beta=opt.beta, kd=loss_kd.item()))
            sys.stdout.flush()

    if not opt.distribution or opt.local_rank == 0:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        f.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                .format(top1=top1, top5=top5))
    return top1.avg, losses.avg
