from __future__ import division, print_function

import os
import pickle
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable

from .util import AverageMeter, accuracy, accuracy_list, reduce_tensor


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
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          idx, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))
                f.write('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                            idx, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        f.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                .format(top1=top1, top5=top5))
    f.close()
    return top1.avg, top5.avg, losses.avg


def validate_multi(val_loader, model_list, criterion, opt):
    """validation milti model using voting"""
    model_num = len(model_list)
    batch_time = AverageMeter()
    # losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time_list = [AverageMeter() for i in range(model_num)]
    losses_list = [AverageMeter() for i in range(model_num)]
    top1_list = [AverageMeter() for i in range(model_num)]
    top5_list = [AverageMeter() for i in range(model_num)]

    # switch to evaluate mode
    [model.eval() for model in model_list]
    f = open(os.path.join(opt.save_folder, 'log.txt'), 'a+')

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output_list = []
            for model_index, model in enumerate(model_list):

                # compute output
                output = model(input)
                output_list.append(output)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses_list[model_index].update(loss.item(), input.size(0))
                top1_list[model_index].update(acc1[0], input.size(0))
                top5_list[model_index].update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % opt.print_freq == 0:
                    print(f'Model {model_index}\t'
                          'Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              idx, len(val_loader), batch_time=batch_time, loss=losses_list[model_index],
                              top1=top1_list[model_index], top5=top5_list[model_index]))
                    f.write(f'Model {model_index}\t'
                            'Test: [{0}/{1}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                                idx, len(val_loader), batch_time=batch_time, loss=losses_list[model_index],
                                top1=top1_list[model_index], top5=top5_list[model_index]))

            acc1, acc5 = accuracy_list(output_list, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            if idx % opt.print_freq == 0:
                print('Model Ensemble\t'
                      'Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          idx, len(val_loader), batch_time=batch_time,
                          top1=top1, top5=top5))
                f.write('Model Ensemble\t'
                        'Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                            idx, len(val_loader), batch_time=batch_time,
                            top1=top1, top5=top5))

        for model_index, model in enumerate(model_list):
            print('Model {model_index} * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(model_index=model_index, top1=top1_list[model_index], top5=top5_list[model_index]))
            f.write('Model {model_index} * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(model_index=model_index, top1=top1_list[model_index], top5=top5_list[model_index]))

        teacher_acc_list = [t.avg for t in top1_list]
        teacher_acc_list = torch.Tensor(teacher_acc_list)
        print('Model Ensemble * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        f.write('Model Ensemble * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    f.close()

    return top1.avg, top5.avg, torch.mean(torch.Tensor([losses.avg for losses in losses_list])), teacher_acc_list
