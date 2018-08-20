import argparse, os, shutil, json, time, sys
import numpy as np

'''
    torch header
'''
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import dataset
from IPython import embed
from tensorboardX import SummaryWriter
from captcha.config import cfg

'''
    Params
'''
USE_TORCHVISION = False

if USE_TORCHVISION:# {{{
    import torchvision.models as models
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
else:   #   use local networks
    import networks as models
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
# }}}

parser = argparse.ArgumentParser(description='PyTorch CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',\
        choices=model_names, help='model architecture: '+\
        ' | '.join(model_names)+' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',\
        help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--save-per-epoch', default=5, type=int)
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--extra', default='', type=str)
parser.add_argument('--logdir', default='train_log', type=str, 
                    help='logdir for tensorboard')

def main():
    print('-'*32)
    global args, best_prec, writer, cnt
    cnt = 0
    best_prec = 0
    args = parser.parse_args()

    if args.logdir == 'train_log':
        args.logdir = '{};{};lr:{};wd:{};'.format(args.arch,
                'pretrained' if args.pretrained else 'not-pretrained',
                args.lr, args.weight_decay)

    if len(args.extra) > 0:
        args.logdir += args.extra


    def touchdir(path):
        if os.path.isdir(path) or os.path.isfile(path):
            for i in range(1000):
                p = '{}({})'.format(path, i)
                if os.path.isdir(p) or os.path.isfile(p):
                    continue
                return p
        return path

    args.logdir = os.path.join('train_log', args.logdir)

    args.logdir = touchdir(args.logdir)
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    print('[LOG]', args, '\n')
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f)
    
    for each in ['tensorboard', 'models']:
        path = os.path.join(args.logdir, each)
        if not os.path.isdir(path):
            os.makedirs(path)

    writer = SummaryWriter(os.path.join(args.logdir, 'tensorboard'))

    model = models.__dict__[args.arch](
        num_classes=cfg.classes+1,
        pretrained=args.pretrained,
    ) 

    def criterion(output, target):
        print(output.shape, target.shape)

    train_dataset = dataset.myDataset(10240, transform=dataset.resizeNormalize((cfg.w, cfg.h)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers)

    val_dataset = dataset.myDataset(256, transform=dataset.resizeNormalize((cfg.w, cfg.h)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.workers)

    input_data = torch.autograd.Variable(
        torch.Tensor(1, cfg.channel, cfg.h, cfg.w),
        requires_grad=True
    )
    writer.add_graph(model, input_data)

    with open(os.path.join(args.logdir, 'network.txt'), 'w') as f:
        f.write('{}'.format(model))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion)
        writer.close()
        return
    
    print('[LOG] ready, GO!')
    cnt = args.start_epoch * len(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if epoch % args.save_per_epoch == args.save_per_epoch - 1\
                or epoch == 0:
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    global cnt
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    top = AverageMeter()

    # switch to train mode
    model.train()

    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        # compute output


        output = model(input)

        loss = criterion(output, target)

        cnt += 1

        # measure accuracy and record loss
        prec = accuracy(output, target, topk=(1))
        losses.update(loss.item(), input.size(0))
        top.update(prec[0], input.size(0))
        writer.add_scalar('loss/train', loss, cnt)
        writer.add_scalar('accuracy/train', prec, cnt)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top.val:.3f} ({top.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top=top))


def validate(val_loader, model, criterion):
    global cnt
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    top = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = torch.autograd.Variable(input.cuda())
            target = torch.autograd.Variable(target.cuda())

            # compute output

            output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target, topk=(1))
            losses.update(loss.item(), input.size(0))
            top.update(prec[0], input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top.val:.3f} ({top.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, 
                       loss=losses, top=top))

        print(' * Prec {top.avg:.3f} Prec@5 {top.avg:.3f}'.format(top=top))

        writer.add_scalar('loss/val', losses.avg, cnt)
        writer.add_scalar('accuracy/val', top.avg, cnt)

    return top.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    realpath = os.path.join(args.logdir, 'models', filename)
    torch.save(state, realpath)
    if not is_best:
        return
    shutil.copyfile(realpath, os.path.join(\
            args.logdir, 'models', 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

