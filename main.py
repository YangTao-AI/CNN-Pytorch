import argparse, os, shutil, json, time, sys

import numpy as np

'''
    torch header
'''
import torch, json
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import pickle
from IPython import embed
from config import *

from tensorboardX import SummaryWriter
from my_folder import MyImageFolder


'''
    Params
'''
USE_TORCHVISION = True
dataset = Al
with open('./train_val.pkl', 'rb') as f:
    train_dic, val_dic = pickle.load(f)

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
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
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
parser.add_argument('--data-cached', default=False, action='store_true')
parser.add_argument('--augmentation', default='crop', type=str,\
        choices=['crop', 'resize'])



def main(get_model=False):
    print('-'*32)
    global args, best_prec1, writer, cnt
    # optionally resume from a checkpoint
    def load_checkpoint(get_model):
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                if not get_model:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
    cnt = 0
    best_prec1 = 0
    args = parser.parse_args()

    model = models.__dict__[args.arch](
        num_classes=dataset.classes, 
        pretrained=args.pretrained
    ) 

    if get_model:
        load_checkpoint(get_model)
        return model
    args.__dict__['dataset'] = dataset.name
    args.__dict__['data'] = dataset.path
    args.__dict__['USE_TORCHVISION'] = USE_TORCHVISION
    args.__dict__['classes'] = dataset.classes
    args.__dict__['std'] = dataset.std
    args.__dict__['mean'] = dataset.mean



    if args.logdir == 'train_log':
        args.logdir = '{},{},lr:{},wd:{}:{}'.format(args.arch,
                'pretrained' if args.pretrained else 'not-pretrained',
                args.lr, args.weight_decay, dataset.name)
    if len(args.extra) > 0:
        args.logdir += ':' + args.extra

    if args.data_cached:
        print('[WRN] --data-cached costs extra memory.\n' + 
              '      Which may cause OOM(out of memory).\n')

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

    input_data = torch.autograd.Variable(
        torch.Tensor(1, dataset.channel, dataset.shape, dataset.shape),
        requires_grad=True
    )
    writer.add_graph(model, input_data)

    with open(os.path.join(args.logdir, 'network.txt'), 'w') as f:
        f.write('{}'.format(model))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    __normal__ = model.parameters()

    optimizer = torch.optim.SGD(
        [{'params': __normal__, 'name': 'normal'}],
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    



    load_checkpoint(get_model)
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train.zip')
    valdir = os.path.join(args.data, 'train.zip')
    normalize = transforms.Normalize(dataset.mean, dataset.std)

    train_dataset = MyImageFolder(
        traindir,
        transforms.Compose(([
            transforms.RandomResizedCrop(dataset.shape)
        ] if args.augmentation == 'crop' else [
            transforms.Resize(dataset.shape),
            transforms.CenterCrop(dataset.shape)
        ]) + [transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), data_cached=args.data_cached, num_workers=args.workers, allow_dic=train_dic)



    with open(os.path.join(args.logdir, 'classes.json'), 'w') as f:
        json.dump(train_dataset.classes, f)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        MyImageFolder(valdir, transforms.Compose([
            transforms.Resize(dataset.shape),
            transforms.CenterCrop(dataset.shape),
            transforms.ToTensor(),
            normalize,
        ]), data_cached=args.data_cached, num_workers=args.workers, allow_dic=val_dic),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    model.cuda()

    if args.evaluate:
        validate(val_loader, model, criterion)
        writer.close()
        return
    
    print('[LOG] ready, GO!')
    cnt = args.start_epoch * len(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if epoch % args.save_per_epoch == args.save_per_epoch - 1\
                or epoch == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec1 = 0
            is_best = False
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    global cnt
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output


        output = model(input)

        loss = criterion(output, target)

        cnt += 1

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        writer.add_scalar('loss/train', loss, cnt)
        writer.add_scalar('accuracy/train', prec1, cnt)

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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    global cnt
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, 
                       loss=losses, top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        writer.add_scalar('loss/val', losses.avg, cnt)
        writer.add_scalar('accuracy/val', top1.avg, cnt)

    return top1.avg


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
    lr = args.lr * (0.1 ** (epoch // 20))

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

