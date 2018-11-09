# @import: {{{
# @commom
import argparse, os, shutil, json, pickle, time
import numpy as np
from IPython import embed

# @pytorch
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

from tensorboardX import SummaryWriter

# @custom
from args import ArgsConfig
from utils import *
from dataset_config import *
from my_folder import MyImageFolder

# }}}

use_torchvision = True
#@ model list {{{
if use_torchvision:
    import torchvision.models as models
else:
    import networks as models
model_names = sorted(name for name in models.__dict__\
        if name.islower() and not name.startswith("__")\
        and callable(models.__dict__[name]))
# }}}


args = ArgsConfig(model_names).args
dataset_cfg = cub


class Dataset(object):# {{{
    def __init__(self, dataset):
        self.__dict__.update(dataset.__dict__)

    def get_train(self, args):
        normalize = transforms.Normalize(dataset.mean, dataset.std)
        train_dataset = MyImageFolder(
            self.train_path,
            transforms.Compose([
                transforms.RandomResizedCrop(self.shape),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dataset.mean, dataset.std)
            ]),
            num_workers=args.workers,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        return train_loader

    def get_val(self, args):
        val_dataset = MyImageFolder(
            self.val_path,
            transforms.Compose([
                transforms.Resize(self.shape),
                transforms.CenterCrop(self.shape),
                transforms.ToTensor(),
                transforms.Normalize(dataset.mean, dataset.std)
            ]),
            num_workers=args.workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )
        return val_loader
# }}}
dataset = Dataset(dataset_cfg)

class Network(object):
    def __init__(self, args, dataset):# {{{
        cudnn.benchmark = True
        self.args = args
        self.other_parameters = {
            'use_torchvision': use_torchvision,
            **dataset.__dict__,
        }
        self.best_prec1 = 0
        self.cnt = 0
        self.dataset = dataset

        cp.log('(#y)RUNNING MODE(#): (#g){}(#)'.format(args.mode))
        
        cp.log('init model')
        self.model = models.__dict__[args.arch](
            num_classes=dataset.classes,
            pretrained=args.pretrained
        )   
        self.loss = nn.CrossEntropyLoss()
        cp.suc('init model', cp.done)
        
        if self.args.mode == 'train':
            self.train()
    # }}}
    def __del__(self):# {{{
        if 'writer' in self.__dict__:
            self.writer.close()
    # }}}
    def train_prepare(self):# {{{
        cp.log('prepare dataset')
        self.train_loader = self.dataset.get_train(self.args)
        self.val_loader = self.dataset.get_val(self.args)
        cp.suc('prepare dataset', cp.done)
        # @train_log 
        cp.log('init train log')
        init_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if args.logdir == 'train_log':
            args.logdir = os.path.join(
                'train_log',
                '{},{},lr:{},wd:{}:{}'.format(
                    args.arch,
                    'pretrained' if args.pretrained else 'not-pretrained',
                    args.lr, args.wd, init_time
                )
            )
            args.logdir = touchdir(args.logdir)
        if file_stat(args.logdir) is None:
            os.makedirs(args.logdir)
        file_type = file_stat(args.logdir)
        assert file_type == 'dir',\
                cp.trans('(#r)[ERR](#) \'(#y){}(#)\' is (a) {}'.format(
                    args.logdir, file_type))

        for each in ['tensorboard', 'models']:
            path = os.path.join(self.args.logdir, each)
            if not os.path.isdir(path):
                os.makedirs(path)
        cp.suc('init train log', cp.done)

        # @tensorboard 
        cp.log('init tensorboad & add graph')
        self.writer = SummaryWriter(os.path.join(self.args.logdir, 'tensorboard'))
        input_data = torch.autograd.Variable(
            torch.Tensor(1, self.dataset.channel, *self.dataset.shape),
            requires_grad=True,
        )
        self.writer.add_graph(self.model, input_data)
        cp.suc('init tensorboad & add graph', cp.done)
        
        self.dump_info()

        cp.log('init loss & optimizer')
        self.loss_func = nn.CrossEntropyLoss()
        __normal__ = self.model.parameters()
        optimizer = torch.optim.SGD([{'params': __normal__, 'name': 'normal'}],
            lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
        )
        cp.suc('init loss & optimizer', cp.done)
        self.load_checkpoint()

    # }}}
    def dump_info(self):# {{{
        args = self.args
        with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
            json.dump(
                {'args': args.__dict__, 'other_parameters': self.other_parameters},
                f,
            )
        with open(os.path.join(args.logdir, 'network.txt'), 'w') as f:
            f.write('{}'.format(self.model))
        with open(os.path.join(args.logdir, 'classes.json'), 'w') as f:
            json.dump(train_dataset.classes, f)
    # }}}
    def load_checkpoint(self):# {{{
        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                cp.log(
                    'loading checkpoint from (#y)\'{}\'(#)'.format(self.args.resume)
                )
                checkpoint = torch.load(self.args.resume)
                self.args.start_epoch = checkpoint.get('epoch', 0)
                self.best_prec1 = checkpoint.get('best_prec1', 0)

                self.model.load_state_dict(checkpoint['state_dict'])
                cp.suc(
                    'successfully loaded model (epoch (#g){}(#))'.format(self.args.start_epoch)
                )
                if self.train:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    cp.suc(
                        'successfully loaded optimizer'
                    )
            else:
                cp.err('no checkpoint found')

   # }}}
    def adjust_learning_rate(self, epoch):# {{{
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        """
        lr = args.lr * (self.args.lr_decay ** (epoch // self.args.lr_decay_freq))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    # }}}
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):# {{{
        model_path = os.path.join(self.args.logdir, 'models')
        realpath = os.path.join(model_path, filename)
        torch.save(state, realpath)
        if not is_best:
            return
        shutil.copyfile(realpath, os.path.join(model_path, 'model_best.pth.tar'))
    # }}}

    def train(self):# {{{
        self.train_prepare()
        self.cnt = self.args.start_epoch * len(self.train_loader)
        for epoch in range(args.start_epoch, args.epochs):
            self.adjust_learning_rate(optimizer, epoch)

            self.train_one(epoch)
            
            # evaluate on validation set
            if (epoch + 1) % self.args.val_freq == 0:
                prec1 = self.validate_one(epoch)
                is_best = prec1 > self.best_prec1
                self.best_prec1 = max(prec1, self.best_prec1)
                save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': self.best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    },
                    is_best, 
                    filename='epoch_%06d.pth,tar' % epoch
                )
    # }}}

    def validate(self):# {{{
        validate_one()
    # }}}


    def train_one(self):# {{{
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        train_loader = self.train_loader
        model = self.model

        model.train()
        end_time = time.time()
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end_time)
            
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()

            output = model(input)

            loss = self.loss(output, target)

            self.cnt += 1

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            self.writer.add_scalar('loss/train', loss, self.cnt)
            self.writer.add_scalar('accuracy/train', prec1, self.cnt)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
    # }}}

    def validate_one(val_loader, model, criterion, epoch=None):# {{{
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model = self.model
        val_loader = self.val_loader

        model.eval()
        with torch.no_grad():
            end = time.time()
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            for i, (input, target) in enumerate(val_loader):
                input = torch.autograd.Variable(input)
                target = torch.autograd.Variable(target)

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

                if i % self.args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           i, len(val_loader), batch_time=batch_time, 
                           loss=losses, top1=top1, top5=top5))

            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

            if epoch:
                writer.add_scalar('loss/val@epoch', losses.avg, epoch)
                writer.add_scalar('accuracy/val@epoch', top1.avg, epoch)
            else:
                writer.add_scalar('loss/val', losses.avg, cnt)
                writer.add_scalar('accuracy/val', top1.avg, cnt)

        return top1.avg
    # }}}
