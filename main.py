# @import: {{{
# @commom
import argparse, os, shutil, json, pickle, time, random
import numpy as np
from IPython import embed

# @pytorch
import torch
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

class Dataset(object):# {{{
    def __init__(self, dataset):
        self.__dict__.update(dataset.__dict__)
        self.train_val = {'train': None, 'val': None}
        if os.path.realpath(dataset.train_path) == os.path.realpath(dataset.val_path):
            self.get_train_val_split()

    def train_val_split(self, rate=0.88):
        dataset = MyImageFolder(
            self.train_path
        )
        samples = dataset.samples
        np.random.shuffle(samples)
        n = len(samples)
        m = int(n*rate)
        split = {'train': [name for name, label in samples[:m]],\
                'val': [name for name, label in samples[m:]]}
        cp.log(
            '(#b)len(split["train"])(##): (#y){}(##), (#b)len(split["val"])(##): (#y){}(##)'.format(len(split['train']), len(split['val']))
        )
        return split

    def get_train_val_split(self):
        intermediate = 'intermediate'
        if file_stat(intermediate) == None:
            os.path.makedirs(intermediate)
        train_val_split_path = os.path.join(intermediate,\
                '{}_train_val_split.json'.format(self.name))

        if os.path.isfile(train_val_split_path):
            cp.wrn('training set is the same as validation set,\n'+\
                    '      use \'(#y){}(#)\' divide dataset'.format(train_val_split_path))
            with open(train_val_split_path, 'r') as f:
                self.train_val = json.load(f)
            cp.suc('\'(#y){}(#)\' loaded'.format(train_val_split_path), cp.done)
        else:
            cp.wrn('training set is the same as validation set,\n'+\
                    '      generating \'(#y){}(#)\''.format(train_val_split_path))
            self.train_val = self.train_val_split()

            with open(train_val_split_path, 'w') as f:
                json.dump(self.train_val, f)
            cp.suc('\'(#y){}(#)\' saved'.format(train_val_split_path), cp.done)

    def get_train(self, args):
        normalize = transforms.Normalize(self.mean, self.std)
        train_dataset = MyImageFolder(
            self.train_path,
            transforms.Compose([
                # transforms.RandomResizedCrop(max(self.shape)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            allow_dict=self.train_val['train'],
            data_cached=False,
            num_workers=args.workers,
        )
        self.classes = train_dataset.classes
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
                # transforms.Resize(max(self.shape)),
                # transforms.CenterCrop(self.shape),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            allow_dict=self.train_val['val'],
            data_cached=False,
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

class Network(object):
    def __init__(self, args, dataset):# {{{
        cp.log('(#y)RUNNING MODE(#): (#g){}(#)'.format(args.mode))
        cudnn.benchmark = True
        self.args = args
        self.best_prec1 = 0
        self.cnt = 0
        self.dataset = dataset
        self.writer = None
        
        # @model & loss
        with procedure('init model'):
            self.model = models.__dict__[args.arch](
                num_classes=dataset.classes,
                pretrained=args.pretrained if args.mode == 'train' else False,
            )   
            self.loss = nn.MultiLabelSoftMarginLoss()
        #################
        
        if args.cuda:
            gpu = None if self.args.gpu is None else self.args.gpu.split(',')
            self.model_cuda = self.model.cuda()
            self.model = nn.DataParallel(self.model_cuda, self.args.gpu)
            self.loss.cuda()
       
        if self.args.mode == 'train':
            self.train()
        if self.args.mode == 'val':
            self.validate()
    # }}}
    def __del__(self):# {{{
        if self.writer is not None:
            self.writer.close()
    # }}}
    def train_prepare(self):# {{{
        # @optimizer
        __normal__ = self.model.parameters()
        self.optimizer = torch.optim.SGD([{'params': __normal__, 'name': 'normal'}],
            lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
        )
        ################

        self.load_checkpoint()
        
        # @dataset
        with procedure('prepare dataset'):
            self.train_loader = self.dataset.get_train(self.args)
            self.val_loader = self.dataset.get_val(self.args)
        ################

        # @train_log 
        with procedure('init train log'):
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
        ################

        # @tensorboard 
        with procedure('init tensorboad & add graph'):
            self.writer = SummaryWriter(os.path.join(self.args.logdir, 'tensorboard'))
            input_data = torch.autograd.Variable(
                torch.Tensor(1, self.dataset.channel, *self.dataset.shape),
                requires_grad=True,
            )
            if self.args.cuda:
                input_data = input_data.cuda()
            self.writer.add_graph(self.model_cuda, input_data)
            self.model_cuda = None
            ################

        self.dump_info()
    # }}}
    def dump_info(self):# {{{
        args = self.args
        other = dataset.__dict__.copy()
        other.pop('train_val')
        other['use_torchvision'] = use_torchvision

        with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
            json.dump({'args': args.__dict__,\
                    'other_parameters': other,\
                    'classes': self.dataset.classes}, f)
        with open(os.path.join(args.logdir, 'network.txt'), 'w') as f:
            f.write('{}'.format(self.model))
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
                cp.suc('successfully loaded model (epoch (#g){}(#))'.format(
                        self.args.start_epoch))
                if self.args.mode == 'train':
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    cp.suc('successfully loaded optimizer')
            else:
                cp.err('no checkpoint found')

   # }}}
    def accuracy(self, output, target):# {{{
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            t = np.array(target)
            o = np.array(output)
            
            o[o>0.5] = 1
            o[o != 1] = 0
            acc = (o == t).mean(axis=1)
            acc[acc<1] = 0
            return acc.mean()

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
        cp.log('(#g)#{}(#) saving to \'(#y){}(#)\''.format(
                state['epoch'], realpath))
        torch.save(state, realpath)
        if not is_best:
            return
        best_model = os.path.join(model_path, 'model_best.pth.tar')
        cp.log('best model saving to \'(#y){}(#)\''.format(best_model))
        shutil.copyfile(realpath, best_model)
    # }}}
    def train(self):# {{{
        self.train_prepare()
        self.cnt = self.args.start_epoch * len(self.train_loader)
        cp.log('(#y)'+'-'*64)
        cp.log('start training at (#b)epoch(##): (#y){}(##) (#b)iter(##): (#y){}(##)'\
                .format(self.args.start_epoch, self.cnt))
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.adjust_learning_rate(epoch)
            self.train_one(epoch)
            # evaluate on validation set
            if (epoch + 1) % self.args.val_freq == 0:
                prec1 = self.validate_one(epoch)
                is_best = prec1 > self.best_prec1
                self.best_prec1 = max(prec1, self.best_prec1)
                self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'best_prec1': self.best_prec1,
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best, filename='epoch_%06d.pth.tar' % epoch,
                )
    # }}}
    def validate(self):# {{{
        self.val_loader = self.dataset.get_val(self.args)
        self.args.__dict__['print_freq'] = 1
        self.load_checkpoint()
        self.validate_one(self.args.start_epoch)
    # }}}
    def train_one(self, epoch):# {{{
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

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
    
            # print(output, target)
            # embed()
            loss = self.loss(output, target)

            self.cnt += 1

            prec1 = self.accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))
            self.writer.add_scalar('loss/train', loss, self.cnt)
            self.writer.add_scalar('accuracy/train', prec1, self.cnt)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            color = 'r' if data_time.val > batch_time.val * 0.5 else\
                    ('y' if data_time.val > batch_time.val * 0.33 else 'g')
            if i % args.print_freq == 0 or epoch == self.args.start_epoch and i < 20:
                cp('(#b)[train](#) '
                    '(#b)Epoch(#) [{0}][{1}/{2}] '
                    '(#b)Time(#) {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                    '(#b)Data(#) (#{color}){data_time.val:.2f}(#) ((#{color}){data_time.avg:.2f}(#)) '
                    '(#b)Loss(#) (#g){loss.val:.4f}(#) ((#g){loss.avg:.4f}(#)) '
                    '(#b)Prec@1(#) (#g){top1.val:.2f}(#) ((#g){top1.avg:.2f}(#))'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, color=color))

        self.writer.add_scalar('loss@epoch/train', losses.avg, epoch)
        self.writer.add_scalar('accuracy@epoch/train', top1.avg, epoch)
    # }}}
    def validate_one(self, epoch=None):# {{{
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model = self.model
        val_loader = self.val_loader

        model.eval()
        with torch.no_grad():
            end_time = time.time()
            for i, (input, target) in enumerate(val_loader):
                if self.args.cuda:
                    input = input.cuda()
                    target = target.cuda()
                input = torch.autograd.Variable(input)
                target = torch.autograd.Variable(target)

                # compute output

                output = model(input)
                loss = self.loss(output, target)

                # measure accuracy and record loss
                prec1 = self.accuracy(output, target)
                losses.update(loss.item(), input.size(0))
                top1.update(prec1, input.size(0))


                # measure elapsed time
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                if i % self.args.print_freq == 0:
                    cp('(#b)[val](#) '
                        '(#b)Epoch(#) [{0}][{1}/{2}] '
                        '(#b)Time(#) {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                        '(#b)Loss(#) (#g){loss.val:.4f}(#) ((#g){loss.avg:.4f}(#)) '
                        '(#b)Prec@1(#) (#g){top1.val:.2f}(#) ((#g){top1.avg:.2f}(#)) '.format(
                        epoch, i, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1))

            cp.log('(#y)Average Accuracy on Validation(#): (#g){top1.avg:.4f}(#)'.format(top1=top1))

            if self.writer is not None:
                self.writer.add_scalar('loss@epoch/val', losses.avg, epoch)
                self.writer.add_scalar('accuracy@epoch/val', top1.avg, epoch)

        return top1.avg
    # }}}



if __name__ == '__main__':
    cp.log('(#y)'+'-'*64)
    use_torchvision = False
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    if not args.cuda and torch.cuda.is_available():
        cp.wrn('run with (#g)--cuda(#) to use gpu')
    dataset_cfg = cell
    dataset = Dataset(dataset_cfg)
    network = Network(args, dataset)
