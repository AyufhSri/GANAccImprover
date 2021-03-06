#from GAN_ACC.CGAN import initialize_weights
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import CGAN
from torch.autograd import Variable
from model_search import Network
from architect import Architect





parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data/',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')

# new hyperparams.
parser.add_argument('--weight_gamma', type=float, default=1.0)
parser.add_argument('--weight_lambda', type=float, default=1.0)
parser.add_argument('--dataset', required=False, help='cifar10 | lsun | mnist')
parser.add_argument('--dataroot', required=False, help='path to data')
parser.add_argument('--model_v_learning_rate', type=float, default=3e-4)
parser.add_argument('--model_v_weight_decay', type=float, default=1e-3)
parser.add_argument('--learning_rate_g', type=float, default=0.025)
parser.add_argument('--learning_rate_d', type=float, default=0.025)
parser.add_argument('--learning_rate_bc', type=float, default=0.025)
parser.add_argument('--learning_rate_rl', type=float, default=0.025)
parser.add_argument('--weight_decay_w', type=float, default=3e-4)
parser.add_argument('--weight_decay_h', type=float, default=3e-4)
parser.add_argument('--is_parallel', type=int, default=1)
parser.add_argument('--teacher_arch', type=str, default='18')
parser.add_argument('--is_cifar100', type=int, default=0)
parser.add_argument('--m',type=int, default= 100)
parser.add_argument('--lambda', type=float, default=1.0)
parser.add_argument('--channelimages', type= int, default=3)
parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--Z_DIM', type=int, default=100)
parser.add_argument('--GEN_EMBEDDING', type=int, default=100)
parser.add_argument('--LAMBDA_GP', type=int, default=10)
parser.add_argument('--FEATURES_GEN', type=int, default=16)
parser.add_argument('--CRITIC_ITERATIONS', type=int, default=5)
parser.add_argument('--device',default="cuda")
args = parser.parse_args()







args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100


class RL(nn.Module):
    def __init__(self,num_classes, img_size):
        super(RL, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4, 3, 1, 1, bias=False,),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.img_size=img_size
        self.emd = nn.Embedding(num_classes, img_size*img_size)
    def forward(self, x,labels):
        emd = self.emd(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, emd], dim=1)
        return self.conv(x)


class Bc(nn.Module):
    def __init__(self, nc):
        super(Bc, self).__init__()
        self.nn = nn.Linear(nc,1)
    def forward(self, input):
        return torch.sigmoid(self.nn(input))


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    if not args.is_parallel:
        torch.cuda.set_device(int(args.gpu))
        logging.info('gpu device = %d' % int(args.gpu))
    else:
        logging.info('gpu device = %s' % args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion2 = nn.BCELoss()
    criterion2 = criterion2.cuda()
    if args.is_cifar100:
        model = Network(args.init_channels, CIFAR100_CLASSES, args.layers, criterion)
        d = Network(args.init_channels, CIFAR100_CLASSES, args.layers, criterion2)
        #e = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
        g = CGAN.Generator(args.Z_DIM, 3, args.FEATURES_GEN, CIFAR100_CLASSES, args.imagesize, args.GEN_EMBEDDING)
        bc = Bc(CIFAR100_CLASSES)
        rl = RL(CIFAR100_CLASSES,args.imagesize)
    else:
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
        d = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion2)
        #e = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
        g = CGAN.Generator(args.Z_DIM, 3, args.FEATURES_GEN, CIFAR_CLASSES, args.imagesize, args.GEN_EMBEDDING)
        bc = Bc(CIFAR_CLASSES)
        rl = RL(CIFAR_CLASSES,args.imagesize)
    
    

    model.cuda()
    d.cuda()
    g.cuda()
    bc.cuda()
    rl.cuda()
    print(type(d))
    CGAN.initialize_weights(g)
    #CGAN.initialize_weights(d)
    CGAN.initialize_weights(rl)


    if args.is_parallel:
        gpus = [int(i) for i in args.gpu.split(',')]
        model = nn.parallel.DataParallel(
            model, device_ids=gpus, output_device=gpus[0])
        g = nn.parallel.DataParallel(
            g, device_ids=gpus, output_device=gpus[0])
        d = nn.parallel.DataParallel(
            d, device_ids=gpus, output_device=gpus[0])
        bc = nn.parallel.DataParallel(
            bc, device_ids=gpus, output_device=gpus[0])
        rl = nn.parallel.DataParallel(
            rl, device_ids=gpus, output_device=gpus[0])
        model = model.module
        g = g.module
        d = d.module
        bc = bc.module
        rl = rl.module

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_g = torch.optim.SGD(
        g.parameters(),
        args.learning_rate_g,
        momentum=args.momentum,
        weight_decay=args.weight_decay_w)
    optimizer_d = torch.optim.SGD(
        d.parameters(),
        args.learning_rate_d,
        momentum=args.momentum,
        weight_decay=args.weight_decay_h)
    optimizer_bc = torch.optim.SGD(
        bc.parameters(),
        args.learning_rate_bc,
        momentum=args.momentum,
        weight_decay=args.weight_decay_h)
    optimizer_rl = torch.optim.SGD(
        rl.parameters(),
        args.learning_rate_rl,
        momentum=args.momentum,
        weight_decay=args.weight_decay_h)


    


    if args.is_cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.is_cifar100:
        train_data = dset.CIFAR100(root=args.data, train=True,
                            download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True,
                            download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=False, num_workers=4)

 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_bc = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_bc, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_rl = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rl, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    #architect= 0


    for epoch in range(args.epochs):
        print(epoch)
        lr = scheduler.get_lr()[0]
        lr_g = scheduler_g.get_lr()[0]
        lr_d = scheduler_d.get_lr()[0]
        lr_bc = scheduler_bc.get_lr()[0]
        lr_rl = scheduler_rl.get_lr()[0]
        logging.info('epoch %d lr %e lr_g %e lr_d %e lr_bc %e lr_rl %e', epoch, lr, lr_g, lr_d,lr_bc,lr_rl)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        train_acc, train_obj = train(
            train_queue, valid_queue,
            model, architect, criterion, optimizer,
            optimizer_g,
            optimizer_g,
            optimizer_bc,
            optimizer_rl,
            lr,
            lr_g, lr_d,
            lr_bc,lr_rl,
            g, d, bc, rl)
        logging.info('train_acc %f', train_acc)
        scheduler.step()
        scheduler_g.step()
        scheduler_d.step()
        scheduler_bc.step()
        scheduler_rl.step()
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # external_acc, external_obj = infer(external_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        # logging.info('external_acc %f', external_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))



def train(train_queue, valid_queue,
          model, architect, criterion, optimizer,
          optimizer_g,
          optimizer_d,
          optimizer_bc,
          optimizer_rl,
          lr,
          lr_g, lr_d,
          lr_bc,lr_rl,
          g, d, bc,rl):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)
        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)


        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        l = utils.label_level_loss(model,input_search, target_search, criterion, args)

        d.set_archparameters(model.arch_parameters)
        optimizer.zero_grad()
        logits = model(input_search)
        loss = criterion(logits, target_search)

        for i in range(CIFAR_CLASSES):
            z = torch.zeros([args.m],dtype=torch.long).cuda()
            for j in range(args.m):
                z[j]=i
            noise = torch.randn(args.m, args.Z_DIM, 1, 1).cuda()
            fake = g(noise,z)
            if i==0:
                ze = (l[i])*(model._loss(fake,z))
            else:
                ze=z+(l[i])*(model._loss(fake,z))
        alpha = 1.0
        loss= loss+ alpha*ze
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        for _ in range(args.CRITIC_ITERATIONS):
            noise = torch.randn(n, args.Z_DIM, 1, 1).cuda()
            fake = g(noise,target)
            d_real = bc(d(rl(input,target))).reshape(-1)
            d_fake = bc(d(rl(fake,target))).reshape(-1)
            gp = CGAN.gradient_penalty2(d, bc,rl, target, input, fake, device=args.device)
            loss_critic = (
                -(torch.mean(d_real) - torch.mean(d_fake)) + args.LAMBDA_GP * gp
            )
            optimizer_rl.zero_grad()
            optimizer_bc.zero_grad()
            optimizer_d.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_bc.step()
            optimizer_d.step()
            optimizer_rl.step()

        noise = torch.randn(n, args.Z_DIM, 1, 1).cuda()        
        fake = g(noise,target)
        gen_fake = bc(d(rl(fake,target_search))).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        optimizer_g.zero_grad()
        loss_gen.backward()
        optimizer_g.step()

        
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg
        


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__=='__main__':
    main()

    




    

        
