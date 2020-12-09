from data import *
import cv2 
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from pytorch_model_summary import summary
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from copy import deepcopy


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Pruned Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--verbosity', default=2, type=int,
                    help='Level of detail in console output, options: 0-2 ')
parser.add_argument('--save_freq', default=100000, type=int,
                    help='How many iterations between saving weights file')
parser.add_argument('--prune', default=0.0, type=float,
                    help='Percentage of global network parameters to prune 0.0-1.0')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    if args.dataset == 'COCO':
        cfg = coco
        dataset = COCODetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], 
                                                                                  MEANS))
    elif args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()
    
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    # Display all blocks and layers of the network
    if args.verbosity > 1:
      print()
      print("The network has been built successfully, the structure is shown below.")
      # print(net)
      # print()
      
    print(summary(net, torch.zeros((16, 3, 300, 300)), batch_size = args.batch_size, show_input = True, show_hierarchical = True))

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))       
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        
        
        
    # Print out all of the things that we can prune 
    # print(net.state_dict().keys())
    
    parameters_to_prune = (
        (net.module.vgg[0], 'weight'),
        (net.module.vgg[2], 'weight'),
        (net.module.vgg[5], 'weight'),
        (net.module.vgg[7], 'weight'),
        (net.module.vgg[10], 'weight'),
        (net.module.vgg[12], 'weight'),
        (net.module.vgg[14], 'weight'),
        (net.module.vgg[17], 'weight'),
        (net.module.vgg[19], 'weight'),
        (net.module.vgg[21], 'weight'),
        (net.module.vgg[24], 'weight'),
        (net.module.vgg[26], 'weight'),
        (net.module.vgg[28], 'weight'),
        (net.module.vgg[31], 'weight'),
        (net.module.vgg[33], 'weight'),     
        (net.module.extras[0], 'weight'),
        (net.module.extras[1], 'weight'),
        (net.module.extras[2], 'weight'),
        (net.module.extras[3], 'weight'),
        (net.module.extras[4], 'weight'),
        (net.module.extras[5], 'weight'),
        (net.module.extras[6], 'weight'),
        (net.module.extras[7], 'weight'),
        (net.module.loc[0], 'weight'),
        (net.module.loc[1], 'weight'),
        (net.module.loc[2], 'weight'),
        (net.module.loc[3], 'weight'),
        (net.module.loc[4], 'weight'),
        (net.module.loc[5], 'weight'),
        (net.module.conf[0], 'weight'),
        (net.module.conf[1], 'weight'),
        (net.module.conf[2], 'weight'),
        (net.module.conf[3], 'weight'),
        (net.module.conf[4], 'weight'),
        (net.module.conf[5], 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=args.prune,
    )
    
    stored = 0
    stored2 = 0
    for x in [0,2,5,7,10,12,14,17,19,21,24,26,28,31,33]:
        print(
            "Sparsity in module.vgg[",x,"].weight: {:.2f}%".format(
                100. * float(torch.sum(net.module.vgg[x].weight == 0))
                / float(net.module.vgg[x].weight.nelement())
            )
        )  
        stored += torch.sum(net.module.vgg[x].weight == 0) # get the number of weights in this layer that are set to 0
        stored2 += net.module.vgg[x].weight.nelement()     # get the total number of elements in this layer
        # if x ==0:
            # print(list(net.module.vgg[x].named_parameters()))
            # print(list(net.module.vgg[x].named_buffers()))
        #prune.remove(net.module.vgg[x], 'weight')          # make the elements pruned in this layer permanent       
        # if x ==0:
            # print("weights have been permanently removed")
            # print(list(net.module.vgg[x].named_parameters()))
            # print(list(net.module.vgg[x].named_buffers()))
    for x in range(8):
        print(
            "Sparsity in module.extras[",x,"].weight: {:.2f}%".format(
                100. * float(torch.sum(net.module.extras[x].weight == 0))
                / float(net.module.extras[x].weight.nelement())
            )
        )  
        stored += torch.sum(net.module.extras[x].weight == 0)
        stored2 += net.module.extras[x].weight.nelement()
        #prune.remove(net.module.extras[x], 'weight')
    for x in range(6):
        print(
            "Sparsity in module.loc[",x,"].weight: {:.2f}%".format(
                100. * float(torch.sum(net.module.loc[x].weight == 0))
                / float(net.module.loc[x].weight.nelement())
            )
        )  
        stored += torch.sum(net.module.loc[x].weight == 0)
        stored2 += net.module.loc[x].weight.nelement()
        #prune.remove(net.module.loc[x], 'weight')
    for x in range(6):
        print(
            "Sparsity in module.conf[",x,"].weight: {:.2f}%".format(
                100. * float(torch.sum(net.module.conf[x].weight == 0))
                / float(net.module.conf[x].weight.nelement())
            )
        )  
        stored += torch.sum(net.module.conf[x].weight == 0)
        stored2 += net.module.conf[x].weight.nelement()
        #prune.remove(net.module.conf[x], 'weight')
 
    print("Pruned Parameters:", stored.item(), "/", stored2)
    print(
        "Global sparsity: {:.2f}%".format(100. * float(stored) / float(stored2) )
    )
        

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters 
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)
    print()
    
    trainStartTime = time.time()

    step_index = 0
    
    # Adjust the current learning rate based on the starting iteration
    for k in range(len(cfg['lr_steps'])):
        if args.start_iter > cfg['lr_steps'][k]:
            print("Adjusting learning rate based on nonzero starting iteration.")
            step_index = step_index + 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    # If using visdom, prepare the plot 
    if args.visdom:
        vis_title = 'SSD ('+str(args.prune*100)+'% Global Pruning) on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_iter_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_epoch_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch =epoch+ 1
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0

        # If this is an iteration where the learning rate should change
        if iteration in cfg['lr_steps']:
            step_index = step_index + 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)	

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=False) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=False) for ann in targets]


        # forward
        t0 = time.time()                          # start a timer
        #print("images shape = ", images.shape)
        out = net(images)                         # run images through the network

        # backprop
        optimizer.zero_grad()                     # reset gradients to zero
        loss_l, loss_c = criterion(out, targets)  # compute both types of loss
        loss = loss_l + loss_c                    # sum both types of loss to get total loss
        loss.backward()                           # compute dloss/dx for every parameter x which has requires_grad=True          
        optimizer.step()                          # Perform a parameter update based on the current gradient 
        t1 = time.time()                          # Get stop time
        loc_loss = loc_loss + loss_l.data              
        conf_loss = conf_loss + loss_c.data

        if iteration % 10 == 0:
            print('Iteration ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data)+' Timer: %.4f sec.' % (t1 - t0))

        if args.visdom:
            update_vis_plot(iteration, loss_l.data, loss_c.data,
                            iter_plot, epoch_plot, 'append')
  
        if iteration-args.start_iter != 0 and iteration % args.save_freq == 0:
            saveName = 'weights/ssd300_'
            saveName += (args.dataset+'_' + repr(iteration) +'_rand_'+str(int(args.prune*100))+ '.pth')
            print('\nSaving state at iteration', iteration,'as:', saveName)
            trainTime = time.time() - trainStartTime
            if trainTime < 3600:
                print('Training Time: %.2f minutes' % (trainTime/60), '\n')
            else:
                print('Training Time: %.2f hours' % (trainTime/3600), '\n')
            net2 = ssd_net
            for x in [0,2,5,7,10,12,14,17,19,21,24,26,28,31,33]:
                prune.remove(net2.vgg[x], 'weight')          # make the elements pruned in this layer permanent
            for x in range(8):
                prune.remove(net2.extras[x], 'weight')
            for x in range(6):
                prune.remove(net2.loc[x], 'weight')
            for x in range(6):
                prune.remove(net2.conf[x], 'weight')            
            torch.save(net2.state_dict(), saveName)
            print("Just finalized the weights after pruning, stopping here..")
            print("Continue with higher args.prune for iterative pruning")
            return
    # You reached the maximum iteration, save before ending
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    print("Lowering learning rate...")
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Done!")


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_iter_plot(_xlabel, _ylabel, _title, _legend):
    x=torch.zeros((1,)).cpu()
    x[0] = args.start_iter     # Make the visdom plot start at args.start_iter rather than 0 every time
    return viz.line(
        X = x,
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def create_epoch_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X = torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
