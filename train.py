import os
import logging
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np

import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system') #容易发生共享内存泄漏

import torch.nn.functional as F

from model.utils import get_model
from metric.utils import calculate_distance, calculate_dice, calculate_dice_split
from training.topoloss_pytorch import getTopoLoss

from torch.autograd import Variable

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from training.boundary_loss import SurfaceLoss, boundary_loss
from training.losses import DiceLoss, soft_cldice, FocalLoss
from training.dataset_CAG import CAGDataset
# from training.dataset_2dCAG import CAGDataset # 3d dataset
from training.utils import update_ema_variables
from training.validation import validation
from training.utils import (
    exp_lr_scheduler_with_warmup,
    log_evaluation_result,
    get_optimizer,)
import yaml
import argparse
import time
import sys
import warnings


from utils import (
    configure_logger,
    save_configure,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,)

warnings.filterwarnings("ignore", category=UserWarning)

def train( net, ema_net, args):

    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")

    
    trainset = CAGDataset(args, mode="train")
    trainLoader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
        persistent_workers=False,
        drop_last=True)

    testset = CAGDataset(args, mode='test')
    testLoader = data.DataLoader(
        testset, 
        batch_size=args.batch_size, 
        pin_memory=False,
        shuffle=True, 
        num_workers=0,
        drop_last=True)
        
    

    logging.info(f"Created Dataset and DataLoader")

    
    # Initialize tensorboard, optimizer and etc
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}")  # tensorboard的保存路径

    optimizer = get_optimizer(args, net)

    if args.resume:
         resume_load_optimizer_checkpoint(optimizer, args)   # 保存dict的模型

    net_for_eval = ema_net if args.ema else net
    
    net.train()

    tic = time.time()
    dice_list = []
    best_Dice = np.zeros(2)
    
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda().float())
    criterion = FocalLoss(2)
    criterion_dl = DiceLoss()
    criterion_cl = soft_cldice()
    # criterion_bd = boundary_loss()
    scaler = torch.cuda.amp.GradScaler() if args.amp else None  # 混合精度
    
    
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        progress = ProgressMeter(
        len(trainLoader),
        [batch_time, epoch_loss],
        prefix="Epoch: [{}]".format(epoch + 1),
    )
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=0,
                                                     # 预热学习warmup_epoch个epoch
                                                     max_epoch=args.epochs)
        
        logging.info(f"Current lr: {exp_scheduler:.4e}")
        for i, inputs in enumerate(trainLoader):
            img, label = inputs[0], inputs[1].long()  # image b,c,t,h,w
            # 输入的图片和标签可视化
            # for i in range(img.shape[2]):
            #     x = img[:,:,i,:,:]
            #     for m in range(x.shape[0]):
            #         show = x.squeeze(1)[m]
            #         plt.subplot(4,4,4*i+m+1)
            #         plt.imshow(show.cpu(),cmap="gray")
            #
            #         showlab = label[m].squeeze(0).cpu()
            #         plt.subplot(4, 4, 12 + m + 1)
            #         plt.imshow(showlab, cmap="gray")
            # plt.show()
    
            step = i + epoch * len(trainLoader)  # global steps
    
            optimizer.zero_grad()
    
            if args.amp:
                with torch.cuda.amp.autocast():
                    result = net(img)
                    # result = net(img)[0]   # medformer

                    # print(result)
                    pre = F.softmax(result, dim=1)
                    # pre_top = pre[:, 1, :, :]
                    _, label_pred = torch.max(pre, dim=1)
                    label_pred = label_pred.to(torch.int8)

                    #######3d loss
                    # total_loss = 0
                    # print(result.shape,label.shape)
                    # for t in range(5):
                    #     result_ = result[:,:,t,:,:]
                    #     label_ = label[:,:,t,:,:]
                    #     # print(result.shape,label.shape)
                    #     loss = criterion(result_, label_.squeeze(1)) + criterion_dl(result_, label_) 
                    #     total_loss = total_loss+loss
                    # loss = total_loss/5
                    #########


                    loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label) 
                    scaler.scale(loss).backward()  # scaler实现的反向误差传播
                    scaler.step(optimizer)  # 优化器中的值也需要进行放缩
                    scaler.update()  # 更新scaler
            else:
                result = net(img)
                pre = F.softmax(result, dim=1)
                pre_top = pre[:, 1, :, :]
                _, label_pred = torch.max(pre, dim=1)
                label_pred = label_pred.to(torch.int8)
                bd_loss = boundary_loss()
                # dice, _, _ = calculate_dice_split(label_pred.view(-1, 1), label.view(-1, 1), args.classes) # 训练集dice
                # dice = dice.cpu().numpy()[1:]
               
                #loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label) + 0.1 * getTopoLoss(pre_top, 
                #                                                                                           label, 512)
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)
                loss.backward(retain_graph=True)
                optimizer.step()
            if args.ema:
                update_ema_variables(net, ema_net, args.ema_alpha, step)
    
            epoch_loss.update(loss.item(), label.shape[0])
            batch_time.update(time.time() - tic)
            tic = time.time()
            
            #dice_list.append(dice) # 训练集dice
    
            torch.cuda.empty_cache()
    
            if i % args.print_freq == 0:
                progress.display(i)
    
            if (step + 1) % args.val_freq == 0:
    
                #dice_list_test, loss_test = validation(net_for_eval, testLoader, args)  # 验证
                dice_list_test = validation(net_for_eval, testLoader, args)  # 验证
                #writer.add_scalars('Dice/%s_AVG', {"test": dice_list_test.mean(), "train": np.array(dice_list).mean()},
                #                  step + 1)
                writer.add_scalar('Dice/%s_AVG',  dice_list_test.mean(), step + 1)
                                 

    
                if dice_list_test.mean() >= best_Dice.mean():
                    best_Dice = dice_list_test
    
                    # Save the checkpoint with best performance
                    # torch.save({
                    #     'epoch': epoch + 1,
                    #     'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                    #     'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    # }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")
                    
                    # Save the model with best performance
                    torch.save({
                        'epoch': step + 1,
                        'model_state_dict': net if not args.torch_compile else net._orig_mod,
                        'ema_model_state_dict': ema_net if args.ema else None,
                        'optimizer_state_dict': optimizer,
                    }, f"{args.cp_path}{args.dataset}/{args.unique_name}/best.pth")
    
                logging.info("Evaluation Done")
                logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")
    
                #writer.add_scalars('Train/Loss', {"train": epoch_loss.avg, "test": loss_test}, step + 1)
                writer.add_scalar('Train/Loss',  epoch_loss.avg, step + 1)
                
                # save the latest checkpoint, including net, ema_net, and optimizer
                torch.save({
                    'epoch': step + 1,
                    'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                    'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_path}{args.dataset}/{args.unique_name}/latest.pth")  # 模型保存路径
                
            
    return best_Dice


def get_parser():

    parser = argparse.ArgumentParser(description='CAG Vessel Segmentation')
    parser.add_argument('--dataset', type=str, default='CAG', help='dataset name')
    parser.add_argument('-d','--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('-m','--model', type=str, default='DE_DCGCN_EE', help='model name')
    parser.add_argument('-n','--frame_N', type=int, default='3', help='frame number of input')
    parser.add_argument('-u','--unique_name', type=str, help='unique experiment name')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile, only supported by pytorch2.0')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default='./exp/CAG/Angionet/latest.pth',
                        help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')

    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    
    config_path = 'normal_config.yaml'
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s" % config_path)

    print('Loading configurations from %s' % config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None

    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)

    if args.torch_compile:
        net = torch.compile(net)
    return net, ema_net


if __name__ == '__main__':


    args = get_parser()
    #if args.dimension == '2d':
    #  from training.dataset_2dCAG import CAGDataset
    #elif args.dimension == '3d':
    #  from training.dataset_CAG import CAGDataset
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #torch.multiprocessing.set_start_method('spawn')
    #torch.multiprocessing.set_sharing_strategy('file_system')

    args.log_path = args.log_path + '%s/' % args.dataset
    Dice_list = []
    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    configure_logger(0, args.cp_dir + ".txt")
    save_configure(args)
    
    logging.info(
          f"\nDimension: {args.dimension},\n"
        + f"Model: {args.model},\n"
        + f"frame_N: {args.frame_N}"
    )

    net, ema_net = init_network(args)
    net.cuda()
    
    if args.ema:
        ema_net.cuda()
    logging.info(f"Created Model")
    
    best_Dice = train(net, ema_net,args)

    # logging.info(f"Training and evaluation is done")

    Dice_list.append(best_Dice)
    

    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt", 'w') as f:
        np.set_printoptions(precision=4, suppress=True)
        f.write('Dice\n')
        f.write(f"{Dice_list}\n")
        f.write("\n")
        
    print(f'training done.')
    sys.exit(0)
