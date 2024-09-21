import builtins
import logging
import os
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import get_model
# from matplotlib import pyplot as plt
from inference.utils import get_inference
# from dataset_conversion.utils import ResampleXYZAxis, ResampleLabelToRef
from torch.utils import data
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings



from utils import (
    configure_logger,
    save_configure,
)
warnings.filterwarnings("ignore", category=UserWarning)



def prediction(model_list, tensor_img, args):

    inference = get_inference(args)


    with torch.no_grad():
        tensor_img = tensor_img.cuda().float()
        B, D, H, W = tensor_img.shape
        # tensor_pred = torch.zeros([args.classes, D, H, W]).to(tensor_img.device)
        tensor_pred = torch.zeros([args.classes, H, W]).to(tensor_img.device)
        if args.dimension == '2d':
            tensor_img = tensor_img.unsqueeze(0).permute(1, 0, 2, 3,4)
            # tensor_img = tensor_img.unsqueeze(0)
        else:
            tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
        start_time = time.time()
        for model in model_list:
            # get_attn = model.get_attn
            pred = inference(model, tensor_img, args)
            end_time = time.time()
            pred = F.softmax(pred)
            elapsed_time = end_time - start_time
            print(f"模型预测一张图片所需的时间: {elapsed_time:.6f} 秒")
            if args.dimension == '2d':
                # pred = pred.permute(1, 0, 2, 3)
                pred = pred.squeeze(0)
            else:
                pred = pred.squeeze(0)

            tensor_pred += pred

        _,label_pred = torch.max(tensor_pred, dim=0)


    return label_pred


def pad_to_training_size(np_img, args):

    z, y, x = np_img.shape
   
    if args.dimension == '3d':
        if z < args.training_size[0]:
            diff = (args.training_size[0]+2 - z) // 2
            np_img = np.pad(np_img, ((diff, diff), (0,0), (0,0)))
            z_start = diff
            z_end = diff + z
        else:
            z_start = 0
            z_end = z

        if y < args.training_size[1]:
            diff = (args.training_size[1]+2 - y) // 2
            np_img = np.pad(np_img, ((0,0), (diff, diff), (0,0)))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[2]:
            diff = (args.training_size[2]+2 -x) // 2
            np_img = np.pad(np_img, ((0,0), (0,0), (diff, diff)))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return np_img, [z_start, z_end, y_start, y_end, x_start, x_end]

    elif args.dimension == '2d':
        
        if y < args.training_size[0]:
            diff = (args.training_size[0]+2 - y) // 2
            np_img = np.pad(np_img, ((0,0), (diff, diff), (0,0)))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[1]:
            diff = (args.training_size[1]+2 -x) // 2
            np_img = np.pad(np_img, ((0,0), (0,0), (diff, diff)))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return np_img, [y_start, y_end, x_start, x_end]

    else:
        raise ValueError




def unpad_img(np_pred, original_idx, args):
    if args.dimension == '3d':
        z_start, z_end, y_start, y_end, x_start, x_end = original_idx
    
        return np_pred[z_start:z_end, y_start:y_end, x_start:x_end]
    elif args.dimension == '2d':
        y_start, y_end, x_start, x_end = original_idx

        return np_pred[:, y_start:y_end, x_start:x_end]
        
    else:
        raise ValueError


def preprocess(input_img, input_lab):
    '''
    This function performs preprocessing to make images to be consistent with training, e.g. spacing resample, redirection and etc.
    Args:
        itk_img: the simpleITK image to be predicted
    Return: the preprocessed image tensor
    '''

    img = input_img.astype(np.float32)
    lab = input_lab.astype(np.uint8)
    
    '''
    Need to modify the following preprocessing steps to be consistent with training. Copy from the dataset_xxx.py
    '''
    #np_img = np.clip(np_img, -79, 304)
    #np_img -= 100.93
    #np_img /= 76.90
    max98 = np.percentile(img, 98)
    img = np.clip(img, 0, max98)
    img = img / max98

    # np_img, original_idx = pad_to_training_size(np_img, args)

    tensor_img = torch.from_numpy(img).float()
    tensor_lab = torch.from_numpy(lab).long()

    return tensor_img, tensor_lab


def postprocess(input):
    img = input.astype(np.float32)


    '''
    Need to modify the following preprocessing steps to be consistent with training. Copy from the dataset_xxx.py
    '''
    # np_img = np.clip(np_img, -79, 304)
    # np_img -= 100.93
    # np_img /= 76.90
    max98 = np.percentile(img, 98)
    img = np.clip(img, 0, max98)
    img = img / max98

    # np_img, original_idx = pad_to_training_size(np_img, args)

    tensor_img = torch.from_numpy(img).float()


    return tensor_img




def init_model(args):

    model_list = []
    for ckp_path in args.load:
        model = get_model(args)
        pth = torch.load(ckp_path, map_location=torch.device('cpu'))

        if args.ema:
            # model.load_state_dict(pth['ema_model_state_dict'])
            model = (pth['ema_model_state_dict'])  # load model
	        # model.load_state_dict(pth['ema_model_state_dict'])   # load model weight
        else:
            model.load_state_dict(pth['model_state_dict'])

        # If you want to load checkpoint trained with previous version code, use the following instead:
        #model.load_state_dict(pth)

        print(model)
        model.cuda()
        model_list.append(model)
        print(f"Model loaded from {ckp_path}")

    return model_list


def get_parser():

    def parse_spacing_list(string):
        return tuple([float(spacing) for spacing in string.split(',')])
    def parse_model_list(string):
        return string.split(',')
    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--dataset', type=str, default='CAG', help='dataset name')
    parser.add_argument('--model', type=str, default='vss', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--frame_N', type=str, default=8, help='stack frame number')

    parser.add_argument('--load', type=parse_model_list, default='./exp/CAG/vss/best.pth', help='the path of trained model checkpoint. Use \',\' as the separator if load multiple checkpoints for ensemble')
    parser.add_argument('--img_path', type=str, default=r"../data/test/label", help='the path of the directory of images to be predicted')
    # parser.add_argument('--img_path', type=str, default='D:\gh\data\prediction\image',help='the path of the directory of images to be predicted')
    parser.add_argument('--save_path', type=str, default='./result/vss', help='the path to save predicted label')
    parser.add_argument('--target_spacing', type=parse_spacing_list, default='1.0,1.0,1.0', help='the spacing that used for training, in x,y,z order for 3d, and x,y order for 2d')

    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    # config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    config_path = 'normal_config.yaml'
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args




if __name__ == '__main__':

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.sliding_window = False
    args.window_size = args.training_size


    model_list = init_model(args)
    # 获取测试数据并预测
    patient_list = os.listdir(args.img_path)
    subdir_path_list = []
    for patient in patient_list:
        dir_path = os.path.join(args.img_path, patient)
        xulie_lsit = os.listdir(dir_path)
        for xulie in xulie_lsit:
            subdir_path = os.path.join(dir_path, xulie)
            subdir_path_list.append(subdir_path)
    trainlab_list = []
    trainimg_list = []
    for path in (subdir_path_list):
        lab_path_list = os.listdir(path)
        img_list = []
        lab_list = []
        labimg_path_list = []
        for item in lab_path_list:
              # 一个序列的lab路径
            lab_path = os.path.join(path, item)
            labimg_path_list.append(lab_path)
            img_path = lab_path.replace("label", "image")
            # img_path = lab_path.replace("trydata", "image")
            img = cv2.imread(img_path, 0)
            lab = cv2.imread(lab_path, 0)
            lab = lab // 255

            img, lab = preprocess(img, lab)
            img_list.append(img)     # 一个序列的img
            lab_list.append(lab)     # 一个序列的lab
            trainlab_list.append(lab)

        n1 = args.frame_N // 2
        n2 = args.frame_N - n1 - 1
        start_image = img_list[0]
        end_image = img_list[-1]
        for i in range(n1):
            img_list.insert(0, start_image)  # 在序列开始复制 N//2张图片
        for j in range(n2):
            img_list.insert(-1, end_image)  # 在序列最后复制N- N//2张图片
        for k,item_path in enumerate(labimg_path_list):
            stack_list = []
            # 获取待stack的图片列表
            for n in range(args.frame_N):
                stack_list.append(img_list[k + n])
            img3d = torch.stack(stack_list, 0)
            img3d = img3d.unsqueeze(0)
            # img3d = torch.from_numpy(img3d)  # 从numpy转换成tensor

            ####################################################################
            pred_label = prediction(model_list, img3d, args) # 3d prediction
            # x0 = x0.cpu().numpy()
            # x0 = x0.squeeze(0)
            # channel_num = x0.shape[0]
            # for i in range(channel_num):
            #     plt.subplot(4,8,i+1)
            #     plt.plot(x0[i,:,:])
            # plt.show()

            pre = pred_label.cpu().numpy()
            pre = pre*255
            labsave_path = os.path.join(args.save_path,path.split("/")[-2],path.split("/")[-1])
            if not os.path.exists(labsave_path):
                os.makedirs(labsave_path)
            save_path = os.path.join(labsave_path,labimg_path_list[k].split("/")[-1])
            cv2.imwrite(os.path.join(labsave_path,labimg_path_list[k].split("/")[-1]), pre)
            print(save_path, 'done')

    # 特征图可视化
    # stack_list = []
    # for img_name in os.listdir("./feature_extra"):
    #     img_path = os.path.join("./feature_extra",img_name)
    #     img = cv2.imread(img_path,0)
    #     img = postprocess(img)
    #     stack_list.append(img)
    # inputimg = np.stack(stack_list,0)
    # inputimg = torch.from_numpy(inputimg)
    # pred_label = prediction(model_list, inputimg, args)
    # pre = pred_label.cpu().numpy()
    # pre = pre * 255






