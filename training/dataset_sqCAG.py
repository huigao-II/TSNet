import os
import torch.multiprocessing
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
import yaml
import random
import sys
import cv2
import argparse
import copy
from torch.utils import data

sys.path.append("dataset")
sys.path.append("training")
import augmentation
sys.path.append("config")

import logging



class CAGDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=5, k=0, seed=0):
        self.frame = args.frame_N
        self.mode = mode
        self.args = args
        subdir_path_list = []
        assert mode in ['train', 'test']
        patient_list = os.listdir(args.data_root)
        for patient in patient_list:
            dir_path = os.path.join(args.data_root,patient)
            xulie_lsit = os.listdir(dir_path)
            for xulie in xulie_lsit:
                subdir_path = os.path.join(dir_path,xulie)
                subdir_path_list.append(subdir_path)

        # 数据集大小划分
        # length = len(subdir_path_list)
        # train_subdir_path_list = []
        # train_len = int(length*0.5)    # 用50%的数据
        #
        # random.seed(1001)
        # index_list = random.sample(range(0,length), train_len)
        # for idx in index_list:
        #     train_subdir_path_list.append(subdir_path_list[idx])

        # length = len(train_subdir_path_list)
        # 训练集、验证集划分
        length = len(subdir_path_list)

        test_path_list = []
        test_len = int(length*0.2)    # 验证集比例0.1

        random.seed(1)
        index_list = random.sample(range(0,length), test_len)
        for idx in index_list:
            test_path_list.append(subdir_path_list[idx])
        # test_path_list = subdir_path_list[k * (length // k_fold): (k + 1) * (length // k_fold)]    # 五折交叉验证
        train_path_list = list(set(subdir_path_list) - set(test_path_list))

        if mode == 'train':
            img_path_list = train_path_list
        else:
            img_path_list = test_path_list
        logging.info(f'Start loading {self.mode} data')

        self.imgpath_list = []
        self.labpath_list = []
        self.augimgpath_list = []
        self.auglabpath_list = []
       
        for path in img_path_list:
            lab_path_list = os.listdir(path)
            self.img_path_list = []          # 每个序列的路径列表
            #self.lab_list = []

            for item in lab_path_list:
                lab_path = os.path.join(path,item)
               
                img_path = lab_path.replace("label", "image")    #############获取图片的路径
                
                self.labpath_list.append(lab_path)
                if args.dimension == '2d':
                    self.imgpath_list.append(img_path)
                
                elif args.dimension == '3d':
                    self.img_path_list.append(img_path)
            
            if args.dimension == '3d':
                l =len(self.img_path_list)
                self.n1 = self.frame//2
                self.n2 = self.frame-self.n1-1
                start_image_path = self.img_path_list[0]
                end_image_path = self.img_path_list[-1]
                for i in range(self.n1):
                  self.img_path_list.insert(0,start_image_path)  # 在序列开始复制 N//2张图片
                for j in range(self.n2):
                  self.img_path_list.insert(-1,end_image_path)   # 在序列最后复制N- N//2张图片
                for k in range(l):
                    stack_path_list = []
                    for n in range(self.frame):
                      stack_path_list.append(self.img_path_list[k+n])
                    self.imgpath_list.append(stack_path_list)
                    
        self.augimgpath_list.extend(self.imgpath_list)
        self.auglabpath_list.extend(self.labpath_list)
        
        self.img_len = len(self.imgpath_list)  
        # 数据增强   
        if mode == 'train':
            random.seed(100)
            self.gau_len = int(self.img_len*0.4)
            gauindex_list = random.sample(range(0,self.img_len),self.gau_len)
            for index1 in gauindex_list:
             gau_path = self.imgpath_list[index1]
             self.augimgpath_list.append(gau_path)
            
            
            random.seed(101)
            self.gam_len = int(self.img_len*0.3)
            gamindex_list = random.sample(range(0,self.img_len), self.gam_len)
            for index2 in gamindex_list:
             gam_path = self.imgpath_list[index2]
             self.augimgpath_list.append(gam_path)
            
            random.seed(103)
            self.bri_len = int(self.img_len*0.3)
            briindex_list = random.sample(range(0,self.img_len), self.bri_len)
            for index3 in briindex_list:
             bri_path = self.imgpath_list[index3]
             self.augimgpath_list.append(bri_path)
            
            # self.augimgpath_list.extend(gaupath_list)
            # self.augimgpath_list.extend(gampath_list)
            # self.augimgpath_list.extend(bripath_list) 
                    
        #assert len(self.trainimg_list) == len(self.trainlab_list)
                    
                


                # img,lab = self.preprocess(img,lab)
                # # print(lab_path)
                # #self.lab_list.append(lab)
                # self.trainlab_list.append(lab)  #用于训练的lab 512,512
                # if args.dimension == '3d':
                #   self.img_list.append(img)
                # elif args.dimension == '2d':
                #   self.trainimg_list.append(img)
                
                
            # if args.dimension == '3d':
            #   l =len(self.img_list)
            #   self.n1 = self.frame//2
            #   self.n2 = self.frame-self.n1-1
            #   start_image = self.img_list[0]
            #   end_image = self.img_list[-1]
            #   for i in range(self.n1):
            #       self.img_list.insert(0,start_image)  # 在序列开始复制 N//2张图片
            #   for j in range(self.n2):
            #       self.img_list.insert(-1,end_image)   # 在序列最后复制N- N//2张图片
            #   for k in range(l):
            #       stack_list = []
            #       # 获取待stack的图片列表
            #       for n in range(self.frame):
            #           stack_list.append(self.img_list[k+n])
            #       # img3d = np.stack(stack_list,0)
            #       # img3d = torch.from_numpy(img3d)   #从numpy转换成tensor
            #       img3d = torch.stack(stack_list, 0) # 3,512,512
            #       self.trainimg_list.append(img3d)  # 用于训练的image

        # print(len(self.trainimg_list),len(self.trainlab_list),len(labimg_path_list),len(patient_list),len(subdir_path_list))
        

        # 镜像
        # rotate_img_list,rotate_lab_list = self.rotate(0.4,self.trainimg_list,self.trainlab_list)
        # self.augimg_list.extend(rotate_img_list)
        # self.auglab_list.extend(rotate_lab_list)
        # 裁剪
        # crop_img_list, crop_lab_list = self.crop(0.5, self.trainimg_list, self.trainlab_list)
        # self.augimg_list.extend(crop_img_list)
        # self.auglab_list.extend(crop_lab_list)
        
        # if mode == 'train':
            
        #   # 高斯噪声
        #   gau_img_list, gau_lab_list = self.gaussian(0.4, self.trainimg_list, self.trainlab_list)
        #   self.augimg_list.extend(gau_img_list)
        #   self.auglab_list.extend(gau_lab_list)
        #   # gamma噪声
        #   gam_img_list, gam_lab_list = self.gamma(0.3, self.trainimg_list, self.trainlab_list)
        #   self.augimg_list.extend(gam_img_list)
        #   self.auglab_list.extend(gam_lab_list)
        #   #亮度
        #   bri_img_list, bri_lab_list = self.brightness(0.3, self.trainimg_list, self.trainlab_list)
        #   self.augimg_list.extend(bri_img_list)
        #   self.auglab_list.extend(bri_lab_list)
        ###############可视化
        # plt.subplot(3,3,1)
        # show1 = gau_img_list[0][0]
        # plt.imshow(show1.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3,3,2)
        # show2 = gau_img_list[0][1]
        # plt.imshow(show2.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3,3,3)
        # show3 = gau_img_list[0][2]
        # plt.imshow(show3.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3,3,4)
        # show4 = gam_img_list[0][0]
        # plt.imshow(show4.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3,3,5)
        # show5 = gam_img_list[0][1]
        # plt.imshow(show5.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3,3,6)
        # show6 = gam_img_list[0][2]
        # plt.imshow(show6.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3, 3, 7)
        # show7 = bri_img_list[0][0]
        # plt.imshow(show7.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3, 3, 8)
        # show8 = bri_img_list[0][1]
        # plt.imshow(show8.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.subplot(3, 3, 9)
        # show9 = bri_img_list[0][2]
        # plt.imshow(show9.cpu(),cmap="gray")
        # plt.axis('off')
        # plt.show()
        
        
        logging.info(f"Load done, length of dataset: {len(self.augimgpath_list)}")





    def __len__(self):
        if self.mode == 'train':
            return len(self.augimgpath_list)
        else:
            return len(self.augimgpath_list)
    def preprocess(self, img):

        max98 = np.percentile(img, 98)
        img = np.clip(img, 0, max98)
        img = img / max98
        tensor_img = torch.from_numpy(img).float()
        

        return tensor_img


    def __getitem__(self, idx):

        # idx = idx % len(self.lab_list)
        
        img_path = self.augimgpath_list[idx] #(D,H,W)
        
        
        if self.args.dimension == '2d':
            lab_path = img_path.replace('image','label') #(H,W)
            
            
            img = cv2.imread(img_path,0)
            lab = cv2.imread(lab_path,0)
            lab = lab//255
            img = self.preprocess(img)
            lab = torch.from_numpy(lab).long()
         
            
            if self.mode == 'train':
                if idx>=self.img_len and idx<self.gau_len:
                    img = augmentation.gaussian_noise(img, std=self.args.gaussian_noise_std)
                
                if idx>=self.gau_len and idx<self.gam_len:
                    img = augmentation.gamma(img.unsqueeze(0).unsqueeze(0), gamma_range=self.args.gamma_range, retain_stats=True).squeeze(0).squeeze(0)
                elif idx>=self.gam_len and idx<self.bri_len:
                    img = augmentation.brightness_multiply(img.unsqueeze(0).unsqueeze(0),
                                                    ).squeeze(0).squeeze(0)
                
                   
            tensor_img = img.unsqueeze(0)  #(1,1,D,H,W)
            tensor_lab = lab.unsqueeze(0)  
                
                
        elif self.args.dimension == '3d':
            lab_path = img_path[len(img_path)//2].replace('image','label') #(H,W)
            lab = cv2.imread(lab_path,0)
            lab = lab//255
            lab = torch.from_numpy(lab).long()
            
            stack_list = []
            for s in img_path:
                img_slice = cv2.imread(s,0)
                img_slice = self.preprocess(img_slice)
                stack_list.append(img_slice)
                
            img =torch.stack(stack_list)
            
            if self.mode == 'train':
            
                if idx>=self.img_len and idx<self.gau_len:
                    img = augmentation.gaussian_noise(img, std=self.args.gaussian_noise_std)
                
                if idx>=self.gau_len and idx<self.gam_len:
                    img = augmentation.gamma(img.unsqueeze(0).unsqueeze(0), gamma_range=self.args.gamma_range, retain_stats=True).squeeze(0).squeeze(0)
                elif idx>=self.gam_len and idx<self.bri_len:
                    img = augmentation.brightness_multiply(img.unsqueeze(0).unsqueeze(0),
                                                    ).squeeze(0).squeeze(0)
            
            tensor_img = img.unsqueeze(0)  #(1,1,D,H,W)
            tensor_lab = lab.unsqueeze(0)
        
        if self.mode == 'train':

            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx)
                tensor_lab = tensor_lab.cuda(self.args.proc_idx)

            # Gaussian Noise
            # tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)
            # # Additive brightness
            # tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)
            # # gamma
            # tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)
            #
            # tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab,
            #                                                                        self.args.scale, self.args.rotate,
            #                                                                        self.args.translate)
            # tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size,mode='random')


        # else:
        #    tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab,self.args.training_size, mode='center')

        # tensor_img = tensor_img.squeeze(0)
        # tensor_lab = tensor_lab.squeeze(0)

        # assert tensor_img.shape == tensor_lab.shape

        # if self.mode == 'train':
        #     return tensor_img, tensor_lab.to(torch.int8)
        # else:
        #     return tensor_img, tensor_lab, np.array(self.spacing_list[idx])
        return tensor_img, tensor_lab.to(torch.int8)


def get_parser():
    def parse_spacing_list(string):
        return tuple([float(spacing) for spacing in string.split(',')])

    def parse_model_list(string):
        return string.split(',')

    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--dataset', type=str, default='CAG', help='dataset name')
    parser.add_argument('--frame_N', type=int, default='3', help='frame number of input')
    parser.add_argument('--model', type=str, default='DE_DCGCN_EE', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')

    parser.add_argument('--load', type=parse_model_list, default=False,
                        help='the path of trained model checkpoint. Use \',\' as the separator if load multiple checkpoints for ensemble')
    parser.add_argument('--img_path', type=str, default=False,
                        help='the path of the directory of images to be predicted')
    parser.add_argument('--save_path', type=str, default='./result/', help='the path to save predicted label')
    parser.add_argument('--target_spacing', type=parse_spacing_list, default='1.0,1.0,1.0',
                        help='the spacing that used for training, in x,y,z order for 3d, and x,y order for 2d')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data_root', type=str, default=' D:/gh/renamedata/test/label')

    args = parser.parse_args()

    # config_path = 'config/%s/%s_%s.yaml' % (args.dataset, args.model, args.dimension)
    config_path = "normal_config.yaml"
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s" % config_path)

    print('Loading configurations from %s' % config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_parser()
    trainset = CAGDataset(args, mode='train', k_fold=args.k_fold)
    trainLoader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4


    )
    print(len(trainLoader))
    for i, inputs in enumerate(trainLoader):
        
        image, label = inputs[0], inputs[1].cuda().to(torch.int8).long()
        print(i,image.shape,label.shape)
        # for i in range(image.shape[0]):
        #     x = image.squeeze(1)
        #     x = x[i]
        #     # plt.plot(4,8,i+1)
        #     for j in range(x.shape[0]):
        #         img = x[j]
        #         show = img.cpu()
        #         plt.subplot(1,3,j+1) # subplot 的图索引必须从1开始
        #         plt.imshow(show,cmap="gray")
        #     plt.show()


