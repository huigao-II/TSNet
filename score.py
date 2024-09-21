import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import metrics
import numpy as np
import Gudhi as gdh
import cv2
import os
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)
def safe_divide(a, b):
    if np.isnan(a) or np.isnan(b):
        print("检测到 NaN 值")
        return np.nan
    if np.isinf(a) or np.isinf(b):
        print("检测到 Inf 值")
        return np.nan
    if b == 0:
        print("检测到除零操作")
        return np.nan
    return a / b

def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    # safe_divide(2*tprec*tsens,tprec+tsens)
    return 2*tprec*tsens/(tprec+tsens+0.00000001)

def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    # a_ = a[k]
    # b_ = b[k]
    hist_ = n * a[k].astype(int) + b[k]
    return np.bincount(hist_, minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_dice(hist):
    return 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def calculate_dice(pred, target, C):
    # pred and target are torch tensor
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.)

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.)

    intersection = pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)

    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.to(pred.device)
    dice = 2 * intersection / summ        #intersection: TP, summ:FN+TP+TP+FP


    return dice, intersection, summ


# 测试代码test
def calculate_score(pred, target, C):
    # pred and target are torch tensor
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.)
    targ_num = target_mask.sum(0).type(torch.float32)  # 得到数据中每类的数量

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.)
    pred_num = pred_mask.sum(0).type(torch.float32)  # 预测数据中每类的数量


    intersection = pred_mask * target_mask  # 得到各类分类正确的数量
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)  # 求每一列的和
    summ = summ.sum(0).type(torch.float32)

    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.to(pred.device)
    dice = 2 * intersection / summ      #intersection: TP, summ:FN+TP+TP+FP
    # IoU = intersection/(summ-intersection)
    IoU = dice/(2-dice)
    R = intersection/targ_num   # R=TP/(TP+FP)
    P = intersection/pred_num   # P=TP/(TP+FN)
    F = 2*P*R/(P+R)
    ACC = 100. * intersection.sum(0) / targ_num.sum(0)

    return dice, IoU,R,P,F,ACC,intersection, summ

def mask_to_boundary(mask, dilation_ratio=0.001):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    mask = mask.astype(np.uint8)
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)

    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]

    # G_d intersects G in the paper.
    return mask - mask_erode


#
def boundary_iou(gt, dt, dilation_ratio=0.001):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)

    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou, gt_boundary, dt_boundary



def betti_err(gt,pr):
    gt_temp = gdh.compute_persistence_diagram(gt, i=1)
    pr_temp = gdh.compute_persistence_diagram(pr, i=1)
    gt_betti_number = len(gt_temp)
    pr_betti_number = len(pr_temp)
    err = abs(gt_betti_number-pr_betti_number)
    return err

# gt_path = r"../data/test/label"
gt_path = 'new_test_label2'
pr_path = 'result/unet_latest'
patient_list = os.listdir(gt_path)
hist = np.zeros((2, 2))
biou_list = []
cl_list = []
err_list = []
for patient in patient_list:
    xulie_list = os.listdir(os.path.join(gt_path,patient))
    for xulie in xulie_list:
        frame_list = os.listdir(os.path.join(gt_path,patient,xulie))
        for frame in frame_list:
            primg_path = os.path.join(pr_path,patient,xulie,frame)
            gtimg_path = os.path.join(gt_path,patient,xulie,frame)
            # print(primg_path)
            # print(gtimg_path)
            gt = cv2.imread(gtimg_path,0)//255
            pr = cv2.imread(primg_path,0)//255
            # 计算混淆矩阵
            hist += fast_hist(gt.flatten(), pr.flatten(), 2)
            biou,_,_ = boundary_iou(gt,pr,dilation_ratio=0.002)
            cl = clDice(pr,gt)
            err = betti_err(gt,pr)
            # print(cl)
            biou_list.append(biou)
            cl_list.append(cl)
            err_list.append(err)

Iou = per_class_iu(hist)[1]
dice = per_class_dice(hist)[1]

Acc = per_Accuracy(hist)  # 2024-01-28
pre = per_class_Precision(hist)[1]  # 2024-01-28
re = per_class_PA_Recall(hist)[1]
biou = np.mean(biou_list)
cl = np.mean(cl_list)
bettierr = np.mean(err_list)

# dice, IoU,R,P,F,ACC,intersection, summ = calculate_score(pr.view(-1, 1), gt.view(-1, 1), 2)
print(dice, Iou, pre, re, Acc, biou, cl, bettierr)
#print(dice, Iou,pre,re,Acc,biou,cl,)
