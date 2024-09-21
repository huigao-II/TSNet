import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import yaml
import torch
from model.utils import get_model
import argparse
import os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import cv2

def init_model(load):

    model_list = []
    for ckp_path in load:

        pth = torch.load(ckp_path, map_location=torch.device('cpu'))
        model = (pth['ema_model_state_dict'])  # load model
        # load model weight
        #model = get_model(args)
        # model.load_state_dict(pth['ema_model_state_dict'])
        print(model)
        model.cuda()
        model_list.append(model)
        print(f"Model loaded from {ckp_path}")

    return model_list

def preprocess(input_img, input_lab):
    img = input_img.astype(np.float32)
    lab = input_lab.astype(np.uint8)
    max98 = np.percentile(img, 98)
    img = np.clip(img, 0, max98)
    img = img / max98
    tensor_img = torch.from_numpy(img).float()
    tensor_lab = torch.from_numpy(lab).long()

    return tensor_img, tensor_lab

class Dataset2d(Dataset):
    def __init__(self, images_root_folder, labels_root_folder, transform=None):
        self.images_root_folder = images_root_folder
        self.labels_root_folder = labels_root_folder
        self.transform = transform
        self.image_label_pairs = self.get_image_label_pairs()
    def get_image_label_pairs(self):
        pairs = []
        for root, _, files in os.walk(self.labels_root_folder):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    label_path = os.path.join(root, file)
                    # Get relative path of the image
                    relative_path = os.path.relpath(label_path, self.images_root_folder)
                    image_path = os.path.join(self.images_root_folder, relative_path)
                    image = Image.open(image_path).convert('L')  # Assuming grayscale images
                    label = Image.open(label_path).convert('L')  # Assuming grayscale labels

                    if os.path.exists(image_path):
                        pairs.append((image, label))
                    else:
                        print(f"Label file not found for {image_path}")
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        # image_path, label_path = self.image_label_pairs[idx]
        #
        # image = Image.open(image_path).convert('L')  # Assuming grayscale images
        # label = Image.open(label_path).convert('L')  # Assuming grayscale labels
        image, label = self.image_label_pairs[idx]

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        label = (label > 0).float()  # Binarize label

        return image, label

class Datasetmulti(Dataset):
    def __init__(self, images_root_folder, labels_root_folder, transform=None):
        self.images_root_folder = images_root_folder
        self.labels_root_folder = labels_root_folder
        self.transform = transform
        self.image_label_pairs = self.get_image_label_pairs()

    def get_image_label_pairs(self):
        pairs = []
        patient_list = os.listdir(self.labels_root_folder)
        subdir_path_list = []
        for patient in patient_list:
            dir_path = os.path.join(self.labels_root_folder, patient)
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
                img_path = os.path.join(self.images_root_folder,lab_path.split("/")[-3],lab_path.split("/")[-2],lab_path.split("/")[-1])
                # print(lab_path,img_path)

                img = cv2.imread(img_path, 0)
                lab = cv2.imread(lab_path, 0)
                lab = lab // 255
                img, lab = preprocess(img, lab)
                img_list.append(img)  # 一个序列的img
                lab_list.append(lab)  # 一个序列的lab
                trainlab_list.append(lab)
            n1 = 5 // 2
            n2 = 5 - n1 - 1
            start_image = img_list[0]
            end_image = img_list[-1]
            for i in range(n1):
                img_list.insert(0, start_image)  # 在序列开始复制 N//2张图片
            for j in range(n2):
                img_list.insert(-1, end_image)  # 在序列最后复制N- N//2张图片
            for k, item_path in enumerate(labimg_path_list):
                stack_list = []
                # 获取待stack的图片列表
                for n in range(5):
                    stack_list.append(img_list[k + n])
                img3d = torch.stack(stack_list, 0)
                img3d = img3d.unsqueeze(0)
                pairs.append((img3d, lab_list[k]))
                # print(f"Label file not found for {image_path}")
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        # image_path, label_path = self.image_label_pairs[idx]
        #
        # image = Image.open(image_path).convert('L')  # Assuming grayscale images
        # label = Image.open(label_path).convert('L')  # Assuming grayscale labels
        image, label = self.image_label_pairs[idx]

        # if self.transform:
        #     # image = self.transform(image)
        #     label = self.transform(label)
        #
        # label = (label > 0).float()  # Binarize label

        return image, label

class Dataset4f(Dataset):
    def __init__(self, images_root_folder, labels_root_folder, transform=None):
        self.images_root_folder = images_root_folder
        self.labels_root_folder = labels_root_folder
        self.transform = transform
        self.image_label_pairs = self.get_image_label_pairs()

    def get_image_label_pairs(self):
        pairs = []
        patient_list = os.listdir(self.labels_root_folder)
        subdir_path_list = []
        for patient in patient_list:
            dir_path = os.path.join(self.labels_root_folder, patient)
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
                img_path = os.path.join(self.images_root_folder,lab_path.split("/")[1],lab_path.split("/")[2],lab_path.split("/")[3])
                # img_path = lab_path.replace("trydata", "image")
                img = cv2.imread(img_path, 0)
                lab = cv2.imread(lab_path, 0)
                lab = lab // 255
                img, lab = preprocess(img, lab)
                img_list.append(img)  # 一个序列的img
                lab_list.append(lab)  # 一个序列的lab
                trainlab_list.append(lab)
            n1 = 4// 2
            n2 = 4- n1 - 1
            start_image = img_list[0]
            end_image = img_list[-1]
            for i in range(n1):
                img_list.insert(0, start_image)  # 在序列开始复制 N//2张图片
            for j in range(n2):
                img_list.insert(-1, end_image)  # 在序列最后复制N- N//2张图片
            for k, item_path in enumerate(labimg_path_list):
                stack_list = []
                # 获取待stack的图片列表
                for n in range(4):
                    stack_list.append(img_list[k + n])
                img3d = torch.stack(stack_list, 0)
                img3d = img3d.unsqueeze(0)
                pairs.append((img3d, lab_list[k]))
                # print(f"Label file not found for {image_path}")
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):

        image, label = self.image_label_pairs[idx]

        return image, label

class Dataset3d(Dataset):
    def __init__(self, images_root_folder, labels_root_folder, transform=None):
        self.images_root_folder = images_root_folder
        self.labels_root_folder = labels_root_folder
        self.transform = transform
        self.image_label_pairs = self.get_image_label_pairs()

    def get_image_label_pairs(self):
        pairs = []
        patient_list = os.listdir(self.labels_root_folder)
        subdir_path_list = []
        for patient in patient_list:
            dir_path = os.path.join(self.labels_root_folder, patient)
            xulie_lsit = os.listdir(dir_path)
            for xulie in xulie_lsit:
                subdir_path = os.path.join(dir_path, xulie)
                subdir_path_list.append(subdir_path)

        for path in (subdir_path_list):
            lab_path_list = os.listdir(path)
            img_list = []
            lab_list = []
            file_name_list = []

            for p, item in enumerate(lab_path_list):
                # 一个序列的lab路径
                lab_path = os.path.join(path, item)

                img_path = os.path.join(self.images_root_folder,lab_path.split("/")[1],lab_path.split("/")[2],lab_path.split("/")[3])
                file_name_list.append(img_path)
                # img_path = lab_path.replace("trydata", "image")
                img = cv2.imread(img_path, 0)
                lab = cv2.imread(lab_path, 0)
                lab = lab // 255


                img, lab = preprocess(img, lab)
                img_list.append(img)  # 一个序列的img
                lab_list.append(lab)

            l = len(img_list)
            w = l % 5
            t = l // 5
            # print(l,w,t)
            for k in range(t):
                stack_img_list = []
                stack_lab_list = []
                img_name_list = []
                for n in range(5):
                    stack_img_list.append(img_list[n + k * 5 + w])
                    stack_lab_list.append(lab_list[n + k * 5 + w])
                    img_name_list.append(file_name_list[n + k * 5 + w])

                img3d = torch.stack(stack_img_list, 0)
                img3d = img3d.unsqueeze(0)
                lab3d = torch.stack(stack_lab_list, 0)
                lab3d = lab3d.unsqueeze(0)
                pairs.append((img3d, lab3d))

        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        image, label = self.image_label_pairs[idx]
        return image, label


# Load the images and labels
def load_test_data(images_root_folder, labels_root_folder, batch_size=4,model_name=str):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    if model_name in ["attention_unet","swinunet","source_DE_DCGCN_EE","Angionet"]:
        dataset = Dataset2d(images_root_folder, labels_root_folder, transform=transform)

    elif model_name == "TSNet":
        dataset = Datasetmulti(images_root_folder, labels_root_folder, transform=transform)

    elif model_name == "SVSNet":
        dataset = Dataset4f(images_root_folder, labels_root_folder, transform=transform)
    elif model_name == "attention_unet_3d":
        dataset = Dataset3d(images_root_folder, labels_root_folder, transform=transform)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Generate ROC curve data
def generate_roc_data(model, dataloader, device='cuda'):
    y_true = []
    y_scores = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)[:,1,:,:]
            labels_np = labels.cpu().numpy().flatten()
            probs_np = probs.cpu().numpy().flatten()

            print(f"Batch labels shape: {labels_np.shape}, Batch predictions shape: {probs_np.shape}")

            y_true.append(labels_np)
            y_scores.append(probs_np)

    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)

    # Debugging information after concatenation
    print(f"Final y_true shape: {y_true.shape}, Final y_scores shape: {y_scores.shape}")

    return y_true, y_scores
# Plot ROC curve
def plot_roc_curve(y_true, y_scores,color):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, color=color, lw=2, label='ROC curve (area = %0.6f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.show()

# Main execution
def main():
    # load_list = ['TSNet',"attention_unet","attention_unet_3d","SVSNet","Angionet","swinunet","source_DE_DCGCN_EE"]\
    # load_list = ['TSNet', "attention_unet_3d","Angionet", "swinunet", "source_DE_DCGCN_EE"]
    model_list = []
    from model.dim2 import AttentionUNet

    model = AttentionUNet(1, 2, 32)
    pth = torch.load('exp/CAG/attention_unet/latest.pth', map_location=torch.device('cpu'))
    model.load_state_dict(pth['ema_model_state_dict'])
    model.cuda()
    model_list.append(model)
    print(f"Model loaded from exp/CAG/attention_unet/latest.pth")
    load_list = ["attention_unet_3d","Angionet","swinunet"]
    # load_list = ["SVSNet"]
    load = [os.path.join("exp/CAG",item,"best.pth") for item in load_list]
    print(load)
    image_path = "../data/test/image"
    label_path = "new_test_label2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_list2 = init_model(load)
    model_list.extend(model_list2)
    from model.dim2.svsnet import SVSNet
    model = SVSNet(1, 4, 512, 512)
    pth = torch.load('exp/CAG/SVSNet/latest.pth', map_location=torch.device('cpu'))
    model.load_state_dict(pth['ema_model_state_dict'])
    model.cuda()
    model_list.append(model)
    print(f"Model loaded from exp/CAG/SVSNet/latest.pth")

    load_list = ['source_DE_DCGCN_EE', "TSNet"]
    # load_list = ["SVSNet"]
    load = [os.path.join("exp/CAG", item, "best.pth") for item in load_list]
    model_list3 = init_model(load)
    model_list.extend(model_list3)

    # from model.dim2 import AttentionUNet
    #
    # model = AttentionUNet(1, 2, 32)
    # pth = torch.load('exp/CAG/attention_unet/latest.pth', map_location=torch.device('cpu'))
    # model.load_state_dict(pth['ema_model_state_dict'])
    # model.cuda()
    # model_list.append(model)
    # print(f"Model loaded from exp/CAG/attention_unet/latest.pth")

    # color = [(66,122,178),(240,145,72),(255,152,150),(219,219,141),(197,157,148),(175,199,232),(72,192,170)]
    color = ['sienna','orange','gold','mediumseagreen','indianred','steelblue','lightskyblue']
    # Replace `YourModelClass` with the actual model class you're using
    fpr_tpr = []
    for i,model in enumerate(model_list):
        # model_name = load_list[i]
        model_name = ["attention_unet", "attention_unet_3d", "Angionet", "swinunet", "SVSNet","source_DE_DCGCN_EE","TSNet" ][i]

        dataloader = load_test_data(image_path, label_path, batch_size=4,model_name=model_name.split("/")[-1])

        y_true, y_scores = generate_roc_data(model, dataloader, device=device)
        print(len(y_true) ,len(y_scores) )
        print(model_name)

        print("Prediction probability range:", np.min(y_scores), np.max(y_scores))
        print("Unique values in y_test:", np.unique(y_true))

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        fpr_tpr.append((fpr,tpr))
    model_name = ["2D-AttentionUnet","3D-AttentionUnet","Angionet","SwinUnet","SVSNet","DE-DCGCN-EE","TSNet(ours)"]

    plt.figure()
    for k,item in enumerate(fpr_tpr):
        plt.plot(item[0], item[1], color=color[k], lw=2, label= model_name[k])  #'(area = %0.6f)' % roc_auc
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
        # plot_roc_curve(y_true, y_scores,color[k])

if __name__ == "__main__":
    main()
