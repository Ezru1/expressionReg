import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import time

class ExpressionDataset(Dataset):
    def __init__(self, root_dir, txt_path, transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        self.root_dir = root_dir
        self.txt_path = txt_path
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据,并预处理,to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('L')
        if self.transform is not None:
            img = self.transform(img)
    
        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取,将硬盘中的数据路径,标签读取进来,存在一个list中
        path, label
        :return:
        """
        # 读取txt，解析txt
        with open(self.txt_path, "r") as f:
            txt_data = f.read().strip()
            txt_data = txt_data.split("\n")
        self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[1]))
                             for i in txt_data]
        
        
class ExpressionDatasetFER2013(Dataset):
    def __init__(self, root_dir, csv_path, mode, transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.mode = mode
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据,并预处理,to Tensor
        :param index:
        :return:
        """
        img, label = self.img_info[index]
        img = Image.fromarray(img, mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取,将硬盘中的数据路径,标签读取进来,存在一个list中
        path, label
        :return:
        """
        # 读取txt，解析txt
        with open(self.csv_path, "r") as f:
            csv_data = f.read().strip()
            csv_data = csv_data.split("\n")[1:]
        self.img_info = [(np.array(list(map(int,i.split(',')[1].split())), dtype = np.uint8).reshape(48,48), int(i.split(',')[0]))
                             for i in csv_data if i.split(',')[2] == self.mode]
        
if __name__ == "__main__":
    
    root_dir = r"C:\\Users\\ycli\\Desktop\\proj\\expressionReg"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir)
    
    csv_path = os.path.join(root_dir, "fer2013.csv")
    transforms_func = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    D = ExpressionDatasetFER2013(root_dir=img_dir, csv_path=csv_path, mode='Training', transform=transforms_func)
    D.__getitem__(0)
    
    # path_txt_train = os.path.join(root_dir, "train.txt")
    # path_txt_valid = os.path.join(root_dir, "val.txt")
    # transforms_func = transforms.Compose([
    #     transforms.Resize((48, 48)),
    #     transforms.ToTensor(),
    # ])
    # a = ExpressionDataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    # for i in range(100):
    #     a,b = D.__getitem__(i)
    #     a.show()
    #     time.sleep(1)