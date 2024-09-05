import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.tinycnn import TinnyCNN
from model.resnet import ResNet18
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from dataset import ExpressionDatasetFER2013
from tqdm import tqdm

def main():
    root_dir = r"C:\\Users\\ycli\\Desktop\\proj\\expressionReg"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir)
    csv_path = os.path.join(root_dir, "fer2013.csv")
    transforms_func = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=35),
        transforms.ToTensor(),
    ])
    train_data = ExpressionDatasetFER2013(root_dir=img_dir, csv_path=csv_path, mode='Training', transform=transforms_func)
    train_loader = DataLoader(dataset=train_data, batch_size=32, drop_last=True, shuffle=True)
    
    model = ResNet18()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    # step 3/4 : 优化模块
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=5)
    
    model.train()
    # step 4/4 : 迭代模块
    for epoch in range(20):
        
        # 训练集训练
        correct_num = 0
        l = 0;s = 0
        for data, labels in tqdm(train_loader):
            # forward & backward
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            optimizer.zero_grad()

            # loss 计算
            loss = loss_f(outputs, labels)
            l += loss
            loss.backward()
            max_norm = 2.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num += (predicted == labels).sum()
            s += labels.shape[0]
        acc = correct_num / s
        print("Epoch:{} Training Loss:{:.2f} Acc:{:.0%}\n".format(epoch, l, acc))
        
        checkpoint = {
            'epoch': epoch,  # current epoch
            'model_state_dict': model.state_dict(),  # model's state
            'optimizer_state_dict': optimizer.state_dict(),  # optimizer's state
            'loss': loss,  # loss value at checkpoint
        }
        torch.save(checkpoint, f'ckpt/fer2013/model_checkpoint_{epoch}.pth')
        
        # 添加停止条件
        # if acc_valid == 1:
        #     break

        # 学习率调整
        scheduler.step()


if __name__ == "__main__":
    main()