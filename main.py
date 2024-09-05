import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.tinycnn import TinnyCNN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from dataset import ExpressionDataset

def main():
    root_dir = r"C:\\Users\\ycli\\Desktop\\proj\\expressionReg"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir)
    path_txt_train = os.path.join(root_dir, "train.txt")
    path_txt_valid = os.path.join(root_dir, "val.txt")
    transforms_func = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    train_data = ExpressionDataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    valid_data = ExpressionDataset(root_dir=img_dir, txt_path=path_txt_valid, transform=transforms_func)
    train_loader = DataLoader(dataset=train_data, batch_size=2, drop_last=True, shuffle=True)
    print(train_data.__getitem__(0)[0].shape)
    
    model = TinnyCNN(7)

    # step 3/4 : 优化模块
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=5)
    # step 4/4 : 迭代模块
    for epoch in range(20):
        # 训练集训练
        model.train()
        for data, labels in train_loader:
            # forward & backward
            outputs = model(data)
            optimizer.zero_grad()

            # loss 计算
            loss = loss_f(outputs, labels)
            loss.backward()
            max_norm = 3.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # for name,p in model.named_parameters():
            #     print(name,p)
            optimizer.step()

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == labels).sum()
            acc = correct_num / labels.shape[0]
            # print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))
    
        # 验证集验证
        model.eval()
        valid_loader = DataLoader(dataset=valid_data, batch_size=2, drop_last=True, shuffle=True)
        correct_num = 0
        s_num = 0
        loss = 0
        for data, labels in valid_loader:
            # forward
            outputs = model(data)

            # loss 计算
            loss += loss_f(outputs, labels)

            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num += (predicted == labels).sum()
            s_num += labels.shape[0]
        acc_valid = correct_num / s_num
        print("Epoch:{} Valid Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc_valid))
        
        checkpoint = {
            'epoch': epoch,  # current epoch
            'model_state_dict': model.state_dict(),  # model's state
            'optimizer_state_dict': optimizer.state_dict(),  # optimizer's state
            'loss': loss,  # loss value at checkpoint
        }
        torch.save(checkpoint, f'ckpt/model_checkpoint_{epoch}.pth')
        
        # 添加停止条件
        if acc_valid == 1:
            break

        # 学习率调整
        scheduler.step()


if __name__ == "__main__":
    main()