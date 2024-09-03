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
    path_txt_test = os.path.join(root_dir, "val.txt")
    transforms_func = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    test_data = ExpressionDataset(root_dir=img_dir, txt_path=path_txt_test, transform=transforms_func)
    checkpoint = torch.load("./ckpt/model_checkpoint_19.pth")
    model = TinnyCNN(7)
    loss_f = nn.CrossEntropyLoss()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    test_loader = DataLoader(dataset=test_data, batch_size=2, drop_last=True, shuffle=True)
    correct_num = 0
    s_num = 0
    loss = 0
    for data, labels in test_loader:
        # forward
        outputs = model(data)

        # loss 计算
        loss += loss_f(outputs, labels)

        # 计算分类准确率
        _, predicted = torch.max(outputs.data, 1)
        correct_num += (predicted == labels).sum()
        s_num += labels.shape[0]
    acc_valid = correct_num / s_num
    print("Valid Loss:{:.2f} Acc:{:.0%}".format(loss, acc_valid))
    return

if __name__ == "__main__":
    main()