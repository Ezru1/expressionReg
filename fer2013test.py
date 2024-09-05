import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.resnet import ResNet18
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from dataset import ExpressionDatasetFER2013

def main():
    root_dir = r"C:\\Users\\ycli\\Desktop\\proj\\expressionReg"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir)
    csv_path = os.path.join(root_dir, "fer2013.csv")
    transforms_func = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        # transforms.RandomRotation(degrees=35),
        transforms.ToTensor(),
    ])
    test_data = ExpressionDatasetFER2013(root_dir=img_dir, csv_path=csv_path, mode='PublicTest', transform=transforms_func)
    test_loader = DataLoader(dataset=test_data, batch_size=128, drop_last=True)
    checkpoint = torch.load("./ckpt/fer2013/model_checkpoint_19.pth")
    
    model = ResNet18()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_f = nn.CrossEntropyLoss()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    correct_num = 0
    s_num = 0
    loss = 0
    A = []
    for data, labels in test_loader:
        # forward
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)

        # loss 计算
        l = loss_f(outputs, labels)
        loss += l
        A.append(l)
        # 计算分类准确率
        _, predicted = torch.max(outputs.data, 1)
        correct_num += (predicted == labels).sum()
        s_num += labels.shape[0]
    acc_valid = correct_num / s_num
    print("Valid Loss:{:.2f} Acc:{:.0%}".format(loss, acc_valid))
    print(A)
    return

if __name__ == "__main__":
    main()