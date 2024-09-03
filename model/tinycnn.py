import torch.nn as nn

class TinnyCNN(nn.Module):
    def __init__(self, cls_num=7):
        super(TinnyCNN, self).__init__()
        self.c1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(2304*8, cls_num)
        # self.fc2 = nn.Linear(1000, cls_num)

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
if __name__ == "__main__":
    m = TinnyCNN(7)
    print(m)