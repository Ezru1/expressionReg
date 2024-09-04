import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.tinycnn import TinnyCNN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

def main():
    with open('fer2013.csv', 'r') as file:
        txt_data = file.read().strip()
        txt_data = txt_data.split("\n")
        print(txt_data)

if __name__ == "__main__":
    main()