import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from model.tinycnn import TinnyCNN
import torch.nn as nn
from torchvision import transforms
from PIL import Image

def load_ckpt(path):
    model = TinnyCNN(7)
    # step 3/4 : 优化模块
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    checkpoint = torch.load(path)

    # Restore the model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model

def get_result(data_in,model):
    model.eval()
    outputs = model(data_in)
    return outputs

def main():
    D = {0:'surprise',1:'sadness',2:'happy',3:'fear',4:'disgust',5:'contempt',6:'anger'}
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    model = load_ckpt("./ckpt/model_checkpoint_13.pth")
    x, y, w, h = 250, 120, 200, 200
    # image = Image.open('1.jpg')
    # print(type(image))
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        roi = frame[y:y+h, x:x+w]
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        # cv2.imshow('Webcam Feed', frame)
        image = Image.fromarray(gray_image)
        preprocess = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
        resized_img = preprocess(image).unsqueeze(0)
        cv2.imshow('Region of Interest', gray_image)
        A = get_result(resized_img,model)
        _, predicted = torch.max(A, 1)
        print(D[int(predicted[0])])
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
