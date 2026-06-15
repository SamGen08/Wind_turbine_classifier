# -*- coding: utf-8 -*-
"""
Wind Turbine Classifier from Satellite Imagery
-----------------------------------------------
Binary CNN classifier to detect wind turbines in satellite image patches.
Dataset: https://www.kaggle.com/datasets/airbusgeo/airbus-wind-turbines-patches
 
Validation accuracy: 94.8%
"""

from google.colab import drive
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

#Paths
drive.mount("/content/drive")
train_path="/content/drive/MyDrive/Wind_project/Dataset/Train"
val_path = "/content/drive/MyDrive/Wind_project/Dataset/Validation"


#Device
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(device)


#Model
class WindTClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
    self.pool= nn.MaxPool2d(2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(8*32*32, 32)
    self.fc2 = nn.Linear(32, 2)

  def forward(self,x):
    out=self.pool(F.relu(self.conv1(x)))
    out=self.pool(F.relu(self.conv2(out)))
    out=self.flatten(out)
    out=self.fc2(F.relu(self.fc1(out)))
    return out
  

#Data
transform_val = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


transform_train = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

train_dataset = ImageFolder(root = train_path, transform = transform_train)
val_dataset = ImageFolder(root = val_path, transform = transform_val)

train_loader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = 32, shuffle = False)


#Training
model = WindTClassifier()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model.train()
for epoch in range(20):
  running_loss = 0
  for batch in train_loader:
    optimizer.zero_grad()
    images, labels = batch
    images,labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  avg_loss = running_loss/len(train_loader)
  print(avg_loss)

  
#Evaluation
model.eval()
miss = 0
with torch.no_grad():
  for batch in val_loader:
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    outputs = torch.argmax(outputs, dim=1)
    for i in range(len(outputs)):
      if outputs[i] != labels[i]:
        miss +=1
accuracy = (len(val_dataset)-miss)/len(val_dataset)*100
print(accuracy)
