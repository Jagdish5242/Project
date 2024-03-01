
import tensorflow_datasets as tfds
import cv2

import numpy as np
import tensorflow as tf




import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms


ds = tfds.data_source('curated_breast_imaging_ddsm')

def iter_tf_data(train_ds):
    x_list = []
    y_list = []
    
    result = train_ds.__getitems__(range(0, len(train_ds), 1))
    
    for data in result:
        x = torch.tensor(data['image'])
        y = torch.tensor(list([data['label']]))
        x_list.append(x) 
        y_list.append(y)
    
    x_list_cat = torch.cat(x_list, axis=-1).T
    y_list_cat = torch.cat(y_list, axis=0)
        
    return [x_list_cat, y_list_cat]

batch_size = 5
train_sampler = torch.utils.data.RandomSampler(ds['train'], num_samples=1000)
train_list = iter_tf_data(ds['train'])
train_dataset = torch.utils.data.TensorDataset(*train_list)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=batch_size
)

test_list = iter_tf_data(ds['test'])
test_dataset = torch.utils.data.TensorDataset(*test_list)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    sampler=None,
    batch_size=batch_size
)

validation_list = iter_tf_data(ds['validation'])
validation_dataset = torch.utils.data.TensorDataset(*validation_list)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    sampler=None,
    batch_size=batch_size
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

def train_model(model, loss_function, optimizer, data_loader):
    
    model.train()
    
    current_loss = 0.0
    current_acc = 0
    
    for i, (inputs, labels) in enumerate(data_loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            outputs = model(inputs.float().unsqueeze(0))
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs.squeeze(0), labels.float())
            
            loss.backward()
            optimizer.step()
            
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
        
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    
    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test_model(model, loss_function, data_loader):
    model.eval()
    
    current_loss = 0.0
    current_acc = 0
    
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs.float().unsqueeze(0))
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs.squeeze(0), labels.float())
            
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
        
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    
    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))

def tl_feature_extractor(epochs=3):
    
    model = torchvision.models.resnet18()
    
    num_input_channels = 5
    new_conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, padding='same')
    model.conv1 = new_conv1
    
    for param in model.parameters():
        param.requires_grad = True
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 5)
    
    model = model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.fc.parameters())
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        
        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, validation_loader)
        test_model(model, loss_function, test_loader)

def tl_fine_tuning(epochs=3):
    
    model = models.resnet18()
    
    num_input_channels = 5
    new_conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, padding='same')
    model.conv1 = new_conv1
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 5) 
    
    model = model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        
        train_model(model, loss_function, optimizer, train_loader)
        test_model(model, loss_function, validation_loader)
        test_model(model, loss_function, test_loader)
    
    
tl_fine_tuning(epochs=10)

tl_feature_extractor(epochs=10)


