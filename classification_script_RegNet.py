import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
from datetime import datetime
from itertools import groupby
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from PIL import Image

#Pytorch
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.utils.data
import torch.utils.data.dataloader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split



#FOLDERPATH_PL = '/Users/davidpichler/GitHubRepo/Thesis/data/segmented/graph/pl'
FOLDERPATH_SPEC = 'data/spec_total' 
TS = datetime.now().strftime('%Y%m%d_%H%M%S')

# AI was used to generate a starting template for this function which was then modified and expanded
def validate(model, device, loader):
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    return correct / total

# AI was used to generate a starting template for this function which was then modified and expanded
def train(model, device, folderpath):
    model = model.to(device)

    # Transofrm the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    # Load the dataset
    dataset = datasets.ImageFolder(folderpath, transform=transform)

    random_seed = 2024
    torch.manual_seed(random_seed)

    # Define the sizes of the train, validate and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train, validate and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print('Train Dataset: ' + str(len(train_dataset)))
    print('Validation Dataset: ' + str(len(val_dataset)))
    print('Test Dataset: ' + str(len(test_dataset)))

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=18)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=18)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=18)

    # Define Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define epochs
    num_epochs = 15

    acc = []
    loss_values = []

    # create csv file
    filename_csv = f'results/RegNet/train_{TS}.csv'
    with open(filename_csv, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss', 'Validation Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_accuracy = validate(model, device, val_loader)
            
            loss_values.append(loss.item())
            acc.append(train_accuracy)

            print(f'Epoch: {epoch + 1} | Loss: {loss.item()} | Validation Accuracy:  {train_accuracy}')

            # Schreiben der Ergebnisse in die CSV-Datei
            writer.writerow({
                'Epoch': epoch + 1,
                'Loss': loss.item(),
                'Validation Accuracy': train_accuracy
            })

    test(model, device, test_loader)

# AI was used to generate a starting template for this function which was then modified and expanded
def test(model, device, data_loader):
    # Test the model
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []

        for X_test_tensor, Y_test_tensor in data_loader:
            X_test_tensor, Y_test_tensor = X_test_tensor.to(device), Y_test_tensor.to(device)
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)

            y_test = Y_test_tensor.cpu().numpy()
            predicted = predicted.cpu().numpy()

            correct += (predicted == y_test).sum().item()
            total += len(y_test)

            true_labels.extend(y_test)
            predicted_labels.extend(predicted)

        accuracy = correct / total
        print(f'Accuracy: {accuracy}')

        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print(f'F1 Score: {f1}')

        precision = precision_score(true_labels, predicted_labels, average='weighted')
        print(f'Precision: {precision}')

        recall = recall_score(true_labels, predicted_labels, average='weighted')
        print(f'Recall: {recall}')

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_df = pd.DataFrame(cm, index=['Actual Corrected', 'Actual Original', 'Actual Smoothed'], columns=['Predicted Corrected', 'Predicted Original', 'Predicted Smoothed'])

    # Display the cm
    print(cm_df)

    # Save the results in a txt file
    filename_txt = f'results/RegNet/test_data_{TS}.txt'
    with open(filename_txt, 'a') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'Confusion Matrix: {cm_df}\n')

# AI was used to generate a starting template for this function which was then modified and expanded
def main():
    # Check Cuda
    print(f"PyTorch version: {torch.__version__}")

    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")


    # Check results folder
    if not os.path.exists(f"{os.getcwd()}/results/RegNet/"): os.makedirs(f"{os.getcwd()}/results/RegNet/")

    # CNN Architectures
    models_dict = {
        "resnet": models.resnet50(),
        "regnet": models.regnet_y_3_2gf(),
        "vgg": models.vgg16()
    }

    # Train
    train(models_dict['regnet'], device, FOLDERPATH_SPEC)



if __name__=='__main__':
    main()
    