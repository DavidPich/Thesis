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
import time
from datetime import timedelta





#DATA_FOLDERPATH = 'data/spec_ten_percent'
#RESULTS_FOLDERPATH = 'results/spec_ten_percent/VGG/'
#MODEL_FOLDERPATH = 'models/spec_ten_percent/VGG/' 

DATA_FOLDERPATH = 'data/dummy'
RESULTS_FOLDERPATH = 'results/dummy/VGG/'
MODEL_FOLDERPATH = 'models/dummy/VGG/'



TS = datetime.now().strftime('%Y%m%d_%H%M%S')
START_TIME = time.monotonic()

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


def train(model, device, folderpath):
#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPU
#        model = nn.DataParallel(model)
#    
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
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Define epochs
    num_epochs = 15

    acc = []
    loss_values = []

    # create csv file
    filename_csv = f'{RESULTS_FOLDERPATH}train_{TS}.csv'
    with open(filename_csv, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss', 'Validation Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Train the model
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_start_time = time.monotonic()
            model.train()
            val_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                val_loss += loss.item()
            
            train_accuracy = validate(model, device, val_loader)
            
            loss_values.append(loss.item())
            acc.append(train_accuracy)

            epoch_end_time = time.monotonic() - epoch_start_time
            print(f'Epoch: {epoch + 1} | Loss: {loss.item()} | Validation Accuracy:  {train_accuracy} | Time Spent: {epoch_end_time}')
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #epochs_no_improve = 0
                torch.save(model.state_dict(), f'{MODEL_FOLDERPATH}best_model.pth')

            # Schreiben der Ergebnisse in die CSV-Datei
            writer.writerow({
                'Epoch': epoch + 1,
                'Loss': loss.item(),
                'Validation Accuracy': train_accuracy
            })

    # Load the best model
    model.load_state_dict(torch.load(f'{MODEL_FOLDERPATH}best_model.pth'))
    test(model, device, test_loader)


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
    filename_txt = f'{RESULTS_FOLDERPATH}test_data_{TS}.txt'
    end_time = time.monotonic()
    with open(filename_txt, 'a') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'Confusion Matrix: {cm_df}\n')
        f.write(f'Time Spent: {timedelta(seconds=end_time - START_TIME)}\n')


def main():
    # Check Cuda
    print(f"PyTorch version: {torch.__version__}")

    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    
    # cuda:1 in order to use second GPU; By default cuda uses cuda:0
    device = "cuda:1" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")


    # Check results and models folder
    if not os.path.exists(f"{os.getcwd()}/{RESULTS_FOLDERPATH}"): os.makedirs(f"{os.getcwd()}/{RESULTS_FOLDERPATH}")
    if not os.path.exists(f"{os.getcwd()}/{MODEL_FOLDERPATH}"): os.makedirs(f"{os.getcwd()}/{MODEL_FOLDERPATH}")

    # CNN Architectures
    models_dict = {
        "resnet": models.resnet50(),
        "regnet": models.regnet_y_3_2gf(),
        "vgg": models.vgg16()
    }

    # Train
    train(models_dict['vgg'], device, DATA_FOLDERPATH)



if __name__=='__main__':
    main()
    