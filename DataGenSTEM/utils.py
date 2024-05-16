import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class PNGDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Assuming file names are in corresponding order
        self.data_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.label_dirs = sorted([d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.images_dir, self.data_files[idx])
        image = Image.open(img_name)
        image = np.array(image).astype('float32') / 255.0

        # Load labels (6 PNGs per label directory)
        label_images = []
        label_dir_path = os.path.join(self.labels_dir, self.label_dirs[idx])
        for label_file in sorted(os.listdir(label_dir_path)):
            label_path = os.path.join(label_dir_path, label_file)
            label_image = Image.open(label_path)
            label_images.append(label_image)
        labels = np.stack([np.array(label).astype('float32') / 255.0 for label in label_images])

        # Apply transformations - don't have any right now
        if self.transform:
            image = self.transform(image)
            label_images = [self.transform(label) for label in label_images]

        image = torch.Tensor(image)
        image = torch.unsqueeze(image, 0)
        labels = torch.Tensor(labels)
        return image, labels


def get_dataloaders(images_dir, labels_dir, batch_size, val_split=0.1, test_split=0.1):
    dataset = PNGDataset(images_dir, labels_dir)

    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - test_size - val_size

    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")


    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, n_epochs, criterion, optimizer, device, save_name, save_loss_history = True):
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')

    model.to(device)
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

            val_epoch_loss = val_running_loss / len(val_loader)
            val_loss_history.append(val_epoch_loss)

        print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

        if save_loss_history:
            np.savez(save_name + 'loss_history.npz', train_loss_history=train_loss_history, val_loss_history=val_loss_history)
        # Save model and optimizer state if validation loss improved
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'best_val_loss': best_val_loss}
            torch.save(checkpoint, save_name)
            print(f"Model saved as validation loss improved to {val_epoch_loss:.4f}")

    return model, train_loss_history, val_loss_history


