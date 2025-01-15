import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from time import time
from collections import defaultdict
from torch import nn
import pandas as pd
import argparse

class Caltech200Dataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        with open(txt_file, 'r') as f:
            for line in f:
                relative_path = line.strip()
                full_path = os.path.join(root_dir, relative_path)
                self.image_paths.append(full_path)
                label = int(relative_path.split('/')[0].split('.')[0]) - 1
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
    
    return running_loss / total, 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
    
    return running_loss / total, 100 * correct / total

def unfreeze_layer(model, layer_name):
    for name, param in model.named_parameters():
        if name.startswith(layer_name):
            param.requires_grad = True

def update_optimizer(model, optimizer, lr_pretrained=1e-5, lr_fc=1e-3):
    return torch.optim.AdamW([
        {'params': model.fc.parameters(), 'lr': lr_fc},
        {'params': (p for n, p in model.named_parameters() if not n.startswith('fc')), 'lr': lr_pretrained}
    ])

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)], p=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Caltech200Dataset(root_dir=args.root_dir, 
                                      txt_file=args.train_txt, 
                                      transform=transform)
    test_dataset = Caltech200Dataset(root_dir=args.root_dir, 
                                     txt_file=args.test_txt, 
                                     transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.pretrained:
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    else:
        model = torchvision.models.resnet50()

    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False

    n_inputs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(n_inputs, 2048),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(2048, 2048),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(2048, 200)
    # )
    model.fc = nn.Linear(n_inputs, 200)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    unfreeze_schedule = {
        100: 'layer4',
        150: 'layer3',
        200: 'layer2'
    }

    best_acc = 0.0
    history = defaultdict(list)

    for epoch in range(args.num_epochs):
        start_time = time()

        if args.scheduled_unfreeze and epoch in unfreeze_schedule:
            layer_to_unfreeze = unfreeze_schedule[epoch]
            print(f"Unfreezing {layer_to_unfreeze}")
            unfreeze_layer(model, layer_to_unfreeze)
            optimizer = update_optimizer(model, optimizer)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time() - start_time
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet50_caltech200.pth')
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

    torch.save(model.state_dict(), 'final_resnet50_caltech200.pth')
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    pd.DataFrame(history).to_csv('history.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet50 on Caltech200 dataset')
    parser.add_argument('--root_dir', type=str, default= '/home/feem/Workspace/caltech_birds/images/', help='Root directory of the dataset')
    parser.add_argument('--train_txt', type=str, default='/home/feem/Workspace/caltech_birds/lists/train.txt', help='Path to train.txt file')
    parser.add_argument('--test_txt', type=str, default= '/home/feem/Workspace/caltech_birds/lists/test.txt', help='Path to test.txt file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--num_epochs', type=int, default=16, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Whether to pretrain the initial weights')
    parser.add_argument('--freeze', action='store_true', help='Whether to freeze the pretrained layers initially')
    parser.add_argument('--scheduled_unfreeze', action='store_true', help='Whether to schedule the freezing of the pretrained layers initially')
    args = parser.parse_args()

    main(args)