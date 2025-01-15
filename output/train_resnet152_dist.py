import argparse
import os
import json
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from time import time
from collections import defaultdict
import matplotlib.pyplot as plt

class Caltech200Dataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None, num_augmentations=1):
        self.root_dir = root_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
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
        return len(self.image_paths) * self.num_augmentations

    def __getitem__(self, idx):
        true_idx = idx // self.num_augmentations
        img_path = self.image_paths[true_idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[true_idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_one_epoch(model, loader, criterion, optimizer, rank):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(rank), labels.to(rank)
        
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
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, rank):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(rank), labels.to(rank)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history_plots.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def print_config(config):
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

def main(rank, world_size, config):
    setup(rank, world_size)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomVerticalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(50),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Caltech200Dataset(root_dir=config['root_dir'], 
                                      txt_file=config['train_txt'], 
                                      transform=train_transform,
                                      num_augmentations=config['num_augmentations'])
    test_dataset = Caltech200Dataset(root_dir=config['root_dir'], 
                                     txt_file=config['test_txt'],
                                     transform=test_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], sampler=test_sampler, num_workers=4)

    if config['pretrained']:
        model = torchvision.models.resnet152(weights="IMAGENET1K_V2")
    else:
        model = torchvision.models.resnet152()
    if config['freeze']:
        for param in model.parameters():
            param.requires_grad = False

    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 200),
    )
    if rank==0:
        print(train_transform)
        print(test_transform)
        print(model.fc)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    best_acc = 0.0
    history = defaultdict(list)

    for epoch in range(config['num_epochs']):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        start_time = time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, rank)
        val_loss, val_acc = evaluate(model, test_loader, criterion, rank)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time() - start_time

        if rank == 0:
            print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.module.state_dict(), 'best_resnet152_caltech200.pth')
                print(f"Best model saved with accuracy: {best_acc:.2f}%")

    if rank == 0:
        torch.save(model.module.state_dict(), 'final_resnet152_caltech200.pth')
        print(f"Training completed. Best accuracy: {best_acc:.2f}%")
        plot_history(history)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train resnet152 on Caltech200 dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()


    config = load_config(args.config)
    print_config(config)


    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config), nprocs=world_size, join=True)