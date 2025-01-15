import argparse
import os
import json
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import random

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

        return image, label, img_path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_model():
    model = torchvision.models.resnet152()
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 200),
    )
    return model

def load_class_names(file_path):
    class_names = {}
    with open(file_path, 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split('.')
            class_names[int(class_id) - 1] = class_name
    return class_names

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_2x2_confusion_matrix(tp, fn, fp, tn):

    cm = np.array([
        [tp, fn],
        [fp, tn]
    ])

    annot = np.array([
        ['TP\n'+str(tp), 'FN\n'+str(fn)],
        ['FP\n'+str(fp), 'TN\n'+str(tn)] 
    ])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.title('2x2 Confusion Matrix')
    plt.tight_layout()
    plt.savefig('2x2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def show_misclassified_examples(model, test_loader, device, class_names, num_examples=5):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append((images[i], labels[i].item(), predicted[i].item(), paths[i]))
                    
                if len(misclassified) >= num_examples:
                    break
            
            if len(misclassified) >= num_examples:
                break
    
    fig, axes = plt.subplots(1, num_examples, figsize=(20, 4))
    for i, (img, true_label, pred_label, path) in enumerate(misclassified[:num_examples]):
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        img = img.astype(np.uint8)
        
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Misclassified examples:")
    for _, true_label, pred_label, _ in misclassified[:num_examples]:
        print(f"True label: {class_names[true_label]}")
        print(f"Predicted label: {class_names[pred_label]}")

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = Caltech200Dataset(root_dir=config['root_dir'], 
                                     txt_file=config['test_txt'],
                                     transform=test_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = create_model()
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model = model.to(device)
    model.eval()

    class_names = load_class_names(config['class_names'])

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, [class_names[i] for i in range(200)])

    tp = np.sum(np.diag(cm))
    fn = np.sum(cm) - tp
    fp = fn
    tn = np.sum(cm) * (cm.shape[0] - 1) - fn

    plot_2x2_confusion_matrix(tp, fn, fp, tn)

    print(f"True Positives: {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")

    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    show_misclassified_examples(model, test_loader, device, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate resnet152 on Caltech200 dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)