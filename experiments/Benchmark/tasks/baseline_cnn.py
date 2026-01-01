
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np

from core.suite import BenchmarkTask
from src.causal_brain_v6 import CausalBrainV6

logger = logging.getLogger(__name__)

class ComparativeExperiment(BenchmarkTask):
    @property
    def name(self):
        return "Comparative"

    def run(self) -> dict:
        config = self.load_config("stress_config.yaml") 
        model_path = self.get_model_config_path(config)
        
        cnn = ResNetBaseline()
        
        # Target 'datasets/train' AND 'datasets/val'
        base_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets'))
        train_dir = os.path.join(base_dataset_dir, 'train')
        val_dir = os.path.join(base_dataset_dir, 'val')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training data not found at {train_dir}")
            
        logger.info(f"Training Baseline CNN (ResNet-50) using YOLO labels from {train_dir} and {val_dir}...")
        
        # Combined Dataset
        ds_train = cnn.get_dataset(train_dir)
        ds_val = cnn.get_dataset(val_dir)
        full_dataset = torch.utils.data.ConcatDataset([ds_train, ds_val])
        
        logger.info(f"Total Images: {len(full_dataset)} (Train: {len(ds_train)}, Val: {len(ds_val)})")
        
        # 1. Train on ALL data (to simulate 'seeing everything')
        cnn.train_model_on_dataset(full_dataset, epochs=5) 
        
        # 2. The Trap
        logger.info("Running Trap Scenario: Asymptomatic Carriers (Pus + No Fever)")
        
        # Filter for Positive images (Label=1) across the full dataset
        trap_indices = []
        for i in range(len(full_dataset)):
            _, label = full_dataset[i]
            if label == 1:
                trap_indices.append(i)
        
        if not trap_indices:
            logger.warning("No images with 'Pus' (Class 2) found! Cannot run Trap Scenario.")
            return {}
            
        trap_loader = DataLoader(Subset(full_dataset, trap_indices), batch_size=1, shuffle=False)
        
        results = []
        nb_model = CausalBrainV6(model_path)
        
        # Carrier Context
        evidence_carrier = {
            'V_white': 'High', # Visual strong signal
            'C_temp': 'Mild',  # Trap: Low fever
            'C_pain': 'Mild', 
            'Age_Group': 'Child'
        }
        
        for i, (inputs, labels) in enumerate(trap_loader):
            # CNN Prediction
            cnn_prob = cnn.predict(inputs)
            
            # NB Prediction
            nb_res = nb_model.diagnose(evidence_carrier)
            nb_prob = nb_res['probs'].get('GAS', 0.0)
            nb_conflict = nb_res.get('cognitive', {}).get('conflict_score', 0.0)
            
            results.append({
                'id': i,
                'cnn_prob': cnn_prob,
                'nb_prob': nb_prob,
                'nb_conflict': nb_conflict
            })
            
        df = pd.DataFrame(results)
        cnn_fpr = (df['cnn_prob'] > 0.5).mean() if not df.empty else 0
        nb_fpr = (df['nb_prob'] > 0.5).mean() if not df.empty else 0
        
        logger.info(f"Trap Results (N={len(df)}) - CNN FPR: {cnn_fpr:.2f}, NB FPR: {nb_fpr:.2f}")
        
        self._plot_comparison(cnn_fpr, nb_fpr)
        df.to_csv(os.path.join(self.output_dir, "comparative_results.csv"), index=False)
        
        return {'cnn_fpr': cnn_fpr, 'nb_fpr': nb_fpr}

    def _plot_comparison(self, cnn_fpr, nb_fpr):
        plt.figure(figsize=(8, 6))
        models = ['Baseline CNN\n(ResNet-50)', 'Neuro-Symbolic\n(CausalBrain)']
        fprs = [cnn_fpr * 100, nb_fpr * 100]
        
        bars = plt.bar(models, fprs, color=['#ff6666', '#66cc99'], edgecolor='black')
        plt.ylabel('False Positive Rate (%) on Carriers')
        plt.title('Vulnerability to "Visual Traps"')
        plt.ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
                     
        self.save_plot(plt.gcf(), "comparative_fpr_plot.png")
        plt.close()


class YOLOClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        # Collect all images
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.samples = [] # (path, label)
        
        for img_file in self.image_files:
            img_path = os.path.join(self.image_dir, img_file)
            
            # Find Label
            # Label file is same name but .txt
            base_name = os.path.splitext(img_file)[0]
            txt_file = base_name + ".txt"
            txt_path = os.path.join(self.label_dir, txt_file)
            
            label = 0 # Default Non-GAS
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        if cls_id == 2: # Class 2 = PUS (White Coating) = GAS
                            label = 1
                            break
            
            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load Image
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return dummy
            image = torch.zeros((3, 224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ResNetBaseline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
             self.model = models.resnet50(pretrained=True)
             
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def get_dataset(self, data_dir):
        # Use Custom YOLO Dataset logic
        return YOLOClassificationDataset(data_dir, transform=self.transform)

    def train_model(self, data_dir, epochs=3):
        dataset = self.get_dataset(data_dir)
        self.train_model_on_dataset(dataset, epochs)

    def train_model_on_dataset(self, dataset, epochs=3):
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            acc = 100 * correct / (total if total > 0 else 1)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Acc: {acc:.2f}% (N={total})")
            
    def predict(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs[0][1].item()

