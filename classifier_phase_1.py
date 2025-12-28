"""
    A StreetCLIP-based Classifier that predicts the 
    state/region from 4 cardinal view images. It uses a Transformer Encoder 
    to fuse features from North, East, South, and West views.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from transformers import CLIPVisionModel 

class Config:
    PROJECT_ROOT = "/scratch/cj2735/project4"
    TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, "train_images")
    TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "test_images")
    TRAIN_CSV = os.path.join(PROJECT_ROOT, "train_ground_truth.csv")
    TEST_CSV = os.path.join(PROJECT_ROOT, "sample_submission.csv")
    
    CHECKPOINT_DIR = "checkpoints_streetclip_classifier"
    
    # Classifier Phase 1: Classifier Checkpoints
    PHASE1_BEST = os.path.join(CHECKPOINT_DIR, "best_model_cls.pth")
    PHASE1_LAST = os.path.join(CHECKPOINT_DIR, "last_checkpoint_cls.pth")
    
    MODEL_NAME = 'geolocal/StreetCLIP'  
    IMG_SIZE = 336                      
    
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 1  
    
    NUM_WORKERS = 8 
    
    EPOCHS_P1 = 15   # Epochs for Classification
    
    LR = 1e-5 
    WD = 1e-4
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# Dataset & Data Processing
# ==========================================

# Standard Normalization stats for OpenAI/StreetCLIP models
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class QuadViewDataset(Dataset):
    """
    Handles loading 4 images (North, East, South, West) for a single location.
    Returns stacked tensor [4, 3, H, W].
    """
    def __init__(self, df, img_dir, idx_to_dense=None, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.idx_to_dense = idx_to_dense
        self.directions = ['image_north', 'image_east', 'image_south', 'image_west']
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        
        # Load all 4 cardinal views
        for d in self.directions:
            path = os.path.join(self.img_dir, row[d])
            if os.path.exists(path):
                img = Image.open(path).convert('RGB') 
            else:
                img = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))
            
            if self.transform: img = self.transform(img)
            images.append(img)
        
        img_tensor = torch.stack(images) 
        
        if self.is_test: return img_tensor, row['sample_id']
        
        label = self.idx_to_dense[row['state_idx']]
        coords = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)
        return img_tensor, label, coords

def get_transforms(is_train=True):
    """Returns augmentation pipeline for training or standard normalization for validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD)
        ])
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

def get_state_mappings(csv_path):
    df = pd.read_csv(csv_path)
    u = sorted(df['state_idx'].unique())
    return {k:i for i,k in enumerate(u)}, {i:k for i,k in enumerate(u)}

# === Loss Function ===
class HaversineLoss(nn.Module):
    """
    Calculates the Great-Circle distance between two points on a sphere.
    Optimizes the model to minimize physical distance (km) rather than Euclidean (MSE).
    """
    def __init__(self): super().__init__()
    def forward(self, pred, target):
        # Input: [Lat, Lon] in degrees -> Convert to Radians
        pred_rad = torch.deg2rad(pred)
        target_rad = torch.deg2rad(target)
        
        dlat = target_rad[:, 0] - pred_rad[:, 0]
        dlon = target_rad[:, 1] - pred_rad[:, 1]
        
        a = torch.sin(dlat/2)**2 + torch.cos(pred_rad[:, 0]) * torch.cos(target_rad[:, 0]) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        
        return (6371.0 * c).mean()

# ==========================================
# Model Architecture
# ==========================================

# --- Phase 1: The Scout ---
class GeoClassifier(nn.Module):
    """
    Multi-View Classifier using StreetCLIP backbone + Transformer Fusion.
    """
    def __init__(self, num_classes, backbone_name=Config.MODEL_NAME):
        super(GeoClassifier, self).__init__()
        print(f"Loading {backbone_name} Classifier...")
        # Backbone: Pre-trained StreetCLIP
        self.backbone = CLIPVisionModel.from_pretrained(backbone_name)
        feat_dim = self.backbone.config.hidden_size
        
        # Fusion: Transformers allow views to attend to each other
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8, batch_first=True, dropout=0.1),
            num_layers=2
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
    def extract_features(self, x):
        """Processes 4 views into a flattened feature vector."""
        B, V, C, H, W = x.shape
        x = x.view(B*V, C, H, W)
        
        # Extract features per image
        feats = self.backbone(pixel_values=x).pooler_output
        feats = feats.view(B, V, -1)
        
        attended = self.transformer(feats)
        return attended.reshape(B, -1)

    def forward(self, x):
        return self.cls_head(self.extract_features(x))

# ==========================================
# Checkpoint Helper Functions
# ==========================================

def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.exists(filename):
        print(f"--> Resuming from checkpoint: {filename}")
        ckpt = torch.load(filename, map_location=Config.DEVICE)
        
        state_dict = ckpt['model_state']
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
            
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler:
            scheduler.load_state_dict(ckpt['scheduler_state'])
            
        return ckpt['epoch'], ckpt['best_score']
    return 0, 0.0

def save_checkpoint(model, optimizer, scheduler, epoch, best_score, filename):
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state': state_dict,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'best_score': best_score
    }, filename)

# ==========================================
# Main Execution Pipeline
# ==========================================

def run():
    print(f"--- GeoMaster Split Training (Smart Resume) ---")
    
    idx2dense, dense2idx = get_state_mappings(Config.TRAIN_CSV)
    full_df = pd.read_csv(Config.TRAIN_CSV)
    
    # Stratified split to ensure all states are represented in Validation
    train_df, val_df = train_test_split(full_df, test_size=0.1, stratify=full_df['state_idx'])
    
    train_csv_path = os.path.join(Config.PROJECT_ROOT, "train_split.csv")
    val_csv_path = os.path.join(Config.PROJECT_ROOT, "val_split.csv")

    # Save to Disk (Ensures we train all the models on same training and validation sets)
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    # Augmentation Pipelines
    train_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])
    val_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)), 
        transforms.ToTensor(), 
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])
    
    train_loader = DataLoader(QuadViewDataset(train_df, Config.TRAIN_IMG_DIR, idx2dense, train_tf), 
                             batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(QuadViewDataset(val_df, Config.TRAIN_IMG_DIR, idx2dense, val_tf), 
                           batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # --- Model Initialization ---
    cls_model = GeoClassifier(len(idx2dense)).to(Config.DEVICE)
    if torch.cuda.device_count() > 1: cls_model = nn.DataParallel(cls_model)
    
    # --- Check Classifier Phase 1 ---
    if os.path.exists(Config.PHASE1_BEST):
        print(">>> Found Best Classifier. Loading weights...")
        st = torch.load(Config.PHASE1_BEST, map_location=Config.DEVICE)['model_state']
        if hasattr(cls_model, 'module'): cls_model.module.load_state_dict(st)
        else: cls_model.load_state_dict(st)
    else:
        print("\n=== PHASE 1: Training Classifier ===")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(cls_model.parameters(), lr=Config.LR, weight_decay=Config.WD)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_P1)
        
        best_acc = 0.0
        
        for epoch in range(Config.EPOCHS_P1):
            cls_model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS_P1} [Train]")
            
            for imgs, labels, _ in pbar: # We ignore 'coords' during classification phase
                imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                optimizer.zero_grad()
                
                # Forward Pass
                logits = cls_model(imgs)
                loss = criterion(logits, labels)
                
                # Backward Pass
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                current_acc = correct / total
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{current_acc:.4f}"})
            
            if scheduler: scheduler.step()
            
            cls_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels, _ in tqdm(val_loader, desc="[Val]"):
                    imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
                    logits = cls_model(imgs)
                    _, predicted = logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1} Summary | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # --- Save Checkpoints ---
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(cls_model, optimizer, scheduler, epoch, best_acc, Config.PHASE1_BEST)
                print(f">>> New Best Classifier Found! Saved to {Config.PHASE1_BEST}")
            
            # Save Last Model (For resuming if crashed)
            save_checkpoint(cls_model, optimizer, scheduler, epoch, val_acc, Config.PHASE1_LAST)

        print("Phase 1 Complete. Loading best weights for Phase 2...")
        st = torch.load(Config.PHASE1_BEST, map_location=Config.DEVICE)['model_state']
        if hasattr(cls_model, 'module'): cls_model.module.load_state_dict(st)
        else: cls_model.load_state_dict(st)

        pass

if __name__ == "__main__":
    run()