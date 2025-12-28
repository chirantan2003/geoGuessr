"""
    Model Branch 2 (ConvNeXt)
    This script trains a ConvNeXt V2 (Large) classifier for the state-level geolocation task.
    It acts as a strong complementary model to the StreetCLIP model (Phase 1).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torchvision.transforms as transforms
import timm  

class Config:
    PROJECT_ROOT = "/scratch/cj2735/project4"
    TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, "train_images")
    TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "test_images")
    TRAIN_CSV = os.path.join(PROJECT_ROOT, "train_ground_truth.csv")
    TEST_CSV = os.path.join(PROJECT_ROOT, "sample_submission.csv")

    TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "train_split.csv")
    VAL_CSV_PATH = os.path.join(PROJECT_ROOT, "val_split.csv")

    # Checkpoint settings
    CKPT_DIR = "checkpoints_convnext"
    BEST_MODEL = os.path.join(CKPT_DIR, "best_convnext_cls.pth")

    MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    IMG_SIZE = 384  

    BATCH_SIZE = 16
    NUM_WORKERS = 8
    EPOCHS = 10

    LR_BACKBONE = 5e-6
    LR_HEAD = 5e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.CKPT_DIR, exist_ok=True)

# ==========================================
# Dataset & State Mapping
# ==========================================

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_state_mappings(csv_path):
    """Generates a mapping from Class Name -> Integer Index and vice versa."""
    df = pd.read_csv(csv_path)
    u = sorted(df['state_idx'].unique())
    return {k:i for i,k in enumerate(u)}, {i:k for i,k in enumerate(u)}

idx2dense, dense2idx = get_state_mappings(Config.TRAIN_CSV)

class QuadViewDataset(Dataset):
    """
    Loads 4 images corresponding to the cardinal directions for a single location.
    Returns them stacked as a single tensor [4, 3, H, W].
    """
    def __init__(self, df, img_dir, idx_to_dense=None, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.idx_to_dense = idx_to_dense
        self.is_test = is_test
        self.views = ['image_north','image_east','image_south','image_west']

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs = []
        
        # Iterate over North, East, South, West columns
        for v in self.views:
            p = os.path.join(self.img_dir, row[v])
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
            else:
                img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))
            
            if self.transform: img = self.transform(img)
            imgs.append(img)

        # Stack into shape [Views, Channels, Height, Width]
        imgs = torch.stack(imgs)
        
        if self.is_test: 
            return imgs, row['sample_id']

        label = self.idx_to_dense[row["state_idx"]]
        return imgs, label

def get_tf(train=True):
    """
    Returns augmentation pipeline.
    Train: Random flips to prevent overfitting.
    Validation: Resize + Normalize only.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

# ==========================================
# ConvNeXt Model Architecture
# ==========================================

class ConvNeXtGeoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load Pre-trained Backbone (feature extractor)
        # num_classes=0 removes the original ImageNet head
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        d = self.backbone.num_features 

        # Custom Head for our specific classes
        self.head = nn.Sequential(
            nn.Linear(d * 4, 2048), # Fusing 4 views
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        B, V, C, H, W = x.shape
        
        x = x.view(B*V, C, H, W)
        
        # Extract features: [B*V, Feature_Dim]
        feats = self.backbone(x)
        
        # Unflatten to separate the views: [B, V, Feature_Dim]
        # Then flatten V into Feature_Dim: [B, V * Feature_Dim]
        # This effectively concatenates North+East+South+West features
        feats = feats.view(B, -1)
        
        # Pass concatenated features to classification head
        return self.head(feats)

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    # Setup Data
    tr = pd.read_csv(Config.TRAIN_CSV_PATH)
    va = pd.read_csv(Config.VAL_CSV_PATH)

    train_loader = DataLoader(
        QuadViewDataset(tr, Config.TRAIN_IMG_DIR, idx2dense, get_tf(True)),
        batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        QuadViewDataset(va, Config.TRAIN_IMG_DIR, idx2dense, get_tf(False)),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    num_classes = len(idx2dense)
    model = ConvNeXtGeoClassifier(num_classes).to(Config.DEVICE)

    if os.path.exists(Config.BEST_MODEL):
        print(f"Found checkpoint at {Config.BEST_MODEL}. Resuming training...")
        checkpoint = torch.load(Config.BEST_MODEL, map_location=Config.DEVICE)
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found. Starting from ImageNet weights.")
    
    # Separate parameters for differential learning rates
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        (head_params if "head" in n else backbone_params).append(p)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": Config.LR_BACKBONE}, # Slower learning for backbone
        {"params": head_params, "lr": Config.LR_HEAD}          # Faster learning for head
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Label smoothing helps prevent the model from becoming overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    scaler = GradScaler()

    best_acc = 0

    print(f"Starting Training for {Config.EPOCHS} Epochs...")
    
    for ep in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1} [Train]")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(Config.DEVICE), lbls.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, lbls)
            
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss_sum / train_batches

        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0
        val_batches = 0
        
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {ep+1} [Val]"):
                imgs, lbls = imgs.to(Config.DEVICE), lbls.to(Config.DEVICE)
                
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                
                val_loss_sum += loss.item()
                val_batches += 1
                
                # Accuracy calculation
                preds = logits.argmax(1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        acc = correct / total
        avg_val_loss = val_loss_sum / val_batches
        
        print(f"\nEpoch {ep+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {acc:.4f}")

        # Save Checkpoint if accuracy improves
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Config.BEST_MODEL)
            print(">>> Saved Best ConvNeXt Model")

        scheduler.step()

    print(f"\nTraining done. Best Val Acc = {best_acc:.4f}")

if __name__ == "__main__":
    train()