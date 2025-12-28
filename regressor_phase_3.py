"""
    This script trains the "Sniper" (GPS Regressor) component of the pipeline.
    
    Methodology:
    1. Load the Pre-trained Classifier (Phase 1) and FREEZE it.
    2. Use the frozen classifier to extract rich visual features from 4 views.
    3. Train a Residual Regressor that predicts the *offset* (difference) 
       between the true GPS coordinates and the center (centroid) of the predicted state.
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
import timm

class Config:
    PROJECT_ROOT = "/scratch/cj2735/project4" 
    TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, "train_images")
    TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "test_images")
    TRAIN_CSV = os.path.join(PROJECT_ROOT, "train_ground_truth.csv")
    TEST_CSV = os.path.join(PROJECT_ROOT, "sample_submission.csv")
    MAPPING_CSV = os.path.join(PROJECT_ROOT, "state_mapping.csv")
    
    PHASE1_MODEL_PATH = "checkpoints_clip/best_model_clip.pth" 
    PHASE2_MODEL_PATH = "best_residual_regressor_clip.pth"

    TRAIN_CSV_PATH = os.path.join(PROJECT_ROOT, "train_split.csv")
    VAL_CSV_PATH = os.path.join(PROJECT_ROOT, "val_split.csv")
    
    MODEL_NAME = 'geolocal/StreetCLIP' 
    IMG_SIZE = 336
    EMBED_DIM = 1024
    NUM_HEADS = 8
    NUM_LAYERS = 2
    
    BATCH_SIZE = 16  
    NUM_WORKERS = 8
    EPOCHS = 7
    LR = 3e-4        
    WD = 1e-4
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EARTH_RADIUS_KM = 6371.0 # Used for Haversine Loss calculation

# ==========================================
# Dataset & Transforms
# ==========================================

# Normalization constants specific to OpenAI CLIP models
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class QuadViewDataset(Dataset):
    def __init__(self, df, img_dir, idx_to_dense=None, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.idx_to_dense = idx_to_dense
        self.directions = ['image_north', 'image_east', 'image_south', 'image_west']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        
        for d in self.directions:
            fname = row[d]
            path = os.path.join(self.img_dir, fname)
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
            else:
                img = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        img_tensor = torch.stack(images)
        
        if self.is_test:
            return img_tensor, row['sample_id']
        
        original_idx = row['state_idx']
        dense_label = self.idx_to_dense[original_idx] if self.idx_to_dense else 0
        coords = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)
        
        return img_tensor, dense_label, coords

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()
        
    def forward(self, pred, target):
        pred_rad = torch.deg2rad(pred)
        target_rad = torch.deg2rad(target)
        
        pred_lat, pred_lon = pred_rad[:, 0], pred_rad[:, 1]
        target_lat, target_lon = target_rad[:, 0], target_rad[:, 1]
        
        dlat = target_lat - pred_lat
        dlon = target_lon - pred_lon
        
        a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon/2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        
        return c.mean() # Returns avg distance in Radians (converted to km later)

# ==========================================
# Centroids
# ==========================================
def calculate_state_centroids(df, idx_to_dense):
    """
    Computes the mean Latitude/Longitude for every state class.
    These act as anchors. The regressor predicts the offset from these anchors.
    """
    num_classes = len(idx_to_dense)
    centroids = torch.zeros(num_classes, 2)
    dense_to_original = {v: k for k, v in idx_to_dense.items()}
    
    for dense_idx in range(num_classes):
        original_idx = dense_to_original[dense_idx]
        state_data = df[df['state_idx'] == original_idx]
        if len(state_data) > 0:
            centroids[dense_idx, 0] = state_data['latitude'].mean()
            centroids[dense_idx, 1] = state_data['longitude'].mean()
            
    return centroids.to(Config.DEVICE)

# ==========================================
# Model Architectures
# ==========================================

# --- PHASE 1 MODEL: StreetCLIP ---
class GeoTransformer(nn.Module):
    def __init__(self, num_classes, backbone_name=Config.MODEL_NAME, embed_dim=Config.EMBED_DIM):
        super(GeoTransformer, self).__init__()
        
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        
        if hasattr(self.backbone, 'num_features'):
            self.num_features = self.backbone.num_features
        else:
            self.num_features = embed_dim
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_features, 
            nhead=Config.NUM_HEADS, 
            batch_first=True, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)
        
        self.fusion_dim = self.num_features * 4
        
        self.cls_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
        self.gps_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2)
        )
        
    def extract_features(self, x):
        """
        Extracts fused features from 4 views. 
        Input: [B, 4, 3, H, W] -> Output: [B, 4096] (if dim=1024)
        """
        B, Views, C, H, W = x.shape
        x = x.view(B * Views, C, H, W)
        
        # 1. Backbone Extraction
        features = self.backbone(x) # [B*4, num_features]
        features = features.view(B, Views, -1)
        
        # 2. Transformer Fusion (Cross-view attention)
        attended_feats = self.transformer(features)
        
        # 3. Flatten
        flat_feats = attended_feats.reshape(B, -1)
        return flat_feats

    def forward(self, x):
        features_flat = self.extract_features(x)
        return self.cls_head(features_flat), self.gps_head(features_flat)

# --- PHASE 2: RESIDUAL REGRESSOR  ---
class ResidualGPSRegressor(nn.Module):
    """
    The 'Sniper'. Takes visual features + predicted state context and outputs
    a refined GPS coordinate.
    """
    def __init__(self, classifier_model, state_centroids, num_states=33, state_emb_dim=64):
        super().__init__()
        # Store the frozen classifier to use as a feature extractor
        self.feature_extractor = classifier_model
        self.state_centroids = state_centroids 
        self.state_embedding = nn.Embedding(num_states, state_emb_dim)
        
        img_feat_dim = self.feature_extractor.num_features * 4
        context_dim = state_emb_dim + 1 
        
        self.regressor = nn.Sequential(
            nn.Linear(img_feat_dim + context_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2) # Output: Latitude Offset, Longitude Offset
        )
        
        # Initialize output layer to near-zero 
        # (So training starts by predicting the centroid exactly)
        nn.init.normal_(self.regressor[-1].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.regressor[-1].bias, 0.0)
        
    def forward(self, img_input, top1_index, top1_prob):
        # 1. Extract Image Features (Gradients disabled for backbone)
        with torch.no_grad(): 
            flat_img_feats = self.feature_extractor.extract_features(img_input)
        
        # 2. Get Context Embeddings
        s1_emb = self.state_embedding(top1_index) 
        p1 = top1_prob.unsqueeze(1) # [B, 1]
        
        # 3. Concatenate (Late Fusion)
        combined_input = torch.cat([flat_img_feats, s1_emb * p1, p1], dim=1)
        
        # 4. Predict Offset
        predicted_offset = self.regressor(combined_input)
        
        # 5. Residual Addition: Prediction = Centroid + Offset
        anchor_coords = self.state_centroids[top1_index] 
        return anchor_coords + predicted_offset

# ==========================================
# Training Engine
# ==========================================
def train_regressor(cls_model, reg_model, loader, optimizer, scaler, criterion):
    reg_model.train()
    cls_model.eval() 
    
    total_loss_rad = 0
    pbar = tqdm(loader, desc="Training Residual Regressor")
    
    for images, _, gps_labels in pbar:
        images = images.to(Config.DEVICE)
        gps_labels = gps_labels.to(Config.DEVICE)
        
        # --- Step 1: Scout Inference ---
        with torch.no_grad():
            with autocast():
                cls_logits, _ = cls_model(images)
                probs = torch.softmax(cls_logits, dim=1)
                
                # Get Top-1
                top1_prob, top1_index = torch.topk(probs, 1, dim=1)
                top1_prob = top1_prob.squeeze(1)
                top1_index = top1_index.squeeze(1)
        
        # --- Step 2: Sniper Training ---
        optimizer.zero_grad()
        with autocast():
            # Pass image + classifier info to regressor
            gps_preds = reg_model(images, top1_index, top1_prob)
            loss = criterion(gps_preds, gps_labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss_rad += loss.item()
        
        loss_km = loss.item() * Config.EARTH_RADIUS_KM
        pbar.set_postfix({'Loss(km)': loss_km})
        
    return total_loss_rad / len(loader)

def validate_regressor(cls_model, reg_model, loader, criterion):
    reg_model.eval()
    cls_model.eval()
    total_dist_km = 0
    total = 0
    
    with torch.no_grad():
        for images, _, gps_labels in tqdm(loader, desc="Validating"):
            images = images.to(Config.DEVICE)
            gps_labels = gps_labels.to(Config.DEVICE)
            
            # 1. Get Context
            cls_logits, _ = cls_model(images)
            probs = torch.softmax(cls_logits, dim=1)
            top1_prob, top1_index = torch.topk(probs, 1, dim=1)
            top1_prob = top1_prob.squeeze(1)
            top1_index = top1_index.squeeze(1)
            
            # 2. Get Prediction
            gps_preds = reg_model(images, top1_index, top1_prob)
            
            # 3. Calculate Error
            dist_rad = criterion(gps_preds, gps_labels)
            dist_km = dist_rad.item() * Config.EARTH_RADIUS_KM
            
            total_dist_km += dist_km * images.size(0)
            total += images.size(0)
            
    return total_dist_km / total

# ==========================================
# Main Execution Pipeline
# ==========================================
def run_phase_2():
    print("--- Starting Phase 2: Residual Regressor (CLIP Backbone) ---")
    
    if os.path.exists(Config.MAPPING_CSV):
        map_df = pd.read_csv(Config.MAPPING_CSV)
        unique_indices = sorted(map_df['state_idx'].unique())
    else:
        full_df = pd.read_csv(Config.TRAIN_CSV)
        unique_indices = sorted(full_df['state_idx'].unique())
        
    idx_to_dense = {original: dense for dense, original in enumerate(unique_indices)}
    full_df = pd.read_csv(Config.TRAIN_CSV)
    
    print("Calculating State Centroids...")
    centroids = calculate_state_centroids(full_df, idx_to_dense)
    
    train_ds = pd.read_csv(Config.TRAIN_CSV_PATH)
    val_ds = pd.read_csv(Config.VAL_CSV_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Load Phase 1 Model
    print(f"Loading Phase 1 Classifier (StreetCLIP) from {Config.PHASE1_MODEL_PATH}...")
    cls_model = GeoTransformer(num_classes=len(unique_indices)).to(Config.DEVICE)
    
    if os.path.exists(Config.PHASE1_MODEL_PATH):
        checkpoint = torch.load(Config.PHASE1_MODEL_PATH, map_location=Config.DEVICE)
        
        # checkpoint loading logic 
        if isinstance(checkpoint, dict):
            if 'ema_state' in checkpoint:
                print("Loading EMA Weights (Best quality)...")
                state_dict = checkpoint['ema_state']
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        clean_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v
                
        cls_model.load_state_dict(clean_dict)
    else:
        print(f"ERROR: Phase 1 model not found at {Config.PHASE1_MODEL_PATH}!")
        return
        
    cls_model.eval() # Freeze Classifier
    
    # Initialize Phase 2 Model (The Sniper)
    print("Initializing Residual Regressor...")
    reg_model = ResidualGPSRegressor(cls_model, centroids, num_states=len(unique_indices)).to(Config.DEVICE)
    
    # Optimize ONLY the Regressor parameters
    optimizer = optim.AdamW(reg_model.parameters(), lr=Config.LR, weight_decay=Config.WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = GradScaler()
    criterion = HaversineLoss()
    
    best_dist = 99999.0
    
    # Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss_rad = train_regressor(cls_model, reg_model, train_loader, optimizer, scaler, criterion)
        val_dist_km = validate_regressor(cls_model, reg_model, val_loader, criterion)
        
        scheduler.step()
        print(f"Epoch {epoch+1} | Train Loss (Rad): {train_loss_rad:.5f} | Val Dist: {val_dist_km:.2f} km")
        
        if val_dist_km < best_dist:
            best_dist = val_dist_km
            torch.save(reg_model.state_dict(), Config.PHASE2_MODEL_PATH)
            print(">>> Saved Best Residual Regressor")
            
    print(f"Phase 2 Complete. Best Distance: {best_dist:.2f} km")

if __name__ == '__main__':
    run_phase_2()