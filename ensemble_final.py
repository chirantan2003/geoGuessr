"""
    This script performs the final model ensemble to generate the submission file.
    
    Ensemble Strategy: "Logit Fusion"
    Instead of averaging probabilities, we average the raw logits 
    from StreetCLIP and ConvNeXt before applying Softmax.
    This preserves the relative confidence magnitude of each model better 
    than averaging probabilities, which can be skewed by calibration issues.
      
    Components:
    1. StreetCLIP: Excellent at semantic scene understanding (text, landmarks).
    2. ConvNeXt V2: Excellent at texture and structural details.
    3. Residual Regressor: Refines the state centroid to exact GPS coordinates.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from transformers import CLIPVisionModel

class Config:
    PROJECT_ROOT = "/scratch/cj2735/project4"
    TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "test_images")
    TEST_CSV = os.path.join(PROJECT_ROOT, "sample_submission.csv")
    TRAIN_CSV = os.path.join(PROJECT_ROOT, "train_ground_truth.csv")
    
    # The Scout (Phase 1 Classifier)
    CKPT_CLIP = "checkpoints_streetclip_classifier/best_finetuned_accuracy.pth"
    # The Partner (Phase 2 Classifier)
    CKPT_CONV = "checkpoints_convnext/best_convnext_cls.pth"
    # The Sniper (Phase 3 Regressor)
    CKPT_REG = "best_residual_regressor_clip.pth"
    
    SUBMISSION_FILE = "submissionfile.csv"
    
    # Logit Weighting: 60% StreetCLIP, 40% ConvNeXt
    # This was determined based on validation accuracy (CLIP is slightly stronger)
    ALPHA = 0.60 
    
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Data Loading & Dual Transforms
# ==========================================
class EnsembleDataset(Dataset):
    """
    Dual-Stream Dataset:
    Returns TWO versions of the same image:
    1. 336x336 Normalized for CLIP
    2. 384x384 Normalized for ConvNeXt
    """
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
        self.views = ['image_north','image_east','image_south','image_west']
        
        # Transform A: StreetCLIP (OpenAI Stats)
        self.tf_clip = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                                 [0.26862954, 0.26130258, 0.27577711])
        ])
        
        # Transform B: ConvNeXt (ImageNet Stats)
        self.tf_conv = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_imgs = []
        conv_imgs = []
        
        # Load all 4 views
        for v in self.views:
            p = os.path.join(self.img_dir, row[v])
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
            else:
                img = Image.new("RGB", (384, 384))
                
            # Apply both transforms to the raw PIL image
            clip_imgs.append(self.tf_clip(img))
            conv_imgs.append(self.tf_conv(img))
            
        return torch.stack(clip_imgs), torch.stack(conv_imgs), row['sample_id']

# ==========================================
# Model Architecture Definitions
# ==========================================

# --- StreetCLIP (Transformer Based)---
class UnifiedGeoModel(nn.Module):
    def __init__(self, num_classes, state_centroids, backbone_name="geolocal/StreetCLIP"):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(backbone_name)
        feat_dim = self.backbone.config.hidden_size
        
        # Transformer Encoder to fuse 4 views
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8, batch_first=True, dropout=0.1),
            num_layers=2
        )
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim*4,1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        self.state_centroids = state_centroids

    def extract_features(self, x):
        B,V,C,H,W = x.shape
        x = x.view(B*V,C,H,W)
        feats = self.backbone(pixel_values=x).pooler_output
        feats = feats.view(B,V,-1)
        return self.transformer(feats).reshape(B,-1) # [B, Dim*4]

    def forward(self, x):
        return self.cls_head(self.extract_features(x))

# --- ConvNeXt (CNN-based) ---
class ConvNeXtGeoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("convnextv2_large.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=0)
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(d * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x):
        B,V,C,H,W = x.shape
        x = x.view(B*V, C, H, W)
        
        # 1. Extract Backbone Features [B*V, D]
        feats = self.backbone(x)
        
        # 2. Reshape to Explicitly Group Views [B, V, D]
        feats = feats.view(B, V, -1)
        
        # 3. Concatenate all 4 views [B, V*D]
        feats = feats.reshape(B, -1)
        
        return self.head(feats)

# --- Residual Regressor ---
class ResidualGPSRegressor(nn.Module):
    def __init__(self, cls_model, state_centroids, num_states, state_emb_dim=64):
        super().__init__()
        self.cls_model = cls_model
        self.state_centroids = state_centroids
        self.state_emb = nn.Embedding(num_states, state_emb_dim)
        self.img_feat_dim = cls_model.backbone.config.hidden_size * 4
        
        self.regressor = nn.Sequential(
            nn.Linear(self.img_feat_dim + state_emb_dim + 1,1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024,512), nn.ReLU(), nn.Linear(512,2)
        )

    def forward(self, imgs, top1_idx, top1_prob):
        # Extract visual features (Frozen from Phase 1)
        with torch.no_grad():
            img_feats = self.cls_model.extract_features(imgs)
        
        # Combine Visuals + State Embed + Probability
        state_emb = self.state_emb(top1_idx)
        top1_p_unsq = top1_prob.unsqueeze(1)
        x_in = torch.cat([img_feats, state_emb*top1_p_unsq, top1_p_unsq], dim=1)
        
        # Predict Offset and add to Centroid
        return self.state_centroids[top1_idx] + self.regressor(x_in)

# ==========================================
# Utilities
# ==========================================
def get_state_mappings(csv_path):
    df = pd.read_csv(csv_path)
    u = sorted(df['state_idx'].unique())
    return {k:i for i,k in enumerate(u)}, {i:k for i,k in enumerate(u)}

def calculate_centroids(df, idx_to_dense):
    cents = torch.zeros(len(idx_to_dense), 2)
    d2o = {v:k for k,v in idx_to_dense.items()}
    for i in range(len(idx_to_dense)):
        grp = df[df['state_idx']==d2o[i]]
        cents[i,0] = grp['latitude'].mean()
        cents[i,1] = grp['longitude'].mean()
    return cents.to(Config.DEVICE)

def apply_tta(imgs):
    return [imgs, torch.flip(imgs, dims=[-1])]

# ==========================================
# Main Inference Loop
# ==========================================
def run_ensemble():
    print("--- Starting Robust Ensemble Inference ---")
    
    idx_to_dense, dense_to_idx = get_state_mappings(Config.TRAIN_CSV)
    centroids = calculate_centroids(pd.read_csv(Config.TRAIN_CSV), idx_to_dense)
    num_classes = len(idx_to_dense)

    print("Loading StreetCLIP...")
    model_clip = UnifiedGeoModel(num_classes, centroids).to(Config.DEVICE)
    st = torch.load(Config.CKPT_CLIP, map_location=Config.DEVICE)
    if 'model_state' in st: st = st['model_state']
    st = {k.replace('module.', ''): v for k, v in st.items()}
    # strict=False handles minor mismatches (e.g., if you dropped a temp layer)
    model_clip.load_state_dict(st, strict=False)
    model_clip.eval()

    print("Loading ConvNeXt...")
    model_conv = ConvNeXtGeoClassifier(num_classes).to(Config.DEVICE)
    model_conv.load_state_dict(torch.load(Config.CKPT_CONV, map_location=Config.DEVICE))
    model_conv.eval()

    print("Loading Regressor...")
    model_reg = ResidualGPSRegressor(model_clip, centroids, num_states=num_classes).to(Config.DEVICE)
    model_reg.load_state_dict(torch.load(Config.CKPT_REG, map_location=Config.DEVICE), strict=False)
    model_reg.eval()

    loader = DataLoader(
        EnsembleDataset(pd.read_csv(Config.TEST_CSV), Config.TEST_IMG_DIR),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    results = []
    print(f"Running Inference (W_CLIP={Config.ALPHA}, Logit Fusion)...")

    # --- Prediction Loop ---
    with torch.no_grad():
        for clip_imgs, conv_imgs, sids in tqdm(loader):
            clip_imgs = clip_imgs.to(Config.DEVICE)
            conv_imgs = conv_imgs.to(Config.DEVICE)

            # --- TTA Loop ---
            clip_tta = apply_tta(clip_imgs)
            conv_tta = apply_tta(conv_imgs)
            
            # Accumulators for TTA Averaging
            accum_probs = 0
            accum_gps = 0
            
            # Iterate through Augmentations (Original, Flipped)
            for i in range(len(clip_tta)):
                c_img = clip_tta[i]
                v_img = conv_tta[i]
                
                # Get Raw Logits (Before Softmax)
                l_clip = model_clip(c_img)
                l_conv = model_conv(v_img)
                
                # LOGIT FUSION
                # Combine logits using the ALPHA weight
                fused_logits = (Config.ALPHA * l_clip) + ((1 - Config.ALPHA) * l_conv)
                
                # Softmax AFTER Fusion
                # This gives the final ensemble probabilities
                final_probs = torch.softmax(fused_logits, dim=1)
                accum_probs += final_probs
                
                # Regress GPS 
                # Use the fused result to pick the state, then regress offset
                top1_p, top1_idx = final_probs.max(1)
                gps_out = model_reg(c_img, top1_idx, top1_p)
                accum_gps += gps_out
            
            # --- Average over TTA ---
            avg_probs = accum_probs / len(clip_tta)
            avg_gps = accum_gps / len(clip_tta)
            
            # --- Convert to Output Format ---
            avg_gps = avg_gps.cpu().numpy()
            top5 = avg_probs.topk(5, 1)[1].cpu().numpy()

            for i, sid in enumerate(sids):
                top5_orig = [dense_to_idx[x] for x in top5[i]]
                results.append({
                    'sample_id': sid.item(),
                    'predicted_state_idx_1': top5_orig[0],
                    'predicted_state_idx_2': top5_orig[1],
                    'predicted_state_idx_3': top5_orig[2],
                    'predicted_state_idx_4': top5_orig[3],
                    'predicted_state_idx_5': top5_orig[4],
                    'predicted_latitude': float(avg_gps[i][0]),
                    'predicted_longitude': float(avg_gps[i][1])
                })

    # --- Save Submission ---
    sub_df = pd.DataFrame(results)
    template = pd.read_csv(Config.TEST_CSV)[['sample_id', 'image_north', 'image_east', 'image_south', 'image_west']]
    final_df = template.merge(sub_df, on='sample_id', how='left')
    final_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"✅ Ensemble Submission Saved: {Config.SUBMISSION_FILE}")

if __name__ == "__main__":
    run_ensemble()