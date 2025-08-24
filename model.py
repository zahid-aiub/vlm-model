# Full PyTorch training pipeline for elbow X-ray fracture (Yes/No) → VLM (T5+LoRA)
# Updated for MURA dataset directory structure

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

# -----------------------------
# 1. Create CSV from directory structure
# -----------------------------
def create_csv_from_directory(train_dir, valid_dir, csv_path='elbow.csv'):
    data = []
    
    # Process train directory
    for patient_dir in os.listdir(train_dir):
        patient_path = os.path.join(train_dir, patient_dir)
        if os.path.isdir(patient_path):
            for study_dir in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_dir)
                if os.path.isdir(study_path):
                    # Extract label from study directory name
                    if 'positive' in study_dir.lower():
                        label = 1
                    elif 'negative' in study_dir.lower():
                        label = 0
                    else:
                        continue  # Skip if cannot determine label
                    
                    # Add all images in this study
                    for img_file in os.listdir(study_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(study_path, img_file)
                            data.append({
                                'filepath': img_path,
                                'label': label,
                                'split': 'train'
                            })
    
    # Process valid directory
    for patient_dir in os.listdir(valid_dir):
        patient_path = os.path.join(valid_dir, patient_dir)
        if os.path.isdir(patient_path):
            for study_dir in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_dir)
                if os.path.isdir(study_path):
                    # Extract label from study directory name
                    if 'positive' in study_dir.lower():
                        label = 1
                    elif 'negative' in study_dir.lower():
                        label = 0
                    else:
                        continue  # Skip if cannot determine label
                    
                    # Add all images in this study
                    for img_file in os.listdir(study_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(study_path, img_file)
                            data.append({
                                'filepath': img_path,
                                'label': label,
                                'split': 'val'
                            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Created CSV with {len(df)} entries. Train: {len(df[df.split=='train'])}, Val: {len(df[df.split=='val'])}")
    return csv_path

# -----------------------------
# 2. Dataset
# -----------------------------
class ElbowDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df.split == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.filepath[idx]
        label = int(self.df.label[idx])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# 3. Transforms
# -----------------------------
img_tf = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# -----------------------------
# 4. Vision Encoder
# -----------------------------
vision_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # Remove classification head
vision_backbone.eval().cuda()

proj = torch.nn.Linear(768, 768).cuda()

# -----------------------------
# 5. T5 + LoRA
# -----------------------------
tok = T5Tokenizer.from_pretrained('t5-base')
# Add special tokens for our task
special_tokens = ['<image>']
tok.add_special_tokens({'additional_special_tokens': special_tokens})

t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
t5.resize_token_embeddings(len(tok))  # Resize for new tokens

# LoRA configuration
config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1,
    target_modules=["q", "v"]
)
t5 = get_peft_model(t5, config).cuda()

# -----------------------------
# 6. Training utilities
# -----------------------------
def make_text(label_batch):
    prompt = ["Question: Is there an elbow fracture in this X-ray? Answer:" for _ in label_batch]
    target = ["Yes" if y == 1 else "No" for y in label_batch]
    return prompt, target

def evaluate(model, proj, vision_backbone, dataloader, tokenizer):
    model.eval()
    proj.eval()
    vision_backbone.eval()
    
    all_y, all_p, all_probs = [], [], []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.cuda()
            
            # Get image features
            with torch.no_grad():
                image_features = vision_backbone(imgs)
            image_features = proj(image_features).unsqueeze(1)
            
            # Create prompts
            prompt, _ = make_text(labels.tolist())
            enc = tokenizer(prompt, padding=True, return_tensors='pt').to('cuda')
            
            # Generate responses
            outputs = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=3,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode outputs
            decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            # Convert to predictions
            preds = []
            for d in decoded:
                if 'yes' in d.lower():
                    preds.append(1)
                elif 'no' in d.lower():
                    preds.append(0)
                else:
                    preds.append(0)  # Default to negative if unclear
            
            all_y.extend(labels.tolist())
            all_p.extend(preds)
    
    # Calculate metrics
    accuracy = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p)
    auroc = roc_auc_score(all_y, all_p)
    cm = confusion_matrix(all_y, all_p)
    
    return accuracy, f1, auroc, cm, all_y, all_p

# -----------------------------
# 7. Main training function
# -----------------------------
def main():
    # Create CSV from directory structure
    print("Creating CSV from directory structure...")
    csv_path = create_csv_from_directory(
        train_dir='train/XR_ELBOW',
        valid_dir='valid/XR_ELBOW',
        csv_path='elbow.csv'
    )
    
    # Create datasets and dataloaders
    train_ds = ElbowDataset(csv_path, 'train', img_tf)
    val_ds = ElbowDataset(csv_path, 'val', img_tf)
    
    # Handle class imbalance
    class_counts = train_ds.df.label.value_counts().to_dict()
    sample_weights = [1.0 / class_counts[y] for y in train_ds.df.label]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_dl = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimizer
    opt = torch.optim.AdamW(
        list(proj.parameters()) + list(t5.parameters()), 
        lr=1e-4,
        weight_decay=0.01
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_f1 = 0
    for epoch in range(15):
        t5.train()
        proj.train()
        vision_backbone.eval()
        
        total_loss = 0
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            imgs, labels = imgs.cuda(), labels.cuda()
            
            # Get image features
            with torch.no_grad():
                feats = vision_backbone(imgs)
            feats = proj(feats).unsqueeze(1)
            
            # Prepare text
            prompt, target = make_text(labels.tolist())
            enc = tok(prompt, padding=True, return_tensors='pt').to('cuda')
            dec = tok(target, padding=True, truncation=True, max_length=8, return_tensors='pt').to('cuda')
            
            # Forward pass
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = t5(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    labels=dec.input_ids
                )
                loss = outputs.loss
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            total_loss += loss.item() * imgs.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_dl.dataset)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # Validation
        accuracy, f1, auroc, cm, _, _ = evaluate(t5, proj, vision_backbone, val_dl, tok)
        print(f"Validation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                't5_state_dict': t5.state_dict(),
                'proj_state_dict': proj.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1
            }, 'best_model.pth')
            print(f"Saved best model with F1: {f1:.4f}")

if __name__ == "__main__":
    main()