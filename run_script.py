# file: run_model.py
# Full PyTorch training pipeline for elbow X-ray fracture (Yes/No) → VLM (ViT → T5 prefix with LoRA)

from __future__ import annotations
import os, random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

import timm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# -----------------------------
# Global config
# -----------------------------
DEVICE = torch.device("cpu")  # Mac CPU/unified memory
SEED = 42
BATCH_SIZE = 24
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = False
IMG_SIZE = 224
VIT_MODEL = 'vit_base_patch16_224'
T5_NAME = 't5-base'
NUM_PREFIX_TOKENS = 8
LEARNING_RATE_T5 = 5e-5
LEARNING_RATE_IMG = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 3
BEST_CKPT = 'best_model.pth'
CSV_PATH = 'elbow.csv'

# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def create_csv_from_directory(train_dir: str, valid_dir: str, csv_path: str = CSV_PATH) -> str:
    data = []

    def walk_split(root: str, split: str):
        if not os.path.isdir(root): return
        for patient in sorted(os.listdir(root)):
            p_path = os.path.join(root, patient)
            if not os.path.isdir(p_path): continue
            for study in sorted(os.listdir(p_path)):
                s_path = os.path.join(p_path, study)
                if not os.path.isdir(s_path): continue
                s_lower = study.lower()
                if 'positive' in s_lower:
                    label = 1
                elif 'negative' in s_lower:
                    label = 0
                else:
                    continue
                for f in sorted(os.listdir(s_path)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        data.append({'filepath': os.path.join(s_path, f), 'label': label, 'split': split})

    walk_split(train_dir, 'train')
    walk_split(valid_dir, 'val')

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"CSV created: {csv_path} | total={len(df)} | train={len(df[df.split=='train'])} | val={len(df[df.split=='val'])}")
    return csv_path


class ElbowDataset(Dataset):
    def __init__(self, csv_path: str, split: str, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df.split == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, int(row.label)


# -----------------------------
# Transforms
# -----------------------------
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# -----------------------------
# Vision → Text prefix projector
# -----------------------------
class ImgToPrefix(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_tokens: int = NUM_PREFIX_TOKENS):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim * num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _ = x.shape
        y = self.proj(x).view(b, self.num_tokens, -1)
        return y


# -----------------------------
# Prompt helpers
# -----------------------------
def make_text_batch(labels: List[int]) -> Tuple[List[str], List[str]]:
    prompts = ["Question: Does this elbow X-ray show a fracture? Answer:" for _ in labels]
    targets = ["Yes." if y == 1 else "No." for y in labels]
    return prompts, targets


def parse_outputs(texts: List[str]) -> List[int]:
    preds = []
    for t in texts:
        s = t.strip().lower()
        if any(k in s for k in ["yes", "fracture", "broken", "positive", "abnormal"]):
            preds.append(1)
        elif any(k in s for k in ["no", "normal", "negative", "intact", "healthy"]):
            preds.append(0)
        else:
            preds.append(0)
    return preds


# -----------------------------
# Build model components
# -----------------------------
def build_models():
    vision = timm.create_model(VIT_MODEL, pretrained=True, num_classes=0).to(DEVICE)
    vision.eval()
    vit_out = 768

    tok = T5Tokenizer.from_pretrained(T5_NAME)
    t5 = T5ForConditionalGeneration.from_pretrained(T5_NAME)
    lora_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
                          lora_dropout=0.1, target_modules=["q", "v"])
    t5 = get_peft_model(t5, lora_cfg).to(DEVICE)

    img2prefix = ImgToPrefix(vit_out, t5.config.d_model, NUM_PREFIX_TOKENS).to(DEVICE)
    return vision, img2prefix, tok, t5


@torch.no_grad()
def extract_img_prefix(vision, img2prefix, imgs):
    feats = vision(imgs)
    return img2prefix(feats)


def build_encoder_inputs(tokenizer, t5_model, prompts, img_prefix):
    enc = tokenizer(prompts, padding=True, return_tensors='pt')
    input_ids, attn_mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    text_emb = t5_model.get_input_embeddings()(input_ids)
    enc_embeds = torch.cat([img_prefix, text_emb], dim=1)
    prefix_mask = torch.ones((img_prefix.size(0), img_prefix.size(1)), dtype=attn_mask.dtype, device=attn_mask.device)
    full_mask = torch.cat([prefix_mask, attn_mask], dim=1)
    return enc_embeds, full_mask


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(t5, img2prefix, vision, dl, tokenizer):
    t5.eval(); img2prefix.eval(); vision.eval()
    all_y, all_p = [], []

    for imgs, labels in dl:
        imgs = imgs.to(DEVICE)
        img_prefix = extract_img_prefix(vision, img2prefix, imgs)
        prompts, _ = make_text_batch(labels.tolist())
        enc_embeds, attn = build_encoder_inputs(tokenizer, t5, prompts, img_prefix)

        encoder_outputs = t5.get_encoder()(inputs_embeds=enc_embeds, attention_mask=attn)
        gen = t5.generate(encoder_outputs=encoder_outputs, attention_mask=attn,
                          max_new_tokens=6, num_beams=3)
        preds = parse_outputs(tokenizer.batch_decode(gen, skip_special_tokens=True))
        all_y.extend(labels.tolist())
        all_p.extend(preds)

    acc = accuracy_score(all_y, all_p)
    f1 = f1_score(all_y, all_p)
    auroc = roc_auc_score(all_y, all_p) if len(set(all_y)) > 1 else float('nan')
    cm = confusion_matrix(all_y, all_p)
    return acc, f1, auroc, cm


# -----------------------------
# Training loop
# -----------------------------
def main():
    seed_everything(SEED)
    create_csv_from_directory('train/XR_ELBOW', 'valid/XR_ELBOW', CSV_PATH)

    train_ds = ElbowDataset(CSV_PATH, 'train', img_tf)
    val_ds = ElbowDataset(CSV_PATH, 'val', img_tf)

    cls_counts = train_ds.df.label.value_counts().to_dict()
    sample_weights = [1.0 / cls_counts[y] for y in train_ds.df.label]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    vision, img2prefix, tok, t5 = build_models()

    opt = torch.optim.AdamW([
        {'params': img2prefix.parameters(), 'lr': LEARNING_RATE_IMG},
        {'params': t5.parameters(), 'lr': LEARNING_RATE_T5},
    ], weight_decay=WEIGHT_DECAY)

    best_f1, patience = 0.0, 0
    for epoch in range(EPOCHS):
        t5.train(); img2prefix.train(); vision.eval()
        total_loss = 0

        for bidx, (imgs, labels) in enumerate(train_dl):
            imgs = imgs.to(DEVICE)
            labels_list = labels.tolist()

            with torch.no_grad():
                img_prefix = extract_img_prefix(vision, img2prefix, imgs)

            prompts, targets = make_text_batch(labels_list)
            enc_embeds, attn = build_encoder_inputs(tok, t5, prompts, img_prefix)
            dec = tok(targets, padding=True, truncation=True, max_length=8, return_tensors='pt').to(DEVICE)

            opt.zero_grad(set_to_none=True)
            encoder_outputs = t5.get_encoder()(inputs_embeds=enc_embeds, attention_mask=attn)
            out = t5(encoder_outputs=encoder_outputs, attention_mask=attn, labels=dec.input_ids)
            loss = out.loss
            loss.backward(); opt.step()

            total_loss += loss.item() * imgs.size(0)
            if bidx % 10 == 0:
                print(f"Epoch {epoch:02d} | Batch {bidx:04d} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_dl.dataset)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f}")

        acc, f1, auroc, cm = evaluate(t5, img2prefix, vision, val_dl, tok)
        print(f"Val → Acc: {acc:.4f} | F1: {f1:.4f} | AUROC: {auroc:.4f}\nCM:\n{cm}")

        if f1 > best_f1:
            best_f1, patience = f1, 0
            torch.save({'t5_state_dict': t5.state_dict(),
                        'img2prefix_state_dict': img2prefix.state_dict(),
                        'epoch': epoch, 'val_f1': f1, 'val_acc': acc}, BEST_CKPT)
            print(f"Saved best checkpoint → {BEST_CKPT} (F1={f1:.4f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"Training complete. Best F1: {best_f1:.4f}")


# -----------------------------
# Single-image prediction
# -----------------------------
@torch.no_grad()
def predict_single(image_path, vision, img2prefix, t5, tokenizer, transform=img_tf):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(DEVICE)
    img_prefix = extract_img_prefix(vision, img2prefix, x)
    prompts = ["Question: Does this elbow X-ray show a fracture? Answer:"]
    enc_embeds, attn = build_encoder_inputs(tokenizer, t5, prompts, img_prefix)
    encoder_outputs = t5.get_encoder()(inputs_embeds=enc_embeds, attention_mask=attn)
    gen = t5.generate(encoder_outputs=encoder_outputs, attention_mask=attn,
                      max_new_tokens=6, num_beams=3)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


if __name__ == "__main__":
    main()

    # Optional quick manual test if sample files exist
    samples = [
        'train/XR_ELBOW/patient00011/study1_negative/image1.png',
        'train/XR_ELBOW/patient00016/study1_positive/image1.png',
    ]
    if all(os.path.exists(p) for p in samples):
        print("\nManual prediction test:")
        # Rebuild models to load best ckpt
        vision, img2prefix, tok, t5 = build_models()
        if os.path.exists(BEST_CKPT):
            ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
            t5.load_state_dict(ckpt['t5_state_dict'])
            img2prefix.load_state_dict(ckpt['img2prefix_state_dict'])
        for p in samples:
            out = predict_single(p, vision, img2prefix, t5, tok)
            print(os.path.basename(p), '→', out)
