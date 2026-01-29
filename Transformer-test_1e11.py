import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

# ===================== Dataset =====================
class AirShowerTransformerDataset(Dataset):
    def __init__(self, hitsE, hitsM, labels, max_hits=2000, only_ED=True):
        self.hitsE = hitsE
        self.hitsM = hitsM
        self.labels = labels
        self.max_hits = max_hits
        self.only_ED = only_ED

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hE = self.hitsE[idx]
        hM = self.hitsM[idx]
        label = int(self.labels[idx])

        if self.only_ED:
            feats = hE[:, :3]
        else:
            feats = np.concatenate([hE[:, :3], hM[:, :3]], axis=0)

        n_hits = len(feats)
        if n_hits < self.max_hits:
            pad = np.zeros((self.max_hits - n_hits, feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        else:
            feats = feats[:self.max_hits]

        feats = torch.tensor(feats, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return feats, label

# ===================== Transformer model =====================
class AirShowerTransformer(nn.Module):
    def __init__(self, in_dim=3, d_model=128, nhead=8, num_layers=4, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        out = self.encoder(x)
        out = out.mean(dim=1)  # global average pooling
        logits = self.classifier(out)
        return logits

# ===================== Train & Evaluate =====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]  # signal 概率
        total_loss += loss.item() * feats.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    acc = ((all_probs > 0.5) == all_labels).mean()
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader.dataset), acc, auc, all_probs, all_labels

# ===================== Main =====================
def main(data_file, model_file, version="Transformer_v1"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    data = np.load(data_file, allow_pickle=True)
    hitsE_all, hitsM_all, labels_all = data['hitsE'], data['hitsM'], data['labels']

    hitsE_train, hitsE_val, hitsM_train, hitsM_val, y_train, y_val = train_test_split(
        hitsE_all, hitsM_all, labels_all, test_size=0.2, random_state=42, stratify=labels_all
    )

    train_dataset = AirShowerTransformerDataset(hitsE_train, hitsM_train, y_train, only_ED=True)
    val_dataset   = AirShowerTransformerDataset(hitsE_val, hitsM_val, y_val, only_ED=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32)

    model = AirShowerTransformer(in_dim=3, d_model=128, nhead=8, num_layers=4, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = np.inf
    patience, epochs_no_improve = 10, 0
    num_epochs = 50

    os.makedirs("/home/zhonghua/Filt_Event/figures", exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_probs, val_labels = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # === signal/background rates ===
        bkg_index = np.where(val_labels == 0)[0]
        sig_index = np.where(val_labels == 1)[0]
        thresholds = [0.8, 0.9, 0.95, 0.99]
        for thr in thresholds:
            bkg_rate = np.sum(val_probs[bkg_index] > thr) / len(bkg_index)
            sig_rate = np.sum(val_probs[sig_index] > thr) / len(sig_index)
            print(f"   threshold={thr:.2f}: bkg={bkg_rate:.4f}, sig={sig_rate:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_file)
            print(f"✅ New best model saved! Val loss={val_loss:.4f}")

            # === 画 softmax 概率直方图 ===
            plt.hist(val_probs[bkg_index], bins=20, alpha=0.5, density=True, label='Background')
            plt.hist(val_probs[sig_index], bins=20, alpha=0.5, density=True, label='Signal')
            plt.xlabel("Signal probability (softmax output)")
            plt.ylabel("Density")
            plt.legend()
            plt.title(f"Transformer softmax output @ epoch {epoch+1}")
            plt.savefig(f"/home/zhonghua/Filt_Event/figures/Transformer_best_model_{version}.png")
            plt.close()

            np.savez(f"/home/zhonghua/Filt_Event/figures/Transformer_val_hist_{version}.npz",
                     bkg_hist=val_probs[bkg_index], sig_hist=val_probs[sig_index])

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"🚨 Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    version='1e11_V03'
    data_file=f"/data/zhonghua/Dataset_Filted/ForTrain/combined_filted_{version}_dataset.npz"
    model_file = f'/home/zhonghua/Filt_Event/models/Transformer_best_model_{version}.pth'
    main(data_file, model_file)