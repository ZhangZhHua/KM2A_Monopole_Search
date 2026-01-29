import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve,  roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, Dataset,random_split,Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingWarmRestarts
import random
import os
import pandas as pd

def hits_to_image(hE, hM, grid_size=128, array_radius=648.0,only_ED=False):
    grid_size = int(grid_size)
    bins = np.linspace(-array_radius, array_radius, grid_size + 1)
    image_em = np.zeros((grid_size, grid_size))
    image_mu = np.zeros((grid_size, grid_size))
    
    # Electromagnetic channel
    if len(hE) > 0:
        x, y, pe = hE[:, 0], hE[:, 1], hE[:, 2]
        image_em, _, _ = np.histogram2d(x, y, bins=[bins, bins], weights=pe)
        image_em = image_em.T  # [grid_size, grid_size]
        image_em = np.log10(image_em+1) + 1e-4
        # if image_em.max() > 0:
        #     image_em = image_em / 1e2  #image_em.max()
    
    # Muon channel
    if len(hM) > 0:
        x, y, pe = hM[:, 0], hM[:, 1], hM[:, 2]
        image_mu, _, _ = np.histogram2d(x, y, bins=[bins, bins], weights=pe)
        image_mu = image_mu.T
        # if image_mu.max() > 0:
        #     image_mu = image_mu / image_mu.max()
    
    # Stack into 2 channels
    if only_ED:
        return np.expand_dims(image_em, axis=0)
    else:
        image = np.stack([image_em, image_mu], axis=0)  # [2, grid_size, grid_size]
        return image

class AirShowerDataset(Dataset):
    def __init__(self, hitsE, hitsM, labels, grid_size=128, only_ED=False):
        self.hitsE = hitsE
        self.hitsM = hitsM
        self.labels = np.array(labels, dtype=np.int8)
        self.grid_size = grid_size
        self.only_ED = only_ED
        unique_labels = np.unique(self.labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            print("⚠️ 警告：发现非法标签 ->", unique_labels)
            # 自动修正：大于 1 的设为 1，小于 0 的设为 0
            self.labels = np.clip(self.labels, 0, 1)
            print("✅ 已修正为 0/1 范围内")
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hE = self.hitsE[idx]
        hM = self.hitsM[idx]
        label = self.labels[idx]
        image = hits_to_image(hE, hM, self.grid_size, only_ED=self.only_ED)
        image_tensor = torch.tensor(image, dtype=torch.float32)  # [2, grid_size, grid_size]
        return image_tensor, torch.tensor(label, dtype=torch.long)



# ----------------------------
# CNN 模型定义
# ----------------------------
class CNNClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 [B,128,1,1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# 训练函数
# ----------------------------
def train_epoch(model, loader, optimizer, device, scaler=None, clip_grad=5.0):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            out = model(data)
            loss = F.cross_entropy(out, target)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# ----------------------------
# 验证函数
# ----------------------------
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            probs = F.softmax(out, dim=1)[:,1]  # 信号概率
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            pred = out.argmax(dim=1)
            correct += (pred==target).sum().item()
    acc = correct / len(all_labels)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, np.array(all_labels), np.array(all_probs)


def test_model(dataset_file, model_file, batch_size=128, version='1e9_V03'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(dataset_file, allow_pickle=True)
    hitsE, hitsM, labels = data['hitsE'], data['hitsM'], data['labels']

    # 创建数据集
    test_dataset = AirShowerDataset(hitsE, hitsM, labels, grid_size=128, only_ED=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if os.path.exists(model_file):
        model = CNNClassifier(in_channels=2).to(device)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        val_loss, val_acc, all_labels, all_probs= evaluate(model, test_loader, device)
        print(f"test:  Best model loaded from {model_file} | Best Val Loss: {val_loss:.4f} | Best Val Acc: {val_acc:.4f} | Best Val AUC: {val_auc:.4f}")
        bkg_index = np.where(labels==0)[0]
        sig_index = np.where(labels==1)[0]
        plt.hist(all_probs[bkg_index], bins=50, alpha=0.5, density=True, label='Background')
        plt.hist(all_probs[sig_index], bins=50, alpha=0.5, density=True, label='Signal')
        plt.xlim(0, 1)
        plt.legend()
        plt.savefig(f'/home/zhonghua/Filt_Event/figures/CNN_test_model_{version}.png')
        plt.close()
        thresholds=[0.8, 0.9, 0.95, 0.99]
        for threshold in thresholds:    
            print(f'    when threshold={threshold:.2f}: bkg rate={len(np.where(all_probs[bkg_index]>threshold)[0])/len(bkg_index):.4f}, sig rate={len(np.where(all_probs[sig_index]>threshold)[0])/len(sig_index):.4f}')



# ----------------------------
# 主函数
# ----------------------------
def main(dataset_file, model_file, only_ED=True, batch_size=128, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = torch.cuda.device_count()
    print(f"可用GPU数量: {world_size}")

    # 加载数据
    data = np.load(dataset_file, allow_pickle=True)
    hitsE, hitsM, labels = data['hitsE'], data['hitsM'], data['labels']

    # 创建数据集
    full_dataset = AirShowerDataset(hitsE, hitsM, labels, grid_size=128, only_ED=only_ED)
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    num_workers = min(25, os.cpu_count()-2) if os.cpu_count() else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f'数据集大小: {len(full_dataset)}, 训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    # ----------------------------
    # 模型与优化器
    # ----------------------------
    in_channels = train_dataset[0][0].shape[0]  # [C,H,W] 的 C
    model = CNNClassifier(in_channels=in_channels).to(device)

    # 多 GPU 支持
    if world_size > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader), T_mult=1, eta_min=1e-7
    )
    scaler = GradScaler()
    best_loss = float("inf")

    # 如果已有模型，先加载并计算 loss
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        best_loss = val_loss
        print(f"✅ Loaded model {model_file}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # ----------------------------
    # 训练循环
    # ----------------------------
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_acc, val_labels, val_probs = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        thresholds=[0.8, 0.9, 0.95, 0.99]
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_file)
            print(f"📌 New best model saved! Val Loss: {best_loss:.4f}")

            # 绘制信号/背景概率直方图
            
            bkg_index = np.where(val_labels==0)[0]
            sig_index = np.where(val_labels==1)[0]
            for threshold in thresholds:
                print(f'    when threshold={threshold:.2f}: bkg rate={len(np.where(val_probs[bkg_index]>threshold)[0])/len(bkg_index):.4f}, sig rate={len(np.where(val_probs[sig_index]>threshold)[0])/len(sig_index):.4f}')

            
            plt.hist(val_probs[bkg_index], bins=50, alpha=0.5, density=True, label='Background')
            plt.hist(val_probs[sig_index], bins=50, alpha=0.5, density=True, label='Signal')
            plt.xlabel("Signal probability")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig("/home/zhonghua/Filt_Event/figures/CNN_best_model_hist.png")
            plt.close()

if __name__ == "__main__":
    version = "1e9_V03"
    model_file = f"/home/zhonghua/Filt_Event/models/CNN_test_model_{version}.pth"
    dataset_file = f"/data/zhonghua/Dataset_Filted/ForTrain/combined_filted_{version}_dataset.npz"
    main(dataset_file, model_file, only_ED=True, batch_size=512, epochs=50)

    test_dataset_file = f"/data/zhonghua/Dataset_Filted/ForTrain/TEST_combined_filted_{version}_dataset.npz"
    test_model(test_dataset_file, model_file, only_ED=True, batch_size=512)
