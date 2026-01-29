import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve,  roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, Dataset,random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingWarmRestarts
import random
import os
import ana

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda')

def plot_output_hist(model,model_file, val_loader,):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_preds_list, val_labels_list, val_probs_list = [], [], []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            out = model(images)
            val_probs = F.softmax(out, dim=1).cpu().numpy()
            val_preds = torch.argmax(out, dim=1).cpu().numpy()
            val_preds_list.extend(val_preds)
            val_labels_list.extend(labels_batch.cpu().numpy())
            val_probs_list.extend(val_probs[:, 1])
    val_acc = accuracy_score(val_labels_list, val_preds_list)
    auc = roc_auc_score(val_labels_list, val_preds_list) if len(set(val_labels_list)) > 1 else 0.0
    print("Training complete. Final validation metrics:")
    print(f"Accuracy: {val_acc:.4f}, AUC: {auc:.4f}")

    val_labels_list = torch.tensor(val_labels_list).numpy()
    val_preds_list = torch.tensor(val_preds_list).numpy()
    val_probs_list = torch.tensor(val_probs_list).numpy()

    # 分信号与背景分别画 hist
    bins=100
    plt.figure(figsize=(8,6))
    plt.hist(val_probs_list[val_labels_list==1], bins=bins, alpha=0.6,density=True,  label=f"Signal (label=1): {len(val_preds_list[val_labels_list==1])}")
    plt.hist(val_probs_list[val_labels_list==0], bins=bins, alpha=0.6, density=True, label=f"Background (label=0): {len(val_preds_list[val_labels_list==0])}")
    plt.xlabel("CNN_2 Output Probability")
    plt.ylabel("Counts")
    plt.legend()
    plt.title("CNN_2 Classification Output")
    plt.grid(True, alpha=0.3)
    plt.savefig('./figures/CNN_2_output_hist.png')
    plt.show()

def plot_metrics(model,train_loss_list,val_loss_list,train_accs_list,val_accs_list,val_labels_list, val_probs_list):
    fpr, tpr, _ = roc_curve(val_labels_list, val_probs_list)
    roc_auc = roc_auc_score(val_labels_list, val_probs_list)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4),) 
    # 损失曲线
    ax[0].plot(train_loss_list, label=f"Train Loss, last={train_loss_list[-1]:.4f}",)
    ax[0].plot(val_loss_list, label=f"Validation Loss, best={np.min(val_loss_list):.4f}",)
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].set_title(f"{model} Training Loss")
    ax[0].legend()
    ax[0].grid(linestyle='--', linewidth=0.5)
    # 准确率曲线
    ax[1].plot(train_accs_list, label=f"Train accuracy, last={train_accs_list[-1]:.4f}", )
    ax[1].plot(val_accs_list, label=f"Vali accuracy, best={np.max(val_accs_list):.4f}", )
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title(f"{model} Accuracy Curve")
    ax[1].legend()
    ax[1].grid(linestyle='--', linewidth=0.5)
    # 混淆矩阵
    ax[2].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})", linewidth=2)
    ax[2].plot([0, 1], [0, 1], linestyle="--", color="gray")  
    ax[2].set_xlabel("False Positive Rate (FPR)")
    ax[2].set_ylabel("True Positive Rate (TPR)")
    ax[2].set_title("ROC Curve")
    ax[2].legend()
    ax[2].grid(linestyle='--', linewidth=0.5)
    # 概率分布
    val_probs_list = np.array(val_probs_list)
    val_labels_list = np.array(val_labels_list)
    bkg_pred = val_probs_list[np.where(val_labels_list == 0)]  # 背景概率
    sig_pred = val_probs_list[np.where(val_labels_list == 1)]  # 信号概率
    threshold = 0.8
    sig_ratio = len(sig_pred[sig_pred > threshold]) / len(sig_pred)  # 信号保有率
    bkg_ratio = len(bkg_pred[bkg_pred > threshold]) / len(bkg_pred)  # 背景误判率
    bins = 100
    
    ax[3].hist(sig_pred, bins=bins, range=(0, 1), density=True, histtype='bar', 
               label=f"$N_{{signal}}$={len(sig_pred)}")
    ax[3].hist(bkg_pred, bins=bins, range=(0, 1), density=True, histtype='bar', 
               label=f"$N_{{bkg}}$={len(bkg_pred)}")
    # ax[3].vlines(threshold, 0, 15, linestyle='--', color='red', label=f'threshold={threshold}')
    ax[3].set_xlabel('Softmax output')
    ax[3].set_ylabel('Distribution')
    ax[3].set_title(f'Validation Data best {model} Output')
    ax[3].legend(loc="upper center")
    ax[3].grid(linestyle="dashed", linewidth=0.5)
    # ax[3].set_ylim(0, 20)  # 调整 y 轴范围以适应文本框
    print(f'when t>{threshold}, 信号保有率={sig_ratio:.5f}, 背景误判率={bkg_ratio:.5f}')
    print(f'when t>{threshold}, Nsig/(Nsig+Nbkg)={sig_ratio/(sig_ratio+bkg_ratio):.4f}, Nsig : Nbkg={sig_ratio/bkg_ratio:.4f}')
 
    plt.tight_layout()
    plt.savefig(f'./figures/{model}_Metrics.png', dpi=300)
    plt.show()
    plt.close()

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.img_size = 128
        self.model = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # Conv 1: 输入通道从 1 改为 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Conv 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv 3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Flatten(),
            nn.Linear(64 * (self.img_size // 8) * (self.img_size // 8), 128),  # Fully Connected Layer
            nn.ReLU(),
            nn.Dropout(0.3),  # 50% 概率丢弃神经元
            nn.Linear(128, 2),  # 2 分类（monopole vs background）
            # nn.Softmax(dim=1)  # 如果需要概率输出，可以取消注释
        )

    def forward(self, x):
        return self.model(x)

def hits_to_image(hE, hM, grid_size=128, array_radius=648.0):
    grid_size = int(grid_size)
    bins = np.linspace(-array_radius, array_radius, grid_size + 1)
    image_em = np.zeros((grid_size, grid_size))
    image_mu = np.zeros((grid_size, grid_size))
    
    # Electromagnetic channel
    if len(hE) > 0:
        x, y, pe = hE[:, 0], hE[:, 1], hE[:, 2]
        image_em, _, _ = np.histogram2d(x, y, bins=[bins, bins], weights=pe)
        image_em = image_em.T  # [grid_size, grid_size]
        if image_em.max() > 0:
            image_em = image_em / image_em.max()
    
    # Muon channel
    if len(hM) > 0:
        x, y, pe = hM[:, 0], hM[:, 1], hM[:, 2]
        image_mu, _, _ = np.histogram2d(x, y, bins=[bins, bins], weights=pe)
        image_mu = image_mu.T
        if image_mu.max() > 0:
            image_mu = image_mu / image_mu.max()
    
    # Stack into 2 channels
    image = np.stack([image_em, image_mu], axis=0)  # [2, grid_size, grid_size]
    return image

class AirShowerDataset(Dataset):
    def __init__(self, hitsE, hitsM, labels, grid_size=128):
        self.hitsE = hitsE
        self.hitsM = hitsM
        self.labels = labels
        self.grid_size = grid_size
   
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hE = self.hitsE[idx]
        hM = self.hitsM[idx]
        label = self.labels[idx]
        image = hits_to_image(hE, hM, self.grid_size)
        image_tensor = torch.tensor(image, dtype=torch.float32)  # [2, grid_size, grid_size]
        return image_tensor, torch.tensor(label, dtype=torch.long)


class CNNClassifier_new(nn.Module):
    def __init__(self, num_classes=2, grid_size=128):
        super(CNNClassifier_new, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)  # 2 input channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * (grid_size // 8) * (grid_size // 8), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # NpzFileList=[
    #             "./Dataset_Filted/Simulation/gamma/1e4_1e5/train_dataset_gamma_1e4_1e5_run000.npz",
    #             "./Dataset_Filted/Simulation/proton/1e4_1e5/train_dataset_proton_1e4_1e5_run000.npz",
    #             "./Dataset_Filted/Simulation/monopole/E1e9/train_dataset_monopole_E1e9.npz",
    #             ]
    # sample_num=(20000,20000,-1)
    # hitsE, hitsM, parID = ana.merge_npzdataset(NpzFileList,sample_num=sample_num)
    # labels = np.copy(parID)
    # labels[labels == 1] = 0
    # labels[labels == 14] = 0
    # labels[labels == 43] = 1
    # np.savez("./Dataset_Filted/ForTrain/train_dataset.npz", hitsE=hitsE, hitsM=hitsM, labels=labels)

    data=np.load("./Dataset_Filted/ForTrain/train_dataset.npz", allow_pickle=True)
    hitsE=data['hitsE']
    hitsM=data['hitsM']
    labels=data['labels']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create datasets
    full_dataset = AirShowerDataset(hitsE, hitsM, labels, grid_size=128)
    train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model, optimizer, scheduler
    model_file="./models/best_cnn_model_new_loss-coswarmre.pt"
    try:
        if os.path.exists(model_file):
            # 加载模型
            model = CNNClassifier_new(num_classes=2).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            print(f"成功从 {model_file} 加载模型")
            val_preds_list, val_labels_list, val_probs_list,total_loss = [], [], [],0
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images = images.to(device)
                    labels_batch = labels_batch.to(device)
                    out = model(images)
                    loss = F.cross_entropy(out, labels_batch)
                    val_probs = F.softmax(out, dim=1).cpu().numpy()
                    val_preds = torch.argmax(out, dim=1).cpu().numpy()
                    val_preds_list.extend(val_preds)
                    val_labels_list.extend(labels_batch.cpu().numpy())
                    val_probs_list.extend(val_probs[:, 1])
                    total_loss += loss.item()
            val_loss = total_loss / len(val_loader)
            val_acc = accuracy_score(val_labels_list, val_preds_list)
            best_auc = roc_auc_score(val_labels_list, val_probs_list) if len(set(val_labels_list)) > 1 else 0.0
            best_loss = val_loss
            print(f"primary model: loss: {best_loss:.4f}, accuracy: {val_acc:.4f}, AUC: {best_auc:.4f}")
            # plot_output_hist(model, val_loader)
            
        else:
            raise FileNotFoundError(f"模型文件 {model_file} 不存在")     
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("创建新模型实例")
        # 创建新模型
        model = CNNClassifier_new(num_classes=2).to(device)
        best_auc = 0
        best_loss = float('inf')


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=6e-4)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader),          # 初始周期长度（步数）
        T_mult=1,        # 周期长度不变
        eta_min=1e-7,     # 学习率最小值
        last_epoch=-1
    )
    # >>>>>>>>>>>>>>>>>>>>Training loop<<<<<<<<<<<<<<<<<<<<<
    train_losses, val_losses, train_accs, val_accs, val_aucs = [], [], [], [], []
    patience = 20
    epochs_no_improve = 0
    num_epochs = 50

    def get_lr(optimizer):
        return optimizer.param_groups[0]['lr']

    print("开始训练：")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_preds, train_labels_list = [], []
        for batch_idx,(images, labels_batch) in enumerate(train_loader):
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = F.cross_entropy(out, labels_batch)
            loss.backward()
            optimizer.step()
            scheduler.step() 
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels_list.extend(labels_batch.cpu().numpy())

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Step: {batch_idx}, LR: {scheduler.get_last_lr()[0]:.8f}")

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_labels_list, train_preds)

        model.eval()
        val_preds_list, val_labels_list,val_probs_list, val_loss_total = [], [],[], 0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                out = model(images)
                loss = F.cross_entropy(out, labels_batch)
                val_loss_total += loss.item()
                val_probs = F.softmax(out, dim=1).cpu().numpy()

                val_preds = torch.argmax(out, dim=1).cpu().numpy()
                val_preds_list.extend(val_preds)
                val_labels_list.extend(labels_batch.cpu().numpy())
                val_probs_list.extend(val_probs[:, 1])
        val_loss = val_loss_total / len(val_loader)

        val_acc = accuracy_score(val_labels_list, val_preds_list)
        auc = roc_auc_score(val_labels_list, val_probs_list) if len(set(val_labels_list)) > 1 else 0.0

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(auc)


        # scheduler.step(auc)
        # if auc > best_auc:
        #     best_auc = auc
        #     epochs_no_improve = 0
        #     torch.save(model.state_dict(), model_file)
        # else:
        #     epochs_no_improve += 1
        
        # scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_file)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, ")
        print(f"    Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}, ")
        print(f"    Val AUC: {auc:.4f}, LR: {get_lr(optimizer):.6f}")

        if epochs_no_improve >= patience: 
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation
    model.load_state_dict(torch.load(model_file))
    model.eval()
    val_preds_list, val_labels_list, val_probs_list = [], [], []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            out = model(images)
            val_probs = F.softmax(out, dim=1).cpu().numpy()
            val_preds = torch.argmax(out, dim=1).cpu().numpy()
            val_preds_list.extend(val_preds)
            val_labels_list.extend(labels_batch.cpu().numpy())
            val_probs_list.extend(val_probs[:, 1])

    val_acc = accuracy_score(val_labels_list, val_preds_list)
    auc = roc_auc_score(val_labels_list, val_probs_list) if len(set(val_labels_list)) > 1 else 0.0
    print("Training complete. Final validation metrics:")
    print(f"Accuracy: {val_acc:.4f}, AUC: {auc:.4f}")

    # print(train_losses, val_losses, train_accs, val_accs, val_labels_list, val_probs_list)
    plot_metrics("CNN",train_losses, val_losses,train_accs, val_accs, val_labels_list, val_probs_list)
    
    # # 额外：绘制输出直方图
    # plot_output_hist(model, val_loader)