import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv,GATConv, GlobalAttention, AttentionalAggregation

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import os
from torch_geometric.nn import knn_graph, radius_graph
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd

# def hits_to_graph(hE, hM, only_ED=False, k=15, r=None):
#     """
#     只用 ED hits (hE)
#     hE: np.array [N_e, 3] (x, y, pe)
#     """
#     hits = hE if len(hE) > 0 else np.zeros((0, 3))

#     if hits.shape[0] == 0:
#         hits = np.zeros((1, 3), dtype=np.float32)

#     # 基础特征
#     x, y, pe = hits[:, 0], hits[:, 1], hits[:, 2]
#     r2 = np.sqrt(x ** 2 + y ** 2)
#     logpe = np.log1p(pe)
#     theta = np.arctan2(y, x)

#     feats = np.stack([x, y, pe, logpe, r2, theta], axis=1)  # [num_nodes, 6]
#     x = torch.tensor(feats, dtype=torch.float32)

#     # 建边
#     if x.size(0) == 1:
#         edge_index = torch.zeros((2, 0), dtype=torch.long)
#     else:
#         if r is not None:
#             edge_index = radius_graph(x[:, :2], r, loop=False)
#         else:
#             edge_index = knn_graph(x[:, :2], k=k, loop=False)

#     return x, edge_index
def hits_to_graph(hE, hM, only_ED=False, k=15, r=None):
    """
    hE: np.array [N_e, 3] (x, y, pe)
    hM: np.array [N_mu, 3] (x, y, pe)
    only_ED: 是否只用电磁探测器
    k: kNN 的邻居数
    r: 半径阈值 (若给定则用 radius_graph，否则用 knn_graph)
    """
    if only_ED:
        hits = hE
    else:
        hits = np.concatenate([hE, hM], axis=0) if (len(hE) > 0 or len(hM) > 0) else np.zeros((0,3))

    if hits.shape[0] == 0:  
        # 空事件 → 虚拟节点
        hits = np.zeros((1, 3), dtype=np.float32)

    # 基础特征
    x, y, pe = hits[:,0], hits[:,1], hits[:,2]
    r2 = np.sqrt(x**2 + y**2)
    logpe = np.log1p(pe)  # log(1+pe)

    feats = np.stack([x, y, pe, logpe, r2], axis=1)  # [num_nodes, 5]
    x = torch.tensor(feats, dtype=torch.float32)

    # 建边
    if x.size(0) == 1:
        edge_index = torch.zeros((2,0), dtype=torch.long)  # 没有边
    else:
        if r is not None:
            edge_index = radius_graph(x[:,:2], r, loop=False)  # 用 (x,y) 建边
        else:
            edge_index = knn_graph(x[:,:2], k=k, loop=False)

    if edge_index.dim() == 1:
        edge_index = edge_index.unsqueeze(0)
    assert edge_index.size(0) == 2, f"edge_index wrong shape: {edge_index.shape}"

    
    return x, edge_index

class AirShowerGraphDataset(Dataset):
    def __init__(self, hitsE, hitsM, labels, only_ED=False, k=8, r=None):
        super().__init__()
        self.hitsE = hitsE
        self.hitsM = hitsM
        self.labels = labels
        self.only_ED = only_ED
        self.k = k
        self.r = r

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hE = self.hitsE[idx]
        hM = self.hitsM[idx]
        label = int(self.labels[idx])

        x, edge_index = hits_to_graph(hE, hM, only_ED=self.only_ED, k=self.k, r=self.r)
        if edge_index.size(0) == 1:
            print(f"Graph idx={idx}, x={x.shape}, edge_index={edge_index.shape}")
        y = torch.tensor(label, dtype=torch.long)  # 交叉熵要求标量 int
        return Data(x=x, edge_index=edge_index, y=y)
class AirShowerGraphDataset_new(Dataset):
    def __init__(self, hitsE, hitsM, labels,
                 csv_file=None, csv_cols=None,
                 only_ED=False, k=15, r=None):
        super().__init__()
        self.hitsE = hitsE
        self.hitsM = hitsM
        self.labels = labels
        self.only_ED = only_ED
        self.k = k
        self.r = r

        # 如果提供了 csv 文件
        if csv_file is not None and csv_cols is not None:
            df = pd.read_csv(csv_file)
            # 只保留需要的列
            self.csv_feats = df[csv_cols].values.astype(np.float32)
        else:
            self.csv_feats = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hE = self.hitsE[idx]
        hM = self.hitsM[idx]
        label = int(self.labels[idx])

        # hit-level graph
        x, edge_index = hits_to_graph(hE, hM,
                                      only_ED=self.only_ED,
                                      k=self.k, r=self.r)

        # event-level features
        if self.csv_feats is not None:
            csv_feat = torch.tensor(self.csv_feats[idx], dtype=torch.float32)
        else:
            csv_feat = torch.zeros(0)  # 如果没给，就返回空

        y = torch.tensor(label, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, csv_feat=csv_feat)
# ----------------------------
# GNN 模型
# ----------------------------
# class GNNClassifier(nn.Module):
#     def __init__(self, in_channels=3, hidden_channels=128, num_layers=4, num_classes=2, dropout=0.3, pool='mean'):
#         """
#         in_channels: 输入节点特征维度 (hitsE或hitsM的[x,y,pe])
#         hidden_channels: GNN隐藏层维度
#         num_layers: GNN卷积层数量
#         num_classes: 分类类别数量
#         dropout: dropout比率
#         pool: 'mean' 或 'max' 全局池化方式
#         """
#         super(GNNClassifier, self).__init__()
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
        
#         # 第一层卷积
#         self.convs.append(GraphConv(in_channels, hidden_channels))
#         self.bns.append(nn.BatchNorm1d(hidden_channels))
        
#         # 中间卷积层
#         for _ in range(num_layers - 1):
#             self.convs.append(GraphConv(hidden_channels, hidden_channels))
#             self.bns.append(nn.BatchNorm1d(hidden_channels))
        
#         self.pool = pool
#         self.dropout = dropout
        
#         # 分类 MLP
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels//2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_channels//2, num_classes)
#         )

#     def forward(self, x, edge_index, batch):
#         """
#         x: 节点特征 [N_total_nodes, in_channels]
#         edge_index: 边索引 [2, num_edges]
#         batch: batch索引 [N_total_nodes]，表示每个节点属于哪一个图
#         """
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
        
#         # 全局池化
#         if self.pool == 'mean':
#             x = global_mean_pool(x, batch)
#         else:
#             x = global_max_pool(x, batch)
        
#         # MLP分类
#         out = self.mlp(x)
#         return out

class GNNClassifier(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=128, num_layers=3,
                 num_classes=2, dropout=0.3, heads=4):
        """
        in_channels: 输入节点特征维度 (默认 6: x,y,pe,logpe,r2,theta)
        hidden_channels: GNN 隐藏层维度
        num_layers: GAT 层数
        num_classes: 分类类别数
        dropout: dropout 比率
        heads: GAT 注意力头数
        """
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 中间层
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 注意力池化
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1)
            )
        )

        self.dropout = dropout

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 注意力池化
        x = self.att_pool(x, batch)

        # 分类
        out = self.mlp(x)
        return out

class GNNClassifier_new(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=128, num_layers=3,
                 num_classes=2, dropout=0.3, heads=4, csv_dim=10):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # GAT 层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 注意力池化
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1)
            )
        )

        self.dropout = dropout

        # MLP 分类 (GNN 表示 + CSV 特征)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels + csv_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, edge_index, batch, csv_feats):
        # GNN 编码
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图池化
        gnn_out = self.att_pool(x, batch)

        # 拼接 event-level 特征
        out = torch.cat([gnn_out, csv_feats], dim=1)

        # 分类
        out = self.mlp(out)
        return out

def compute_avg_distance(x):
    # x: (N, d) 节点特征，只取坐标部分，比如前三维 (x,y,z) 或 (x,y)
    coords = x[:, :2]   # 如果 x 前三列是坐标
    r = torch.cdist(coords, coords, p=2)
    avg_dist = r.mean().item()
    max_dist = r.max().item()
    print(f"平均距离 = {avg_dist:.3f}, 最大距离 = {max_dist:.3f}")
    return avg_dist, max_dist

# ----------------------------
# 2. 训练函数
# ----------------------------
def train(model, loader, optimizer, device, scaler=None, clip_grad=5.0, num_classes=None):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        # 平均距离 = 575.972, 最大距离 = 4588.601
        # compute_avg_distance(data.x) 

        # 🔍 Debug 检查 edge_index 是否越界
        if data.edge_index.max().item() >= data.x.size(0):
            raise RuntimeError(
                f"[EdgeIndexError] batch={batch_idx}, "
                f"x={data.x.size()}, "
                f"edge_index max={data.edge_index.max().item()}"
            )

        # 🔍 Debug 检查 label 是否越界
        if num_classes is not None and data.y.max().item() >= num_classes:
            raise RuntimeError(
                f"[LabelError] batch={batch_idx}, "
                f"y_max={data.y.max().item()}, num_classes={num_classes}"
            )

        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)

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
# 3. 验证函数
# ----------------------------
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    correct, total = 0, 0
    val_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            val_loss += loss.item()
            probs = F.softmax(out, dim=1)[:,1]
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
            all_labels.extend(data.y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    acc = correct / total
    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)
    mask = ~np.isnan(probs_np)
    auc = roc_auc_score(labels_np[mask], probs_np[mask]) if len(set(labels_np[mask])) > 1 else 0.0
    return val_loss / len(loader), acc, auc, labels_np[mask], probs_np[mask]

# ----------------------------
# 4. 主训练循环
# ----------------------------
def main(data_file,csv_file,  model_file,version='1e9_V04'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(data_file, allow_pickle=True)
    hitsE_all, hitsM_all, labels_all = data['hitsE'], data['hitsM'], data['labels']
    
    from sklearn.model_selection import train_test_split
    
    num_events = len(labels_all)
    indices = np.arange(num_events)

    train_idx, val_idx = train_test_split(indices, test_size=0.2,
                                        random_state=42, stratify=labels_all)

    # 用 indices 取子集
    hitsE_train, hitsM_train, labels_train = hitsE_all[train_idx], hitsM_all[train_idx], labels_all[train_idx]
    hitsE_val, hitsM_val, labels_val = hitsE_all[val_idx], hitsM_all[val_idx], labels_all[val_idx]

    # 这里也能保证 CSV 对应，因为 CSV 里第 i 行就是 event i
    csv_feats = pd.read_csv(csv_file)
    train_csv = csv_feats[train_idx]
    val_csv   = csv_feats[val_idx]

    train_dataset = AirShowerGraphDataset(hitsE_train, hitsM_train, labels_train, only_ED=True,k=15)
    val_dataset   = AirShowerGraphDataset(hitsE_val, hitsM_val, labels_val, only_ED=True,k=15)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=256)
    print(f'✅ Dataset loaded! Train: {len(train_dataset)}, Val: {len(val_dataset)}, Total: {len(labels_all)}')
    # dist.init_process_group("nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 会自动传

    # torch.cuda.set_device(local_rank)
    # device = torch.device(f"cuda:{local_rank}")

    # model = GNNClassifier(in_channels=5).to(device)
    model = GNNClassifier_new()
    # model = DDP(model, device_ids=[local_rank])
    # 加载模型
    best_loss, best_acc, best_auc = 1, 0, 0
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print(f"✅ Model loaded from {model_file}")
        best_loss, best_acc, best_auc, _, _ =evaluate(model, val_loader, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader), T_mult=1, eta_min=1e-7
    )
    scaler = GradScaler()

    patience, epochs_no_improve =10, 0
    num_epochs = 50
    logs = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device, scaler=scaler, num_classes=2)
        val_loss, val_acc, val_auc, val_labels, val_probs = evaluate(model, val_loader, device)
        scheduler.step()

        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["val_acc"].append(val_acc)
        logs["val_auc"].append(val_auc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_file)
            print(f"✅ New best model saved! Val loss: {best_loss:.4f}")
            bkg_index = np.where(val_labels==0)[0]
            sig_index = np.where(val_labels==1)[0]
            plt.hist(val_probs[bkg_index], bins=50, alpha=0.5, density=True, label='Background')
            plt.hist(val_probs[sig_index], bins=50, alpha=0.5, density=True, label='Signal')
            plt.xlim(0, 1)
            plt.legend()
            plt.savefig(f'/home/zhonghua/Filt_Event/figures/GNN_best_model_{version}.png')
            plt.close()
            epochs_no_improve = 0
            thresholds=[0.8, 0.9, 0.95, 0.99]
            for threshold in thresholds:
                print(f'    when threshold={threshold:.2f}: bkg rate={len(np.where(val_probs[bkg_index]>threshold)[0])/len(bkg_index):.4f}, sig rate={len(np.where(val_probs[sig_index]>threshold)[0])/len(sig_index):.4f}')

            np.savez(f"/home/zhonghua/Filt_Event/figures/GNN_Val_hist_{version}.npz", bkg_hist=val_probs[bkg_index], sig_hist=val_probs[sig_index])
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"🚨 Early stopping triggered at epoch {epoch+1}")
            break

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 左边画 Loss 曲线
    axs[0].plot(logs["train_loss"], label="Train Loss")
    axs[0].plot(logs["val_loss"], label="Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("Training vs Validation Loss")

    # 右边画 Val Acc 和 Val AUC
    axs[1].plot(logs["val_acc"], label="Val Acc")
    axs[1].plot(logs["val_auc"], label="Val AUC")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Score")
    axs[1].legend()
    axs[1].set_title("Validation Metrics")

    plt.tight_layout()
    plt.savefig(f"/home/zhonghua/Filt_Event/figures/GNN_training_summary_{version}.png")
    plt.close()


def test_model(data_file, model_file, version):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(data_file, allow_pickle=True)
    hitsE_all, hitsM_all, labels_all = data['hitsE'], data['hitsM'], data['labels']
    # train_dataset = AirShowerGraphDataset(hitsE_train, hitsM_train, labels_train, only_ED=True,k=15)
    val_dataset   = AirShowerGraphDataset(hitsE_all, hitsM_all, labels_all, only_ED=True,k=15)

    # train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # dist.init_process_group("nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 会自动传

    # torch.cuda.set_device(local_rank)
    # device = torch.device(f"cuda:{local_rank}")

    model = GNNClassifier(in_channels=5).to(device)

    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print(f"✅ Model loaded from {model_file}")
        best_loss, best_acc, best_auc, labels, probs =evaluate(model, val_loader, device)
        print(f"test:  Best model loaded from {model_file} | Best Val Loss: {best_loss:.4f} | Best Val Acc: {best_acc:.4f} | Best Val AUC: {best_auc:.4f}")
        bkg_index = np.where(labels==0)[0]
        sig_index = np.where(labels==1)[0]
        plt.hist(probs[bkg_index], bins=50, alpha=0.5, density=True, label='Background')
        plt.hist(probs[sig_index], bins=50, alpha=0.5, density=True, label='Signal')
        plt.xlim(0, 1)
        plt.legend()
        plt.savefig(f'/home/zhonghua/Filt_Event/figures/GNN_test_model_{version}.png')
        plt.close()
        thresholds=[0.8, 0.9, 0.95, 0.99]
        for threshold in thresholds:    
            print(f'    when threshold={threshold:.2f}: bkg rate={len(np.where(probs[bkg_index]>threshold)[0])/len(bkg_index):.4f}, sig rate={len(np.where(probs[sig_index]>threshold)[0])/len(sig_index):.4f}')

        np.savez(f"/home/zhonghua/Filt_Event/figures/GNN_Test_hist_{version}.npz", bkg_hist=probs[bkg_index], sig_hist=probs[sig_index])
        

if __name__ == "__main__":
    version='1e10_V03'
    data_file=f"/data/zhonghua/Dataset_Filted/ForTrain/combined_filted_{version}_dataset.npz"
    model_file = f'/home/zhonghua/Filt_Event/models/GNN_best_model_{version}_new.pth'
    
    main(data_file, model_file, version)

    test_data_file=f"/data/zhonghua/Dataset_Filted/ForTrain/TEST_combined_filted_{version}_dataset.npz"
    test_model(test_data_file, model_file, version)