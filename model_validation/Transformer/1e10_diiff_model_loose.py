# Transformer_LHAASO_v2.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR
import pandas as pd

class EASDataset(Dataset):
    def __init__(self, signal_file, bkg_file, param_keys, use_hitsM=False):
        """
        Args:
            signal_file: path to signal.npz (Label will be 1)
            bkg_file: path to bkg.npz (Label will be 0)
            param_keys: list of parameter names to use as features.
                        Note: 'label' in this list will be ignored for feature extraction
                        since label is determined by file source.
            use_hitsM: whether to use hitsM data.
        """
        self.use_hitsM = use_hitsM
        
        # --- 1. Load Data ---
        sig_data = np.load(signal_file, allow_pickle=True)
        bkg_data = np.load(bkg_file, allow_pickle=True)
        
        # --- 2. Process Hits (Concatenate) ---
        self.hitsE = np.concatenate([sig_data["hitsE"], bkg_data["hitsE"]], axis=0)
        
        self.hitsM = None
        if use_hitsM:
            # check if hitsM exists in both
            if "hitsM" in sig_data and "hitsM" in bkg_data:
                self.hitsM = np.concatenate([sig_data["hitsM"], bkg_data["hitsM"]], axis=0)
            else:
                print("Warning: use_hitsM=True but 'hitsM' missing in one of the files. Ignoring hitsM.")
                self.use_hitsM = False

        # --- 3. Process Parameters (Feature Alignment) ---
        # 我们只提取 param_keys 中指定的参数，且忽略 "label" 字符串
        # 因为我们现在通过文件来源确定 label
        self.feature_keys = [k for k in param_keys if k != "label"]
        
        # Helper function to extract specific columns by name
        def extract_features(data_source, requested_keys):
            source_names = list(data_source["param_names"])
            source_params = data_source["params"]
            
            # Find indices for the requested keys
            indices = []
            valid_keys = []
            for k in requested_keys:
                if k in source_names:
                    indices.append(source_names.index(k))
                    valid_keys.append(k)
                else:
                    print(f"Warning: Key '{k}' not found in file param_names. Skipping.")
            
            # Extract and return
            if len(indices) == 0:
                return np.zeros((len(source_params), 0), dtype=np.float32)
            
            return source_params[:, indices].astype(np.float32)

        # 分别提取 Signal 和 Bkg 的特征（解决了信号参数更多的问题，因为我们只按需提取）
        sig_features = extract_features(sig_data, self.feature_keys)
        bkg_features = extract_features(bkg_data, self.feature_keys)
        
        # 拼接特征
        self.params = np.concatenate([sig_features, bkg_features], axis=0)
        
        # --- 4. Create Labels ---
        # Signal file -> 1, Bkg file -> 0
        n_sig = len(sig_data["hitsE"])
        n_bkg = len(bkg_data["hitsE"])
        
        sig_labels = np.ones(n_sig, dtype=np.int64)
        bkg_labels = np.zeros(n_bkg, dtype=np.int64)
        self.labels = np.concatenate([sig_labels, bkg_labels], axis=0)

        print(f"Dataset loaded. Signal: {n_sig}, Bkg: {n_bkg}, Total: {n_sig+n_bkg}")
        print(f"Features used: {self.feature_keys}")

    def __len__(self):
        return len(self.hitsE)

    def __getitem__(self, idx):
        # 1. Get Hits
        hits = self.hitsE[idx].astype(np.float32)
        if self.use_hitsM and self.hitsM is not None:
            other = self.hitsM[idx]
            if other is not None and len(other) > 0:
                hits = np.concatenate([hits, other.astype(np.float32)], axis=0)
        
        # 2. Get Params (Features)
        params = self.params[idx] # 已经是处理好的 float32
        
        # 3. Get Label
        label = self.labels[idx]
        
        return hits, params, int(label)

def collate_fn(batch, max_hits=None, pad_value=0.0):
    hits_list, params_list, labels_list = zip(*batch)
    lens = [h.shape[0] for h in hits_list]
    if max_hits is None:
        max_len = max(lens)
    else:
        max_len = min(max(lens), max_hits)
    feat_dim = hits_list[0].shape[1]
    B = len(hits_list)
    padded = torch.full((B, max_len, feat_dim), pad_value, dtype=torch.float32)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i, h in enumerate(hits_list):
        L = min(h.shape[0], max_len)
        padded[i, :L] = torch.from_numpy(h[:L])
        mask[i, :L] = 1
    params = torch.tensor(np.stack([p if p.shape[0] > 0 else np.zeros(1, dtype=np.float32) for p in params_list], axis=0), dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.long)
    # params = params[1:]  # drop original label from params vector
    return padded, mask, params, labels

class PairwiseAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, pair_feat_dim=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_heads)
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask, pair_feat):
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0,2,3,1)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)

        attn_logits = torch.matmul(q, k) * self.scale
        bias = self.pair_mlp(pair_feat).permute(0,3,1,2)
        attn_logits = attn_logits + bias

        key_mask = (~mask).unsqueeze(1).unsqueeze(2)
        attn_logits = attn_logits.masked_fill(key_mask, float("-1e9"))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B, N, D)
        out = self.out_proj(out)

        x = x + out
        x = self.norm1(x)

        x2 = self.ffn(x)
        x = x + x2
        x = self.norm2(x)
        return x

class EdgeConv(nn.Module):
    """局部几何卷积层：捕捉 shower hit 局部空间模式"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x, mask):
        B, N, C = x.shape
        x_i = x.unsqueeze(2).repeat(1, 1, N, 1)
        x_j = x.unsqueeze(1).repeat(1, N, 1, 1)
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)
        edge_feat = self.mlp(edge_feat)
        edge_feat = edge_feat.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        return edge_feat.mean(dim=2)  # 聚合局部信息


class StrongParT(nn.Module):
    def __init__(self, in_dim=5, param_dim=5, embed_dim=128, num_heads=8, num_layers=4, pair_feat_dim=5, dropout=0.3):
        super().__init__()

        # === 基础嵌入 ===
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.pos_mlp = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU())

        # === 局部几何增强层 ===
        self.edge_conv = EdgeConv(embed_dim, embed_dim)

        # === 全局 Pairwise Transformer 层 ===
        self.blocks = nn.ModuleList([
            PairwiseAttentionBlock(embed_dim, num_heads, pair_feat_dim=pair_feat_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # === Param 特征 ===
        self.param_fc = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # === 门控融合机制 ===
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # === 分类头 ===
        self.cls = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, hits, mask, params):
        B, N, _ = hits.shape
        pos = hits[:, :, :2]
        feat = self.embed(hits) + self.pos_mlp(pos)

        # 局部几何强化
        feat = feat + self.edge_conv(feat, mask)

        # === Pairwise features ===
        x, y, r, pe, dt = [hits[:, :, i:i+1] for i in range(5)]
        pair_feat = torch.stack([
            x - x.permute(0,2,1),
            y - y.permute(0,2,1),
            r - r.permute(0,2,1),
            pe - pe.permute(0,2,1),
            dt - dt.permute(0,2,1)
        ], dim=-1)
        pair_feat = pair_feat / (pair_feat.abs().mean(dim=(1,2), keepdim=True) + 1e-6)

        # === Transformer 层 + 多尺度残差 ===
        skip = feat
        for block in self.blocks:
            feat = block(feat, mask, pair_feat)
            feat = feat + skip  # residual bridge

        # === Global pooling ===
        mask_f = mask.unsqueeze(-1).float()
        feat_global = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)

        # === 参数特征 ===
        if params.shape[1] > 0:
            param_feat = self.param_fc(params)
        else:
            param_feat = torch.zeros_like(feat_global)

        # === 门控融合 ===
        gate = self.gate(torch.cat([feat_global, param_feat], dim=1))
        fusion = gate * feat_global + (1 - gate) * param_feat

        # === 分类 ===
        out = torch.cat([fusion, feat_global], dim=1)
        logits = self.cls(out)
        return logits

class ParT_LHAASO_Small(nn.Module):
    def __init__(self, in_dim=5, param_dim=5, embed_dim=64, num_heads=4, num_layers=3,
                 pair_feat_dim=3, dropout=0.3, max_pool=False):
        super().__init__()
        
        # === 1. Embedding 层 ===
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码 MLP (使用 x,y 坐标做位置嵌入)
        # 注意：虽然pairwise特征不用x,y，但点特征(pos)通常还是保留x,y作为空间参考
        # 如果你想把这里的pos也改成r,pe等，请说明。这里暂时保持原逻辑：用前两列(x,y)做位置编码
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # === 2. Pairwise 投影 ===
        # 输入维度应为3 (dr, dpe, ddt)，投影到2维以供AttentionBlock使用（为了轻量化）
        self.pair_proj = nn.Linear(pair_feat_dim, 2)

        # === 3. Transformer Blocks ===
        self.blocks = nn.ModuleList([
            # 注意：这里 pair_feat_dim=2 对应 pair_proj 的输出维度
            PairwiseAttentionBlock(embed_dim, num_heads,
                                   pair_feat_dim=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # === 4. 参数分支 ===
        self.param_fc = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        self.max_pool = max_pool

        # === 5. 分类头 ===
        self.cls = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, hits, mask, params):
        B, N, _ = hits.shape
        
        # 1. Point Feature Embedding
        # 依然使用 x,y (前两列) 作为绝对位置编码输入
        pos = hits[:, :, :2] 
        feat = self.embed(hits) + self.pos_mlp(pos)

        # 2. Pairwise Feature Construction (修改部分)
        # ---------------------------------------------------------
        # 不使用 x, y (idx 0, 1)，改用 r (idx 2), pe (idx 3), dt (idx 4)
        r  = hits[:, :, 2].unsqueeze(2)  # [B, N, 1]
        pe = hits[:, :, 3].unsqueeze(2)  # [B, N, 1]
        dt = hits[:, :, 4].unsqueeze(2)  # [B, N, 1]

        # 计算两两之间的差值 (Broadcasting)
        # shape: [B, N, N] -> [B, N, N, 1]
        dr  = r - r.permute(0, 2, 1)
        dpe = pe - pe.permute(0, 2, 1)
        ddt = dt - dt.permute(0, 2, 1)

        # Normalization (参考 ParT_LHAASO 的逻辑)
        dr  /= (dr.abs().mean(dim=(1, 2), keepdim=True) + 1e-6)
        dpe /= (dpe.abs().mean(dim=(1, 2), keepdim=True) + 1e-6)
        ddt /= (ddt.abs().mean(dim=(1, 2), keepdim=True) + 1e-6)

        # 堆叠特征: [B, N, N, 3]
        pair_feat = torch.stack([dr, dpe, ddt], dim=-1)  
        # ---------------------------------------------------------

        # 3. Projection (3 -> 2)
        pair_feat = self.pair_proj(pair_feat)

        # 4. Transformer Blocks
        for block in self.blocks:
            feat = block(feat, mask, pair_feat)

        # 5. Global Pooling
        mask_f = mask.unsqueeze(-1).float()
        if self.max_pool:
            masked_feat = feat.clone()
            masked_feat[~mask] = float("-1e9")
            feat_global, _ = masked_feat.max(dim=1)
            
            # 安全检查：防止全空事件导致NaN (虽然理论上mask不应全空)
            invalid = (mask.sum(dim=1) == 0)
            if invalid.any():
                 mean_f = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)
                 feat_global[invalid] = mean_f[invalid]
        else:
            feat_global = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)

        # 6. Parameter Fusion
        if params.shape[1] == 0:
            param_feat = torch.zeros((params.shape[0], feat_global.shape[1]), device=feat_global.device)
        else:
            # 简单的维度对齐保护 (可选)
            if params.shape[1] != self.param_fc[0].in_features:
                # 如果实际参数比模型定义的少，补0；如果多，截断
                p_dim = self.param_fc[0].in_features
                if params.shape[1] < p_dim:
                    pad = torch.zeros((params.shape[0], p_dim - params.shape[1]), device=params.device)
                    p_in = torch.cat([params, pad], dim=1)
                else:
                    p_in = params[:, :p_dim]
                param_feat = self.param_fc(p_in)
            else:
                param_feat = self.param_fc(params)

        # 7. Classification
        out = torch.cat([feat_global, param_feat], dim=1)
        logits = self.cls(out)
        return logits


class ParT_LHAASO(nn.Module):
    def __init__(self, in_dim=5, param_dim=5, embed_dim=128, num_heads=8, num_layers=4, pair_feat_dim=3, dropout=0.3, max_pool=True):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.pos_mlp = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU())
        self.blocks = nn.ModuleList([PairwiseAttentionBlock(embed_dim, num_heads, pair_feat_dim=pair_feat_dim, dropout=dropout) for _ in range(num_layers)])
        self.param_fc = nn.Sequential(nn.Linear(param_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.max_pool = max_pool
        self.cls = nn.Sequential(nn.Linear(embed_dim * 2, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, hits, mask, params):
        B, N, _ = hits.shape
        pos = hits[:, :, :2]
        feat = self.embed(hits) + self.pos_mlp(pos)
        x_pos = pos[..., 0].unsqueeze(2)
        y_pos = pos[..., 1].unsqueeze(2)
        r = hits[:, :, 2].unsqueeze(2)
        pe = hits[:, :, 3].unsqueeze(2)
        dt = hits[:, :, 4].unsqueeze(2)

        dx = x_pos - x_pos.permute(0,2,1)
        dy = y_pos - y_pos.permute(0,2,1)
        dr = r - r.permute(0,2,1)
        dpe = pe - pe.permute(0,2,1)
        ddt = dt - dt.permute(0,2,1)
        
        # per-feature normalization
        # dx /= (dx.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        # dy /= (dy.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dr /= (dr.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dpe /= (dpe.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        ddt /= (ddt.abs().mean(dim=(1,2), keepdim=True) + 1e-6)

        # pair_feat = torch.stack([dx, dy, dr, dpe, ddt], dim=-1)  # [B, N, N, 5]
        pair_feat = torch.stack([dr, dpe, ddt], dim=-1)
        for block in self.blocks:
            feat = block(feat, mask, pair_feat)
        mask_f = mask.unsqueeze(-1).float()
        if self.max_pool:
            masked_feat = feat.clone()
            masked_feat[~mask] = float("-1e9")
            feat_global, _ = masked_feat.max(dim=1)
            invalid = (mask.sum(dim=1) == 0)
            if invalid.any():
                mean_f = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)
                feat_global[invalid] = mean_f[invalid]
        else:
            feat_global = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)

        if params.shape[1] == 0:
            param_feat = torch.zeros((params.shape[0], feat_global.shape[1]), device=feat_global.device)
        else:
            if params.shape[1] != self.param_fc[0].in_features:
                p = params
                if p.shape[1] < self.param_fc[0].in_features:
                    pad = torch.zeros((p.shape[0], self.param_fc[0].in_features - p.shape[1]), device=p.device)
                    p = torch.cat([p, pad], dim=1)
                else:
                    p = p[:, :self.param_fc[0].in_features]
                param_feat = self.param_fc(p)
            else:
                param_feat = self.param_fc(params)

        out = torch.cat([feat_global, param_feat], dim=1)
        logits = self.cls(out)
        return logits

def plot_and_save_hist(probs, labels,  out_dir, name,):
    os.makedirs(out_dir, exist_ok=True)
    sig = probs[labels == 1]
    bkg = probs[labels == 0]
    r_sig= len(sig[sig>0.5])/len(sig)
    r_bkg= len(bkg[bkg>0.5])/len(bkg)
    # np.savez(os.path.join(out_dir, f"val_preds_best.npz"), sig=sig, bkg=bkg)
    plt.figure(figsize=(6,4))
    plt.hist(bkg, bins=50, range=(0,1), histtype='step', density=True, label=f'Background: {len(bkg)}')
    plt.hist(sig, bins=50, range=(0,1), histtype='step', density=True, label=f'Monopole: {len(sig)}')
    plt.xlabel("Predicted signal probability")
    plt.ylabel("Probility density")
    plt.title(f">0.5: Signal={r_sig*100:.2f}%, Background={r_bkg*100:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name))
    plt.close()

def plot_Scaled(probs, labels, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    sig_num = 5.84e2
    ga_nums = 1.84e4
    pr_nums = 9.89e4
    bkg_num = ga_nums + pr_nums

    sig = probs[labels == 1]
    bkg = probs[labels == 0]
    bins = 50
    bkg_pdf, bins_edge = np.histogram(bkg, bins=bins, density=True, range=(0,1))
    sig_pdf, _ = np.histogram(sig, bins=bins, density=True, range=(0,1))
    bkg_counts = bkg_pdf * bkg_num * np.diff(bins_edge)
    sig_counts = sig_pdf * sig_num * np.diff(bins_edge)

    # 创建一行两列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 第一个图 - 对数y轴
    ax1.bar(bins_edge[:-1], bkg_counts, width=np.diff(bins_edge), 
            align='edge', alpha=0.7, label='Background')
    ax1.bar(bins_edge[:-1], sig_counts, width=np.diff(bins_edge), 
            align='edge', alpha=0.7, label='Signal',  
            bottom=bkg_counts)
    ax1.set_yscale('log')
    ax1.set_xlabel('Variable')
    ax1.set_ylabel('Counts (log scale)')
    ax1.set_title("Expected Background + Parker's Signal (Log Scale)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 第二个图 - 线性y轴
    ax2.bar(bins_edge[:-1], bkg_counts, width=np.diff(bins_edge), 
            align='edge', alpha=0.7, label='Background')
    ax2.bar(bins_edge[:-1], sig_counts, width=np.diff(bins_edge), 
            align='edge', alpha=0.7, label='Signal',  
            bottom=bkg_counts)
    ax2.set_xlabel('Variable')
    ax2.set_ylabel('Counts (linear scale)')
    ax2.set_title("Expected Background + Parker's Signal (Linear Scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name))
    plt.close()


def warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return -(epoch-warmup_epochs) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: focusing parameter, 常取 1.5~2.5
        alpha: 类别平衡系数 (list或tensor)，例如 [0.5, 0.5] 或自动计算
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = prob of true class
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_and_validate(signal_file,bkg_file,param_keys,use_hitsM=False,epochs=50,batch_size=128,lr=2e-4,device='cuda',max_hits=1024,val_fraction=0.2,
                       out_dir="./transformer_out",patience=6,seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    # === Change: Instantiate with two files ===
    ds = EASDataset(signal_file, bkg_file, param_keys=param_keys, use_hitsM=use_hitsM)
    # ==========================================
    
    N = len(ds)
    n_val = int(N * val_fraction)
    n_train = N - n_val

    indices = torch.randperm(N).tolist()  # 随机打乱索引，这就混合了信号和背景
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds, val_indices)

    # 假设 collate_fn 已经在外部定义好了
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, max_hits=max_hits))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, max_hits=max_hits))
    sample_hits = ds.hitsE[0]
    in_dim = sample_hits.shape[1]
    param_dim =len(param_keys)-1  # params stored exclude label in model earlier; keep >=1
    
    # model = ParT_LHAASO(in_dim=in_dim, param_dim=param_dim, dropout=0.3, max_pool=False)
    # model = StrongParT(in_dim=in_dim, param_dim=param_dim, dropout=0.3)
    model = ParT_LHAASO_Small(in_dim=in_dim, param_dim=param_dim,dropout=0.3,max_pool=False)
    
    # model_path = os.path.join(out_dir, "par_transformer_best.pth")
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = warmup_cosine_schedule(opt, warmup_epochs=5, total_epochs=30, base_lr=lr)
    # scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, )
    # scheduler = CosineAnnealingLR(opt, T_max=30, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = FocalLoss(gamma=2.0)
    
    best_auc = -1.0
    best_loss = 0.6
    bad = 0
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for hits, mask, params, labels in tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}"):
            hits = hits.to(device); mask = mask.to(device); params = params.to(device); labels = labels.to(device)
           
            opt.zero_grad()
            logits = model(hits, mask, params)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
           
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if len(train_losses)>0 else 0.0

        # validation
        model.eval()
        all_probs = []
        all_labels = []
        val_losses = []
        with torch.no_grad():
            for hits, mask, params, labels in tqdm(val_loader, desc="   Validate"):
                hits = hits.to(device); mask = mask.to(device); params = params.to(device); labels = labels.to(device)
                logits = model(hits, mask, params)
                loss = loss_fn(logits, labels)
                # with torch.cuda.amp.autocast():
                #     logits = model(hits, mask, params)
                #     loss = loss_fn(logits, labels)
                val_losses.append(loss.item())
                probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
        probs = np.concatenate(all_probs) if len(all_probs)>0 else np.array([])
        labels = np.concatenate(all_labels) if len(all_labels)>0 else np.array([])
        val_loss = float(np.mean(val_losses)) if len(val_losses)>0 else 0.0
        val_auc = roc_auc_score(labels, probs) if len(np.unique(labels))>1 else 0.5
        print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}/{best_auc:.4f}, val_auc={val_auc:.4f}")

        # save + plot when improved
        # if val_auc > best_auc:
        #     best_auc = val_auc
        if val_loss < best_loss:
            best_loss = val_loss
            bad = 0
            model_path = os.path.join(out_dir, "par_transformer_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"    ✅Saved best model: auc={val_auc} lr={scheduler.get_last_lr()[0]:.3e}")
            # print(f"       path: {model_path}")
            sig = probs[labels == 1]
            bkg = probs[labels == 0]
            np.savez(os.path.join(out_dir, f"val_preds_best.npz"), sig=sig, bkg=bkg, indices=np.array(val_indices))
            plot_and_save_hist(probs, labels, out_dir, name="val_preds_best.png", )
            # plot_Scaled(probs, labels, out_dir, name="val_preds_best_scaled.png", )
            # test_exp_data("/home/zhonghua/data/Dataset_Filted/Experiment/2022/1e10_V03_2022_dataset_optimized.npz",  param_keys, model_path="par_transformer_best.pth",out_dir=out_dir, use_hitsM=use_hitsM, max_hits=256, device=device)
        else:
            bad += 1
            plot_and_save_hist(probs, labels, out_dir, name="val_preds_epoch.png", )
            # plot_Scaled(probs, labels, out_dir, name="val_preds_epoch_scaled.png", )
            print(f"    No improvement (bad={bad}/{patience}), lr={scheduler.get_last_lr()[0]:.3e}")

        if bad >= patience:
            print("Early stopping triggered.")
            break

        # scheduler.step(val_loss)
        scheduler.step()

    return model

def test_exp_data(npz_file, param_keys, model_path="par_transformer_best.pth",
                  out_dir="./transformer", use_hitsM=False, max_hits=1024, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)

    # === 1. 加载实验数据 ===
    ds = EASDataset(npz_file, use_hitsM=use_hitsM, param_keys=param_keys)
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, max_hits=max_hits))

    # === 2. 构建模型结构 ===
    sample_hits = ds.hitsE[0]
    in_dim = sample_hits.shape[1]
    param_dim = max(1, len(ds.param_index) - 1)

    # model = ParT_LHAASO(in_dim=in_dim, param_dim=param_dim,)
    # model = StrongParT(in_dim=in_dim, param_dim=param_dim,embed_dim=128, num_heads=8, num_layers=4,pair_feat_dim=5, dropout=0.3)
    model = ParT_LHAASO_Small(in_dim=in_dim, param_dim=param_dim,)
    
    # if torch.cuda.device_count() > 1:
    #     # print(f"    ✅ Using {torch.cuda.device_count()} GPUs for inference.")
    #     model = torch.nn.DataParallel(model)

    # === 3. 加载训练好的权重 ===
    model_path = os.path.join(out_dir, model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    # === 4. 前向推理 ===
    all_probs = []
    all_indices = []  

    with torch.no_grad():
        for batch_idx, (hits, mask, params, _) in enumerate(tqdm(loader, desc="   EXP Test:")):
            hits = hits.to(device); mask = mask.to(device); params = params.to(device)
            logits = model(hits, mask, params)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

            # 保存全局 index（DataLoader 保留了顺序，所以可以用偏移量恢复）
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + len(probs)
            indices = np.arange(start_idx, end_idx)
            all_indices.append(indices)

    all_probs = np.concatenate(all_probs)
    all_indices = np.concatenate(all_indices)

    # === 5. 保存为 npz 文件 ===
    # save_path = os.path.join(out_dir, "exp_probs_little.npz")
    # save_path = os.path.join(out_dir, "exp_probs_no_muon.npz")
    save_path = os.path.join(out_dir, "exp_probs_optimized.npz")
    np.savez(save_path, probs=all_probs, indices=all_indices)

    # === 6. 加载训练集对比用概率 ===
    train_best_probs = np.load(os.path.join(out_dir, "val_preds_best.npz"))
    train_sig = train_best_probs["sig"]
    train_bkg = train_best_probs["bkg"]

    # === 7. 绘制分布图 ===
    plt.figure(figsize=(6, 4))
    plt.hist(all_probs, bins=50, range=(0, 1), color="steelblue", alpha=0.7, density=True,
             label=f"Experiment: {len(all_probs)}")
    plt.hist(train_bkg, bins=50, range=(0, 1), histtype="step", density=True,
             label=f"Background: {len(train_bkg)}")
    plt.hist(train_sig, bins=50, range=(0, 1), histtype="step", density=True,
             label=f"Monopole: {len(train_sig)}")
    plt.xlabel("Model Output Probability")
    plt.ylabel("Density")
    plt.title("Experimental Data Probability Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # fig_path = os.path.join(out_dir, "exp_probs_distribution_little_muon.png")
    fig_path = os.path.join(out_dir, "EF_loose.png")
    # fig_path = os.path.join(out_dir, "diff_model_exp_probs_distribution_loose.png")
    plt.savefig(fig_path)
    plt.close()


    return all_probs, all_indices

def test_diff_models(model_config_path, 
                     exp_files_dict, 
                     param_keys, 
                     out_dir="./comparison_results", 
                     use_hitsM=False, 
                     max_hits=1024, 
                     device='cuda'):
    """
    Args:
        model_config_path: 训练好的模型权重路径 (.pth)
        exp_files_dict: 字典，格式 {"ModelName": "path/to/npz"}
        param_keys: 模型训练时使用的参数列表
        out_dir: 输出目录
    """
    class InferenceDataset(Dataset):
        def __init__(self, npz_file, param_keys, use_hitsM=False):
            """
            专门用于单文件推理的 Dataset
            """
            data = np.load(npz_file, allow_pickle=True)
            self.hitsE = data["hitsE"]
            
            # 处理 HitsM
            self.hitsM = None
            self.use_hitsM = use_hitsM
            if use_hitsM and "hitsM" in data:
                self.hitsM = data["hitsM"]
            
            # --- 提取模型需要的特征 (Feature Extraction) ---
            # 逻辑：只提取 param_keys 里的列作为网络输入
            source_names = list(data["param_names"])
            source_params = data["params"]
            
            # 保存原始参数和名称，用于最后生成 CSV
            self.raw_params = source_params 
            self.raw_param_names = source_names
            
            # 找到模型所需特征的索引
            self.feature_keys = [k for k in param_keys if k != "label"] # 排除label
            idxs = []
            for k in self.feature_keys:
                if k in source_names:
                    idxs.append(source_names.index(k))
                else:
                    # 如果缺少某个特征，可能需要报错或填0，这里简单打印警告
                    print(f"Warning: Key '{k}' missing in inference file {npz_file}")
            
            if len(idxs) > 0:
                self.features = source_params[:, idxs].astype(np.float32)
            else:
                self.features = np.zeros((len(source_params), 0), dtype=np.float32)
                
        def __len__(self):
            return len(self.hitsE)

        def __getitem__(self, idx):
            # 1. Hits
            hits = self.hitsE[idx].astype(np.float32)
            if self.use_hitsM and self.hitsM is not None:
                other = self.hitsM[idx]
                if other is not None and len(other) > 0:
                    hits = np.concatenate([hits, other.astype(np.float32)], axis=0)
            
            # 2. Params (Network Input)
            params = self.features[idx]
            
            # 3. Dummy Label (为了兼容 collate_fn)
            label = -1 
            
            return hits, params, int(label)
    os.makedirs(out_dir, exist_ok=True)
    
    # === 1. 初始化模型 (只初始化一次) ===
    # 为了确定输入维度，先临时读取第一个文件的一个样本
    first_key = list(exp_files_dict.keys())[0]
    temp_ds = InferenceDataset(exp_files_dict[first_key], param_keys, use_hitsM)
    sample_hits = temp_ds.hitsE[0]
    
    in_dim = sample_hits.shape[1]
    # param_dim 是输入给网络的特征数量
    param_dim = temp_ds.features.shape[1] 
    
    print(f"Model Input Dims -> Hit Feats: {in_dim}, Scalar Feats: {param_dim}")

    # --- 这里实例化你的模型 ---
    # 请确保这里的模型类定义与训练时一致
    model = ParT_LHAASO_Small(in_dim=in_dim, param_dim=param_dim)
    # model = ParT_LHAASO(in_dim=in_dim, param_dim=param_dim,)
    # model = StrongParT(...) 
    
    # 加载权重
    print(f"Loading model weights from: {model_config_path}")
    model.load_state_dict(torch.load(model_config_path, map_location=device))
    model.to(device)
    model.eval()

    def plot(log=False):
        # === 1. 准备绘图布局 ===
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
        
        colors = ['blue', 'green', 'orange', 'purple', 'cyan']
        
        # 定义统一的 bins，为了计算比值，必须锁定 bins
        bins = np.linspace(0, 1, 21) 
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0] # 计算 bin 宽度
        
        # === 优先加载 Validation Reference (基准) ===
        val_pred_path = os.path.join(os.path.dirname(model_config_path), "val_preds_best.npz")
        
        # 初始化分母占位符
        hist_val_bkg_density = None
        hist_val_bkg_safe = None
        
        if os.path.exists(val_pred_path):
            print(f"\nLoading validation reference from: {val_pred_path}")
            val_data = np.load(val_pred_path)
            # 假设 val_bkg 是背景事件的模型分数数组
            val_bkg = val_data["bkg"] 
            val_sig = val_data["sig"]
            # 1.1. 计算 Background 的【原始计数】和【总计数】
            hist_val_bkg_count, _ = np.histogram(val_bkg, bins=bins, density=False) # density=False: 原始计数
            total_val_bkg_count = len(val_bkg)
            
            # 1.2. 计算 Background 的【归一化密度】
            hist_val_bkg_density = hist_val_bkg_count / (total_val_bkg_count * bin_width)
            
            # 1.3. 计算归一化后的【泊松误差】
            # 原始误差 sigma_count = sqrt(Count)
            # 归一化后的误差 sigma_density = sigma_count / (total_count * bin_width)
            # 为了防止除以0，在 total_count=0 或 count=0 时，误差为0
            sigma_count = np.sqrt(hist_val_bkg_count)
            # 这里的 total_val_bkg_count 应该是大于 0 的，否则整个 hist 都是 0，下面的代码会失败
            if total_val_bkg_count > 0:
                sigma_density = sigma_count / (total_val_bkg_count * bin_width)
            else:
                sigma_density = np.zeros_like(hist_val_bkg_count)

            # 1.4. 创建 safe 分母用于比值图
            hist_val_bkg_safe = hist_val_bkg_density.copy()
            hist_val_bkg_safe[hist_val_bkg_safe == 0] = np.nan 

            # 1.5. 绘制 Background 主图 (使用误差条和 step 风格)
            ax_main.errorbar(bin_centers, hist_val_bkg_density, yerr=sigma_density, 
                             fmt='o', color='black', markersize=4, capsize=3,
                             )
            
            # 同时使用 histtype='step' 绘制连接线
            ax_main.hist(bins[:-1], bins, weights=hist_val_bkg_density, 
                         histtype='step', color='black', linestyle='--', linewidth=2, label=f"Val Background (Base)")
            
            # Ratio Plot: 背景除以背景 = 1 (画一条参考线)
            ax_ratio.axhline(1, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

            # 2. 计算并绘制 Signal (保持原样，可根据需要添加误差)
            hist_val_sig, _ = np.histogram(val_sig, bins=bins, density=True)
            ax_main.hist(bins[:-1], bins, weights=hist_val_sig, 
                        histtype='step', color='red', linestyle='--', linewidth=2,
                        label=f"Val Signal")
            
            ratio_sig = hist_val_sig / hist_val_bkg_safe
            ax_ratio.plot(bin_centers, ratio_sig, color='red', linestyle='--', linewidth=1, alpha=0.5)

        else:
            print(f"Warning: Validation reference file not found at {val_pred_path}")
            hist_val_bkg_safe = None

        # === 3. 循环处理每个实验文件 ===
        inference_results = {} 

        for i, (exp_name, npz_path) in enumerate(exp_files_dict.items()):
            print(f"\nProcessing {exp_name} -> {npz_path} ...")
            
            ds = InferenceDataset(npz_path, param_keys=param_keys, use_hitsM=use_hitsM)
            loader = DataLoader(ds, batch_size=128, shuffle=False,
                                collate_fn=lambda b: collate_fn(b, max_hits=max_hits),
                                num_workers=4)
            
            all_probs = []
            with torch.no_grad():
                for hits, mask, params, _ in tqdm(loader, desc=f"   Infer {exp_name}"):
                    hits = hits.to(device); mask = mask.to(device); params = params.to(device)
                    logits = model(hits, mask, params)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.append(probs)
            
            all_probs = np.concatenate(all_probs)
            inference_results[exp_name] = all_probs
            
            # --- 保存 CSV (保持原逻辑) ---
            raw_df = pd.DataFrame(ds.raw_params, columns=ds.raw_param_names)
            raw_df['model_score'] = all_probs
            csv_name = f"{exp_name}_predictions.csv"
            raw_df.to_csv(os.path.join(out_dir, csv_name), index=False)

            # === 4. 添加到主分布图 (ax_main) ===
            # 计算当前数据的直方图
            hist_exp, _ = np.histogram(all_probs, bins=bins, density=True)
            
            curr_color = colors[i % len(colors)]
            
            # 绘制主图 (Fill + Step)
            ax_main.hist(bins[:-1], bins, weights=hist_exp,
                        histtype='stepfilled', alpha=0.3, color=curr_color,
                        label=f"{exp_name}")
            ax_main.hist(bins[:-1], bins, weights=hist_exp,
                        histtype='step', linewidth=1.5, color=curr_color)
            
            # === 5. 添加到比值图 (ax_ratio) ===
            if hist_val_bkg_safe is not None:
                # 计算比值: Exp / Val_Background
                ratio = hist_exp / hist_val_bkg_safe
                
                # 绘制比值 (通常用 step 或 plot)
                # where='mid' 表示点在台阶中间
                ax_ratio.step(bin_centers, ratio, where='mid', color=curr_color, linewidth=1.5)
                # 或者用 scatter 画点：
                # ax_ratio.plot(bin_centers, ratio, '.', color=curr_color, markersize=4)

        # === 6. 设置绘图细节 ===
        
        # --- Main Plot Settings ---
        ax_main.set_ylabel("Density")
        ax_main.set_title("Score Distribution & Ratio to Background")
        ax_main.legend(loc='upper center', frameon=False, ncol=2) # 图例分两列更紧凑
        ax_main.grid(alpha=0.3)
        # 隐藏主图 x 轴刻度标签 (因为共享了 x 轴)
        plt.setp(ax_main.get_xticklabels(), visible=False) 

        if log:
            ax_main.set_yscale('log')
            fig_name = "comparison_distribution_ratio_log.png"
            # Log下通常需要设置最小y值，防止看到负无穷的噪音
            # ax_main.set_ylim(bottom=1e-4) 
        else:
            fig_name = "comparison_distribution_ratio_linear.png"

        # --- Ratio Plot Settings ---
        ax_ratio.set_ylabel("Ratio / Val Bkg")
        ax_ratio.set_xlabel("Model Output Score (Signal Probability)")
        ax_ratio.set_ylim(0, 5) # 设置比值图的 Y 轴范围，通常 0 到 2 或 3 足够，视偏差大小调整
        ax_ratio.grid(alpha=0.3, axis='y') # 主要看横线
        ax_ratio.grid(alpha=0.3, axis='x')

        # 保存
        plt.savefig(os.path.join(out_dir, fig_name), dpi=300, bbox_inches='tight')
        plt.close()
    plot(log=False)
    plot(log=True)
    
    # print(f"\nAll tasks finished. Plot saved to {os.path.join(out_dir, fig_name)}")
# /home/zhonghua/miniconda3/bin/python /home/zhonghua/Filt_Event/model_validation/Transformer/1e10_diiff_model_loose.py
if __name__ == "__main__":
    # configure paths and keys here "recE",

    param_keys = ["label", "R_mean", "Eage",  "rec_theta", "rec_phi"]
    # param_keys = ["label", "recE", "rec_theta", "rec_phi"]
    use_hitsM = False
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    out_dir = "/home/zhonghua/Filt_Event/model_validation/Transformer/1e10_QG_based_loose"
    signal_file = "/home/zhonghua/data/Dataset_Filted/filted_Monopole_1e10_merged_1000_70_new_dataset_loose.npz"
    bkg_file = "/home/zhonghua/data/Dataset_Filted/1e10_V03/Proton_all_combined_filted_1e10_V03_12_16_dataset_loose.npz"
    os.makedirs(out_dir, exist_ok=True)
    model = train_and_validate(signal_file,bkg_file, param_keys, use_hitsM=use_hitsM, epochs=200, batch_size=64, lr=1e-4,
                               device=device, max_hits=256, val_fraction=0.2, out_dir=out_dir, patience=100, seed=42)

 
    # 2. 放到字典里，键名会作为图例 label 和 csv 文件名前缀
    files_to_test = {
        "EF_Proton": "/home/zhonghua/data/Dataset_Filted/CosmicRay/Npz/0818_filted_EF_Proton_4e13_1e15_dataset_loose.npz",
        "QF_Proton":  "/home/zhonghua/data/Dataset_Filted/CosmicRay/Npz/0818_filted_QF_Proton_4e13_1e15_dataset_loose.npz",
        # "QG_Proton": "/home/zhonghua/data/Dataset_Filted/CosmicRay/Npz/Proton_1000_70_1e10_V03_dataset_loose.npz",
        "2022":"/home/zhonghua/data/Dataset_Filted/Experiment/2022/all_combined_2022_dataset_1e10_for_diffmodels_loose.npz",
    }
    # files_to_test = {
    #     "EF_Proton": "/home/zhonghua/data/Dataset_Filted/CosmicRay/Npz/0818_filted_EF_Proton_4e13_1e15_dataset_strict.npz",
    #     "QF_Proton":  "/home/zhonghua/data/Dataset_Filted/CosmicRay/Npz/0818_filted_QF_Proton_4e13_1e15_dataset_strict.npz",
    #     "QG_Proton": "/home/zhonghua/data/Dataset_Filted/CosmicRay/Npz/Proton_1000_70_1e10_V03_dataset_strict.npz",
    #     "2022":"/home/zhonghua/data/Dataset_Filted/Experiment/2022/all_combined_2022_dataset_1e10_for_diffmodels_strict.npz",
    # }
    
    # 4. 模型路径
    model_dir = out_dir
    model_path = os.path.join(model_dir, "par_transformer_best.pth")
    
    # 5. 运行
    test_diff_models(
        model_config_path=model_path,
        exp_files_dict=files_to_test,
        param_keys=param_keys,
        out_dir=os.path.join(model_dir, "model_comparison_output"),
        use_hitsM=False,
        max_hits=256, # 确保与训练时一致
        device="cuda:3"
    )