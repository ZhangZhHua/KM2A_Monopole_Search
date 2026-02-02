# Transformer_LHAASO_v2.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler 
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP                # <<< modified >>>
from torch.cuda.amp import autocast, GradScaler                             # <<< modified >>>
import torch.multiprocessing as mp                                          # <<< modified >>>
from torch.utils.data import Subset
class EASDataset(Dataset):
    def __init__(self, npz_file, use_hitsM=False, param_keys=None):
        data = np.load(npz_file, allow_pickle=True)
        self.hitsE = data["hitsE"]
        self.hitsM = data["hitsM"] if ("hitsM" in data and use_hitsM) else None
        self.params = data["params"]
        self.param_names = list(data["param_names"])
        if param_keys is None:
            param_keys = self.param_names
        # build param indices robustly: only keep keys that exist
        self.param_keys = [k for k in param_keys if k in self.param_names]
        self.param_index = [self.param_names.index(k) for k in self.param_keys]
        self.use_hitsM = use_hitsM

    def __len__(self):
        return len(self.hitsE)

    def __getitem__(self, idx):
        hits = self.hitsE[idx].astype(np.float32)
        if self.use_hitsM and self.hitsM is not None:
            other = self.hitsM[idx]
            if other is not None and len(other) > 0:
                hits = np.concatenate([hits, other.astype(np.float32)], axis=0)
        params_all = np.array(self.params[idx], dtype=np.float32)
        # be defensive: if param vector shorter than param_index, clip indices
        if np.max(self.param_index) >= params_all.shape[0]:
            idxs = [i for i in self.param_index if i < params_all.shape[0]]
        else:
            idxs = self.param_index
        params = params_all[idxs].astype(np.float32)
        # original label expected to be first param in param_names; fallback to params_all[0]
        label_raw = int(params_all[0]) if params_all.shape[0] > 0 else 0
        label = 1 if label_raw == 43 else 0
        params = params[1:]
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




class StrongParT(nn.Module):
    def __init__(self, 
                 in_dim=5, 
                 param_dim=5, 
                 embed_dim=256,     # <--- MODIFIED: 增加了 embedding 维度
                 num_heads=8, 
                 num_layers=6,      # <--- MODIFIED: 增加了层数
                 pair_feat_dim=6,   # <--- MODIFIED: 增加了欧氏距离特征 (dx, dy, dist, dr, dpe, ddt)
                 dropout=0.4, 
                 max_pool=True):     # max_pool 参数现在被忽略，我们总是用 max+mean
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.pos_mlp = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU())
        
        # <--- MODIFIED: 确保这里传入了正确的 pair_feat_dim
        self.blocks = nn.ModuleList([
            PairwiseAttentionBlock(embed_dim, num_heads, pair_feat_dim=pair_feat_dim, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        self.param_fc = nn.Sequential(nn.Linear(param_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())
        
        # <--- MODIFIED: 分类头现在接收 max_pool + mean_pool + param_feat
        # (embed_dim * 3)
        self.cls = nn.Sequential(
            nn.Linear(embed_dim * 3, 512), # <--- MODIFIED: 输入维度 * 3
            nn.GELU(), 
            nn.Dropout(0.3), 
            nn.Linear(512, 256),           # <--- MODIFIED: 增大了中间层
            nn.GELU(), 
            nn.Dropout(0.3), 
            nn.Linear(256, 2)              # <--- MODIFIED: 128 -> 256
        )

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
        
        # <--- MODIFIED: 增加欧氏距离 (Euclidean distance)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-6)
        
        # per-feature normalization
        dx /= (dx.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dy /= (dy.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dr /= (dr.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dpe /= (dpe.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        ddt /= (ddt.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dist /= (dist.mean(dim=(1,2), keepdim=True) + 1e-6) # <--- MODIFIED: 也归一化

        # <--- MODIFIED: 堆叠6个特征
        pair_feat = torch.stack([dx, dy, dist, dr, dpe, ddt], dim=-1)  # [B, N, N, 6]

        for block in self.blocks:
            feat = block(feat, mask, pair_feat)
            
        mask_f = mask.unsqueeze(-1).float()
        
        # --- MODIFIED: 总是同时计算 Max 和 Mean Pooling ---
        
        # 1. Mean Pooling (safe)
        mean_feat = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)
        
        # 2. Max Pooling (safe)
        masked_feat = feat.clone()
        masked_feat[~mask] = -1e9 # 用负无穷来 mask
        max_feat, _ = masked_feat.max(dim=1)
        
        # 3. 处理没有 hit 的空事件
        invalid = (mask.sum(dim=1) == 0)
        if invalid.any():
            max_feat[invalid] = mean_feat[invalid] # 在空事件时，max_feat 回退到 mean_feat (全0)
        
        # --- End Modification ---

        # (param_feat 的逻辑保持不变)
        if params.shape[1] == 0:
            param_feat = torch.zeros((params.shape[0], self.param_fc[0].out_features), device=feat.device)
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

        # <--- MODIFIED: 串联 max, mean, 和 param 特征
        out = torch.cat([max_feat, mean_feat, param_feat], dim=1)
        logits = self.cls(out)
        return logits

class ParT_LHAASO_Small(nn.Module):
    def __init__(self, in_dim=5, param_dim=5, embed_dim=64, num_heads=4, num_layers=3,
                 pair_feat_dim=3, dropout=0.3, max_pool=False):
        super().__init__()
        # embedding层轻量化 + dropout
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 用3个pairwise特征（dx, dy, dr）
        self.pair_proj = nn.Linear(pair_feat_dim, 2)

        self.blocks = nn.ModuleList([
            PairwiseAttentionBlock(embed_dim, num_heads,
                                   pair_feat_dim=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 参数分支减小深度
        self.param_fc = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        self.max_pool = max_pool

        # 分类头更轻
        self.cls = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, hits, mask, params):
        B, N, _ = hits.shape
        pos = hits[:, :, :2]
        feat = self.embed(hits) + self.pos_mlp(pos)

        # pairwise简化
        x = hits[:, :, 0].unsqueeze(2)
        y = hits[:, :, 1].unsqueeze(2)
        r = hits[:, :, 2].unsqueeze(2)

        dx = x - x.permute(0, 2, 1)
        dy = y - y.permute(0, 2, 1)
        dr = r - r.permute(0, 2, 1)

        # normalization
        for d in [dx, dy, dr]:
            d /= (d.abs().mean(dim=(1, 2), keepdim=True) + 1e-6)

        pair_feat = torch.stack([dx, dy, dr], dim=-1)  # [B, N, N, 3]
        pair_feat = self.pair_proj(pair_feat)

        # attention block
        for block in self.blocks:
            feat = block(feat, mask, pair_feat)

        mask_f = mask.unsqueeze(-1).float()
        if self.max_pool:
            masked_feat = feat.clone()
            masked_feat[~mask] = float("-1e9")
            feat_global, _ = masked_feat.max(dim=1)
        else:
            feat_global = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)

        # param
        if params.shape[1] == 0:
            param_feat = torch.zeros((params.shape[0], feat_global.shape[1]), device=feat_global.device)
        else:
            param_feat = self.param_fc(params)

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

        # dx = x_pos - x_pos.permute(0,2,1)
        # dy = y_pos - y_pos.permute(0,2,1)
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
        pair_feat = torch.stack([ dr, dpe, ddt], dim=-1)  # [B, N, N, 5]
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
            return (warmup_epochs-epoch)*2 / warmup_epochs
        else:
            progress = ((epoch-warmup_epochs) % total_epochs) / (total_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def adaptive_cosine_schedule(optimizer, total_epochs, base_lr, 
                             warmup_epochs=5, 
                             high_lr_factor=2,
                             num_cos_cycles=2, 
                             steady_epochs=10):
    """
    自定义学习率调度：
    1. 前 warmup_epochs 用较高常量学习率
    2. 接着 num_cos_cycles 次 cosine 衰减（每个周期总长为 total_epochs / num_cos_cycles）
    3. 最后 steady_epochs 保持中等学习率再进入新一轮

    Args:
        optimizer: torch optimizer
        total_epochs: 总训练轮数
        base_lr: 基础学习率
        warmup_epochs: 热启动阶段
        high_lr_factor: 热启动阶段的倍率
        num_cos_cycles: cosine下降的轮数
        steady_epochs: 每次cos之后保持高学习率的轮数
    """
    
    # 计算单个cos周期长度
    cosine_period = (total_epochs - warmup_epochs - steady_epochs) // num_cos_cycles
    total_cycle_len = cosine_period * num_cos_cycles + steady_epochs + warmup_epochs

    def lr_lambda(epoch):
        # ===== 阶段1：warmup阶段（高LR恒定） =====
        if epoch < warmup_epochs:
            return high_lr_factor

        # ===== 阶段2：cosine 衰减循环 =====
        elif epoch < warmup_epochs + cosine_period * num_cos_cycles:
            progress = (epoch - warmup_epochs) % cosine_period / cosine_period
            return 0.5 * (1.0 + np.cos(np.pi * progress))  # 从1→0平滑下降
        
        # ===== 阶段3：steady 阶段（中等LR恒定） =====
        elif epoch < total_cycle_len:
            return 1  # 稍微降低的稳定学习率
        
        # ===== 之后可自动重复循环 =====
        else:
            phase_epoch = (epoch - total_cycle_len) % total_cycle_len
            return lr_lambda(phase_epoch)

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



def test_exp_data(npz_file, param_keys, model_path="par_transformer_best.pth",
                  out_dir="./transformer", use_hitsM=False, max_hits=1024):
    # === 初始化 DDP 环境 ===
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    os.makedirs(out_dir, exist_ok=True)

    # === 1. 加载实验数据 ===
    ds = EASDataset(npz_file, use_hitsM=use_hitsM, param_keys=param_keys)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(ds, batch_size=128, sampler=sampler,
                        collate_fn=lambda b: collate_fn(b, max_hits=max_hits))

    # === 2. 构建模型 ===
    sample_hits = ds.hitsE[0]
    in_dim = sample_hits.shape[1]
    param_dim = max(1, len(ds.param_index) - 1)
    model = ParT_LHAASO(in_dim=in_dim, param_dim=param_dim)
    # model = ParT_LHAASO_Small(in_dim=in_dim, param_dim=param_dim)
    # === 3. 加载权重 ===
    model_path = os.path.join(out_dir, model_path)
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # === 4. 每个 rank 推理一部分 ===
    local_probs, local_indices = [], []
    with torch.no_grad():
        for batch_idx, (hits, mask, params, _) in enumerate(tqdm(loader, desc="   EXP Test:")):
            hits = hits.to(device); mask = mask.to(device); params = params.to(device)
            logits = model(hits, mask, params)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            local_probs.append(probs)

            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + len(probs)
            local_indices.append(np.arange(start_idx, end_idx))

    local_probs = np.concatenate(local_probs)
    local_indices = np.concatenate(local_indices)

    # === 5. 收集所有 GPU 的结果 ===
    local_probs_t = torch.tensor(local_probs, device=device)
    local_indices_t = torch.tensor(local_indices, device=device)

    gather_probs = [torch.zeros_like(local_probs_t) for _ in range(world_size)]
    gather_indices = [torch.zeros_like(local_indices_t) for _ in range(world_size)]

    dist.all_gather(gather_probs, local_probs_t)
    dist.all_gather(gather_indices, local_indices_t)

    # === 6. rank 0 保存结果 ===
    if rank == 0:
        all_probs = torch.cat(gather_probs).cpu().numpy()
        all_indices = torch.cat(gather_indices).cpu().numpy()
        save_path = os.path.join(out_dir, "exp_probs_optimized.npz")
        np.savez(save_path, probs=all_probs, indices=all_indices)

        # 绘制分布图（只在 rank0 画）
        train_best_probs = np.load(os.path.join(out_dir, "val_preds_best.npz"))
        train_sig = train_best_probs["sig"]
        train_bkg = train_best_probs["bkg"]

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
        plt.savefig(os.path.join(out_dir, "exp_probs_distribution_optimized.png"))
        plt.close()

    dist.barrier()
    dist.destroy_process_group()

from functools import partial

def train_and_validate_ddp(rank, world_size, npz_file, param_keys,use_hitsM=False,epochs=50,batch_size=128,lr=2e-4,
                       device='cuda',max_hits=1024,val_fraction=0.2,
                       out_dir="./transformer_out",patience=6,seed=42):
    # <<< modified >>>
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if rank == 0: os.makedirs(out_dir, exist_ok=True)

    ds = EASDataset(npz_file, use_hitsM=use_hitsM, param_keys=param_keys)
    N = len(ds)
    n_val = int(N * val_fraction)
    n_train = N - n_val
    # 手动分割并保存索引
    indices = torch.randperm(N).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # 子集
    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)

    # <<< modified >>> use distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,collate_fn=partial(collate_fn, max_hits=max_hits),)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        collate_fn=partial(collate_fn, max_hits=max_hits),
    )
    sample_hits = ds.hitsE[0]
    in_dim = sample_hits.shape[1]
    param_dim = max(1, len(ds.param_index)-1)
    # model = StrongParT(in_dim=in_dim, param_dim=param_dim,).to(device)
    model = ParT_LHAASO(in_dim=in_dim, param_dim=param_dim, dropout=0.3).to(device)
    # model = ParT_LHAASO_Small(in_dim=in_dim, param_dim=param_dim,dropout=0.1).to(device)

    # <<< modified >>> wrap model
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = warmup_cosine_schedule(opt, warmup_epochs=5, total_epochs=20, base_lr=lr)
    # scheduler = adaptive_cosine_schedule(
    #     opt, 
    #     total_epochs=20, 
    #     base_lr=1e-3,
    #     warmup_epochs=5,
    #     high_lr_factor=1.5,
    #     num_cos_cycles=2,
    #     steady_epochs=10
    # )
    # loss_fn = FocalLoss(gamma=2.0)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = 0.6
    bad = 0
    for epoch in range(1, epochs+1):
        train_sampler.set_epoch(epoch)         # <<< modified >>>
        model.train()
        train_losses = []
        for hits, mask, params, labels in tqdm(train_loader, disable=(rank!=0), desc=f"Train epoch {epoch}/{epochs}"):
            hits = hits.to(device); mask = mask.to(device); params = params.to(device); labels = labels.to(device)
            opt.zero_grad()

            # <<< modified >>> mixed precision
            logits = model(hits, mask, params)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if len(train_losses)>0 else 0.0

        # validation only on rank 0
        model.eval()
        all_probs = []; all_labels = []; val_losses = []
        all_indices = []   # collect original dataset indices for this rank

        with torch.no_grad():
            for batch_idx, (hits, mask, params, labels) in enumerate(tqdm(val_loader, desc="   Validate", disable=(rank!=0))):
                hits = hits.to(device); mask = mask.to(device); params = params.to(device); labels = labels.to(device)

                logits = model(hits, mask, params)
                loss = loss_fn(logits, labels)
                val_losses.append(loss.item())

                probs_np = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()    # CPU numpy
                labels_np = labels.cpu().numpy()

                all_probs.append(probs_np)
                all_labels.append(labels_np)

                # get original indices from Subset (val_ds is Subset)
                # Note: val_loader yields samples in the order given by val_sampler.
                # val_ds.indices is the list of original indices for this Subset.
                # We must know which slice of val_ds.indices corresponds to this batch.
                # DistributedSampler yields contiguous batches within each epoch; DataLoader will follow sampler order.
                start = batch_idx * val_loader.batch_size
                end = start + len(probs_np)
                batch_indices = np.array(val_ds.indices[start:end], dtype=np.int64)
                all_indices.append(batch_indices)

        # concat per-rank results to numpy arrays (still per-rank)
        if len(all_probs) > 0:
            probs_local = np.concatenate(all_probs)
            labels_local = np.concatenate(all_labels)
            indices_local = np.concatenate(all_indices)
        else:
            probs_local = np.array([], dtype=np.float32)
            labels_local = np.array([], dtype=np.int64)
            indices_local = np.array([], dtype=np.int64)

        # ---------- distributed gather using all_gather_object (works for variable-length lists) ----------
        # prepare python objects to gather
        probs_list = [None]
        labels_list = [None]
        indices_list = [None]

        # make containers to receive from all ranks on each process (must be same shape)
        gathered_probs = [None for _ in range(world_size)]
        gathered_labels = [None for _ in range(world_size)]
        gathered_indices = [None for _ in range(world_size)]

        # all_gather_object collects arbitrary Python objects (no GPU allocation)
        dist.all_gather_object(gathered_probs, probs_local)
        dist.all_gather_object(gathered_labels, labels_local)
        dist.all_gather_object(gathered_indices, indices_local)

        # only rank 0 merges and computes metrics / saving
        if rank == 0:
            # concatenate in rank order
            probs = np.concatenate(gathered_probs) if len(gathered_probs) > 0 else np.array([], dtype=np.float32)
            labels = np.concatenate(gathered_labels) if len(gathered_labels) > 0 else np.array([], dtype=np.int64)
            indices = np.concatenate(gathered_indices) if len(gathered_indices) > 0 else np.array([], dtype=np.int64)
        else:
            probs = np.array([], dtype=np.float32)
            labels = np.array([], dtype=np.int64)
            indices = np.array([], dtype=np.int64)

        # compute val_loss and AUC using only local val_losses (optionally we could gather/average val_loss)
        val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0

        # rank 0 prints and saves
        if rank == 0:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
            print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}/{best_loss:.4f}, val_auc={val_auc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                bad = 0
                model_path = os.path.join(out_dir, "par_transformer_best.pth")
                torch.save(model.module.state_dict(), model_path)
                print(f"    ✅Saved best model: auc={val_auc}")
                sig = probs[labels == 1]; bkg = probs[labels == 0]
                np.savez(os.path.join(out_dir, f"val_preds_best.npz"), sig=sig, bkg=bkg, indices=indices)
                plot_and_save_hist(probs, labels, out_dir, name="val_preds_best.png")
            else:
                bad += 1
                plot_and_save_hist(probs, labels, out_dir, name="val_preds_epoch.png")
                print(f"    No improvement (bad={bad}/{patience}, lr={scheduler.get_last_lr()[0]:.2e})")

        stop_tensor = torch.tensor(0, device=device, dtype=torch.uint8)
        if rank == 0 and bad >= patience:        # <<< modified >>>
            stop_tensor.fill_(1)

        # 把 stop 信号广播给所有进程（所有进程都要调用）
        dist.broadcast(stop_tensor, src=0)       # <<< modified >>>

        if stop_tensor.item() == 1:
            if rank == 0:
                print("Early stopping triggered (distributed).")
            break

        scheduler.step()

    torch.distributed.destroy_process_group()   # <<< modified >>>

# torchrun --nproc_per_node=4 /home/zhonghua/Filt_Event/Transformer_LHAASO_1e10_strong.py
# <<< modified >>> main entry for DDP
if __name__ == "__main__":
    import torch.distributed as dist
    import torch
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    npz_file = "/home/zhonghua/data/Dataset_Filted/1e10_V03/sampled_Ponly_1e10_V03_dataset_optimized.npz"
    param_keys = ["label", "R_mean", "Eage", "recE", "rec_theta", "rec_phi"]
    use_hitsM = False
    out_dir = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized_strict"
    os.makedirs(out_dir, exist_ok=True)

    train_and_validate_ddp(local_rank, world_size, npz_file, param_keys, use_hitsM, 
                           epochs=200, batch_size=32, lr=1e-3, device="cuda", 
                           max_hits=256, val_fraction=0.2, out_dir=out_dir, patience=100, seed=42)
    
    exp_npz="/home/zhonghua/data/Dataset_Filted/Experiment/2022/1e10_V03_2022_dataset_optimized.npz" # 经过muon cut 以及优化的
    test_exp_data(exp_npz, param_keys, model_path="par_transformer_best.pth",
                  out_dir=out_dir, use_hitsM=use_hitsM, max_hits=256)