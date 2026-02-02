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
            ed = self.hitsE[idx].astype(np.float32)
            ed = np.concatenate([ed, np.zeros((ed.shape[0], 1), dtype=np.float32)], axis=1)  # tag=0
            md = self.hitsM[idx].astype(np.float32)
            md = np.concatenate([md, np.ones((md.shape[0], 1), dtype=np.float32)], axis=1)   # tag=1
            hits = np.concatenate([ed, md], axis=0)
        else:
            hits = self.hitsE[idx].astype(np.float32)
            hits = np.concatenate([hits, np.zeros((hits.shape[0], 1), dtype=np.float32)], axis=1)
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
    def __init__(self, in_dim=5, param_dim=5, embed_dim=128, num_heads=8, num_layers=4, pair_feat_dim=5, dropout=0.3, max_pool=True):
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
        dx /= (dx.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dy /= (dy.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dr /= (dr.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        dpe /= (dpe.abs().mean(dim=(1,2), keepdim=True) + 1e-6)
        ddt /= (ddt.abs().mean(dim=(1,2), keepdim=True) + 1e-6)

        pair_feat = torch.stack([dx, dy, dr, dpe, ddt], dim=-1)  # [B, N, N, 5]

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
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_and_validate(npz_file,param_keys,use_hitsM=False,epochs=50,batch_size=128,lr=2e-4,device='cuda',max_hits=1024,val_fraction=0.2,
                       out_dir="./transformer_out",patience=6,seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    ds = EASDataset(npz_file, use_hitsM=use_hitsM, param_keys=param_keys)
    N = len(ds)
    n_val = int(N * val_fraction)
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, max_hits=max_hits))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, max_hits=max_hits))
    sample_hits = ds.hitsE[0]
    in_dim = sample_hits.shape[1]
    param_dim = max(1, len(ds.param_index)-1)  # params stored exclude label in model earlier; keep >=1
    
    # model = ParT_LHAASO(in_dim=in_dim, param_dim=param_dim, embed_dim=128, num_heads=8, num_layers=4, pair_feat_dim=5, dropout=0.3)
    model = ParT_LHAASO_Small(in_dim=in_dim,param_dim=param_dim,embed_dim=64,num_heads=4,num_layers=3,pair_feat_dim=3,dropout=0.4,max_pool=False)
    model.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = warmup_cosine_schedule(opt, warmup_epochs=5, total_epochs=epochs, base_lr=lr)
    # scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, )
    # scheduler = CosineAnnealingLR(opt, T_max=20, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()
    
    scaler = torch.cuda.amp.GradScaler()
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
            # 混合精度训练前向传播
            # with torch.cuda.amp.autocast():
            #     logits = model(hits, mask, params)
            #     loss = loss_fn(logits, labels)
            
            # # 混合精度训练反向传播
            # scaler.scale(loss).backward()
            # scaler.step(opt)
            # scaler.update()
            #
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
        if val_auc > best_auc:
            best_auc = val_auc
        # if val_loss < best_loss:
        #     best_loss = val_loss
            bad = 0
            model_path = os.path.join(out_dir, "par_transformer_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"    ✅Saved best model: auc={val_auc} lr={scheduler.get_last_lr()[0]:.3e}")
            # print(f"       path: {model_path}")
            sig = probs[labels == 1]
            bkg = probs[labels == 0]
            np.savez(os.path.join(out_dir, f"val_preds_best.npz"), sig=sig, bkg=bkg)
            plot_and_save_hist(probs, labels, out_dir, name="val_preds_best.png", )
            plot_Scaled(probs, labels, out_dir, name="val_preds_best_scaled.png", )
            test_exp_data("/home/zhonghua/data/Dataset_Filted/Experiment/2022/1e10_V03_2022_dataset.npz", 
                          param_keys, model_path="par_transformer_best.pth",
                          out_dir=out_dir, use_hitsM=use_hitsM, max_hits=256, device=device)
        else:
            bad += 1
            plot_and_save_hist(probs, labels, out_dir, name="val_preds_epoch.png", )
            plot_Scaled(probs, labels, out_dir, name="val_preds_epoch_scaled.png", )
            print(f"    No improvement (bad={bad}/{patience}), lr={scheduler.get_last_lr()[0]:.3e}")

        if bad >= patience:
            print("Early stopping triggered.")
            break

        # scheduler.step(val_loss)
        scheduler.step()
    # final save last
    # last_path = os.path.join(out_dir, "par_transformer_last.pth")
    # torch.save(model.state_dict(), last_path)
    # print("Training finished. Last model saved to", last_path)
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

    model = ParT_LHAASO(in_dim=in_dim, param_dim=param_dim,
                        embed_dim=128, num_heads=8, num_layers=4,
                        pair_feat_dim=5, dropout=0.3)
    model.to(device)

    # === 3. 加载训练好的权重 ===
    model_path = os.path.join(out_dir, model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 4. 前向推理 ===
    all_probs = []
    all_indices = []  # <--- 新增

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
    save_path = os.path.join(out_dir, "exp_probs.npz")
    np.savez(save_path, probs=all_probs, indices=all_indices)
    print(f"✅ 概率与索引已保存至 {save_path}")

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

    fig_path = os.path.join(out_dir, "exp_probs_distribution.png")
    plt.savefig(fig_path)
    plt.close()


    return all_probs, all_indices

if __name__ == "__main__":
    # configure paths and keys here
    # npz_file = "/home/zhonghua/data/Dataset_Filted/1e10_V03/sampled_1e10_V03_dataset.npz"
    npz_file = "/home/zhonghua/data/Dataset_Filted/1e10_V03/sampled_1e10_V03_dataset.npz"
    param_keys = ["label", "R_mean", "Eage", "recE", "rec_theta", "rec_phi"]
    # param_keys = ["label", "recE", "rec_theta", "rec_phi"]
    use_hitsM = False
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    out_dir = "/home/zhonghua/Filt_Event/transformer/1e10_MD"
    os.makedirs(out_dir, exist_ok=True)
    model = train_and_validate(npz_file, param_keys, use_hitsM=use_hitsM, epochs=100, batch_size=64, lr=1e-4,
                               device=device, max_hits=512, val_fraction=0.2, out_dir=out_dir, patience=10, seed=42)

    exp_npz="/home/zhonghua/data/Dataset_Filted/Experiment/2022/1e10_V03_2022_dataset.npz"
    test_exp_data(exp_npz, param_keys, model_path="par_transformer_best.pth",
                  out_dir=out_dir, use_hitsM=use_hitsM, max_hits=512, device=device)