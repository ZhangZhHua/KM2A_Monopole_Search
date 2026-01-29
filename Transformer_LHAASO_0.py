# train_par_transformer_with_bias.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# ---------- Dataset ----------
class EASDataset(Dataset):
    def __init__(self, npz_file, use_hitsM=False,
                 param_keys=["label", "R_mean", "Eage", "recE", "rec_theta", "rec_phi"]):
        data = np.load(npz_file, allow_pickle=True)
        self.hitsE = data["hitsE"]
        self.hitsM = data["hitsM"] if use_hitsM and "hitsM" in data else None
        self.params = data["params"]
        self.param_names = list(data["param_names"])
        self.param_keys = param_keys
        self.use_hitsM = use_hitsM
        self.param_index = [self.param_names.index(k) for k in param_keys]

    def __len__(self):
        return len(self.hitsE)

    def __getitem__(self, idx):
        hits = self.hitsE[idx].astype(np.float32)
        if self.use_hitsM and self.hitsM is not None:
            other = self.hitsM[idx]
            if other is not None and len(other) > 0:
                hits = np.concatenate([hits, other.astype(np.float32)], axis=0)
        params = np.array(self.params[idx], dtype=np.float32)[self.param_index]
        label_raw = params[0]
        label = 1 if label_raw == 43 else 0
        params = params[1:]  # drop original label from params vector
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
    params = torch.tensor(np.stack(params_list, axis=0), dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return padded, mask, params, labels

# ---------- Pairwise bias attention block ----------
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

        # embed pairwise features to per-head bias
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_heads)
        )

        # feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask, pair_feat):
        # x: [B, N, D], mask: [B, N] bool, pair_feat: [B, N, N, pair_feat_dim]
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B, H, N, Hd]
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0,2,3,1)  # [B, H, Hd, N]
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B, H, N, Hd]

        # raw attention scores
        attn_logits = torch.matmul(q, k) * self.scale  # [B, H, N, N]

        # pairwise bias -> project to [B, N, N, H] then permute to [B, H, N, N]
        bias = self.pair_mlp(pair_feat)  # [B, N, N, H]
        bias = bias.permute(0, 3, 1, 2)

        attn_logits = attn_logits + bias

        # mask: True for valid; we need attn_mask where True means to ignore
        # build key padding mask (for keys), shape [B, 1, 1, N] broadcastable
        key_mask = (~mask).unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        attn_logits = attn_logits.masked_fill(key_mask, float("-1e9"))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, N, Hd]
        out = out.permute(0,2,1,3).contiguous().view(B, N, D)
        out = self.out_proj(out)
        x = x + out
        x = self.norm1(x)
        # FFN
        x2 = self.ffn(x)
        x = x + x2
        x = self.norm2(x)
        return x

# ---------- Full model ----------
class ParT_LHAASO(nn.Module):
    def __init__(self, in_dim=5, param_dim=5, embed_dim=128, num_heads=8, num_layers=4,
                 pair_feat_dim=3, dropout=0.1, max_pool=True):
        super().__init__()
        self.in_dim = in_dim
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            PairwiseAttentionBlock(embed_dim, num_heads, pair_feat_dim=pair_feat_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.param_fc = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.max_pool = max_pool
        self.cls = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, hits, mask, params):
        # hits: [B,N,in_dim], mask: [B,N], params: [B,param_dim]
        B, N, _ = hits.shape
        pos = hits[:, :, :2]  # x,y expected in first two columns
        feat = self.embed(hits) + self.pos_mlp(pos)
        # construct pairwise features: dx, dy, dPE (assuming pe at index 2)
        # shape [B, N, N, 3]
        x_pos = pos[..., 0].unsqueeze(2)  # [B,N,1]
        y_pos = pos[..., 1].unsqueeze(2)
        dx = x_pos - x_pos.permute(0,2,1)
        dy = y_pos - y_pos.permute(0,2,1)
        if hits.shape[2] > 2:
            pe = hits[:, :, 2].unsqueeze(2)
            dpe = pe - pe.permute(0,2,1)
        else:
            dpe = torch.zeros(B, N, N, device=hits.device)
        pair_feat = torch.stack([dx, dy, dpe], dim=-1)
        # normalize pairwise features (small trick)
        pair_feat = pair_feat / (pair_feat.abs().mean(dim=(1,2,3), keepdim=True) + 1e-6)

        for block in self.blocks:
            feat = block(feat, mask, pair_feat)

        # pooling
        mask_f = mask.unsqueeze(-1).float()
        if self.max_pool:
            masked_feat = feat.clone()
            masked_feat[~mask] = float("-1e9")
            feat_global, _ = masked_feat.max(dim=1)
            # if all masked for an event, fallback to mean (rare)
            invalid = (mask.sum(dim=1) == 0)
            if invalid.any():
                mean_f = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)
                feat_global[invalid] = mean_f[invalid]
        else:
            feat_global = (feat * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)

        param_feat = self.param_fc(params)
        out = torch.cat([feat_global, param_feat], dim=1)
        logits = self.cls(out)
        return logits

# ---------- Training & utility ----------
def train(npz_file, param_keys, use_hitsM=False, epochs=20, batch_size=32, lr=2e-4,
          device='cuda', max_hits=None, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds = EASDataset(npz_file, use_hitsM=use_hitsM, param_keys=param_keys)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, max_hits=max_hits))
    model = ParT_LHAASO(in_dim=ds.hitsE[0].shape[1], param_dim=len(param_keys)-1,
                       embed_dim=128, num_heads=8, num_layers=4, pair_feat_dim=3, dropout=0.1)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_auc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        all_preds, all_labels = [], []
        for hits, mask, params, labels in tqdm(loader, desc=f"Train epoch {epoch}/{epochs}"):
            hits = hits.to(device); mask = mask.to(device); params = params.to(device); labels = labels.to(device)
            opt.zero_grad()
            logits = model(hits, mask, params)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            probs = F.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            all_preds.append(probs); all_labels.append(labels.detach().cpu().numpy())
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        auc = roc_auc_score(labels, preds) if len(np.unique(labels))>1 else 0.5
        mean_loss = np.mean(losses)
        print(f"Epoch {epoch} loss {mean_loss:.4f} auc {auc:.4f}")
        # save best
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "par_transformer_best.pth")
            print("Saved best model, auc=", best_auc)
    return model

def predict_and_plot(npz_file, model_path, param_keys, use_hitsM=False, batch_size=64, device='cuda', max_hits=None):
    ds = EASDataset(npz_file, use_hitsM=use_hitsM, param_keys=param_keys)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, max_hits=max_hits))
    model = ParT_LHAASO(in_dim=ds.hitsE[0].shape[1], param_dim=len(param_keys)-1,
                       embed_dim=128, num_heads=8, num_layers=4, pair_feat_dim=3, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for hits, mask, params, labels in tqdm(loader, desc="Predict"):
            hits = hits.to(device); mask = mask.to(device); params = params.to(device)
            logits = model(hits, mask, params)
            probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_preds.append(probs); all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    sig_hist = preds[labels==1]
    bkg_hist = preds[labels==0]
    np.savez("/home/zhonghua/Filt_Event/transformer/1e10/preds_labels.npz", sig_hist=sig_hist, bkg_hist=bkg_hist)

    plt.figure(figsize=(7,5))
    plt.hist(preds[labels==0], bins=50, histtype='step', density=True, label='Background')
    plt.hist(preds[labels==1], bins=50, histtype='step', density=True, label='Monopole')
    plt.xlabel("Predicted Signal Probability")
    plt.ylabel("Normalized count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/zhonghua/Filt_Event/transformer/1e10/preds_hist.png")
    plt.show()
    if len(np.unique(labels))>1:
        auc = roc_auc_score(labels, preds)
        print("AUC:", auc)
    return preds, labels

# ---------- Main ----------
if __name__ == "__main__":
    # 用户按需配置这里的路径与 param_keys
    npz_file = "/home/zhonghua/data/Dataset_Filted/1e10_V03/sampled_1e10_V03_dataset.npz"
    param_keys = ["label", "R_mean", "Eage", "recE", "rec_theta", "rec_phi"]
    use_hitsM = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 训练
    print("param_keys:", param_keys)
    # print("len(params):", len(dataset.params[0]))
    model = train(npz_file, param_keys, use_hitsM=use_hitsM, epochs=20, batch_size=32, lr=2e-4, device=device, max_hits=1024)
    # 预测并画图（加载保存的最优模型）
    if os.path.exists("par_transformer_best.pth"):
        predict_and_plot(npz_file, "/home/zhonghua/Filt_Event/transformer/1e10/par_transformer_best.pth", param_keys, use_hitsM=use_hitsM, device=device, max_hits=1024)
    else:
        torch.save(model.state_dict(), "par_transformer_last.pth")
        predict_and_plot(npz_file, "/home/zhonghua/Filt_Event/transformer/1e10/par_transformer_last.pth", param_keys, use_hitsM=use_hitsM, device=device, max_hits=1024)