
import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 4
ITEM_DIM   = 64
MAYOR_DIM  = 8
DROPOUT    = 0.2
LR         = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
N_FEATURES = 11

ANCHOR_HOURS = [168, 72, 24, 1, 0]
PREDICT_HOURS = [12, 24]


MAYORS = [
    "Aatrox", "Cole", "Diana", "Diaz", "Foxy", "Marina", "Paul",
    "Derpy", "Jerry", "Scorpius", "Finnegan", "Unknown"
]
MAYOR_TO_IDX = {m: i for i, m in enumerate(MAYORS)}

def get_mayor_idx(name: str) -> int:
    for m in MAYORS:
        if m in str(name):
            return MAYOR_TO_IDX[m]
    return MAYOR_TO_IDX["Unknown"]


def load_db(db_path: str, days: int) -> pd.DataFrame:
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM snapshots WHERE ts >= ? ORDER BY product_id, ts",
        con, params=(cutoff_ms,)
    )
    con.close()
    print(f"[Data] Loaded {len(df):,} rows | {df['product_id'].nunique()} items | "
          f"last {days} days")
    return df


def build_item_index(df: pd.DataFrame) -> dict:
    items = sorted(df["product_id"].unique())
    return {pid: i for i, pid in enumerate(items)}


def build_anchor_dataset(df: pd.DataFrame, item_index: dict,
                          scalers: dict) -> list:
    """
    For each item × time window, sample 5 anchor snapshots and build
    one training example: (anchor_features, item_id, mayor_id, label).

    anchor_features shape: (5, 11)
    label: scaled buyPrice array for each PREDICT_HOURS
    """
    examples = []
    ANCHOR_MS = [int(h * 3600 * 1000) for h in ANCHOR_HOURS]
    PREDICT_MS = [int(h * 3600 * 1000) for h in PREDICT_HOURS]

    for pid, grp in tqdm(df.groupby("product_id"), desc="Building dataset"):
        if pid not in item_index:
            continue
        item_id = item_index[pid]

        scaler = scalers[pid]
        raw = grp[["sellPrice", "buyPrice", "sellVolume", "buyVolume",
                    "sellMovingWeek", "buyMovingWeek"]].values.astype(np.float32)
        scaled = scaler.transform(raw)

        ts_arr  = grp["ts"].values
        mayors  = grp["mayor"].values

        STEP = 10
        for i in range(0, len(grp) - 1, STEP):
            t_now = ts_arr[i]

            labels = []
            for p_ms in PREDICT_MS:
                future_mask = ts_arr >= (t_now + p_ms)
                if future_mask.any():
                    j_label = np.argmax(future_mask)
                    labels.append(scaled[j_label, 1])
                else:
                    labels.append(-1.0)

            if all(lbl == -1.0 for lbl in labels):
                continue

            anchors = []
            for ah, anchor_ms in zip(ANCHOR_HOURS, ANCHOR_MS):
                t_anchor = t_now - anchor_ms
                if t_anchor < ts_arr[0]:
                    k = 0
                else:
                    diff = np.abs(ts_arr - t_anchor)
                    k = np.argmin(diff)

                s = scaled[k]
                row = grp.iloc[k]
                spread   = float(row["spread"])
                spreadPct = float(row["spreadPct"])
                spread_n   = np.clip(spread   / 50000, 0, 1)
                spreadPct_n = np.clip(spreadPct / 100,  0, 1)
                hours_delta = ah / 168.0

                anchors.append([
                    s[0],
                    s[1],
                    spread_n,
                    spreadPct_n,
                    s[2],
                    s[3],
                    s[4],
                    s[5],
                    0.0, 0.0,
                    hours_delta,
                ])

            mayor_id = get_mayor_idx(mayors[i])
            examples.append((
                np.array(anchors, dtype=np.float32),
                item_id,
                mayor_id,
                np.array(labels, dtype=np.float32),
            ))

    return examples


def fit_scalers(df: pd.DataFrame) -> dict:
    """Fit a per-item MinMaxScaler on the 6 numeric columns."""
    cols = ["sellPrice", "buyPrice", "sellVolume", "buyVolume",
            "sellMovingWeek", "buyMovingWeek"]
    scalers = {}
    for pid, grp in df.groupby("product_id"):
        sc = MinMaxScaler()
        sc.fit(grp[cols].values.astype(np.float32))
        scalers[pid] = sc
    return scalers


class BazaarDataset(Dataset):
    def __init__(self, examples):
        self.X      = torch.tensor(np.stack([e[0] for e in examples]))
        self.items  = torch.tensor([e[1] for e in examples], dtype=torch.long)
        self.mayors = torch.tensor([e[2] for e in examples], dtype=torch.long)
        self.y      = torch.tensor(np.stack([e[3] for e in examples]))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.items[i], self.mayors[i], self.y[i]


class BazaarTransformer(nn.Module):
    def __init__(self, n_items: int, n_mayors: int):
        super().__init__()
        self.n_items  = n_items
        self.n_mayors = n_mayors

        self.item_embed  = nn.Embedding(n_items,  ITEM_DIM)
        self.mayor_embed = nn.Embedding(n_mayors, MAYOR_DIM)

        self.input_proj = nn.Linear(N_FEATURES, D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT, batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        self.item_query_proj = nn.Linear(ITEM_DIM, D_MODEL)
        self.cross_attn      = nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=N_HEADS,
            dropout=DROPOUT, batch_first=True
        )

        self.film_scale = nn.Linear(MAYOR_DIM, D_MODEL)
        self.film_shift = nn.Linear(MAYOR_DIM, D_MODEL)

        self.fc = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128,      64), nn.GELU(),
            nn.Linear(64,        2),
        )

        self.skip_proj = nn.Linear(ITEM_DIM, 2)

        self.alpha = nn.Parameter(torch.tensor(0.65))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_query_proj.weight)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.skip_proj.weight)
        nn.init.zeros_(self.skip_proj.bias)

    def forward(self, x_anchors, item_ids, mayor_ids):

        item_emb  = self.item_embed(item_ids)
        mayor_emb = self.mayor_embed(mayor_ids)

        x = self.input_proj(x_anchors)

        x = self.transformer(x)

        query = self.item_query_proj(item_emb).unsqueeze(1)
        attended, _ = self.cross_attn(query, x, x)
        attended = attended.squeeze(1)

        scale     = torch.sigmoid(self.film_scale(mayor_emb))
        shift     = self.film_shift(mayor_emb)
        modulated = attended * scale + shift

        global_pred = self.fc(modulated)

        item_pred   = self.skip_proj(item_emb)

        alpha = torch.clamp(self.alpha, 0.5, 0.85)
        return alpha * item_pred + (1 - alpha) * global_pred


def train(model, loader_train, loader_val, epochs, device):
    opt      = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=2
    )
    loss_fn  = nn.HuberLoss(delta=0.1)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, ib, mb, yb in loader_train:
            xb, ib, mb, yb = xb.to(device), ib.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, ib, mb)
            
            mask = (yb != -1.0)
            loss = loss_fn(pred[mask], yb[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(loader_train.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, ib, mb, yb in loader_val:
                xb, ib, mb, yb = xb.to(device), ib.to(device), mb.to(device), yb.to(device)
                pred = model(xb, ib, mb)
                mask = (yb != -1.0)
                val_loss += loss_fn(pred[mask], yb[mask]).item() * len(yb)
        val_loss /= len(loader_val.dataset)

        schedule.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)
    return best_val


def tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().numpy().tolist()


def export_weights(model: BazaarTransformer, item_index: dict,
                   scalers: dict, val_loss: float, out_path: str):
    """Serialise everything the JS inference engine needs."""
    sd = model.state_dict()

    scaler_min   = {}
    scaler_scale = {}
    for pid, sc in scalers.items():
        scaler_min[pid]   = sc.data_min_.tolist()
        scaler_scale[pid] = sc.scale_.tolist()

    weights = {
        "meta": {
            "trained_at":    pd.Timestamp.now().isoformat(timespec="seconds"),
            "n_items":       model.n_items,
            "n_mayors":      model.n_mayors,
            "anchor_hours":  ANCHOR_HOURS,
            "predict_hours": PREDICT_HOURS,
            "val_loss":      round(val_loss, 6),
            "d_model":       D_MODEL,
            "n_heads":       N_HEADS,
            "n_layers":      N_LAYERS,
            "item_dim":      ITEM_DIM,
            "mayor_dim":     MAYOR_DIM,
        },
        "item_index":   item_index,
        "mayor_index":  MAYOR_TO_IDX,
        "scaler_min":   scaler_min,
        "scaler_scale": scaler_scale,

        "item_embeddings":  tensor_to_list(sd["item_embed.weight"]),
        "mayor_embeddings": tensor_to_list(sd["mayor_embed.weight"]),

        "input_proj_w": tensor_to_list(sd["input_proj.weight"]),
        "input_proj_b": tensor_to_list(sd["input_proj.bias"]),

        "transformer": [],

        "cross_attn_q_w": tensor_to_list(sd["cross_attn.in_proj_weight"][:D_MODEL]),
        "cross_attn_q_b": tensor_to_list(sd["cross_attn.in_proj_bias"][:D_MODEL]),
        "cross_attn_k_w": tensor_to_list(sd["cross_attn.in_proj_weight"][D_MODEL:2*D_MODEL]),
        "cross_attn_k_b": tensor_to_list(sd["cross_attn.in_proj_bias"][D_MODEL:2*D_MODEL]),
        "cross_attn_v_w": tensor_to_list(sd["cross_attn.in_proj_weight"][2*D_MODEL:]),
        "cross_attn_v_b": tensor_to_list(sd["cross_attn.in_proj_bias"][2*D_MODEL:]),
        "cross_attn_out_w": tensor_to_list(sd["cross_attn.out_proj.weight"]),
        "cross_attn_out_b": tensor_to_list(sd["cross_attn.out_proj.bias"]),

        "item_query_w": tensor_to_list(sd["item_query_proj.weight"]),
        "item_query_b": tensor_to_list(sd["item_query_proj.bias"]),

        "film_scale_w": tensor_to_list(sd["film_scale.weight"]),
        "film_scale_b": tensor_to_list(sd["film_scale.bias"]),
        "film_shift_w": tensor_to_list(sd["film_shift.weight"]),
        "film_shift_b": tensor_to_list(sd["film_shift.bias"]),

        "skip_proj_w": tensor_to_list(sd["skip_proj.weight"]),
        "skip_proj_b": tensor_to_list(sd["skip_proj.bias"]),

        "fc_ln_w":  tensor_to_list(sd["fc.0.weight"]),
        "fc_ln_b":  tensor_to_list(sd["fc.0.bias"]),
        "fc_0_w":   tensor_to_list(sd["fc.1.weight"]),
        "fc_0_b":   tensor_to_list(sd["fc.1.bias"]),
        "fc_1_w":   tensor_to_list(sd["fc.4.weight"]),
        "fc_1_b":   tensor_to_list(sd["fc.4.bias"]),
        "fc_2_w":   tensor_to_list(sd["fc.6.weight"]),
        "fc_2_b":   tensor_to_list(sd["fc.6.bias"]),

        "alpha": float(torch.clamp(model.alpha, 0.5, 0.85).item()),
    }

    for i in range(N_LAYERS):
        prefix = f"transformer.layers.{i}"
        layer = {
            "sa_q_w": tensor_to_list(sd[f"{prefix}.self_attn.in_proj_weight"][:D_MODEL]),
            "sa_q_b": tensor_to_list(sd[f"{prefix}.self_attn.in_proj_bias"][:D_MODEL]),
            "sa_k_w": tensor_to_list(sd[f"{prefix}.self_attn.in_proj_weight"][D_MODEL:2*D_MODEL]),
            "sa_k_b": tensor_to_list(sd[f"{prefix}.self_attn.in_proj_bias"][D_MODEL:2*D_MODEL]),
            "sa_v_w": tensor_to_list(sd[f"{prefix}.self_attn.in_proj_weight"][2*D_MODEL:]),
            "sa_v_b": tensor_to_list(sd[f"{prefix}.self_attn.in_proj_bias"][2*D_MODEL:]),
            "sa_out_w": tensor_to_list(sd[f"{prefix}.self_attn.out_proj.weight"]),
            "sa_out_b": tensor_to_list(sd[f"{prefix}.self_attn.out_proj.bias"]),
            "ff_1_w": tensor_to_list(sd[f"{prefix}.linear1.weight"]),
            "ff_1_b": tensor_to_list(sd[f"{prefix}.linear1.bias"]),
            "ff_2_w": tensor_to_list(sd[f"{prefix}.linear2.weight"]),
            "ff_2_b": tensor_to_list(sd[f"{prefix}.linear2.bias"]),
            "ln1_w": tensor_to_list(sd[f"{prefix}.norm1.weight"]),
            "ln1_b": tensor_to_list(sd[f"{prefix}.norm1.bias"]),
            "ln2_w": tensor_to_list(sd[f"{prefix}.norm2.weight"]),
            "ln2_b": tensor_to_list(sd[f"{prefix}.norm2.bias"]),
        }
        weights["transformer"].append(layer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(weights, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[Export] Saved {out_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Bazaar Intel — Trainer")
    parser.add_argument("--db",     default="bazaar.db",             help="SQLite DB")
    parser.add_argument("--days",   default=90,  type=int,           help="Days of data to train on")
    parser.add_argument("--epochs", default=100, type=int,           help="Training epochs")
    parser.add_argument("--out",    default="../website/weights.json",help="Output weights path")
    parser.add_argument("--device", default="auto",                  help="cpu / cuda / auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Train] Device: {device}")

    df = load_db(args.db, args.days)
    if len(df) < 1000:
        print("[Error] Not enough data. Run collect.py for at least a few hours first.")
        sys.exit(1)

    item_index = build_item_index(df)
    n_items    = len(item_index)
    n_mayors   = len(MAYORS)
    print(f"[Train] {n_items} items | {n_mayors} mayors")

    print("[Train] Fitting per-item scalers...")
    scalers = fit_scalers(df)

    print("[Train] Building anchor dataset...")
    examples = build_anchor_dataset(df, item_index, scalers)
    if not examples:
        print("[Error] Dataset is empty — need more data with time range.")
        sys.exit(1)
    print(f"[Train] {len(examples):,} training examples")

    split = int(len(examples) * 0.8)
    ds_train = BazaarDataset(examples[:split])
    ds_val   = BazaarDataset(examples[split:])
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)
    print(f"[Train] Train: {len(ds_train):,}  Val: {len(ds_val):,}")

    model = BazaarTransformer(n_items=n_items, n_mayors=n_mayors).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {n_params:,}")

    print(f"[Train] Training for {args.epochs} epochs...\n")
    best_val = train(model, loader_train, loader_val, args.epochs, device)
    print(f"\n[Train] Best validation loss: {best_val:.6f}")

    print("[Export] Serialising weights...")
    export_weights(model, item_index, scalers, best_val, args.out)
    print("[Done] Update the website by pushing the new weights.json to GitHub.")


if __name__ == "__main__":
    main()
