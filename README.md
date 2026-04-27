# Skyblock Bazaar AI Dashboard

A browser-based investment dashboard for Hypixel SkyBlock that uses a Temporal Transformer model to predict future Bazaar prices. There is no backend — everything runs statically via GitHub Pages, with the model executing entirely inside the user's browser.

---

## How it works

### Step 1 — Data Collection (`trainer/collect.py`)

Run this script continuously in the background while you want to build up your training dataset:

```
cd trainer
python collect.py
```

Every 60 seconds it hits the Hypixel Bazaar API and logs a snapshot of every active product into a local SQLite database (`bazaar.db`). Each row stores:

- `sellPrice`, `buyPrice` — current order book prices
- `sellVolume`, `buyVolume` — immediate order depth
- `sellMovingWeek`, `buyMovingWeek` — 7-day rolling volume
- `spread`, `spreadPct` — computed buy/sell gap
- `mayor` — the active Skyblock Mayor + Minister (fetched from the Election API every 5 minutes)

The session has automatic retry logic for transient network errors (429 rate limits, 5xx failures). There is no cap on how long you can run it — the more history you have, the better the model trains.

---

### Step 2 — Training (`trainer/train.py`)

Once you have a few days of data (minimum ~1000 rows), run:

```
cd trainer
python train.py
```

Default flags:
- `--db bazaar.db` — path to the SQLite database
- `--days 90` — how many days of history to use
- `--epochs 100` — training iterations
- `--out ../website/weights.json` — where to save the exported model
- `--device auto` — uses CUDA if available, otherwise CPU

#### What the model does

Instead of treating price prediction as a simple regression problem, the model encodes the full recent history of an item using a series of "anchor" snapshots at fixed lookback points:

| Anchor | Lookback |
|--------|----------|
| 0 | Now (live from Hypixel) |
| 1 | 1 hour ago |
| 2 | 1 day ago |
| 3 | 3 days ago |
| 4 | 1 week ago |

Each anchor is a vector of 11 normalized features (prices, volumes, spread, time delta). These 5 vectors form a short sequence that a Transformer encoder reads with full self-attention.

#### Model architecture (`BazaarTransformer`)

```
Input anchors (5, 11)
  → Linear projection → (5, 256)
  → Transformer Encoder (4 layers, 4 heads, GELU, d_ff=1024)
  → (5, 256)

Item embedding (64-dim)
  → Query projection → (1, 256)
  → Cross-attention over Transformer output → (256,)       ← "what parts of history matter for this item?"

Mayor embedding (8-dim)
  → FiLM modulation: scale = sigmoid(Wm·emb), shift = Wm'·emb
  → attended = attended * scale + shift                     ← "how does the active Mayor distort this?"

Global head: LayerNorm → Linear(256→128) → GELU → Linear(128→64) → GELU → Linear(64→2)
Skip path:   Linear(item_emb → 2)                                          ← item-specific baseline

Output = α · skip + (1-α) · global    where α ∈ [0.5, 0.85] (learned)
```

The dual output predicts the scaled buyPrice at **+12 hours** and **+24 hours** simultaneously. If either future timestamp is missing from the database (e.g. the very end of the collected data), that target is masked out of the loss using a sentinel value of `-1.0` — this is the Masked Huber Loss that lets partial training examples contribute without poisoning the gradients.

Training uses AdamW with cosine annealing restarts, gradient clipping at 1.0, and keeps the best validation checkpoint.

#### Scalers

Each item gets its own `MinMaxScaler` fitted on its historical price range. The scaler parameters (`data_min_`, `scale_`) are exported alongside the weights so the browser can inverse-transform the model's predictions back into real coin values.

---

### Step 3 — Export (`weights.json`)

At the end of training, `export_weights()` serializes everything into a single JSON file:

- Model hyperparameters and training metadata
- All weight matrices (Transformer layers, projections, heads)
- Item and Mayor embedding tables
- Per-item scaler min/scale arrays
- The learned blending weight `alpha`

This file is committed to the `website/` directory and served statically.

---

### Step 4 — Browser Inference (`website/inference.js`)

`inference.js` implements the full forward pass of `BazaarTransformer` in vanilla JavaScript with zero dependencies. When the page loads:

1. `weights.json` is fetched once and kept in memory.
2. For each item, `buildAnchorFeatures()` in `app.js` assembles the 5×11 anchor matrix using live Hypixel prices and cached CoflNet history.
3. `BazaarInference.predict(pid, anchorFeatures, mayorName)` runs the JS forward pass and returns `{ pred_buy_12h, pred_buy_24h }` in real coin units.

Predictions are scored and ranked on the **Picks** tab using a composite formula that weighs predicted price change, weekly volume, and current spread.

---

### Step 5 — Dashboard (`website/`)

The frontend is plain HTML + CSS + JavaScript, deployable anywhere as a static site.

- **Picks tab** — ranked table of the best predicted buys, filterable by price/spread
- **Market tab** — card grid showing current spread and live predictions per item
- **Stats tab** — Chart.js volume charts for the top items
- **Settings tab** — configure min/max price and spread filters

Historical price data for the graph modal is pulled live from the [CoflNet API](https://sky.coflnet.com/) on demand and cached in memory for the session.

---

## Requirements

```
cd trainer
pip install -r requirements.txt
```

Key dependencies: `torch`, `pandas`, `scikit-learn`, `tqdm`, `requests`

No GPU required, but training is significantly faster with CUDA.
