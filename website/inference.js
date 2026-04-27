
const BazaarInference = (() => {
  'use strict';

  let W    = null;  // parsed weights.json
  let meta = null;

  
  const sigmoid = x => 1 / (1 + Math.exp(-x));

  // GELU approximation matching PyTorch default
  const gelu = x => {
    const t = Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1 + t);
  };

  
  // Element-wise add of two 1D arrays
  const addVec = (a, b) => a.map((v, i) => v + b[i]);

  // Element-wise multiply of two 1D arrays
  const mulVec = (a, b) => a.map((v, i) => v * b[i]);

  // Element-wise add of two 2D arrays
  const addSeq = (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j]));

  // Transpose a 2D array: (m, n) → (n, m)
  function transpose(A) {
    const m = A.length, n = A[0].length;
    const T = Array.from({ length: n }, () => new Array(m));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        T[j][i] = A[i][j];
    return T;
  }

  // Linear: vector (d_in,) → (d_out,)
  // W: (d_out, d_in), b: (d_out,)
  function linearVec(v, W_mat, b) {
    return W_mat.map((row, i) => b[i] + row.reduce((s, w, k) => s + w * v[k], 0));
  }

  // Linear: sequence (seq, d_in) → (seq, d_out)
  function linearSeq(X, W_mat, b) {
    const Wt = transpose(W_mat);
    const d_out = b.length;
    return X.map(row => {
      const out = b.slice();
      for (let k = 0; k < row.length; k++) {
        const rk = row[k];
        for (let j = 0; j < d_out; j++) out[j] += rk * Wt[k][j];
      }
      return out;
    });
  }

  // Softmax over 1D array
  function softmax(arr) {
    const max  = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum  = exps.reduce((s, v) => s + v, 0);
    return exps.map(v => v / sum);
  }

  // Layer normalisation on 1D vector
  function layernormVec(x, w, b) {
    const n    = x.length;
    const mean = x.reduce((s, v) => s + v, 0) / n;
    const vari = x.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
    const std  = Math.sqrt(vari + 1e-5);
    return x.map((v, i) => w[i] * (v - mean) / std + b[i]);
  }

  // Layer normalisation on 2D sequence
  const layernormSeq = (X, w, b) => X.map(row => layernormVec(row, w, b));

  
  /**
   * Computes multi-head attention.
   * Q: (seq_q, d_model) — already projected
   * K: (seq_k, d_model) — already projected
   * V: (seq_k, d_model) — already projected
   * W_out: (d_model, d_model), b_out: (d_model,)
   * Returns: (seq_q, d_model)
   */
  function multiheadAttention(Q, K, V, W_out, b_out, n_heads) {
    const d_model = Q[0].length;
    const d_k     = Math.floor(d_model / n_heads);
    const scale   = 1 / Math.sqrt(d_k);
    const seq_q   = Q.length;
    const seq_k   = K.length;

    // Allocate output buffer
    const concat = Array.from({ length: seq_q }, () => new Array(d_model).fill(0));

    for (let h = 0; h < n_heads; h++) {
      const start = h * d_k;
      const end   = start + d_k;

      // Extract head slice: (seq, d_k)
      const Qh = Q.map(row => row.slice(start, end));
      const Kh = K.map(row => row.slice(start, end));
      const Vh = V.map(row => row.slice(start, end));

      // Attention weights: (seq_q, seq_k)
      const scores = Qh.map(qRow => {
        const raw = Kh.map(kRow => qRow.reduce((s, v, l) => s + v * kRow[l], 0) * scale);
        return softmax(raw);
      });

      // Weighted sum of V → fill into concat at correct head positions
      for (let i = 0; i < seq_q; i++)
        for (let j = 0; j < d_k; j++) {
          let sum = 0;
          for (let k = 0; k < seq_k; k++) sum += scores[i][k] * Vh[k][j];
          concat[i][start + j] = sum;
        }
    }

    // Output projection: (seq_q, d_model)
    return linearSeq(concat, W_out, b_out);
  }

  
  function transformerLayer(x, layer, n_heads) {
    // Self-attention: project Q, K, V from input
    const Q = linearSeq(x, layer.sa_q_w, layer.sa_q_b);
    const K = linearSeq(x, layer.sa_k_w, layer.sa_k_b);
    const V = linearSeq(x, layer.sa_v_w, layer.sa_v_b);

    const attn = multiheadAttention(Q, K, V, layer.sa_out_w, layer.sa_out_b, n_heads);

    // Post-norm residual
    x = layernormSeq(addSeq(x, attn), layer.ln1_w, layer.ln1_b);

    // Feed-forward: GELU(x W1 + b1) W2 + b2
    const ffMid = linearSeq(x, layer.ff_1_w, layer.ff_1_b).map(row => row.map(gelu));
    const ffOut  = linearSeq(ffMid, layer.ff_2_w, layer.ff_2_b);

    // Post-norm residual
    x = layernormSeq(addSeq(x, ffOut), layer.ln2_w, layer.ln2_b);
    return x;
  }

  
  /**
   * predict(pid, anchorFeatures, mayorName)
   *
   * @param {string} pid            - Hypixel product ID, e.g. "ENCHANTED_IRON"
   * @param {number[][]} anchorFeats - 5×11 array, oldest anchor first.
   *   Column order per row matches training:
   *   [sellPrice, buyPrice, spread_norm, spreadPct_norm,
   *    sellVolume, buyVolume, sellMovingWeek, buyMovingWeek,
   *    0, 0, hours_delta_norm]
   * @param {string} mayorName      - Active mayor name (e.g. "Finnegan")
   *
   * @returns {{ pred_buy_12h: number|null, pred_buy_24h: number|null, item_found: boolean }}
   */
  function predict(pid, anchorFeats, mayorName) {
    if (!W) return { pred_buy_12h: null, pred_buy_24h: null, item_found: false };

    const item_id = W.item_index[pid];
    if (item_id === undefined) return { pred_buy_12h: null, pred_buy_24h: null, item_found: false };

    // Mayor lookup — substring match
    let mayor_id = W.mayor_index['Unknown'] ?? 11;
    for (const [name, idx] of Object.entries(W.mayor_index)) {
      if (name !== 'Unknown' && String(mayorName).includes(name)) {
        mayor_id = idx;
        break;
      }
    }

    const n_heads  = meta.n_heads;
    const item_emb  = W.item_embeddings[item_id];   // (64,)
    const mayor_emb = W.mayor_embeddings[mayor_id]; // (8,)

    // ── 1. Input projection: (5, 11) → (5, 256)
    let x = linearSeq(anchorFeats, W.input_proj_w, W.input_proj_b);

    // ── 2. Transformer self-attention (N_LAYERS layers)
    for (const layer of W.transformer) {
      x = transformerLayer(x, layer, n_heads);
    }
    // x: (5, 256)

    // ── 3. Cross-attention: item embedding queries the price sequence
    const item_query_raw = linearVec(item_emb, W.item_query_w, W.item_query_b); // (256,)
    const Q_cross = linearSeq([item_query_raw], W.cross_attn_q_w, W.cross_attn_q_b); // (1, 256)
    const K_cross = linearSeq(x, W.cross_attn_k_w, W.cross_attn_k_b);               // (5, 256)
    const V_cross = linearSeq(x, W.cross_attn_v_w, W.cross_attn_v_b);               // (5, 256)

    const attended_seq = multiheadAttention(
      Q_cross, K_cross, V_cross,
      W.cross_attn_out_w, W.cross_attn_out_b,
      n_heads
    );
    const attended = attended_seq[0]; // (256,) — squeeze seq=1

    // ── 4. FiLM modulation (mayor → scale + shift attended)
    const film_scale = linearVec(mayor_emb, W.film_scale_w, W.film_scale_b).map(sigmoid);
    const film_shift = linearVec(mayor_emb, W.film_shift_w, W.film_shift_b);
    const modulated  = addVec(mulVec(attended, film_scale), film_shift); // (256,)

    // ── 5. Global prediction path (FC head)
    const h0 = layernormVec(modulated, W.fc_ln_w, W.fc_ln_b);
    const h1 = linearVec(h0, W.fc_0_w, W.fc_0_b).map(gelu);
    const h2 = linearVec(h1, W.fc_1_w, W.fc_1_b).map(gelu);
    const global_pred = linearVec(h2, W.fc_2_w, W.fc_2_b); // [12h, 24h]

    // ── 6. Item-specific skip path
    const item_pred = linearVec(item_emb, W.skip_proj_w, W.skip_proj_b); // [12h, 24h]

    // ── 7. Blend (alpha clipped to [0.5, 0.85])
    const alpha      = Math.max(0.5, Math.min(0.85, W.alpha));
    const pred_scaled_12h = alpha * item_pred[0] + (1 - alpha) * global_pred[0];
    const pred_scaled_24h = alpha * item_pred[1] + (1 - alpha) * global_pred[1];

    // ── 8. Inverse-scale to real coin value
    // sklearn MinMaxScaler: X_scaled = (X - min) * scale_ → X = X_scaled / scale_ + min
    const sc_min   = W.scaler_min[pid];
    const sc_scale = W.scaler_scale[pid];
    if (!sc_min || !sc_scale) return { pred_buy_12h: null, pred_buy_24h: null, item_found: true };

    // Column 1 = buyPrice in the scaler
    const pred_raw_12h = pred_scaled_12h / sc_scale[1] + sc_min[1];
    const pred_raw_24h = pred_scaled_24h / sc_scale[1] + sc_min[1];

    return { 
      pred_buy_12h: Math.max(0, pred_raw_12h), 
      pred_buy_24h: Math.max(0, pred_raw_24h), 
      item_found: true 
    };
  }

  
  function load(weightsJson) {
    W    = weightsJson;
    meta = weightsJson.meta;
    console.log(`[Inference] Loaded weights: ${meta.n_items} items, trained ${meta.trained_at}`);
  }

  const isLoaded     = () => W !== null;
  const getItemIndex = () => W ? W.item_index : {};
  const getMeta      = () => meta;

  return { load, predict, isLoaded, getItemIndex, getMeta };
})();
