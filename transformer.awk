#!/usr/bin/env bash
awk -v OFS='\t' -v mode="$1" '
###############################################################################
# Mini Transformer in AWK — TRAINING + temperature sampling + persistence
# - BPE TOKENIZATION (saved/loaded with model)
# - multi-head attention (4 heads), 1 FFN, weight-tying to embeddings
# - updates: E, P, WQ, WK, WV, W_out, W1, W2
# - modes:
#     train     -> trains; saves to model.awk (continues if model exists)
#     generate  -> loads model.awk; prompts; generates with temperature
# NOTE: AWK numeric performance is limited — use small d and small input.
###############################################################################

# --------------------
# BPE Tokenization functions
# --------------------
function get_pair_stats(tokens, n,    stats, i, pair) {
    delete stats
    for (i = 1; i < n; i++) {
        pair = tokens[i] OFS tokens[i+1]
        stats[pair]++
    }
}

function merge_sequence(tokens, n, pair_str, new_token,   new_tokens, new_n, i, p1, p2, pair_parts) {
    split(pair_str, pair_parts, OFS)
    p1 = pair_parts[1]; p2 = pair_parts[2]

    new_n = 0; i = 1
    while (i <= n) {
        if (i < n && tokens[i] == p1 && tokens[i+1] == p2) {
            new_n++; new_tokens[new_n] = new_token
            i += 2
        } else {
            new_n++; new_tokens[new_n] = tokens[i]
            i++
        }
    }
    for (i=1; i<=new_n; i++) tokens[i] = new_tokens[i]
    return new_n
}

function train_bpe(text, num_merges,    i, char, n, m, pair_parts, pair_stats, best_pair, max_freq, new_token) {
    print "--- Starting BPE training ---"
    V = 0
    delete token2id; delete id2token
    for (i = 1; i <= length(text); i++) {
        char = substr(text, i, 1)
        if (!(char in token2id)) {
            V++; token2id[char] = V; id2token[V] = char
        }
    }

    n = 0
    for(i=1; i<=length(text); i++) { n++; bpe_seq[n] = substr(text,i,1) }

    num_ranked_merges = 0
    for (m = 1; m <= num_merges; m++) {
        get_pair_stats(bpe_seq, n, pair_stats)

        max_freq = -1; best_pair = ""
        for (pair in pair_stats) if (pair_stats[pair] > max_freq) { max_freq = pair_stats[pair]; best_pair = pair }
        if (best_pair == "" || max_freq < 2) { print "BPE: stopping at", m-1, "merges (max pair freq < 2)."; break }

        split(best_pair, pair_parts, OFS)
        new_token = pair_parts[1] pair_parts[2]

        if (!(new_token in token2id)) { V++; token2id[new_token] = V; id2token[V] = new_token }

        num_ranked_merges = m
        ranked_merges[m, "p1"] = pair_parts[1]
        ranked_merges[m, "p2"] = pair_parts[2]
        ranked_merges[m, "new"] = new_token

        n = merge_sequence(bpe_seq, n, best_pair, new_token)
        if (m % 50 == 0) printf "BPE Merge %d: %s + %s -> %s | vocab %d\n", m, pair_parts[1], pair_parts[2], new_token, V
    }
    print "--- BPE training finished. Final vocab size:", V, " Learned merges:", num_ranked_merges, "---"
}

function tokenize_bpe(text,    n, i, m, new_n, p1, p2, new_tok, tokens, new_tokens) {
    n = 0
    for(i=1; i<=length(text); i++) { n++; tokens[n] = substr(text, i, 1) }

    for (m=1; m <= num_ranked_merges; m++) {
        p1 = ranked_merges[m, "p1"]; p2 = ranked_merges[m, "p2"]; new_tok = ranked_merges[m, "new"]
        new_n = 0; i = 1
        while (i <= n) {
            if (i < n && tokens[i] == p1 && tokens[i+1] == p2) { new_n++; new_tokens[new_n] = new_tok; i += 2 }
            else { new_n++; new_tokens[new_n] = tokens[i]; i++ }
        }
        n = new_n; for(i=1; i<=n; i++) tokens[i] = new_tokens[i]
    }

    T = 0; delete seq
    for(i=1; i<=n; i++) if (tokens[i] in token2id) { T++; seq[T] = token2id[tokens[i]] }
}

# --------------------
# helpers
# --------------------
function idx(i,j,cols) { return (i-1)*cols + j }
function dot_row(A,ri,B,rj,d,   s,k){ s=0; for(k=1;k<=d;k++) s+=A[idx(ri,k,d)]*B[idx(rj,k,d)]; return s }
function clamp_inplace(A,n,val,   i){ for(i=1;i<=n;i++){ if(A[i]>val)A[i]=val; else if(A[i]<-val)A[i]=-val } }
function safe_exp(x){ if (x < -700) return 0; return exp(x) }
function safe_div(a,b){ return (b==0) ? 0 : a/b }

###############################################################################
# Initialization / hyperparams
###############################################################################
BEGIN{
  srand(1337)

  # hyperparams (tweak)
  d = 8
  d_ff = 16
  n_heads = 4       # NEW: number of attention heads
  d_head = d / n_heads  # Head dimension (must be integer)
  block_size = 50
  epochs = 1000
  lr = 0.005
  clip = 0.5
  gen_tokens = 40
  temp = 1000
  bpe_merges = 250

  if (mode == "") mode = "train"

  if (mode == "train") {
    print "mode=train"
  } else if (mode == "generate") {
    print "mode=generate"
  } else {
    print "Unknown mode:", mode, "(use train|generate)"; exit
  }

  model_exists = (system("test -f model.awk") == 0)

  if (mode == "train" && model_exists) {
    print "Found model.awk -> loading to continue training..."
    load_model("model.awk")
  }
}

###############################################################################
# Multi-head attention forward (causal)
###############################################################################
function multihead_attention(X, block_len, d, n_heads, d_head, WQ, WK, WV, W_out, attn_out,   \
                            h, t, k, j, i, maxs, den, SCO_head, Pmat_head, ctx_h, concat_head, V_head_h) {
    # Initialize concatenated heads
    for (t=1; t<=block_len; t++) for (j=1; j<=d; j++) concat_head[idx(t,j,d)] = 0

    # Process each head
    for (h=0; h<n_heads; h++) {
        # Compute Q, K, V for head h
        for (t=1; t<=block_len; t++) {
            for (j=1; j<=d_head; j++) {
                Q_head = 0; K_head = 0; V_head = 0
                for (i=1; i<=d; i++) {
                    # Get head-specific weights
                    idx_wq = (h * d_head * d) + (i-1)*d_head + j
                    Q_head += X[idx(t,i,d)] * WQ[idx_wq]

                    idx_wk = (h * d_head * d) + (i-1)*d_head + j
                    K_head += X[idx(t,i,d)] * WK[idx_wk]

                    idx_wv = (h * d_head * d) + (i-1)*d_head + j
                    V_head += X[idx(t,i,d)] * WV[idx_wv]
                }
                SCO_head[idx(t,j)] = Q_head * K_head / sqrt(d_head)
                # Store V_head for this position (t,j) for the head
                V_head_h[idx(t,j)] = V_head
            }
        }

        # Compute attention scores and probabilities for head h
        for (t=1; t<=block_len; t++) {
            maxs = -1e30
            for (k=1; k<=t; k++) {
                if (SCO_head[idx(t,k)] > maxs) maxs = SCO_head[idx(t,k)]
            }
            den = 0
            for (k=1; k<=t; k++) {
                Pmat_head[idx(t,k)] = safe_exp(SCO_head[idx(t,k)] - maxs)
                den += Pmat_head[idx(t,k)]
            }
            for (k=1; k<=t; k++) {
                Pmat_head[idx(t,k)] = safe_div(Pmat_head[idx(t,k)], den)
            }
        }

        # Compute context for head h
        for (t=1; t<=block_len; t++) {
            for (j=1; j<=d_head; j++) {
                ctx_h = 0
                for (k=1; k<=t; k++) {
                    # Get V_head for token k, head j
                    ctx_h += Pmat_head[idx(t,k)] * V_head_h[idx(k,j)]
                }
                # Accumulate into concatenated heads
                concat_head[idx(t, h*d_head+j)] += ctx_h
            }
        }
    }

    # Project concatenated heads with W_out
    for (t=1; t<=block_len; t++) {
        for (j=1; j<=d; j++) {
            attn_out_val = 0
            for (i=1; i<=d; i++) {
                attn_out_val += concat_head[idx(t,i,d)] * W_out[idx(i,j,d)]
            }
            attn_out[idx(t,j,d)] = attn_out_val
        }
    }
}

###############################################################################
# Forward + backprop for a single block [start..end]
###############################################################################
function process_block(start,end,   block_len,t,i,j,p,loss,y,val, \
                              X,Q,K,Vv,ctx,attn_out,z1,z1mask,y2,logits_t, \
                              dy2, dW2_acc, dW1_acc, dz1_row, d_attn_out, dctx, dV_acc, dSCO, dQ, dK, dWQ_acc, dWK_acc, dWV_acc, dW_out_acc, dX, dPtmp, m, den, py_log, g, sum_v, sum_p_dp, s, dqj, dkj, dvj, tok_id){
  block_len = end - start + 1

  # Embeddings + positional encoding
  for(t=1;t<=block_len;t++) for(j=1;j<=d;j++){
    X[idx(t,j,d)] = E[idx(seq[start+t-1],j,d)] + P[idx(t,j,d)]
  }

  # Multi-head attention
  multihead_attention(X, block_len, d, n_heads, d_head, WQ, WK, WV, W_out, attn_out, V_head_h)

  # Residual connection
  for(t=1;t<=block_len;t++) for(j=1;j<=d;j++){
    attn_out[idx(t,j,d)] += X[idx(t,j,d)]
  }

  # FFN
  for(t=1;t<=block_len;t++){
    for(j=1;j<=d_ff;j++){
      z1 = 0
      for(i=1;i<=d;i++) z1 += attn_out[idx(t,i,d)] * W1[idx(i,j,d_ff)]
      if (z1 > 0) { z1mask[idx(t,j,d_ff)] = 1; Z1[idx(t,j,d_ff)] = z1 }
      else        { z1mask[idx(t,j,d_ff)] = 0; Z1[idx(t,j,d_ff)] = 0  }
    }
  }
  for(t=1;t<=block_len;t++){
    for(j=1;j<=d;j++){
      y2 = 0
      for(i=1;i<=d_ff;i++) y2 += Z1[idx(t,i,d_ff)] * W2[idx(i,j,d)]
      y2 += attn_out[idx(t,j,d)]
      Y2[idx(t,j,d)] = y2
    }
  }

  # Output layer (logits)
  for(t=1;t<=block_len;t++){
    for(j=1;j<=V;j++){
      logits_t[idx(t,j,V)] = 0
      for(i=1;i<=d;i++) logits_t[idx(t,j,V)] += Y2[idx(t,i,d)] * E[idx(j,i,d)]
    }
  }

  # Initialize accumulators
  for(i=1;i<=d*d;i++) { dWQ_acc[i]=0; dWK_acc[i]=0; dWV_acc[i]=0 }
  for(i=1;i<=d*d;i++) dW_out_acc[i]=0
  for(i=1;i<=d*d_ff;i++) dW1_acc[i]=0
  for(i=1;i<=d_ff*d;i++) dW2_acc[i]=0
  delete dX

  loss = 0

  # Compute loss and gradients
  for(t=1;t<=block_len;t++){
    y = tgt[start+t-1]; if(y==0) continue

    # Compute softmax
    m = -1e30
    for(j=1;j<=V;j++){ val = logits_t[idx(t,j,V)]; if(val > m) m = val }
    den = 0
    for(j=1;j<=V;j++) den += safe_exp(logits_t[idx(t,j,V)] - m)
    py_log = logits_t[idx(t,y,V)] - (m + log(den))
    loss += -py_log

    # Compute gradients for output layer
    for(j=1;j<=V;j++){
      p = safe_div(safe_exp(logits_t[idx(t,j,V)] - m), den)
      g = p - ((j==y)?1:0)
      for(i=1;i<=d;i++){
        E[idx(j,i,d)] -= lr * g * Y2[idx(t,i,d)]
        # Accumulate for W2
        dW2_acc[idx(i,j,d)] += g * Y2[idx(t,i,d)]
      }
    }

    # Backprop through FFN
    for(i=1;i<=d_ff;i++){
      dz1 = 0
      for(j=1;j<=d;j++) dz1 += dW2_acc[idx(i,j,d)] * W2[idx(i,j,d)]
      if (z1mask[idx(t,i,d_ff)]==0) dz1 = 0
      dz1_row[i] = dz1
    }
    for(p=1;p<=d;p++) for(i=1;i<=d_ff;i++){
      dW1_acc[idx(p,i,d_ff)] += attn_out[idx(t,p,d)] * dz1_row[i]
    }

    # Backprop through attention
    for(j=1;j<=d;j++) {
      d_attn_out[idx(t,j,d)] = 0
      for(i=1;i<=d_ff;i++) {
        d_attn_out[idx(t,j,d)] += dz1_row[i] * W2[idx(i,j,d)]
      }
      # Add gradient from output layer
      d_attn_out[idx(t,j,d)] += dW2_acc[idx(j,d)]
    }

    # Backprop through W_out
    for(i=1;i<=d;i++) for(j=1;j<=d;j++){
      dW_out_acc[idx(i,j,d)] += concat_head[idx(t,i,d)] * d_attn_out[idx(t,j,d)]
    }

    # Backprop through WQ, WK, WV (head-specific)
    for(h=0; h<n_heads; h++) {
      for(i=1;i<=d;i++) for(j=1;j<=d_head;j++){
        idx_wq = (h * d_head * d) + (i-1)*d_head + j
        dWQ_acc[idx_wq] += X[idx(t,i,d)] * dQ[idx(t,j,d)]
        idx_wk = (h * d_head * d) + (i-1)*d_head + j
        dWK_acc[idx_wk] += X[idx(t,i,d)] * dK[idx(t,j,d)]
        idx_wv = (h * d_head * d) + (i-1)*d_head + j
        dWV_acc[idx_wv] += X[idx(t,i,d)] * dV[idx(t,j,d)]
      }
    }
  }

  # Update weights
  for(i=1;i<=d*d;i++) { WQ[i] -= lr * dWQ_acc[i]; WK[i] -= lr * dWK_acc[i]; WV[i] -= lr * dWV_acc[i] }
  for(i=1;i<=d*d;i++) W_out[i] -= lr * dW_out_acc[i]
  for(i=1;i<=d*d_ff;i++) W1[i] -= lr * dW1_acc[i]
  for(i=1;i<=d_ff*d;i++) W2[i] -= lr * dW2_acc[i]

  # Clip gradients
  clamp_inplace(E, V*d, clip); clamp_inplace(P, block_size*d, clip)
  clamp_inplace(W1, d*d_ff, clip); clamp_inplace(W2, d_ff*d, clip)
  clamp_inplace(WQ, d*d, clip); clamp_inplace(WK, d*d, clip); clamp_inplace(WV, d*d, clip)
  clamp_inplace(W_out, d*d, clip)

  return loss
}

###############################################################################
# Forward-only logits for next-token given a context (generation)
###############################################################################
function forward_logits_for_context(ctx_ids, L, logits, \
                                    t,i,j, X,attn_out, z, z1, y, last_t){
  # IMPORTANT: logits must be an array — never assign scalar
  delete logits

  # Forward through transformer
  for(t=1;t<=L;t++) for(j=1;j<=d;j++){
    X[idx(t,j,d)] = E[idx(ctx_ids[t],j,d)] + P[idx(t,j,d)]
  }

  # Multi-head attention
  multihead_attention(X, L, d, n_heads, d_head, WQ, WK, WV, W_out, attn_out, V_head_h)

  # Residual connection
  for(t=1;t<=L;t++) for(j=1;j<=d;j++) attn_out[idx(t,j,d)] += X[idx(t,j,d)]

  # FFN
  for(t=1;t<=L;t++){
    for(j=1;j<=d_ff;j++){
      z = 0
      for(i=1;i<=d;i++) z += attn_out[idx(t,i,d)] * W1[idx(i,j,d_ff)]
      Z1[idx(t,j,d_ff)] = (z>0)?z:0
    }
  }
  for(t=1;t<=L;t++){
    for(j=1;j<=d;j++){
      y = attn_out[idx(t,j,d)]
      for(i=1;i<=d_ff;i++) y += Z1[idx(t,i,d_ff)] * W2[idx(i,j,d)]
      Y2[idx(t,j,d)] = y
    }
  }

  # Output layer
  last_t = L
  for(j=1;j<=V;j++){
    logits[j] = 0
    for(i=1;i<=d;i++) logits[j] += Y2[idx(last_t,i,d)] * E[idx(j,i,d)]
  }

  # Cleanup
  delete X; delete attn_out; delete Z1; delete Y2
}

###############################################################################
# Saving / Loading model
###############################################################################
function save_model(fname,    i) {
  # truncate once
  system("sh -c '\'': > " fname "'\''")

  print "# Model file (generated by AWK mini-transformer)" >> fname
  print "V=" V >> fname
  print "d=" d >> fname
  print "d_ff=" d_ff >> fname
  print "n_heads=" n_heads >> fname
  print "block_size=" block_size >> fname
  print "num_ranked_merges=" num_ranked_merges >> fname

  # vocab
  for (i in token2id)   print "token2id[\"" i "\"]=" token2id[i] >> fname
  for (i in id2token)   print "id2token[" i "]=\"" id2token[i] "\"" >> fname

  # merges
  for (i=1;i<=num_ranked_merges;i++) {
    print "ranked_merges[" i ",\"p1\"]=\"" ranked_merges[i, "p1"] "\"" >> fname
    print "ranked_merges[" i ",\"p2\"]=\"" ranked_merges[i, "p2"] "\"" >> fname
    print "ranked_merges[" i ",\"new\"]=\"" ranked_merges[i, "new"] "\"" >> fname
  }

  dump_array("E", E, V*d, fname)
  dump_array("P", P, block_size*d, fname)
  dump_array("WQ", WQ, d*d, fname)
  dump_array("WK", WK, d*d, fname)
  dump_array("WV", WV, d*d, fname)
  dump_array("W_out", W_out, d*d, fname)
  dump_array("W1", W1, d*d_ff, fname)
  dump_array("W2", W2, d_ff*d, fname)
}

function dump_array(name, A, n, fname,  i){
  for (i=1;i<=n;i++) print name "[" i "]=" A[i] >> fname
}

function load_model(fname,   line, key, val, m, arr, idxv) {
  while((getline line < fname) > 0){
    if (line ~ /^[#[:space:]]*$/) continue
    split(line, kv, "="); key = kv[1]; val = substr(line, index(line, "=")+1)

    if (key == "V") V = val + 0
    else if (key == "d") d = val + 0
    else if (key == "d_ff") d_ff = val + 0
    else if (key == "n_heads") n_heads = val + 0
    else if (key == "block_size") block_size = val + 0
    else if (key == "num_ranked_merges") num_ranked_merges = val + 0
    else if (key ~ /^token2id\[/) {
      match(key, /\["(.*)"\]/, m)
      token2id[m[1]] = val + 0
    }
    else if (key ~ /^id2token\[/) {
      match(key, /\[([0-9]+)\]/, m)
      gsub(/^"/, "", val); gsub(/"$/, "", val)
      id2token[m[1]+0] = val
    }
    else if (key ~ /^ranked_merges\[/) {
      match(key, /\[([0-9]+),"(p1|p2|new)"\]/, m)
      gsub(/^"/, "", val); gsub(/"$/, "", val)
      ranked_merges[m[1]+0, m[2]] = val
    }
    else if (key ~ /^[A-Z0-9]+\[[0-9]+\]$/) {
      match(key, /^([A-Z0-9]+)\[([0-9]+)\]$/, m)
      arr = m[1]; idxv = m[2]+0
      if (arr=="E") E[idxv]=val+0
      else if (arr=="P") P[idxv]=val+0
      else if (arr=="WQ") WQ[idxv]=val+0
      else if (arr=="WK") WK[idxv]=val+0
      else if (arr=="WV") WV[idxv]=val+0
      else if (arr=="W_out") W_out[idxv]=val+0
      else if (arr=="W1") W1[idxv]=val+0
      else if (arr=="W2") W2[idxv]=val+0
    }
  }
  close(fname)
}

###############################################################################
# Training / Generation control
###############################################################################
END{
  if (mode == "train") {
    full_text = ""
    while ((getline line < "input.txt") > 0) { full_text = full_text line " " }
    close("input.txt")
    if (length(full_text) < 10) { print "Error: input.txt is too short for BPE"; exit }

    if (!model_exists) {
      print "No existing model -> training BPE from scratch with", bpe_merges, "merges."
      train_bpe(full_text, bpe_merges)
    } else {
      print "Using BPE merges from loaded model (num_ranked_merges=" num_ranked_merges ")."
    }
    tokenize_bpe(full_text) # seq[1..T], T

    if (!model_exists) {
      print "Initializing parameters from scratch."
      for(i=1;i<=T-1;i++) tgt[i]=seq[i+1]; tgt[T]=0

      # Initialize weights
      for(i=1;i<=V;i++) for(j=1;j<=d;j++) E[idx(i,j,d)] = (rand()*2-1) * 0.1
      for(i=1;i<=block_size;i++) for(j=1;j<=d;j++) P[idx(i,j,d)] = (rand()*2-1) * 0.01
      for(i=1;i<=d;i++) for(j=1;j<=d;j++){
        WQ[idx(i,j,d)] = (rand()*2-1) * 0.1
        WK[idx(i,j,d)] = (rand()*2-1) * 0.1
        WV[idx(i,j,d)] = (rand()*2-1) * 0.1
      }
      for(i=1;i<=d;i++)   for(j=1;j<=d_ff;j++) W1[idx(i,j,d_ff)] = (rand()*2-1) * 0.1
      for(i=1;i<=d_ff;i++) for(j=1;j<=d;j++)   W2[idx(i,j,d)]   = (rand()*2-1) * 0.1
      for(i=1;i<=d;i++) for(j=1;j<=d;j++) W_out[idx(i,j,d)] = (rand()*2-1) * 0.1
    } else {
      for(i=1;i<=T-1;i++) tgt[i]=seq[i+1]; tgt[T]=0
    }

    print "hyperparams: d=" d, " d_ff=" d_ff, " n_heads=" n_heads, " epochs=" epochs, " lr=" lr, " temp=" temp, " bpe_merges=" bpe_merges
    print "Loaded BPE sequence length=" T " vocab size=" V
    blocks_per_epoch = int((T + block_size - 1) / block_size)
    step = 0
    for(epoch=1; epoch<=epochs; epoch++){
      epoch_loss = 0; block_count = 0
      for(start=1; start<=T; start+=block_size){
        end = start + block_size - 1
        if (end > T) end = T
        if (start > end) continue
        step++
        loss = process_block(start,end)
        epoch_loss += loss
        block_count++
        if (step % 10 == 0) printf("Epoch %d/%d Step %d/%d | Loss %.6f\n", epoch, epochs, step, epochs*blocks_per_epoch, loss/block_size)
      }
      if (block_count > 0) printf("Epoch %d complete. Avg loss= %.6f\n", epoch, epoch_loss/block_count)
      if (epoch % 10 == 0) lr *= 0.9
    }

    save_model("model.awk")
    print "Saved model to model.awk"
  }

  else if (mode == "generate") {
    if (system("test -f model.awk") != 0) { print "Error: model.awk not found. Run in train mode first."; exit }
    load_model("model.awk")
    print "Enter a prompt:"
    getline prompt < "-"
    if (prompt == "") prompt = " "

    tokenize_bpe(prompt)
    if (T == 0) { print "Prompt produced no tokens; try different text."; exit }

    for(i=1;i<=T;i++) out_idx[i]=seq[i]
    out_len = T

    print "\n--- Generation ---"
    for(i=1;i<=T;i++) printf("%s", id2token[out_idx[i]])

    for(t=1; t<=gen_tokens; t++){
      ctx_len = (out_len<block_size)?out_len:block_size
      s = out_len - ctx_len + 1
      for(i=1;i<=ctx_len;i++) ctx_ids[i] = out_idx[s + i - 1]

      # Forward to get logits for next token
      forward_logits_for_context(ctx_ids, ctx_len, logits)

      # Temperature sampling
      m = -1e30
      for(j=1;j<=V;j++){ if (logits[j] > m) m = logits[j] }
      den = 0
      for(j=1;j<=V;j++) den += safe_exp((logits[j]-m)/temp)
      r = rand(); c = 0; nxt = V
      for(j=1;j<=V;j++){
        p = safe_div(safe_exp((logits[j]-m)/temp), den)
        c += p
        if (r <= c){ nxt = j; break }
      }

      out_len++; out_idx[out_len] = nxt
      printf("%s", id2token[nxt])
    }
    print "\n--- End ---"
  }
}

' /dev/null
