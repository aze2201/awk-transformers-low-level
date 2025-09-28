#!/usr/bin/env bash
awk -v OFS='\t' -v mode="$1" '
###############################################################################
# GPT-2 style Transformer in AWK (Portable Version)
# - BPE TOKENIZATION (saved/loaded with model)
# - multi-head attention (4 heads), 1 FFN, weight-tying to embeddings
# - pre-norm architecture with LayerNorm, Dropout, AdamW, and biases
# - modes:
#     train     -> trains; saves to model.awk (continues if model exists)
#     generate  -> loads model.awk; prompts; generates with temperature
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
            new_n++; new_tokens[new_n] = new_token; i += 2
        } else {
            new_n++; new_tokens[new_n] = tokens[i]; i++
        }
    }
    for (i=1; i<=new_n; i++) tokens[i] = new_tokens[i]
    return new_n
}

function train_bpe(text, num_merges,    i, char, n, m, pair_parts, pair_stats, best_pair, max_freq, new_token) {
    print "Starting BPE training"
    V = 0
    delete token2id; delete id2token
    for (i = 1; i <= length(text); i++) {
        char = substr(text, i, 1)
        if (!(char in token2id)) { V++; token2id[char] = V; id2token[V] = char }
    }
    n = 0
    for(i=1; i<=length(text); i++) { n++; bpe_seq[n] = substr(text,i,1) }
    num_ranked_merges = 0
    for (m = 1; m <= num_merges; m++) {
        get_pair_stats(bpe_seq, n, pair_stats)
        max_freq = -1; best_pair = ""
        for (pair in pair_stats) if (pair_stats[pair] > max_freq) { max_freq = pair_stats[pair]; best_pair = pair }
        if (best_pair == "" || max_freq < 2) { print "BPE: stopping at", m-1, "merges."; break }
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
    print "--- BPE training finished. Final vocab size:", V, " Merges:", num_ranked_merges, "---"
}

function tokenize_bpe(text, seq_out,    n, i, m, new_n, p1, p2, new_tok, tokens, new_tokens, T) {
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

    T = 0
    for(i=1; i<=n; i++) {
        if (tokens[i] in token2id) {
            T++
            if (seq_out == "seq_full") {
                seq_full[T] = token2id[tokens[i]]
            } else if (seq_out == "out_idx") {
                out_idx[T] = token2id[tokens[i]]
            }
        }
    }
    return T
}


# --------------------
# helpers
# --------------------
function idx(i,j,cols) { return (i-1)*cols + j }
function dot_row(A,ri,B,rj,d,   s,k){ s=0; for(k=1;k<=d;k++) s+=A[idx(ri,k,d)]*B[idx(rj,k,d)]; return s }
function clamp_inplace(A,n,val,   i){ for(i=1;i<=n;i++){ if(A[i]>val)A[i]=val; else if(A[i]<-val)A[i]=-val } }
function safe_exp(x){ if (x < -700) return 0; return exp(x) }
function safe_div(a,b){ return (b==0) ? 0 : a/b }

# --------------------
# AdamW optimizer
# --------------------
function adamw_init_optimizer() {
    adam_t = 0; beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8
    init_moments(m_E, v_E, V*d); init_moments(m_P, v_P, block_size*d)
    init_moments(m_WQ, v_WQ, d*d); init_moments(m_WK, v_WK, d*d)
    init_moments(m_WV, v_WV, d*d); init_moments(m_W_out, v_W_out, d*d)
    init_moments(m_W1, v_W1, d*d_ff); init_moments(m_W2, v_W2, d_ff*d)
    init_moments(m_LN1_gamma, v_LN1_gamma, d); init_moments(m_LN1_beta, v_LN1_beta, d)
    init_moments(m_LN2_gamma, v_LN2_gamma, d); init_moments(m_LN2_beta, v_LN2_beta, d)
    init_moments(m_b1, v_b1, d_ff); init_moments(m_b2, v_b2, d)
}
function init_moments(m,v,n, i){ for(i=1;i<=n;i++){m[i]=0;v[i]=0} }
function adamw_update(param, grad, m, v, n, decay,  i, m_hat, v_hat, update) {
    for (i=1; i<=n; i++) {
        if (grad[i] == 0) continue
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]
        m_hat = m[i] / (1 - beta1^adam_t)
        v_hat = v[i] / (1 - beta2^adam_t)
        update = lr * m_hat / (sqrt(v_hat) + epsilon)
        param[i] -= update
        if (decay) param[i] -= lr * weight_decay * param[i]
    }
}
function adamw_update_all() {
    adam_t++
    adamw_update(E, dE_acc, m_E, v_E, V*d, 1); adamw_update(P, dP_acc, m_P, v_P, block_size*d, 1)
    adamw_update(WQ, dWQ_acc, m_WQ, v_WQ, d*d, 1); adamw_update(WK, dWK_acc, m_WK, v_WK, d*d, 1)
    adamw_update(WV, dWV_acc, m_WV, v_WV, d*d, 1); adamw_update(W_out, dW_out_acc, m_W_out, v_W_out, d*d, 1)
    adamw_update(W1, dW1_acc, m_W1, v_W1, d*d_ff, 1); adamw_update(W2, dW2_acc, m_W2, v_W2, d_ff*d, 1)
    adamw_update(LN1_gamma, dLN1_gamma, m_LN1_gamma, v_LN1_gamma, d, 1)
    adamw_update(LN1_beta, dLN1_beta, m_LN1_beta, v_LN1_beta, d, 0)
    adamw_update(LN2_gamma, dLN2_gamma, m_LN2_gamma, v_LN2_gamma, d, 1)
    adamw_update(LN2_beta, dLN2_beta, m_LN2_beta, v_LN2_beta, d, 0)
    adamw_update(b1, db1_acc, m_b1, v_b1, d_ff, 0); adamw_update(b2, db2_acc, m_b2, v_b2, d, 0)
}
function reset_gradients() {
    delete dE_acc; delete dP_acc; delete dWQ_acc; delete dWK_acc; delete dWV_acc
    delete dW_out_acc; delete dW1_acc; delete dW2_acc; delete dLN1_gamma
    delete dLN1_beta; delete dLN2_gamma; delete dLN2_beta; delete db1_acc; delete db2_acc
}

# --------------------
# Neural Network Layers
# --------------------
function layernorm1_forward(block_len,    t, j, sum, sum_sq, mean, var, std_dev, inv_std, val, norm_val) {
    for (t=1; t<=block_len; t++) {
        sum = 0; sum_sq = 0
        for (j=1; j<=d; j++) { val = X[idx(t,j,d)]; sum += val; sum_sq += val*val }
        mean = sum/d; var = sum_sq/d - mean*mean; std_dev = sqrt(var + 1e-5); inv_std = 1/std_dev
        LN1_mean[t] = mean; LN1_inv_std[t] = inv_std
        for (j=1; j<=d; j++) {
            norm_val = (X[idx(t,j,d)] - mean) * inv_std
            LN1_out[idx(t,j,d)] = norm_val * LN1_gamma[j] + LN1_beta[j]
        }
    }
}
function layernorm2_forward(block_len,    t, j, sum, sum_sq, mean, var, std_dev, inv_std, val, norm_val) {
    for (t=1; t<=block_len; t++) {
        sum = 0; sum_sq = 0
        for (j=1; j<=d; j++) { val = res1_out[idx(t,j,d)]; sum += val; sum_sq += val*val }
        mean = sum/d; var = sum_sq/d - mean*mean; std_dev = sqrt(var + 1e-5); inv_std = 1/std_dev
        LN2_mean[t] = mean; LN2_inv_std[t] = inv_std
        for (j=1; j<=d; j++) {
            norm_val = (res1_out[idx(t,j,d)] - mean) * inv_std
            LN2_out[idx(t,j,d)] = norm_val * LN2_gamma[j] + LN2_beta[j]
        }
    }
}
function layernorm1_backward(block_len,    t, j, d_norm_val, sum_d_norm, sum_d_norm_x_norm, mean, inv_std, norm_val, d_out_val) {
    for(j=1;j<=d;j++) { dLN1_gamma[j]=0; dLN1_beta[j]=0 }
    for (t=1; t<=block_len; t++) {
        sum_d_norm=0; sum_d_norm_x_norm=0
        for (j=1; j<=d; j++) {
            mean=LN1_mean[t]; inv_std=LN1_inv_std[t]
            norm_val = (X[idx(t,j,d)]-mean)*inv_std; d_out_val = dLN1_out[idx(t,j,d)]
            dLN1_gamma[j] += d_out_val*norm_val; dLN1_beta[j] += d_out_val
            d_norm_val = d_out_val*LN1_gamma[j]; sum_d_norm += d_norm_val; sum_d_norm_x_norm += d_norm_val*norm_val
        }
        for (j=1; j<=d; j++) {
            mean=LN1_mean[t]; inv_std=LN1_inv_std[t]
            norm_val = (X[idx(t,j,d)]-mean)*inv_std; d_norm_val = dLN1_out[idx(t,j,d)]*LN1_gamma[j]
            dX_from_attn[idx(t,j,d)] = inv_std * (d_norm_val - sum_d_norm/d - norm_val*sum_d_norm_x_norm/d)
        }
    }
}
function layernorm2_backward(block_len,    t, j, d_norm_val, sum_d_norm, sum_d_norm_x_norm, mean, inv_std, norm_val, d_out_val) {
    for(j=1;j<=d;j++) { dLN2_gamma[j]=0; dLN2_beta[j]=0 }
    for (t=1; t<=block_len; t++) {
        sum_d_norm=0; sum_d_norm_x_norm=0
        for (j=1; j<=d; j++) {
            mean=LN2_mean[t]; inv_std=LN2_inv_std[t]
            norm_val = (res1_out[idx(t,j,d)]-mean)*inv_std; d_out_val = dLN2_out[idx(t,j,d)]
            dLN2_gamma[j] += d_out_val*norm_val; dLN2_beta[j] += d_out_val
            d_norm_val = d_out_val*LN2_gamma[j]; sum_d_norm += d_norm_val; sum_d_norm_x_norm += d_norm_val*norm_val
        }
        for (j=1; j<=d; j++) {
            mean=LN2_mean[t]; inv_std=LN2_inv_std[t]
            norm_val = (res1_out[idx(t,j,d)]-mean)*inv_std; d_norm_val = dLN2_out[idx(t,j,d)]*LN2_gamma[j]
            dX_from_ffn[idx(t,j,d)] = inv_std * (d_norm_val - sum_d_norm/d - norm_val*sum_d_norm_x_norm/d)
        }
    }
}
function dropout_forward(inp, out, d, block_len, p, mask,   t, j) {
    if (mode == "generate" || p == 0) {
        # This is the corrected block. The invalid `if (inp != out)` is removed.
        for(t=1; t<=block_len*d; t++) out[t] = inp[t]
        return
    }
    scale = 1 / (1-p)
    for (t=1; t<=block_len*d; t++) {
        if (rand() < p) { out[t] = 0; mask[t] = 0 }
        else { out[t] = inp[t] * scale; mask[t] = scale }
    }
}
function dropout_backward(d_out, d_inp, mask, n,   i) {
    for (i=1; i<=n; i++) d_inp[i] = d_out[i] * mask[i]
}

###############################################################################
# Initialization / hyperparams
###############################################################################
BEGIN{
    srand(1337)
    d = 16; d_ff = 32; n_heads = 4; d_head = d / n_heads
    block_size = 64; epochs = 500; lr = 0.001; clip = 1.0
    bpe_merges = 500; dropout_p = 0.1; weight_decay = 0.01
    gen_tokens = 50; temp = 1.0

    if (mode == "") mode = "train"
    if (mode == "train") print "mode=train"
    else if (mode == "generate") print "mode=generate"
    else { print "Unknown mode:", mode, "(use train|generate)"; exit }
    model_exists = (system("test -f model.awk") == 0)
    if (mode == "train" && model_exists) {
        print "Found model.awk -> loading to continue training..."
        load_model("model.awk")
    }
}

###############################################################################
# Multi-head attention forward (causal)
###############################################################################
function multihead_attention(block_len, cache_for_bwd,    h, t, k, j, i, val, maxs, den, Q_h, K_h, V_h, d_head, score, prob, ctx_hj, out_val) {
    d_head = d/n_heads
    for(i=1;i<=block_len*d;i++) attn_out[i] = 0

    for (t=1; t<=block_len; t++) {
        for(h=0; h<n_heads; h++) {
            for(j=1; j<=d_head; j++) {
                Q_h=0; K_h=0; V_h=0
                for(i=1; i<=d; i++) {
                    val = LN1_out[idx(t,i,d)]
                    Q_h += val * WQ[idx(i, h*d_head+j, d)]
                    K_h += val * WK[idx(i, h*d_head+j, d)]
                    V_h += val * WV[idx(i, h*d_head+j, d)]
                }
                cache_for_bwd["Q",h,t,j] = Q_h; cache_for_bwd["K",h,t,j] = K_h; cache_for_bwd["V",h,t,j] = V_h
            }
        }
    }

    for (h=0; h<n_heads; h++) {
        for (t=1; t<=block_len; t++) {
            for (k=1; k<=t; k++) {
                score = 0
                for(j=1; j<=d_head; j++) score += cache_for_bwd["Q",h,t,j] * cache_for_bwd["K",h,k,j]
                cache_for_bwd["S",h,t,k] = score / sqrt(d_head)
            }
        }
        for (t=1; t<=block_len; t++) {
            maxs = -1e30
            for (k=1; k<=t; k++) if (cache_for_bwd["S",h,t,k] > maxs) maxs = cache_for_bwd["S",h,t,k]
            den = 0
            for (k=1; k<=t; k++) {
                prob = safe_exp(cache_for_bwd["S",h,t,k] - maxs)
                cache_for_bwd["P",h,t,k] = prob; den += prob
            }
            for (k=1; k<=t; k++) cache_for_bwd["P",h,t,k] /= den
        }
        for (t=1; t<=block_len; t++) {
            for (j=1; j<=d_head; j++) {
                ctx_hj = 0
                for (k=1; k<=t; k++) ctx_hj += cache_for_bwd["P",h,t,k] * cache_for_bwd["V",h,k,j]
                cache_for_bwd["C",h,t,j] = ctx_hj
            }
        }
    }

    for (t=1; t<=block_len; t++) {
        for (j=1; j<=d; j++) {
            out_val = 0
            for (h=0; h<n_heads; h++) {
                for (i=1; i<=d_head; i++) {
                    out_val += cache_for_bwd["C",h,t,i] * W_out[idx(h*d_head+i, j, d)]
                }
            }
            attn_out[idx(t,j,d)] = out_val
        }
    }
}


###############################################################################
# Forward + backprop for a single block
###############################################################################
function process_block(start,end, is_training,    block_len, t, j, i, p, loss, y, val, m, den, py_log, g, z1, y2_ffn, d_val, tok_id) {
    block_len = end - start + 1
    if (is_training) reset_gradients()

    for(t=1;t<=block_len;t++) for(j=1;j<=d;j++){ X[idx(t,j,d)] = E[idx(seq[start+t-1],j,d)] + P[idx(t,j,d)] }

    layernorm1_forward(block_len)
    multihead_attention(block_len, attn_cache)
    dropout_forward(attn_out, attn_drop_out, d, block_len, dropout_p, attn_drop_mask)
    for(t=1;t<=block_len*d;t++) res1_out[t] = X[t] + attn_drop_out[t]

    layernorm2_forward(block_len)
    for(t=1;t<=block_len;t++) for(j=1;j<=d_ff;j++){
        z1 = b1[j]
        for(i=1;i<=d;i++) z1 += LN2_out[idx(t,i,d)] * W1[idx(i,j,d_ff)]
        ffn_z1[idx(t,j,d_ff)] = z1
        if (z1>0) { ffn_relu[idx(t,j,d_ff)]=z1; ffn_relu_mask[idx(t,j,d_ff)]=1 }
        else { ffn_relu[idx(t,j,d_ff)]=0; ffn_relu_mask[idx(t,j,d_ff)]=0 }
    }
    for(t=1;t<=block_len;t++) for(j=1;j<=d;j++){
        y2_ffn = b2[j]
        for(i=1;i<=d_ff;i++) y2_ffn += ffn_relu[idx(t,i,d_ff)] * W2[idx(i,j,d)]
        Y2[idx(t,j,d)] = y2_ffn
    }
    dropout_forward(Y2, ffn_drop_out, d, block_len, dropout_p, ffn_drop_mask)
    for(t=1;t<=block_len*d;t++) Y2[t] = res1_out[t] + ffn_drop_out[t]

    for(t=1;t<=block_len;t++) for(j=1;j<=V;j++){ logits[idx(t,j,V)] = dot_row(Y2,t,E,j,d) }

    loss = 0
    for(t=1;t<=block_len;t++){
        y = tgt[start+t-1]; if(y==0) continue
        m = -1e30
        for(j=1;j<=V;j++) if(logits[idx(t,j,V)] > m) m = logits[idx(t,j,V)]
        den = 0
        for(j=1;j<=V;j++) den += safe_exp(logits[idx(t,j,V)] - m)
        py_log = logits[idx(t,y,V)] - (m + log(den)); loss -= py_log

        if (!is_training) continue

        for(j=1;j<=V;j++){
            p = safe_div(safe_exp(logits[idx(t,j,V)] - m), den)
            g = p - (j==y); dY2[idx(t,j,d)] = 0
            for(i=1;i<=d;i++){ dE_acc[idx(j,i,d)] += g * Y2[idx(t,i,d)]; dY2[idx(t,i,d)] += g * E[idx(j,i,d)] }
        }
    }
    if (!is_training) return loss/block_len

    for(t=1;t<=block_len*d;t++) { dres1_out[t] = dY2[t]; d_ffn_drop_out[t] = dY2[t] }
    dropout_backward(d_ffn_drop_out, d_ffn_drop_out, ffn_drop_mask, block_len*d)

    for(t=1;t<=block_len;t++){
        for(i=1;i<=d_ff;i++){
            d_ffn_relu[idx(t,i,d_ff)] = 0
            for(j=1;j<=d;j++){
                d_val = d_ffn_drop_out[idx(t,j,d)]
                dW2_acc[idx(i,j,d)] += ffn_relu[idx(t,i,d_ff)] * d_val
                d_ffn_relu[idx(t,i,d_ff)] += W2[idx(i,j,d)] * d_val
            }
        }
        for(j=1;j<=d;j++) db2_acc[j] += d_ffn_drop_out[idx(t,j,d)]
    }

    for(t=1;t<=block_len*d_ff;t++) d_ffn_z1[t] = d_ffn_relu[t] * ffn_relu_mask[t]

    for(t=1;t<=block_len;t++){
        for(i=1;i<=d;i++){
            dLN2_out[idx(t,i,d)] = 0
            for(j=1;j<=d_ff;j++){
                d_val = d_ffn_z1[idx(t,j,d_ff)]
                dW1_acc[idx(i,j,d_ff)] += LN2_out[idx(t,i,d)] * d_val
                dLN2_out[idx(t,i,d)] += W1[idx(i,j,d_ff)] * d_val
            }
        }
        for(j=1;j<=d_ff;j++) db1_acc[j] += d_ffn_z1[idx(t,j,d_ff)]
    }

    layernorm2_backward(block_len)
    for(t=1;t<=block_len*d;t++) dres1_out[t] += dX_from_ffn[t]

    for(t=1;t<=block_len*d;t++) { dX[t] = dres1_out[t]; d_attn_drop_out[t] = dres1_out[t] }
    dropout_backward(d_attn_drop_out, d_attn_drop_out, attn_drop_mask, block_len*d)

    for(i=1;i<=block_len*d;i++) dLN1_out[i] = d_attn_drop_out[i]
    layernorm1_backward(block_len)
    for(t=1;t<=block_len*d;t++) dX[t] += dX_from_attn[t]

    for(t=1;t<=block_len;t++) {
        tok_id = seq[start+t-1]
        for(j=1;j<=d;j++) {
            d_val = dX[idx(t,j,d)]
            dE_acc[idx(tok_id, j, d)] += d_val; dP_acc[idx(t, j, d)] += d_val
        }
    }

    clamp_inplace(dE_acc, V*d, clip); clamp_inplace(dP_acc, block_size*d, clip)
    clamp_inplace(dW1_acc, d*d_ff, clip); clamp_inplace(dW2_acc, d_ff*d, clip)
    adamw_update_all()

    return loss/block_len
}

function forward_logits_for_context(ctx_ids, L, logits_out,    t, j, i, z1, y2_ffn, last_t) {
    for(t=1;t<=L;t++) for(j=1;j<=d;j++) X[idx(t,j,d)] = E[idx(ctx_ids[t],j,d)] + P[idx(t,j,d)]

    layernorm1_forward(L)
    multihead_attention(L, attn_cache)
    dropout_forward(attn_out, attn_drop_out, d, L, 0, _)
    for(t=1;t<=L*d;t++) res1_out[t] = X[t] + attn_drop_out[t]

    layernorm2_forward(L)
    for(t=1;t<=L;t++) for(j=1;j<=d_ff;j++){
        z1 = b1[j]
        for(i=1;i<=d;i++) z1 += LN2_out[idx(t,i,d)] * W1[idx(i,j,d_ff)]
        ffn_relu[idx(t,j,d_ff)] = (z1>0)?z1:0
    }
    for(t=1;t<=L;t++) for(j=1;j<=d;j++){
        y2_ffn = b2[j]
        for(i=1;i<=d_ff;i++) y2_ffn += ffn_relu[idx(t,i,d_ff)] * W2[idx(i,j,d)]
        Y2[idx(t,j,d)] = y2_ffn
    }
    dropout_forward(Y2, ffn_drop_out, d, L, 0, _)
    for(t=1;t<=L*d;t++) Y2[t] = res1_out[t] + ffn_drop_out[t]

    last_t = L
    for(j=1;j<=V;j++) logits_out[j] = dot_row(Y2,last_t,E,j,d)
}

# Saving / Loading model
function save_model(fname,    i) {
    printf "" > fname
    print "V=" V >> fname; print "d=" d >> fname; print "d_ff=" d_ff >> fname
    print "n_heads=" n_heads >> fname; print "block_size=" block_size >> fname
    print "dropout_p=" dropout_p >> fname; print "weight_decay=" weight_decay >> fname
    print "num_ranked_merges=" num_ranked_merges >> fname; print "adam_t=" adam_t >> fname
    print "beta1=" beta1 >> fname; print "beta2=" beta2 >> fname; print "epsilon=" epsilon >> fname
    for (i in token2id) print "token2id[\"" i "\"]=" token2id[i] >> fname
    for (i in id2token) print "id2token[" i "]=\"" id2token[i] "\"" >> fname
    for (i=1;i<=num_ranked_merges;i++) {
        print "ranked_merges[" i ",\"p1\"]=\"" ranked_merges[i, "p1"] "\"" >> fname
        print "ranked_merges[" i ",\"p2\"]=\"" ranked_merges[i, "p2"] "\"" >> fname
        print "ranked_merges[" i ",\"new\"]=\"" ranked_merges[i, "new"] "\"" >> fname
    }
    dump_array("E", E, V*d, fname); dump_array("P", P, block_size*d, fname)
    dump_array("WQ", WQ, d*d, fname); dump_array("WK", WK, d*d, fname); dump_array("WV", WV, d*d, fname)
    dump_array("W_out", W_out, d*d, fname); dump_array("W1", W1, d*d_ff, fname); dump_array("W2", W2, d_ff*d, fname)
    dump_array("LN1_gamma", LN1_gamma, d, fname); dump_array("LN1_beta", LN1_beta, d, fname)
    dump_array("LN2_gamma", LN2_gamma, d, fname); dump_array("LN2_beta", LN2_beta, d, fname)
    dump_array("b1", b1, d_ff, fname); dump_array("b2", b2, d, fname)
    dump_array("m_E", m_E, V*d, fname); dump_array("v_E", v_E, V*d, fname)
}
function dump_array(name, A, n, fname,  i){ for (i=1;i<=n;i++) print name "[" i "]=" A[i] >> fname }
function load_model(fname,   line, key, val, kv, m, arr, idxv, idx_str) {
    while((getline line < fname) > 0){
        if (line ~ /^[#[:space:]]*$/) continue
        if (split(line, kv, "=") < 2) continue
        key = kv[1]; val = substr(line, index(line, "=")+1)
        if (match(key, /^[a-zA-Z_][a-zA-Z0-9_]*\[/)) {
            idx_str = substr(key, RSTART, RLENGTH)
            arr = substr(idx_str, 1, length(idx_str)-1)
            if (arr == "token2id") { match(key, /\["(.*)"\]/, m); token2id[m[1]] = val+0 }
            else if (arr == "id2token") { match(key, /\[([0-9]+)\]/, m); gsub(/"/,"",val); id2token[m[1]+0]=val }
            else if (arr == "ranked_merges") { match(key, /\[([0-9]+),"(.*)"\]/, m); gsub(/"/,"",val); ranked_merges[m[1]+0,m[2]]=val }
            else {
                match(key, /\[([0-9]+)\]/, m); idxv = m[1]+0
                if(arr=="E")E[idxv]=val+0; else if(arr=="P")P[idxv]=val+0
                else if(arr=="WQ")WQ[idxv]=val+0; else if(arr=="WK")WK[idxv]=val+0
                else if(arr=="WV")WV[idxv]=val+0; else if(arr=="W_out")W_out[idxv]=val+0
                else if(arr=="W1")W1[idxv]=val+0; else if(arr=="W2")W2[idxv]=val+0
                else if(arr=="LN1_gamma")LN1_gamma[idxv]=val+0; else if(arr=="LN1_beta")LN1_beta[idxv]=val+0
                else if(arr=="LN2_gamma")LN2_gamma[idxv]=val+0; else if(arr=="LN2_beta")LN2_beta[idxv]=val+0
                else if(arr=="b1")b1[idxv]=val+0; else if(arr=="b2")b2[idxv]=val+0
                else if(arr=="m_E")m_E[idxv]=val+0; else if(arr=="v_E")v_E[idxv]=val+0
            }
        } else {
            if(key=="V")V=val+0; else if(key=="d")d=val+0; else if(key=="d_ff")d_ff=val+0
            else if(key=="n_heads")n_heads=val+0; else if(key=="block_size")block_size=val+0
            else if(key=="dropout_p")dropout_p=val+0; else if(key=="weight_decay")weight_decay=val+0
            else if(key=="num_ranked_merges")num_ranked_merges=val+0; else if(key=="adam_t")adam_t=val+0
            else if(key=="beta1")beta1=val+0; else if(key=="beta2")beta2=val+0; else if(key=="epsilon")epsilon=val+0
        }
    }
    close(fname)
}


# Training / Generation control
END{
    if (mode == "train") {
        full_text = ""
        while ((getline line < "input.txt") > 0) { full_text = full_text line " " }
        close("input.txt")
        if (length(full_text) < 100) { print "Error: input.txt is too short"; exit }

        if (!model_exists) {
            print "No existing model -> training BPE from scratch with", bpe_merges, "merges."
            train_bpe(full_text, bpe_merges)
            adamw_init_optimizer()
        } else {
            print "Using BPE merges from loaded model (num_ranked_merges=" num_ranked_merges ")."
            print "Loaded AdamW optimizer state: t=" adam_t
        }

        delete seq_full; T_full = tokenize_bpe(full_text, "seq_full")
        split_idx = int(T_full * 0.9)
        T = split_idx; T_val = T_full - split_idx
        for(i=1; i<=T; i++) { seq[i] = seq_full[i]; tgt[i]=seq_full[i+1] }
        for(i=1; i<=T_val; i++) { val_seq[i] = seq_full[split_idx+i]; val_tgt[i]=seq_full[split_idx+i+1] }
        tgt[T] = 0; val_tgt[T_val] = 0

        if (!model_exists) {
            print "Initializing parameters from scratch."
            for(i=1;i<=V*d;i++) E[i]=(rand()*2-1)*0.1; for(i=1;i<=block_size*d;i++) P[i]=(rand()*2-1)*0.01
            for(i=1;i<=d*d;i++) { WQ[i]=(rand()*2-1)*0.1; WK[i]=(rand()*2-1)*0.1; WV[i]=(rand()*2-1)*0.1; W_out[i]=(rand()*2-1)*0.1 }
            for(i=1;i<=d*d_ff;i++) W1[i]=(rand()*2-1)*0.1; for(i=1;i<=d_ff*d;i++) W2[i]=(rand()*2-1)*0.1
            for(i=1;i<=d;i++) { LN1_gamma[i]=1; LN1_beta[i]=0; LN2_gamma[i]=1; LN2_beta[i]=0 }
            for(i=1;i<=d_ff;i++) b1[i]=0; for(i=1;i<=d;i++) b2[i]=0
        }

        printf "hyperparams: d=%d d_ff=%d n_heads=%d epochs=%d lr=%.4f dropout=%.2f\n", d, d_ff, n_heads, epochs, lr, dropout_p
        printf "Data: train tokens=%d, val tokens=%d, vocab size=%d\n", T, T_val, V
        step = 0
        for(epoch=1; epoch<=epochs; epoch++){
            epoch_loss = 0; block_count = 0
            for(start=1; start<T; start+=block_size){
                end = start + block_size - 1
                if (end >= T) end = T - 1
                if (start > end) continue
                step++
                loss = process_block(start, end, 1)
                epoch_loss += loss; block_count++
            }
            if (epoch % 5 == 0 || epoch == 1) {
                val_loss = 0; val_blocks = 0
                for(start_val=1; start_val<T_val; start_val+=block_size) {
                    end_val = start_val + block_size - 1; if (end_val >= T_val) end_val = T_val-1
                    if(start_val>end_val) continue
                    tmp_T=T; T=T_val; for(i=1;i<=T_val;i++){tmp_s[i]=seq[i];tmp_t[i]=tgt[i];seq[i]=val_seq[i];tgt[i]=val_tgt[i]}
                    val_loss += process_block(start_val, end_val, 0)
                    T=tmp_T; for(i=1;i<=T_val;i++){seq[i]=tmp_s[i];tgt[i]=tmp_t[i]}
                    val_blocks++
                }
                if (block_count > 0 && val_blocks > 0) printf("Epoch %d | Train Loss %.4f | Val Loss %.4f\n", epoch, epoch_loss/block_count, val_loss/val_blocks)
            }
        }
        save_model("model.awk")
        print "Saved model to model.awk with AdamW optimizer state"
    }

    else if (mode == "generate") {
        if (system("test -f model.awk") != 0) { print "Error: model.awk not found."; exit }
        load_model("model.awk")
        print "Enter a prompt:"
        getline prompt < "-"
        if (prompt == "") prompt = " "

        delete out_idx; out_len = tokenize_bpe(prompt, "out_idx")
        if (out_len == 0) { print "Prompt produced no tokens."; exit }

        print "\n--- Generation ---"
        for(i=1;i<=out_len;i++) printf("%s", id2token[out_idx[i]])

        for(t_gen=1; t_gen<=gen_tokens; t_gen++){
            ctx_len = (out_len<block_size)?out_len:block_size
            s = out_len - ctx_len + 1
            for(i=1;i<=ctx_len;i++) ctx_ids[i] = out_idx[s + i - 1]

            forward_logits_for_context(ctx_ids, ctx_len, logits)

            m = -1e30
            for(j=1;j<=V;j++) if (logits[j] > m) m = logits[j]
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
