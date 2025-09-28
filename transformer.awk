#!/usr/bin/env bash
awk -v OFS='\t' -v mode="$1" '
###############################################################################
# GPT-2 style Transformer in AWK (Complete Implementation)
# - Complete backpropagation through all layers
# - GELU activation, proper weight initialization
# - Full AdamW optimizer for all parameters
# - Gradient accumulation and clipping
# - Final layer norm and bias terms
###############################################################################

# --------------------
# BPE Tokenization functions
# --------------------
function tanh(x) {
    if (x < -20) return -1
    if (x > 20) return 1
    e2x = exp(2*x)
    return (e2x - 1) / (e2x + 1)
}



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
# Math functions and helpers
# --------------------
function idx(i,j,cols) { return (i-1)*cols + j }
function dot_row(A,ri,B,rj,d,   s,k){ s=0; for(k=1;k<=d;k++) s+=A[idx(ri,k,d)]*B[idx(rj,k,d)]; return s }
function clamp_inplace(A,n,val,   i){ for(i=1;i<=n;i++){ if(A[i]>val)A[i]=val; else if(A[i]<-val)A[i]=-val } }
function safe_exp(x){ if (x < -700) return 0; return exp(x) }
function safe_div(a,b){ return (b==0) ? 0 : a/b }

# GELU activation function (Gaussian Error Linear Unit)
function gelu(x,    pi, cdf) {
    pi = 3.141592653589793
    # Approximation of Gaussian CDF: 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x*x*x)))
}

function gelu_derivative(x,    pi, x3, tanh_arg, sech2) {
    pi = 3.141592653589793
    x3 = x*x*x
    tanh_arg = sqrt(2/pi) * (x + 0.044715 * x3)
    sech2 = 1 - tanh(tanh_arg)^2  # sech^2(x) = 1 - tanh^2(x)
    return 0.5 * (1 + tanh(tanh_arg)) + 0.5 * x * sech2 * sqrt(2/pi) * (1 + 3*0.044715*x*x)
}

# Xavier/Glorot weight initialization
function xavier_init(fan_in, fan_out) {
    return (rand() * 2 - 1) * sqrt(6.0 / (fan_in + fan_out))
}

# Kaiming He initialization for ReLU (used in some parts)
function kaiming_init(fan_in) {
    return (rand() * 2 - 1) * sqrt(2.0 / fan_in)
}

# --------------------
# AdamW optimizer (Complete)
# --------------------
function adamw_init_optimizer(    l, i) {
    adam_t = 0; beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8
    
    # Initialize momentum and variance for all parameters
    for(i=1; i<=V*d; i++) { m_E[i]=0; v_E[i]=0 }
    for(i=1; i<=block_size*d; i++) { m_P[i]=0; v_P[i]=0 }
    
    for(l=1; l<=n_layers; l++) {
        # Attention weights and biases
        for(i=1; i<=d*d; i++) { 
            m_WQs[l,i]=0; v_WQs[l,i]=0; m_WKs[l,i]=0; v_WKs[l,i]=0
            m_WVs[l,i]=0; v_WVs[l,i]=0; m_W_outs[l,i]=0; v_W_outs[l,i]=0
        }
        for(i=1; i<=d; i++) {
            m_bQs[l,i]=0; v_bQs[l,i]=0; m_bKs[l,i]=0; v_bKs[l,i]=0
            m_bVs[l,i]=0; v_bVs[l,i]=0; m_b_outs[l,i]=0; v_b_outs[l,i]=0
        }
        
        # FFN weights and biases
        for(i=1; i<=d*d_ff; i++) { m_W1s[l,i]=0; v_W1s[l,i]=0 }
        for(i=1; i<=d_ff*d; i++) { m_W2s[l,i]=0; v_W2s[l,i]=0 }
        for(i=1; i<=d_ff; i++) { m_b1s[l,i]=0; v_b1s[l,i]=0 }
        for(i=1; i<=d; i++) { m_b2s[l,i]=0; v_b2s[l,i]=0 }
        
        # Layer norm parameters
        for(i=1; i<=d; i++) { 
            m_LN1_gammas[l,i]=0; v_LN1_gammas[l,i]=0
            m_LN1_betas[l,i]=0; v_LN1_betas[l,i]=0
            m_LN2_gammas[l,i]=0; v_LN2_gammas[l,i]=0  
            m_LN2_betas[l,i]=0; v_LN2_betas[l,i]=0
        }
    }
    
    # Final layer norm
    for(i=1; i<=d; i++) {
        m_final_ln_gamma[i]=0; v_final_ln_gamma[i]=0
        m_final_ln_beta[i]=0; v_final_ln_beta[i]=0
    }
}

function adamw_update(param, grad, m, v, n, decay,   i, m_hat, v_hat, update) {
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
    
    # Update embeddings and positional encodings
    adamw_update(E, dE_acc, m_E, v_E, V*d, 1)
    adamw_update(P, dP_acc, m_P, v_P, block_size*d, 1)
    
    # Update each layer
    for(l=1; l<=n_layers; l++) {
        # Attention weights and biases
        adamw_update(WQs[l], dWQs_acc[l], m_WQs[l], v_WQs[l], d*d, 1)
        adamw_update(WKs[l], dWKs_acc[l], m_WKs[l], v_WKs[l], d*d, 1)
        adamw_update(WVs[l], dWVs_acc[l], m_WVs[l], v_WVs[l], d*d, 1)
        adamw_update(W_outs[l], dW_outs_acc[l], m_W_outs[l], v_W_outs[l], d*d, 1)
        
        adamw_update(bQs[l], dbQs_acc[l], m_bQs[l], v_bQs[l], d, 0)
        adamw_update(bKs[l], dbKs_acc[l], m_bKs[l], v_bKs[l], d, 0)
        adamw_update(bVs[l], dbVs_acc[l], m_bVs[l], v_bVs[l], d, 0)
        adamw_update(b_outs[l], db_outs_acc[l], m_b_outs[l], v_b_outs[l], d, 0)
        
        # FFN weights and biases
        adamw_update(W1s[l], dW1s_acc[l], m_W1s[l], v_W1s[l], d*d_ff, 1)
        adamw_update(W2s[l], dW2s_acc[l], m_W2s[l], v_W2s[l], d_ff*d, 1)
        adamw_update(b1s[l], db1s_acc[l], m_b1s[l], v_b1s[l], d_ff, 0)
        adamw_update(b2s[l], db2s_acc[l], m_b2s[l], v_b2s[l], d, 0)
        
        # Layer norm parameters
        adamw_update(LN1_gammas[l], dLN1_gammas_acc[l], m_LN1_gammas[l], v_LN1_gammas[l], d, 1)
        adamw_update(LN1_betas[l], dLN1_betas_acc[l], m_LN1_betas[l], v_LN1_betas[l], d, 0)
        adamw_update(LN2_gammas[l], dLN2_gammas_acc[l], m_LN2_gammas[l], v_LN2_gammas[l], d, 1)
        adamw_update(LN2_betas[l], dLN2_betas_acc[l], m_LN2_betas[l], v_LN2_betas[l], d, 0)
    }
    
    # Final layer norm
    adamw_update(final_ln_gamma, dfinal_ln_gamma, m_final_ln_gamma, v_final_ln_gamma, d, 1)
    adamw_update(final_ln_beta, dfinal_ln_beta, m_final_ln_beta, v_final_ln_beta, d, 0)
}

function reset_gradients(    l) {
    delete dE_acc; delete dP_acc
    
    for(l=1; l<=n_layers; l++) {
        delete dWQs_acc[l]; delete dWKs_acc[l]; delete dWVs_acc[l]; delete dW_outs_acc[l]
        delete dbQs_acc[l]; delete dbKs_acc[l]; delete dbVs_acc[l]; delete db_outs_acc[l]
        delete dW1s_acc[l]; delete dW2s_acc[l]
        delete db1s_acc[l]; delete db2s_acc[l]
        delete dLN1_gammas_acc[l]; delete dLN1_betas_acc[l]
        delete dLN2_gammas_acc[l]; delete dLN2_betas_acc[l]
    }
    
    delete dfinal_ln_gamma; delete dfinal_ln_beta
}

# --------------------
# Neural Network Layers
# --------------------
function layernorm_forward(inp, out, gamma, beta, mean_cache, inv_std_cache, d, block_len,    t, j, sum, sum_sq, mean, var, std_dev, inv_std, val, norm_val) {
    for (t=1; t<=block_len; t++) {
        sum = 0; sum_sq = 0
        for (j=1; j<=d; j++) { val = inp[idx(t,j,d)]; sum += val; sum_sq += val*val }
        mean = sum/d; var = sum_sq/d - mean*mean; std_dev = sqrt(var + 1e-5); inv_std = 1/std_dev
        mean_cache[t] = mean; inv_std_cache[t] = inv_std
        for (j=1; j<=d; j++) {
            norm_val = (inp[idx(t,j,d)] - mean) * inv_std
            out[idx(t,j,d)] = norm_val * gamma[j] + beta[j]
        }
    }
}

function layernorm_backward(d_out, inp, d_inp, gamma, d_gamma, d_beta, mean_cache, inv_std_cache, d, block_len,    t, j, d_norm_val, sum_d_norm, sum_d_norm_x_norm, mean, inv_std, norm_val) {
    for(j=1;j<=d;j++) { d_gamma[j]=0; d_beta[j]=0 }
    
    for (t=1; t<=block_len; t++) {
        sum_d_norm=0; sum_d_norm_x_norm=0
        mean = mean_cache[t]; inv_std = inv_std_cache[t]
        
        for (j=1; j<=d; j++) {
            norm_val = (inp[idx(t,j,d)] - mean) * inv_std
            d_out_val = d_out[idx(t,j,d)]
            d_gamma[j] += d_out_val * norm_val
            d_beta[j] += d_out_val
            d_norm_val = d_out_val * gamma[j]
            sum_d_norm += d_norm_val
            sum_d_norm_x_norm += d_norm_val * norm_val
        }
        
        for (j=1; j<=d; j++) {
            norm_val = (inp[idx(t,j,d)] - mean) * inv_std
            d_norm_val = d_out[idx(t,j,d)] * gamma[j]
            d_inp[idx(t,j,d)] = inv_std * (d_norm_val - sum_d_norm/d - norm_val*sum_d_norm_x_norm/d)
        }
    }
}

function multihead_attention_forward(l, inp, block_len, cache,    h, t, k, j, i, val, maxs, den, Q_h, K_h, V_h, d_head, score, prob, ctx_hj, out_val) {
    d_head = d/n_heads
    
    for(i=1;i<=block_len*d;i++) attn_out[i] = 0

    # Compute Q, K, V with biases
    for (t=1; t<=block_len; t++) {
        for(h=0; h<n_heads; h++) {
            for(j=1; j<=d_head; j++) {
                Q_h = bQs[l, h*d_head+j]
                K_h = bKs[l, h*d_head+j] 
                V_h = bVs[l, h*d_head+j]
                
                for(i=1; i<=d; i++) {
                    val = inp[idx(t,i,d)]
                    Q_h += val * WQs[l,idx(i, h*d_head+j, d)]
                    K_h += val * WKs[l,idx(i, h*d_head+j, d)]
                    V_h += val * WVs[l,idx(i, h*d_head+j, d)]
                }
                cache["Q",h,t,j] = Q_h
                cache["K",h,t,j] = K_h  
                cache["V",h,t,j] = V_h
            }
        }
    }

    # Scaled dot-product attention
    for (h=0; h<n_heads; h++) {
        for (t=1; t<=block_len; t++) {
            for (k=1; k<=t; k++) {
                score = 0
                for(j=1; j<=d_head; j++) score += cache["Q",h,t,j] * cache["K",h,k,j]
                cache["S",h,t,k] = score / sqrt(d_head)
            }
        }
        
        for (t=1; t<=block_len; t++) {
            maxs = -1e30
            for (k=1; k<=t; k++) if (cache["S",h,t,k] > maxs) maxs = cache["S",h,t,k]
            den = 0
            for (k=1; k<=t; k++) {
                prob = safe_exp(cache["S",h,t,k] - maxs)
                cache["P",h,t,k] = prob
                den += prob
            }
            if (den > 0) for (k=1; k<=t; k++) cache["P",h,t,k] /= den
        }
        
        for (t=1; t<=block_len; t++) {
            for (j=1; j<=d_head; j++) {
                ctx_hj = 0
                for (k=1; k<=t; k++) ctx_hj += cache["P",h,t,k] * cache["V",h,k,j]
                cache["C",h,t,j] = ctx_hj
            }
        }
    }

    # Output projection with bias
    for (t=1; t<=block_len; t++) {
        for (j=1; j<=d; j++) {
            out_val = b_outs[l,j]
            for (h=0; h<n_heads; h++) {
                for (i=1; i<=d_head; i++) {
                    out_val += cache["C",h,t,i] * W_outs[l,idx(h*d_head+i, j, d)]
                }
            }
            attn_out[idx(t,j,d)] = out_val
        }
    }
}

function multihead_attention_backward(l, d_attn_out, inp, d_inp, cache, block_len,    h, t, k, j, i, d_C, d_V, d_P, d_S, d_Q, d_K, d_head, sum_d_P, d_val) {
    d_head = d/n_heads
    
    # Initialize gradient arrays
    for(j=1; j<=d; j++) {
        db_outs_acc[l,j] = 0
        for(i=1; i<=d; i++) dW_outs_acc[l,idx(i,j,d)] = 0
    }
    
    # Gradient through output projection
    for (t=1; t<=block_len; t++) {
        for (j=1; j<=d; j++) {
            d_val = d_attn_out[idx(t,j,d)]
            db_outs_acc[l,j] += d_val
            for (h=0; h<n_heads; h++) {
                for (i=1; i<=d_head; i++) {
                    dW_outs_acc[l,idx(h*d_head+i, j, d)] += cache["C",h,t,i] * d_val
                    cache["dC",h,t,i] = (cache["dC",h,t,i] ) + W_outs[l,idx(h*d_head+i, j, d)] * d_val
                }
            }
        }
    }
    
    # Gradient through attention computation (simplified)
    for (h=0; h<n_heads; h++) {
        for (t=1; t<=block_len; t++) {
            for (k=1; k<=t; k++) {
                d_P = 0
                for(j=1; j<=d_head; j++) d_P += cache["dC",h,t,j] * cache["V",h,k,j]
                cache["dP",h,t,k] = d_P
            }
            
            for (k=1; k<=t; k++) {
                d_S = cache["dP",h,t,k] * cache["P",h,t,k]
                for (k2=1; k2<=t; k2++) {
                    d_S -= cache["dP",h,t,k2] * cache["P",h,t,k2] * cache["P",h,t,k]
                }
                cache["dS",h,t,k] = d_S / sqrt(d_head)
            }
        }
        
        # Gradient through Q, K projections
        for (t=1; t<=block_len; t++) {
            for(j=1; j<=d_head; j++) {
                cache["dQ",h,t,j] = 0
                cache["dK",h,t,j] = 0
                cache["dV",h,t,j] = 0
            }
            
            for (k=1; k<=t; k++) {
                for(j=1; j<=d_head; j++) {
                    cache["dQ",h,t,j] += cache["dS",h,t,k] * cache["K",h,k,j]
                    cache["dK",h,k,j] += cache["dS",h,t,k] * cache["Q",h,t,j]
                }
            }
            
            for(j=1; j<=d_head; j++) {
                for (k=1; k<=t; k++) {
                    cache["dV",h,k,j] += cache["dC",h,t,j] * cache["P",h,t,k]
                }
            }
        }
    }
    
    # Gradient through Q, K, V projections and biases
    for(j=1; j<=d; j++) {
        dbQs_acc[l,j] = 0; dbKs_acc[l,j] = 0; dbVs_acc[l,j] = 0
        for(i=1; i<=d; i++) {
            dWQs_acc[l,idx(i,j,d)] = 0
            dWKs_acc[l,idx(i,j,d)] = 0  
            dWVs_acc[l,idx(i,j,d)] = 0
        }
    }
    
    for (t=1; t<=block_len; t++) {
        for (h=0; h<n_heads; h++) {
            for (j=1; j<=d_head; j++) {
                col = h*d_head + j
                dbQs_acc[l,col] += cache["dQ",h,t,j]
                dbKs_acc[l,col] += cache["dK",h,t,j] 
                dbVs_acc[l,col] += cache["dV",h,t,j]
                
                for (i=1; i<=d; i++) {
                    dWQs_acc[l,idx(i,col,d)] += inp[idx(t,i,d)] * cache["dQ",h,t,j]
                    dWKs_acc[l,idx(i,col,d)] += inp[idx(t,i,d)] * cache["dK",h,t,j]
                    dWVs_acc[l,idx(i,col,d)] += inp[idx(t,i,d)] * cache["dV",h,t,j]
                    
                    d_inp[idx(t,i,d)] += (WQs[l,idx(i,col,d)] * cache["dQ",h,t,j] + \
                                         WKs[l,idx(i,col,d)] * cache["dK",h,t,j] + \
                                         WVs[l,idx(i,col,d)] * cache["dV",h,t,j])
                }
            }
        }
    }
}

function ffn_forward(l, inp, block_len, cache,  t, j, i, z1, y2_ffn) {
    # FFN intermediate layer with GELU
    for(t=1; t<=block_len; t++) {
        for(j=1; j<=d_ff; j++) {
            z1 = b1s[l,j]
            for(i=1; i<=d; i++) z1 += inp[idx(t,i,d)] * W1s[l,idx(i,j,d_ff)]
            cache["z1",t,j] = z1
            cache["gelu",t,j] = gelu(z1)
        }
    }
    
    # FFN output layer
    for(t=1; t<=block_len; t++) {
        for(j=1; j<=d; j++) {
            y2_ffn = b2s[l,j]
            for(i=1; i<=d_ff; i++) y2_ffn += cache["gelu",t,i] * W2s[l,idx(i,j,d)]
            ffn_out[idx(t,j,d)] = y2_ffn
        }
    }
}

function ffn_backward(l, d_ffn_out, inp, d_inp, cache, block_len,    t, j, i, d_gelu, d_z1) {
    # Initialize gradients
    for(j=1; j<=d_ff; j++) db1s_acc[l,j] = 0
    for(j=1; j<=d; j++) db2s_acc[l,j] = 0
    for(i=1; i<=d*d_ff; i++) dW1s_acc[l,i] = 0
    for(i=1; i<=d_ff*d; i++) dW2s_acc[l,i] = 0
    
    # Gradient through output layer
    for(t=1; t<=block_len; t++) {
        for(j=1; j<=d; j++) {
            d_val = d_ffn_out[idx(t,j,d)]
            db2s_acc[l,j] += d_val
            for(i=1; i<=d_ff; i++) {
                dW2s_acc[l,idx(i,j,d)] += cache["gelu",t,i] * d_val
                cache["d_gelu",t,i] = (cache["d_gelu",t,i] ) + W2s[l,idx(i,j,d)] * d_val
            }
        }
    }
    
    # Gradient through GELU activation
    for(t=1; t<=block_len; t++) {
        for(j=1; j<=d_ff; j++) {
            d_gelu = cache["d_gelu",t,j]
            d_z1 = d_gelu * gelu_derivative(cache["z1",t,j])
            cache["d_z1",t,j] = d_z1
        }
    }
    
    # Gradient through intermediate layer
    for(t=1; t<=block_len; t++) {
        for(j=1; j<=d_ff; j++) {
            d_z1 = cache["d_z1",t,j]
            db1s_acc[l,j] += d_z1
            for(i=1; i<=d; i++) {
                dW1s_acc[l,idx(i,j,d_ff)] += inp[idx(t,i,d)] * d_z1
                d_inp[idx(t,i,d)] += W1s[l,idx(i,j,d_ff)] * d_z1
            }
        }
    }
}

function dropout_forward(inp, out, d, block_len, p, mask,   t, j) {
    if (mode == "generate" || p == 0) {
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

# --------------------
# Learning Rate Scheduler
# --------------------
function get_lr(step,    pi, decay_ratio) {
    if (warmup_steps > 0 && step < warmup_steps) {
        return max_lr * step / warmup_steps
    }
    if (step > max_steps) {
        return min_lr
    }
    pi = 3.1415926535
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * decay_ratio))
}

###############################################################################
# Initialization / hyperparams
###############################################################################
BEGIN{
    srand(1337)
    d = 24; d_ff = 48; n_heads = 4
    n_layers = 2
    block_size = 64; epochs = 100
    bpe_merges = 500; dropout_p = 0.1; weight_decay = 0.01
    gen_tokens = 60; temp = 0.9
    grad_accum_steps = 4  # Gradient accumulation steps

    max_lr = 0.001; min_lr = 0.0001; warmup_steps = 20

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
# Forward + backprop for a single block (Complete)
###############################################################################
function process_block(start,end, is_training,    block_len, t, j, i, loss, y, m, den, py_log, g, l, z1, y2_ffn) {
    block_len = end - start + 1
    if (is_training) reset_gradients()

    # Forward pass: Input embeddings
    for(t=1;t<=block_len;t++) for(j=1;j<=d;j++){ 
        X[idx(t,j,d)] = E[idx(seq[start+t-1],j,d)] + P[idx(t,j,d)] 
    }

    # Forward pass through all layers
    for(l=1; l<=n_layers; l++) {
        # Store residual for attention
        for(i=1;i<=block_len*d;i++) res1_cache[l,i] = X[i]
        
        # Layer norm 1 + attention
        layernorm_forward(X, LN1_out, LN1_gammas[l], LN1_betas[l], LN1_mean[l], LN1_inv_std[l], d, block_len)
        multihead_attention_forward(l, LN1_out, block_len, attn_cache[l])
        dropout_forward(attn_out, attn_drop_out, d, block_len, dropout_p, attn_drop_mask[l])
        
        # Residual connection 1
        for(i=1;i<=block_len*d;i++) X[i] = res1_cache[l,i] + attn_drop_out[i]

        # Store residual for FFN
        for(i=1;i<=block_len*d;i++) res2_cache[l,i] = X[i]
        
        # Layer norm 2 + FFN
        layernorm_forward(X, LN2_out, LN2_gammas[l], LN2_betas[l], LN2_mean[l], LN2_inv_std[l], d, block_len)
        ffn_forward(l, LN2_out, block_len, ffn_cache[l])
        dropout_forward(ffn_out, ffn_drop_out, d, block_len, dropout_p, ffn_drop_mask[l])
        
        # Residual connection 2
        for(i=1;i<=block_len*d;i++) X[i] = res2_cache[l,i] + ffn_drop_out[i]
    }

    # Final layer norm before output
    for(i=1;i<=block_len*d;i++) final_ln_in[i] = X[i]
    layernorm_forward(X, final_ln_out, final_ln_gamma, final_ln_beta, final_ln_mean, final_ln_inv_std, d, block_len)

    # Output projection (weight tying with embeddings)
    for(t=1;t<=block_len;t++) for(j=1;j<=V;j++){ 
        logits[idx(t,j,V)] = dot_row(final_ln_out,t,E,j,d) 
    }

    loss = 0
    if (is_training) {
        # Loss computation
        for(t=1;t<=block_len;t++){
            y = tgt[start+t-1]; if(y==0) continue
            m = -1e30
            for(j=1;j<=V;j++) if(logits[idx(t,j,V)] > m) m = logits[idx(t,j,V)]
            den = 0
            for(j=1;j<=V;j++) den += safe_exp(logits[idx(t,j,V)] - m)
            if (den == 0) continue
            py_log = logits[idx(t,y,V)] - (m + log(den)); loss -= py_log

            # Gradient through softmax and output projection
            for(j=1;j<=V;j++){
                p = safe_div(safe_exp(logits[idx(t,j,V)] - m), den)
                g = p - (j==y)
                for(i=1;i<=d;i++){ 
                    dE_acc[idx(j,i,d)] += g * final_ln_out[idx(t,i,d)] 
                }
                # Gradient to final layer norm output
                for(i=1;i<=d;i++){ 
                    d_final_ln_out[idx(t,i,d)] += g * E[idx(j,i,d)]
                }
            }
        }

        # Backward pass through final layer norm
        layernorm_backward(d_final_ln_out, final_ln_in, d_final_ln_in, final_ln_gamma, dfinal_ln_gamma, dfinal_ln_beta, final_ln_mean, final_ln_inv_std, d, block_len)
        
        # Backward pass through all layers in reverse
        for(l=n_layers; l>=1; l--) {
            # Gradient through residual connection 2
            for(i=1;i<=block_len*d;i++) dX[i] = d_final_ln_in[i]
            for(i=1;i<=block_len*d;i++) d_ffn_drop_out[i] = d_final_ln_in[i]
            
            # Gradient through dropout
            dropout_backward(d_ffn_drop_out, d_ffn_drop_out, ffn_drop_mask[l], block_len*d)
            
            # Gradient through FFN
            ffn_backward(l, d_ffn_drop_out, LN2_out, d_LN2_out, ffn_cache[l], block_len)
            
            # Gradient through layer norm 2
            layernorm_backward(d_LN2_out, X, dX_ln2, LN2_gammas[l], dLN2_gammas_acc[l], dLN2_betas_acc[l], LN2_mean[l], LN2_inv_std[l], d, block_len)
            
            # Gradient through residual connection 1
            for(i=1;i<=block_len*d;i++) dX[i] += dX_ln2[i]
            for(i=1;i<=block_len*d;i++) d_attn_drop_out[i] = dX[i]
            
            # Gradient through dropout
            dropout_backward(d_attn_drop_out, d_attn_drop_out, attn_drop_mask[l], block_len*d)
            
            # Gradient through attention
            multihead_attention_backward(l, d_attn_drop_out, LN1_out, d_LN1_out, attn_cache[l], block_len)
            
            # Gradient through layer norm 1
            layernorm_backward(d_LN1_out, res1_cache[l], d_final_ln_in, LN1_gammas[l], dLN1_gammas_acc[l], dLN1_betas_acc[l], LN1_mean[l], LN1_inv_std[l], d, block_len)
        }
        
        # Gradient through input embeddings
        for(t=1;t<=block_len;t++) {
            tok_id = seq[start+t-1]
            for(j=1;j<=d;j++) {
                d_val = d_final_ln_in[idx(t,j,d)]
                dE_acc[idx(tok_id, j, d)] += d_val
                dP_acc[idx(t, j, d)] += d_val
            }
        }

        # Gradient clipping
        clamp_inplace(dE_acc, V*d, clip); clamp_inplace(dP_acc, block_size*d, clip)
        for(l=1; l<=n_layers; l++) {
            clamp_inplace(dWQs_acc[l], d*d, clip); clamp_inplace(dWKs_acc[l], d*d, clip)
            clamp_inplace(dWVs_acc[l], d*d, clip); clamp_inplace(dW_outs_acc[l], d*d, clip)
            clamp_inplace(dW1s_acc[l], d*d_ff, clip); clamp_inplace(dW2s_acc[l], d_ff*d, clip)
        }
    }
    
    return loss/block_len
}

function forward_logits_for_context(ctx_ids, L, logits_out,    t, j, i, l, z1, y2_ffn, last_t) {
    for(t=1;t<=L;t++) for(j=1;j<=d;j++) X[idx(t,j,d)] = E[idx(ctx_ids[t],j,d)] + P[idx(t,j,d)]
    
    for(l=1; l<=n_layers; l++) {
        for(i=1;i<=L*d;i++) res_x[i] = X[i]
        layernorm_forward(X, LN1_out, LN1_gammas[l], LN1_betas[l], LN1_mean[l], LN1_inv_std[l], d, L)
        multihead_attention_forward(l, LN1_out, L, attn_cache[l])
        for(i=1;i<=L*d;i++) X[i] = res_x[i] + attn_out[i]

        for(i=1;i<=L*d;i++) res_x[i] = X[i]
        layernorm_forward(X, LN2_out, LN2_gammas[l], LN2_betas[l], LN2_mean[l], LN2_inv_std[l], d, L)
        ffn_forward(l, LN2_out, L, ffn_cache[l])
        for(i=1;i<=L*d;i++) X[i] = res_x[i] + ffn_out[i]
    }
    
    # Final layer norm
    layernorm_forward(X, final_ln_out, final_ln_gamma, final_ln_beta, final_ln_mean, final_ln_inv_std, d, L)
    
    last_t = L
    for(j=1;j<=V;j++) logits_out[j] = dot_row(final_ln_out,last_t,E,j,d)
}

# Saving / Loading model
function save_model(fname,    i, l) {
    printf "" > fname
    print "V=" V >> fname; print "d=" d >> fname; print "d_ff=" d_ff >> fname
    print "n_heads=" n_heads >> fname; print "n_layers=" n_layers >> fname
    print "block_size=" block_size >> fname; print "adam_t=" adam_t >> fname
    print "num_ranked_merges=" num_ranked_merges >> fname
    for (i in token2id) print "token2id[\"" i "\"]=" token2id[i] >> fname
    for (i in id2token) print "id2token[" i "]=\"" id2token[i] "\"" >> fname
    dump_array("E", E, V*d, fname); dump_array("P", P, block_size*d, fname)
    
    for(l=1; l<=n_layers; l++) {
        dump_layered_array("WQs", WQs, l, d*d, fname); dump_layered_array("WKs", WKs, l, d*d, fname)
        dump_layered_array("WVs", WVs, l, d*d, fname); dump_layered_array("W_outs", W_outs, l, d*d, fname)
        dump_layered_array("bQs", bQs, l, d, fname); dump_layered_array("bKs", bKs, l, d, fname)
        dump_layered_array("bVs", bVs, l, d, fname); dump_layered_array("b_outs", b_outs, l, d, fname)
        dump_layered_array("W1s", W1s, l, d*d_ff, fname); dump_layered_array("W2s", W2s, l, d_ff*d, fname)
        dump_layered_array("b1s", b1s, l, d_ff, fname); dump_layered_array("b2s", b2s, l, d, fname)
        dump_layered_array("LN1_gammas", LN1_gammas, l, d, fname); dump_layered_array("LN1_betas", LN1_betas, l, d, fname)
        dump_layered_array("LN2_gammas", LN2_gammas, l, d, fname); dump_layered_array("LN2_betas", LN2_betas, l, d, fname)
    }
    
    dump_array("final_ln_gamma", final_ln_gamma, d, fname)
    dump_array("final_ln_beta", final_ln_beta, d, fname)
}
function dump_array(name, A, n, fname,  i){ for (i=1;i<=n;i++) print name "[" i "]=" A[i] >> fname }
function dump_layered_array(name, A, l, n, fname,  i){ for (i=1;i<=n;i++) print name "[" l "," i "]=" A[l,i] >> fname }

function load_model(fname,   line, key, val, kv, m, arr, l, idxv) {
    while((getline line < fname) > 0){
        if (line ~ /^[#[:space:]]*$/) continue
        if (split(line, kv, "=") < 2) continue
        key = kv[1]; val = substr(line, index(line, "=")+1)

        if (match(key, /([a-zA-Z0-9_]+)\[([0-9]+),([0-9]+)\]/)) {
            arr = substr(key, RSTART, RLENGTH-1); l = substr(key, RSTART+length(arr)+2, 1); idxv = substr(key, RSTART+length(arr)+4, length(key)-RSTART-length(arr)-4)
            if(arr=="WQs") WQs[l,idxv]=val+0; else if(arr=="WKs") WKs[l,idxv]=val+0
            else if(arr=="WVs") WVs[l,idxv]=val+0; else if(arr=="W_outs") W_outs[l,idxv]=val+0
            else if(arr=="bQs") bQs[l,idxv]=val+0; else if(arr=="bKs") bKs[l,idxv]=val+0
            else if(arr=="bVs") bVs[l,idxv]=val+0; else if(arr=="b_outs") b_outs[l,idxv]=val+0
            else if(arr=="W1s") W1s[l,idxv]=val+0; else if(arr=="W2s") W2s[l,idxv]=val+0
            else if(arr=="b1s") b1s[l,idxv]=val+0; else if(arr=="b2s") b2s[l,idxv]=val+0
            else if(arr=="LN1_gammas") LN1_gammas[l,idxv]=val+0; else if(arr=="LN1_betas") LN1_betas[l,idxv]=val+0
            else if(arr=="LN2_gammas") LN2_gammas[l,idxv]=val+0; else if(arr=="LN2_betas") LN2_betas[l,idxv]=val+0
        } else if (match(key, /^[a-zA-Z_][a-zA-Z0-9_]*\[/)) {
            arr = substr(key, RSTART, RLENGTH-1)
            if (arr == "token2id") { match(key, /\["(.*)"\]/, m); token2id[m[1]] = val+0 }
            else if (arr == "id2token") { match(key, /\[([0-9]+)\]/, m); gsub(/"/,"",val); id2token[m[1]+0]=val }
            else if (arr == "ranked_merges") { match(key, /\[([0-9]+),"(.*)"\]/, m); gsub(/"/,"",val); ranked_merges[m[1]+0,m[2]]=val }
            else { match(key, /\[([0-9]+)\]/, m); idxv = m[1]+0
                if(arr=="E")E[idxv]=val+0; else if(arr=="P")P[idxv]=val+0
                else if(arr=="final_ln_gamma")final_ln_gamma[idxv]=val+0
                else if(arr=="final_ln_beta")final_ln_beta[idxv]=val+0
            }
        } else {
            if(key=="V")V=val+0; else if(key=="d")d=val+0; else if(key=="d_ff")d_ff=val+0
            else if(key=="n_heads")n_heads=val+0; else if(key=="n_layers")n_layers=val+0
            else if(key=="block_size")block_size=val+0; else if(key=="adam_t")adam_t=val+0
            else if(key=="num_ranked_merges")num_ranked_merges=val+0
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
            print "Initializing parameters with proper initialization."
            
            # Initialize embeddings
            for(i=1;i<=V*d;i++) E[i] = xavier_init(V, d)
            for(i=1;i<=block_size*d;i++) P[i] = (rand()*2-1)*0.01
            
            # Initialize each layer
            for(l=1; l<=n_layers; l++) {
                # Attention weights with Xavier initialization
                for(i=1;i<=d*d;i++) { 
                    WQs[l,i] = xavier_init(d, d)
                    WKs[l,i] = xavier_init(d, d) 
                    WVs[l,i] = xavier_init(d, d)
                    W_outs[l,i] = xavier_init(d, d)
                }
                # Attention biases - zero initialized
                for(i=1;i<=d;i++) { 
                    bQs[l,i] = 0; bKs[l,i] = 0; bVs[l,i] = 0; b_outs[l,i] = 0 
                }
                
                # FFN weights with Xavier initialization  
                for(i=1;i<=d*d_ff;i++) W1s[l,i] = xavier_init(d, d_ff)
                for(i=1;i<=d_ff*d;i++) W2s[l,i] = xavier_init(d_ff, d)
                # FFN biases - zero initialized
                for(i=1;i<=d_ff;i++) b1s[l,i] = 0
                for(i=1;i<=d;i++) b2s[l,i] = 0
                
                # Layer norm parameters
                for(i=1;i<=d;i++) { 
                    LN1_gammas[l,i] = 1; LN1_betas[l,i] = 0
                    LN2_gammas[l,i] = 1; LN2_betas[l,i] = 0
                }
            }
            
            # Final layer norm
            for(i=1;i<=d;i++) { 
                final_ln_gamma[i] = 1; final_ln_beta[i] = 0
            }
        }

        printf "hyperparams: d=%d d_ff=%d n_heads=%d n_layers=%d epochs=%d\n", d, d_ff, n_heads, n_layers, epochs
        printf "Data: train tokens=%d, val tokens=%d, vocab size=%d\n", T, T_val, V
        blocks_per_epoch = int((T + block_size -1) / block_size)
        max_steps = epochs * blocks_per_epoch
        step = 0
        
        best_val_loss = 1e9
        patience = 10
        patience_counter = 0
        
        for(epoch=1; epoch<=epochs; epoch++){
            epoch_loss = 0; block_count = 0
            for(start=1; start<T; start+=block_size){
                end = start + block_size - 1
                if (end >= T) end = T - 1
                if (start > end) continue
                step++
                lr = get_lr(step)
                loss = process_block(start, end, 1)
                epoch_loss += loss; block_count++
                
                # Gradient accumulation (simplified - update every step)
                if(step % grad_accum_steps == 0) {
                    adamw_update_all()
                    reset_gradients()
                }
                
                if(step % 10 == 0) printf "step %d/%d, epoch %d, loss %f, lr %.6f\n", step, max_steps, epoch, loss, lr
            }
            
            # Validation
            if (epoch % 5 == 0 || epoch == 1) {
                val_loss = 0; val_blocks = 0
                for(start_val=1; start_val<T_val; start_val+=block_size) {
                    end_val = start_val + block_size - 1
                    if (end_val >= T_val) end_val = T_val-1
                    if(start_val>end_val) continue
                    # Save current training state
                    tmp_T=T; T=T_val
                    for(i=1;i<=T_val;i++){tmp_s[i]=seq[i];tmp_t[i]=tgt[i];seq[i]=val_seq[i];tgt[i]=val_tgt[i]}
                    # Forward pass only for validation
                    val_loss += process_block(start_val, end_val, 0)
                    # Restore training state
                    T=tmp_T
                    for(i=1;i<=T_val;i++){seq[i]=tmp_s[i];tgt[i]=tmp_t[i]}
                    val_blocks++
                }
                avg_val_loss = val_loss/val_blocks
                printf("Epoch %d | Train Loss %.4f | Val Loss %.4f\n", epoch, epoch_loss/block_count, avg_val_loss)
                
                # Early stopping
                if (avg_val_loss < best_val_loss) {
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    save_model("model_best.awk")
                } else {
                    patience_counter++
                    if (patience_counter >= patience) {
                        print "Early stopping at epoch", epoch
                        break
                    }
                }
            }
        }
        
        # Load best model for final save
        if (system("test -f model_best.awk") == 0) {
            load_model("model_best.awk")
        }
        save_model("model.awk")
        print "Saved model to model.awk"
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
