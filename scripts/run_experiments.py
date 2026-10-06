import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

# Configure matplotlib aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 13

FIG_DIR = "docs/informe/figures"
os.makedirs(FIG_DIR, exist_ok=True)

print("=== 1. Scaled Dot-Product Attention Verification & Scaling ===")
# Mathematical verification of variance
d_k_list = [16, 64, 128, 256, 512, 1024]
N_samples = 20000
empirical_vars = []
softmax_max_unscaled = []
softmax_max_scaled = []

for dk in d_k_list:
    q = torch.randn(N_samples, dk)
    k = torch.randn(N_samples, dk)
    dots = (q * k).sum(dim=-1) # dot product
    var = dots.var().item()
    empirical_vars.append(var)
    
    # Softmax behavior on random keys
    # batch of 1 query, 50 keys
    q_sample = torch.randn(100, 1, dk)
    k_sample = torch.randn(100, 50, dk)
    scores = torch.bmm(q_sample, k_sample.transpose(1, 2)).squeeze(1) # shape: [100, 50]
    
    sm_unscaled = F.softmax(scores, dim=-1)
    sm_scaled = F.softmax(scores / math.sqrt(dk), dim=-1)
    
    softmax_max_unscaled.append(sm_unscaled.max(dim=-1).values.mean().item())
    softmax_max_scaled.append(sm_scaled.max(dim=-1).values.mean().item())

print("d_k values:", d_k_list)
print("Theoretical variances (d_k):", d_k_list)
print("Empirical variances:", [round(v, 2) for v in empirical_vars])
print("Max softmax prob (unscaled):", [round(v, 3) for v in softmax_max_unscaled])
print("Max softmax prob (scaled):", [round(v, 3) for v in softmax_max_scaled])

# Computational scaling benchmark: sequence lengths n
seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
d_model = 768
n_heads = 12
d_k = d_model // n_heads

theoretical_quad_ops = [(n**2) * d_model * 2 for n in seq_lengths] # QK^T + Attention*V
theoretical_linear_ops = [n * (d_model**2) * 2 for n in seq_lengths]
mem_quad_mb = [(n**2 * n_heads * 4) / (1024**2) for n in seq_lengths] # 4 bytes float32

# Plot Scaling Complexity & Variance
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Subplot 1: Variance vs d_k and Softmax Saturation
ax1 = axes[0]
ax1.plot(d_k_list, d_k_list, 'k--', label=r'Teórica: $\mathrm{Var}(q \cdot k) = d_k$', linewidth=1.5)
ax1.plot(d_k_list, empirical_vars, 'ro', label=r'Empírica ($N=20\,000$)', markersize=6)
ax1.set_xlabel(r'Dimensión de proyección $d_k$')
ax1.set_ylabel(r'Varianza del producto punto $\mathrm{Var}(q \cdot k)$')
ax1.set_title('Verificación de Varianza de Scaled Dot-Product')
ax1.legend(loc='upper left', frameon=True)

# Inset / secondary axis for softmax peak probability
ax1_twin = ax1.twinx()
ax1_twin.plot(d_k_list, softmax_max_unscaled, 'b-s', label=r'Máx Softmax (Sin Escalar, $\approx 1.0$)', markersize=5)
ax1_twin.plot(d_k_list, softmax_max_scaled, 'g-^', label=r'Máx Softmax (Escalado $1/\sqrt{d_k}$)', markersize=5)
ax1_twin.set_ylabel('Probabilidad máxima promedio')
ax1_twin.set_ylim(0, 1.1)
ax1_twin.legend(loc='lower right', frameon=True)

# Subplot 2: Complexity O(n^2) vs O(n)
ax2 = axes[1]
ax2.plot(seq_lengths, [mem for mem in mem_quad_mb], 'r-o', label=r'Memoria Matriz Atención $O(n^2)$ (MB)', linewidth=1.8)
ax2.set_xlabel('Longitud de secuencia $n$')
ax2.set_ylabel('Memoria de atención (MB) [12 cabezas, FP32]', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_title('Escalamiento Cuadrático $O(n^2)$ vs Longitud $n$')

ax2_twin = ax2.twinx()
ax2_twin.plot(seq_lengths, [ops / 1e9 for ops in theoretical_quad_ops], 'b--d', label=r'FLOPs Atención $O(n^2 d)$ (GFLOPs)', linewidth=1.8)
ax2_twin.set_ylabel('FLOPs Atención (GFLOPs)', color='b')
ax2_twin.tick_params(axis='y', labelcolor='b')
ax2_twin.set_yscale('log')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_scaling_complexity.pdf", bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/fig_scaling_complexity.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_scaling_complexity")

print("\n=== 2. Loading Pretrained Models: BERT and GPT-2 ===")
tok_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True, attn_implementation="eager")
bert.eval()

tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")
gpt2 = AutoModel.from_pretrained("gpt2", output_attentions=True, attn_implementation="eager")
gpt2.eval()

print("=== 3. BERT Attention Patterns Heatmaps ===")
sentence = "The clever scientist designed an intelligent algorithm for NLP."
inputs = tok_bert(sentence, return_tensors="pt")
tokens = tok_bert.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = bert(**inputs)
    # outputs.attentions is a tuple of 12 layers, each [batch=1, num_heads=12, seq_len, seq_len]
    attentions = outputs.attentions

# We select 4 interesting heads showcasing distinct patterns:
# 1. Previous / Next token attention (e.g. Layer 1 Head 0 or Layer 2 Head 3)
# 2. Delimiter token [CLS] / [SEP] (e.g. Layer 5 Head 1 or Layer 8 Head 10)
# 3. Punctuation attention (e.g. Layer 7 Head 6)
# 4. Identity / Broad semantic attention (e.g. Layer 10 Head 4 or Layer 0 Head 2)

selected_heads = [
    (0, 4, "L0-H4: Atención a sí mismo / identidad léxica"),
    (1, 0, "L1-H0: Atención al token siguiente/anterior"),
    (7, 2, "L7-H2: Atención a tokens delimitadores ([CLS]/[SEP])"),
    (10, 4, "L10-H4: Atención semántica distribuida profunda")
]

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()

for idx, (layer, head, title) in enumerate(selected_heads):
    attn_matrix = attentions[layer][0, head].numpy()
    ax = axes[idx]
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues", ax=ax,
                cbar_kws={'shrink': 0.8}, vmin=0, vmax=np.max(attn_matrix))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_bert_heatmaps.pdf", bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/fig_bert_heatmaps.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_bert_heatmaps")

print("\n=== 4. Winograd Coreference Resolution Experiment ===")
s1 = "The animal didn't cross the street because it was too tired."
s2 = "The animal didn't cross the street because it was too wide."

in1 = tok_bert(s1, return_tensors="pt")
in2 = tok_bert(s2, return_tensors="pt")
toks1 = tok_bert.convert_ids_to_tokens(in1["input_ids"][0])
toks2 = tok_bert.convert_ids_to_tokens(in2["input_ids"][0])

with torch.no_grad():
    out1 = bert(**in1)
    out2 = bert(**in2)

it_idx_1 = toks1.index("it")
animal_idx_1 = toks1.index("animal")
street_idx_1 = toks1.index("street")

it_idx_2 = toks2.index("it")
animal_idx_2 = toks2.index("animal")
street_idx_2 = toks2.index("street")

print(f"Tokens: it={it_idx_1}, animal={animal_idx_1}, street={street_idx_1}")

# Search for the head maximizing Winograd disambiguation:
# Score = (Attn1(it -> animal) - Attn1(it -> street)) + (Attn2(it -> street) - Attn2(it -> animal))
best_score = -999.0
best_layer, best_head = -1, -1

for layer in range(12):
    for head in range(12):
        a1 = out1.attentions[layer][0, head].numpy()
        a2 = out2.attentions[layer][0, head].numpy()
        
        diff1 = a1[it_idx_1, animal_idx_1] - a1[it_idx_1, street_idx_1]
        diff2 = a2[it_idx_2, street_idx_2] - a2[it_idx_2, animal_idx_2]
        
        score = diff1 + diff2
        if score > best_score:
            best_score = score
            best_layer = layer
            best_head = head

print(f"Best Winograd head: Layer {best_layer}, Head {best_head} (Score: {best_score:.4f})")
a1_best = out1.attentions[best_layer][0, best_head].numpy()
a2_best = out2.attentions[best_layer][0, best_head].numpy()

print(f"Sentence 1 (tired): it -> animal = {a1_best[it_idx_1, animal_idx_1]:.4f}, it -> street = {a1_best[it_idx_1, street_idx_1]:.4f}")
print(f"Sentence 2 (wide):  it -> animal = {a2_best[it_idx_2, animal_idx_2]:.4f}, it -> street = {a2_best[it_idx_2, street_idx_2]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(a1_best, xticklabels=toks1, yticklabels=toks1, cmap="YlGnBu", ax=axes[0], cbar_kws={'shrink': 0.8})
axes[0].set_title(f"S1: '...too tired'\nCapa {best_layer}, Cabeza {best_head}: it $\\rightarrow$ animal ({a1_best[it_idx_1, animal_idx_1]:.3f})", fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(a2_best, xticklabels=toks2, yticklabels=toks2, cmap="YlGnBu", ax=axes[1], cbar_kws={'shrink': 0.8})
axes[1].set_title(f"S2: '...too wide'\nCapa {best_layer}, Cabeza {best_head}: it $\\rightarrow$ street ({a2_best[it_idx_2, street_idx_2]:.3f})", fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_winograd_resolution.pdf", bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/fig_winograd_resolution.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_winograd_resolution")

print("\n=== 5. GPT-2 Causal Attention & Attention Sinks ===")
gpt2_text = "Deep learning and transformer architectures have completely revolutionized natural language processing."
gpt2_inputs = tok_gpt2(gpt2_text, return_tensors="pt")
gpt2_toks = [tok_gpt2.decode([t]) for t in gpt2_inputs["input_ids"][0]]

with torch.no_grad():
    gpt2_out = gpt2(**gpt2_inputs)
    gpt2_attentions = gpt2_out.attentions

# Check causal triangular structure:
is_triangular = True
for l, attn in enumerate(gpt2_attentions):
    mat = attn[0].numpy()
    for h in range(mat.shape[0]):
        upper = np.triu(mat[h], k=1)
        if not np.allclose(upper, 0.0, atol=1e-5):
            is_triangular = False
print("Is GPT-2 attention strictly causal (lower triangular)?:", is_triangular)

# Attention Sink Analysis:
# Xiao et al. (2023): In autoregressive models, token 0 receives massive attention even if semantically irrelevant.
sink_proportions = [] # per layer, average proportion of attention received by token 0 for pos >= 1
for layer in range(12):
    layer_attn = gpt2_attentions[layer][0].numpy() # shape [12, seq_len, seq_len]
    # For every query position i from 1 to seq_len-1, examine attention to key 0
    t0_attn = layer_attn[:, 1:, 0] # shape [12, seq_len-1]
    sink_proportions.append(t0_attn.mean())

print("Attention sink proportion per layer:", [round(float(p), 4) for p in sink_proportions])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Subplot 1: Heatmap of GPT-2 causal attention (Layer 6, Head 3)
sample_head = gpt2_attentions[6][0, 3].numpy()
sns.heatmap(sample_head, xticklabels=gpt2_toks, yticklabels=gpt2_toks, cmap="Purples", ax=axes[0], cbar_kws={'shrink': 0.8})
axes[0].set_title("GPT-2 (Capa 6, Cabeza 3): Matriz Triangular Inferior\n(Notar columna 0 oscura: Attention Sink)", fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Subplot 2: Attention to token 0 per layer
axes[1].plot(range(12), sink_proportions, 'm-o', linewidth=2, markersize=7, label='Proporción a Token 0')
axes[1].axhline(1.0 / len(gpt2_toks), color='gray', linestyle='--', label=r'Línea Base Uniforme ($1/n$)')
axes[1].set_xlabel('Capa del Modelo GPT-2')
axes[1].set_ylabel('Proporción Promedio de Atención al Token 0')
axes[1].set_title('Fenómeno de Attention Sinks (Xiao et al., 2023)', fontsize=11, fontweight='bold')
axes[1].set_xticks(range(12))
axes[1].set_ylim(0, max(sink_proportions) * 1.15)
axes[1].legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_gpt2_causal_sinks.pdf", bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/fig_gpt2_causal_sinks.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_gpt2_causal_sinks")

print("\n=== 6. Attention Entropy per Layer (BERT vs GPT-2) ===")
# Entropy calculation: H = -sum(p * log2(p + eps))
def compute_layer_entropies(attentions, is_causal=False):
    layer_entropies = []
    eps = 1e-12
    for layer_attn in attentions:
        # shape: [batch=1, heads=12, seq, seq]
        attn = layer_attn[0].numpy()
        num_heads, seq_len, _ = attn.shape
        head_ents = []
        for h in range(num_heads):
            mat = attn[h]
            if is_causal:
                # for causal, each row i only distributes over i+1 elements
                row_ents = []
                for i in range(1, seq_len):
                    p = mat[i, :i+1]
                    p = p / (p.sum() + eps)
                    ent = -np.sum(p * np.log2(p + eps))
                    # normalize by max entropy log2(i+1)
                    norm_ent = ent / np.log2(i + 1) if i > 0 else 0.0
                    row_ents.append(norm_ent)
                head_ents.append(np.mean(row_ents))
            else:
                p = mat
                ent = -np.sum(p * np.log2(p + eps), axis=-1)
                # normalize by max entropy log2(seq_len)
                norm_ent = ent / np.log2(seq_len)
                head_ents.append(np.mean(norm_ent))
        layer_entropies.append(np.mean(head_ents))
    return layer_entropies

bert_entropy = compute_layer_entropies(attentions, is_causal=False)
gpt2_entropy = compute_layer_entropies(gpt2_attentions, is_causal=True)

print("BERT normalized entropy per layer:", [round(float(e), 3) for e in bert_entropy])
print("GPT-2 normalized entropy per layer:", [round(float(e), 3) for e in gpt2_entropy])

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(range(12), bert_entropy, 'b-s', linewidth=2, markersize=6, label='BERT (Bidireccional)')
ax.plot(range(12), gpt2_entropy, 'g-^', linewidth=2, markersize=6, label='GPT-2 (Causal Autoregresivo)')
ax.set_xlabel('Capa del Modelo')
ax.set_ylabel('Entropía de Shannon Normalizada $H / H_{\\max}$')
ax.set_title('Evolución de la Entropía de Atención por Capa', fontsize=12, fontweight='bold')
ax.set_xticks(range(12))
ax.set_ylim(0.2, 1.0)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_attention_entropy.pdf", bbox_inches='tight')
plt.savefig(f"{FIG_DIR}/fig_attention_entropy.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_attention_entropy")

print("\nAll experiments and figures completed successfully!")
