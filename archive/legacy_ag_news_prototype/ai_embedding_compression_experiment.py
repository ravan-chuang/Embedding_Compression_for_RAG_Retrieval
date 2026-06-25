"""
# Data Compression Final Project: Enhanced PQ Experiment

Changes in this version:

1. Use 1,000 AG News samples per class, about 4,000 samples in total.
2. Test PQ with `m=8/16/32` and `k=32/64`.
3. Apply L2 normalization after PQ / Rotation + PQ reconstruction.
4. Add an analysis for the best compression method under Top-5 Recall ≥ 0.9.
5. Also output a PQ / Rotation PQ ranking table.

Recommended to run in Colab. Install the required packages before the first run:

```python
!pip install -q sentence-transformers datasets scikit-learn pandas numpy matplotlib
```

"""

# ============================================================
# AI Embedding Compression Project - PQ Enhanced Experiment Version
# Scalar Quantization vs VQ vs PQ vs Random Rotation
#
# Key changes in this version：
# 1. Use 1,000 AG News samples per class, about 4,000 samples in total
# 2. Test PQ with m=8/16/32 and k=32/64
# 3. Apply L2 normalization after PQ / Rotation + PQ reconstruction
# 4. Add best practical method analysis under Top-5 Recall >= 0.9

# ============================================================
# ----------------------------
# 0. Install packages
# ----------------------------
# If running in Colab, run this first:
# !pip install -q sentence-transformers datasets scikit-learn pandas numpy matplotlib

import os
import math
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ============================================================
# 1. Global settings

# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# This version uses 1,000 samples per class: about 4,000 AG News samples in total
SAMPLES_PER_CLASS = 1000

TEST_SIZE = 0.2
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Keep Single VQ as a baseline only
SINGLE_VQ_K_LIST = [4, 8, 16]

# Enhanced PQ settings: add m=32 and k=32/64
# D=384 when：
# m=8  -> each sub-vector has 48 dimensions
# m=16 -> each sub-vector has 24 dimensions
# m=32 -> each sub-vector has 12 dimensions
PQ_SETTINGS = [
    (8, 32), (8, 64),
    (16, 32), (16, 64),
    (32, 32), (32, 64),
]

SAVE_RESULTS = True
RESULT_DIR = "results_pq_l2norm_ag1000"
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 2. Load dataset: AG News

# ============================================================
def load_ag_news_subset(samples_per_class=1000, seed=42):
    try:
        from datasets import load_dataset

        dataset = load_dataset("ag_news", split="train")
        df = pd.DataFrame(dataset)

        label_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }
        df["label_name"] = df["label"].map(label_names)

        sampled = (
            df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=seed))
              .reset_index(drop=True)
        )

        texts = sampled["text"].astype(str).tolist()
        labels = sampled["label"].to_numpy()
        label_names_list = sampled["label_name"].tolist()

        print("Loaded AG News subset")
        print(f"Total samples: {len(texts)}")
        print(sampled["label_name"].value_counts())

        return texts, labels, label_names_list

    except Exception as e:
        print("AG News loading failed. Fallback to sklearn 20 Newsgroups.")
        print("Reason:", repr(e))

        from sklearn.datasets import fetch_20newsgroups

        categories = [
            "comp.graphics",
            "rec.sport.baseball",
            "sci.space",
            "talk.politics.misc",
        ]

        data = fetch_20newsgroups(
            subset="train",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )

        df = pd.DataFrame({
            "text": data.data,
            "label": data.target,
        })
        df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False)
        df = df[df["text"].str.len() > 50].reset_index(drop=True)

        sampled = (
            df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=seed))
              .reset_index(drop=True)
        )

        texts = sampled["text"].tolist()
        labels = sampled["label"].to_numpy()
        label_names_list = [data.target_names[i] for i in labels]

        print("Loaded 20 Newsgroups subset")
        print(f"Total samples: {len(texts)}")
        print(pd.Series(label_names_list).value_counts())

        return texts, labels, label_names_list


texts, labels, label_names = load_ag_news_subset(SAMPLES_PER_CLASS, SEED)

# ============================================================
# 3. Train / Test Split

# ============================================================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=labels,
)

print("\nTrain samples:", len(train_texts))
print("Test samples :", len(test_texts))

# ============================================================
# 4. Convert texts to embeddings

# ============================================================
def make_embeddings(texts, model_name=EMBEDDING_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


print("\nEncoding train embeddings...")
X_train = make_embeddings(train_texts)

print("\nEncoding test embeddings...")
X_test = make_embeddings(test_texts)

N_train, D = X_train.shape
N_test = X_test.shape[0]

print("\nEmbedding shape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Dimension D:", D)

# ============================================================
# 5. Evaluation helper functions

# ============================================================
def l2_normalize(x, eps=1e-12):
    """
    Apply L2 normalization to vectors.
    After PQ / VQ reconstruction, vector norms may change; if retrieval uses cosine / inner product,
    normalizing reconstructed vectors usually makes semantic retrieval more stable.
    """
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)).astype(np.float32)


def mse(x, x_hat):
    return float(np.mean((x - x_hat) ** 2))


def avg_cosine_to_original(x, x_hat, eps=1e-12):
    x_norm = l2_normalize(x, eps)
    h_norm = l2_normalize(x_hat, eps)
    cos = np.sum(x_norm * h_norm, axis=1)
    return float(np.mean(cos))


def mean_inner_product_error(x, x_hat):
    """
    Here we compare the pairwise inner product matrices after normalization,
    so the metric better matches the cosine retrieval scenario.
    """
    x_n = l2_normalize(x)
    h_n = l2_normalize(x_hat)
    original_scores = x_n @ x_n.T
    approx_scores = h_n @ h_n.T
    return float(np.mean(np.abs(original_scores - approx_scores)))


def topk_recall_database_compression(x_query, x_db_original, x_db_approx, k=5):
    """
    The query uses the original float32 embedding;
    the database uses original vs compressed-reconstructed embeddings;
    compare the top-k overlap.
    """
    xq = l2_normalize(x_query)
    xo = l2_normalize(x_db_original)
    xa = l2_normalize(x_db_approx)

    true_scores = xq @ xo.T
    approx_scores = xq @ xa.T

    recalls = []

    for i in range(x_query.shape[0]):
        true_rank = np.argsort(-true_scores[i])
        approx_rank = np.argsort(-approx_scores[i])

        if x_query.shape[0] == x_db_original.shape[0]:
            true_rank = true_rank[true_rank != i]
            approx_rank = approx_rank[approx_rank != i]

        true_topk = set(true_rank[:k])
        approx_topk = set(approx_rank[:k])

        recalls.append(len(true_topk & approx_topk) / k)

    return float(np.mean(recalls))


def original_size_bytes(n, d):
    return n * d * 4


def add_metric(metrics, method, x_original, x_hat, compressed_bytes, notes=""):
    orig_bytes = original_size_bytes(x_original.shape[0], x_original.shape[1])
    ratio = orig_bytes / compressed_bytes if compressed_bytes > 0 else np.inf

    metrics.append({
        "method": method,
        "original_bytes": orig_bytes,
        "compressed_bytes": int(compressed_bytes),
        "compression_ratio": float(ratio),
        "mse": mse(x_original, x_hat),
        "avg_cosine": avg_cosine_to_original(x_original, x_hat),
        "inner_product_error": mean_inner_product_error(x_original, x_hat),
        f"top{TOP_K}_recall": topk_recall_database_compression(
            x_original,
            x_original,
            x_hat,
            k=TOP_K
        ),
        "notes": notes,
    })


def check_duplicate_methods(metrics):
    names = [m["method"] for m in metrics]
    dup = pd.Series(names).value_counts()
    dup = dup[dup > 1]
    if len(dup) > 0:
        raise ValueError(f"Duplicated method names found:\n{dup}")

# ============================================================
# 6. Scalar Quantization

# ============================================================
def symmetric_quantize_train_scale(x_train, bits):
    qmax = (2 ** (bits - 1)) - 1
    max_abs = np.max(np.abs(x_train))
    scale = max_abs / qmax if max_abs > 0 else 1.0
    return np.float32(scale)


def symmetric_quantize_with_scale(x, bits, scale):
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax

    q = np.round(x / scale)
    q = np.clip(q, qmin, qmax)
    q = q.astype(np.int8)

    x_hat = (q.astype(np.float32) * scale).astype(np.float32)
    return q, x_hat


def scalar_compressed_bytes(n, d, bits, num_scales=1):
    value_bytes = math.ceil(n * d * bits / 8)
    scale_bytes = num_scales * 4
    return value_bytes + scale_bytes

# ============================================================
# 7. Random Rotation

# ============================================================
def random_orthogonal_matrix(d, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


R = random_orthogonal_matrix(D, SEED)


def rotate(x, R):
    return (x @ R).astype(np.float32)


def inverse_rotate(y, R):
    return (y @ R.T).astype(np.float32)

# ============================================================
# 8. Single Codebook VQ

# ============================================================
@dataclass
class SingleVQModel:
    kmeans: KMeans
    codebook: np.ndarray


def train_single_vq(x_train, k, seed=42):
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(x_train)
    codebook = kmeans.cluster_centers_.astype(np.float32)
    return SingleVQModel(kmeans=kmeans, codebook=codebook)


def encode_decode_single_vq(model, x, normalize_output=False):
    indices = model.kmeans.predict(x)
    x_hat = model.codebook[indices].astype(np.float32)
    if normalize_output:
        x_hat = l2_normalize(x_hat)
    return indices.astype(np.int32), x_hat


def single_vq_compressed_bytes(n, d, k):
    index_bits = math.ceil(math.log2(k))
    index_bytes = math.ceil(n * index_bits / 8)
    codebook_bytes = k * d * 4
    return index_bytes + codebook_bytes

# ============================================================
# 9. Product Quantization

# ============================================================
@dataclass
class PQModel:
    m: int
    k: int
    sub_dim: int
    kmeans_list: list
    codebooks: list


def train_pq(x_train, m, k, seed=42):
    n, d = x_train.shape
    assert d % m == 0, f"D={d} must be divisible by m={m}"

    sub_dim = d // m
    kmeans_list = []
    codebooks = []

    for i in range(m):
        start = i * sub_dim
        end = (i + 1) * sub_dim
        sub_x = x_train[:, start:end]

        kmeans = KMeans(
            n_clusters=k,
            random_state=seed + i,
            n_init=5,       # k=64/m=32 can be slower, so n_init=5 is more practical
            max_iter=300,
        )
        kmeans.fit(sub_x)

        kmeans_list.append(kmeans)
        codebooks.append(kmeans.cluster_centers_.astype(np.float32))

    return PQModel(
        m=m,
        k=k,
        sub_dim=sub_dim,
        kmeans_list=kmeans_list,
        codebooks=codebooks,
    )


def encode_decode_pq(model, x, normalize_output=True):
    n, d = x.shape
    x_hat = np.zeros_like(x, dtype=np.float32)
    all_indices = []

    for i in range(model.m):
        start = i * model.sub_dim
        end = (i + 1) * model.sub_dim
        sub_x = x[:, start:end]

        indices = model.kmeans_list[i].predict(sub_x)
        all_indices.append(indices)

        x_hat[:, start:end] = model.codebooks[i][indices]

    indices = np.stack(all_indices, axis=1).astype(np.int32)

    if normalize_output:
        x_hat = l2_normalize(x_hat)

    return indices, x_hat


def pq_compressed_bytes(n, d, m, k):
    assert d % m == 0
    index_bits = math.ceil(math.log2(k))
    index_bytes = math.ceil(n * m * index_bits / 8)
    codebook_bytes = k * d * 4
    return index_bytes + codebook_bytes

# ============================================================
# 10. Run all experiments

# ============================================================
metrics = []

# 10.1 Float32 baseline
add_metric(
    metrics,
    method="float32_original",
    x_original=X_test,
    x_hat=X_test.copy(),
    compressed_bytes=original_size_bytes(N_test, D),
    notes="Original float32 embeddings."
)

# 10.2 Scalar Quantization
for bits in [8, 4, 2]:
    scale = symmetric_quantize_train_scale(X_train, bits)
    _, X_hat = symmetric_quantize_with_scale(X_test, bits, scale)

    add_metric(
        metrics,
        method=f"scalar_int{bits}",
        x_original=X_test,
        x_hat=X_hat,
        compressed_bytes=scalar_compressed_bytes(N_test, D, bits, num_scales=1),
        notes="Per-tensor symmetric scalar quantization. Scale trained from train set."
    )

# 10.3 Random Rotation + Scalar Quantization
X_train_rot = rotate(X_train, R)
X_test_rot = rotate(X_test, R)

for bits in [8, 4, 2]:
    scale = symmetric_quantize_train_scale(X_train_rot, bits)
    _, X_hat_rot_space = symmetric_quantize_with_scale(X_test_rot, bits, scale)
    X_hat = inverse_rotate(X_hat_rot_space, R)

    add_metric(
        metrics,
        method=f"rotation_scalar_int{bits}",
        x_original=X_test,
        x_hat=X_hat,
        compressed_bytes=scalar_compressed_bytes(N_test, D, bits, num_scales=1),
        notes="Random rotation matrix generated by shared seed; matrix storage not counted."
    )

# 10.4 Single VQ baseline
for k in SINGLE_VQ_K_LIST:
    model = train_single_vq(X_train, k, SEED)
    _, X_hat = encode_decode_single_vq(model, X_test, normalize_output=True)

    add_metric(
        metrics,
        method=f"single_vq_k{k}_l2norm",
        x_original=X_test,
        x_hat=X_hat,
        compressed_bytes=single_vq_compressed_bytes(N_test, D, k),
        notes="Single VQ trained on train set; reconstructed vectors are L2-normalized."
    )

# 10.5 Product Quantization with L2 normalize
for m, k in PQ_SETTINGS:
    if D % m != 0:
        print(f"Skip PQ m={m}, k={k}: D={D} not divisible by m.")
        continue

    if k > N_train:
        print(f"Skip PQ m={m}, k={k}: k > N_train.")
        continue

    print(f"Training PQ m={m}, k={k} ...")
    model = train_pq(X_train, m, k, SEED)
    _, X_hat = encode_decode_pq(model, X_test, normalize_output=True)

    add_metric(
        metrics,
        method=f"pq_m{m}_k{k}_l2norm",
        x_original=X_test,
        x_hat=X_hat,
        compressed_bytes=pq_compressed_bytes(N_test, D, m, k),
        notes="Product Quantization trained on train set; reconstructed vectors are L2-normalized."
    )

# 10.6 Random Rotation + Product Quantization with L2 normalize
for m, k in PQ_SETTINGS:
    if D % m != 0:
        print(f"Skip Rotation PQ m={m}, k={k}: D={D} not divisible by m.")
        continue

    if k > N_train:
        print(f"Skip Rotation PQ m={m}, k={k}: k > N_train.")
        continue

    print(f"Training Rotation + PQ m={m}, k={k} ...")
    model = train_pq(X_train_rot, m, k, SEED)
    _, X_hat_rot_space = encode_decode_pq(model, X_test_rot, normalize_output=False)
    X_hat = inverse_rotate(X_hat_rot_space, R)
    X_hat = l2_normalize(X_hat)

    add_metric(
        metrics,
        method=f"rotation_pq_m{m}_k{k}_l2norm",
        x_original=X_test,
        x_hat=X_hat,
        compressed_bytes=pq_compressed_bytes(N_test, D, m, k),
        notes="PQ after random rotation; reconstructed vectors are inverse-rotated and L2-normalized."
    )

check_duplicate_methods(metrics)

metrics_df = pd.DataFrame(metrics)

recall_col = f"top{TOP_K}_recall"

metrics_df = metrics_df.sort_values(
    by=[recall_col, "compression_ratio"],
    ascending=[False, False],
).reset_index(drop=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n===== Metrics Summary =====")
display(metrics_df)

# ============================================================
# 11. Automatically analyze the best methods

# ============================================================
df_no_float = metrics_df[metrics_df["method"] != "float32_original"].copy()

best_recall = df_no_float.sort_values(recall_col, ascending=False).iloc[0]
best_compression = df_no_float.sort_values("compression_ratio", ascending=False).iloc[0]

df_no_float["balance_score"] = df_no_float["compression_ratio"] * df_no_float[recall_col]
best_balance = df_no_float.sort_values("balance_score", ascending=False).iloc[0]

print("\n===== Best Methods =====")
print("Best Top-k Recall:")
print(best_recall[["method", "compression_ratio", "avg_cosine", recall_col]])

print("\nBest Compression Ratio:")
print(best_compression[["method", "compression_ratio", "avg_cosine", recall_col]])

print("\nBest Balance Score = Compression Ratio × Top-k Recall:")
print(best_balance[["method", "compression_ratio", "avg_cosine", recall_col, "balance_score"]])

# ============================================================
# 12. Practical threshold analysis：Top-5 Recall >= 0.9

# ============================================================
practical_df = df_no_float[df_no_float[recall_col] >= 0.9].copy()

print("\n===== Practical Best Method: Top-5 Recall >= 0.9 =====")
if len(practical_df) == 0:
    print("No method satisfies Top-5 Recall >= 0.9.")
else:
    practical_df = practical_df.sort_values(
        by=["compression_ratio", recall_col, "avg_cosine"],
        ascending=[False, False, False]
    )
    display(practical_df[["method", "compression_ratio", "avg_cosine", recall_col, "mse", "inner_product_error", "notes"]])
    practical_best = practical_df.iloc[0]
    print("\nBest practical method:")
    print(practical_best[["method", "compression_ratio", "avg_cosine", recall_col, "mse"]])

# ============================================================
# 13. PQ / Rotation PQ Ranking

# ============================================================
pq_rank_df = metrics_df[
    metrics_df["method"].str.startswith("pq_") |
    metrics_df["method"].str.startswith("rotation_pq_")
].sort_values(by=recall_col, ascending=False)

print("\n===== PQ / Rotation PQ Ranking =====")
display(pq_rank_df[["method", "compression_ratio", "avg_cosine", recall_col, "mse", "inner_product_error"]])

# ============================================================
# 14. Plot figures

# ============================================================
def plot_bar(df, x_col, y_col, title, ylabel, filename=None):
    plt.figure(figsize=(13, 5))
    plt.bar(df[x_col], df[y_col])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=200, bbox_inches="tight")

    plt.show()


plot_df = metrics_df[metrics_df["method"] != "float32_original"].copy()

plot_bar(
    plot_df,
    "method",
    "compression_ratio",
    "Compression Ratio by Method",
    "Compression Ratio",
    os.path.join(RESULT_DIR, "compression_ratio.png") if SAVE_RESULTS else None
)

plot_bar(
    plot_df,
    "method",
    "avg_cosine",
    "Average Cosine Similarity by Method",
    "Avg Cosine Similarity",
    os.path.join(RESULT_DIR, "avg_cosine.png") if SAVE_RESULTS else None
)

plot_bar(
    plot_df,
    "method",
    recall_col,
    f"Top-{TOP_K} Recall by Method",
    f"Top-{TOP_K} Recall",
    os.path.join(RESULT_DIR, "topk_recall.png") if SAVE_RESULTS else None
)

plt.figure(figsize=(9, 6))
plt.scatter(plot_df["compression_ratio"], plot_df[recall_col])

for _, row in plot_df.iterrows():
    plt.text(
        row["compression_ratio"],
        row[recall_col],
        row["method"],
        fontsize=8,
        ha="left",
        va="bottom"
    )

plt.xlabel("Compression Ratio")
plt.ylabel(f"Top-{TOP_K} Recall")
plt.title("Compression Ratio vs Retrieval Quality")
plt.grid(True, alpha=0.3)
plt.tight_layout()

if SAVE_RESULTS:
    plt.savefig(os.path.join(RESULT_DIR, "compression_vs_retrieval.png"), dpi=200, bbox_inches="tight")

plt.show()

# ============================================================
# 15. Retrieval Demo

# ============================================================
def get_reconstructed_by_method(method_name):
    if method_name == "float32_original":
        return X_test.copy()

    if method_name.startswith("scalar_int"):
        bits = int(method_name.replace("scalar_int", ""))
        scale = symmetric_quantize_train_scale(X_train, bits)
        _, X_hat = symmetric_quantize_with_scale(X_test, bits, scale)
        return X_hat

    if method_name.startswith("rotation_scalar_int"):
        bits = int(method_name.replace("rotation_scalar_int", ""))
        scale = symmetric_quantize_train_scale(X_train_rot, bits)
        _, X_hat_rot_space = symmetric_quantize_with_scale(X_test_rot, bits, scale)
        return inverse_rotate(X_hat_rot_space, R)

    if method_name.startswith("pq_m"):
        # format: pq_m32_k64_l2norm
        tmp = method_name.replace("pq_m", "").replace("_l2norm", "")
        m_str, k_str = tmp.split("_k")
        m = int(m_str)
        k = int(k_str)
        model = train_pq(X_train, m, k, SEED)
        _, X_hat = encode_decode_pq(model, X_test, normalize_output=True)
        return X_hat

    if method_name.startswith("rotation_pq_m"):
        # format: rotation_pq_m32_k64_l2norm
        tmp = method_name.replace("rotation_pq_m", "").replace("_l2norm", "")
        m_str, k_str = tmp.split("_k")
        m = int(m_str)
        k = int(k_str)
        model = train_pq(X_train_rot, m, k, SEED)
        _, X_hat_rot_space = encode_decode_pq(model, X_test_rot, normalize_output=False)
        return l2_normalize(inverse_rotate(X_hat_rot_space, R))

    raise ValueError(f"Unknown method: {method_name}")


def show_retrieval_demo(query_index=0, methods=None, k=5):
    if methods is None:
        methods = [
            "float32_original",
            "scalar_int4",
            "rotation_scalar_int4",
            "scalar_int2",
            "pq_m32_k32_l2norm",
            "pq_m32_k64_l2norm",
            "rotation_pq_m32_k64_l2norm",
        ]

    query = l2_normalize(X_test[query_index:query_index+1])
    print("=" * 80)
    print("Query text:")
    print(test_texts[query_index])
    print("Label:", test_labels[query_index])
    print("=" * 80)

    for method in methods:
        X_db = l2_normalize(get_reconstructed_by_method(method))
        scores = (query @ X_db.T).flatten()

        rank = np.argsort(-scores)
        rank = rank[rank != query_index]
        top_indices = rank[:k]

        print(f"\n--- {method} Top-{k} ---")
        for r, idx in enumerate(top_indices, 1):
            text_preview = test_texts[idx].replace("\n", " ")
            if len(text_preview) > 130:
                text_preview = text_preview[:130] + "..."
            print(f"{r}. score={scores[idx]:.4f}, label={test_labels[idx]} | {text_preview}")


demo_indices = [0, min(5, N_test - 1), min(10, N_test - 1)]
for idx in demo_indices:
    show_retrieval_demo(query_index=idx, k=TOP_K)

# ============================================================
# 16. Save results

# ============================================================
if SAVE_RESULTS:
    metrics_path = os.path.join(RESULT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    pq_path = os.path.join(RESULT_DIR, "pq_ranking.csv")
    pq_rank_df.to_csv(pq_path, index=False, encoding="utf-8-sig")

    config_path = os.path.join(RESULT_DIR, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("AI Embedding Compression Project - PQ L2 Norm AG News 1000/class\n")
        f.write(f"Dataset: AG News subset, samples_per_class={SAMPLES_PER_CLASS}\n")
        f.write(f"Embedding model: {EMBEDDING_MODEL_NAME}\n")
        f.write(f"Train samples: {N_train}\n")
        f.write(f"Test samples: {N_test}\n")
        f.write(f"Embedding dimension: {D}\n")
        f.write(f"Top-k: {TOP_K}\n")
        f.write(f"PQ settings: {PQ_SETTINGS}\n")
        f.write("\nNotes:\n")
        f.write("- PQ / Rotation PQ reconstructed vectors are L2-normalized.\n")
        f.write("- Scalar quantization compression size includes one float32 scale.\n")
        f.write("- Random rotation matrix storage is not counted because it is generated from a shared seed.\n")
        f.write("- VQ/PQ codebook storage is counted in compressed size.\n")

    print("\nSaved results to:", RESULT_DIR)

# ============================================================
# 17. Report-ready summary template

# ============================================================
print("\n===== Report Summary Template =====")
print(f"""
This experiment uses an AG News subset，sampling {SAMPLES_PER_CLASS} texts per class，for a total of about {len(texts)} samples，
and converts the texts into {D} dimensions sentence embeddings。This version enhances Product Quantization:
it adds m=32 and k=32/64 settings, and applies L2 normalization after PQ reconstruction to improve cosine retrieval stability.

The main evaluation metrics include compression ratio, MSE, average cosine similarity,
inner product error, and Top-{TOP_K} retrieval recall.

Key observations:
1. int8 / int4 scalar quantization remain high-quality baselines.
2. PQ m=32/k=32,64 tests whether more fine-grained sub-vector codebook improve retrieval recall.
3. PQ reconstructionafter L2 normalization can reduce the effect of reconstructed vector norm shifts on cosine retrieval.
4. If Top-{TOP_K} Recall >= 0.9 methodare still mainly int4 ，indicates that on this dataset scalar int4 is the most practicalcompression method。
5. If PQ recall improves significantly, it indicates that more data, larger m/k values, and normalization help PQ.
""")