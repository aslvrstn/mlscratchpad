import plotly.express as px
import torch as t
import numpy as np
from typing import Tuple, Any
from easy_transformer import EasyTransformer

MAIN = __name__ == "__main__"

device = "cpu"

cache_2 = {}
if MAIN:
    tokens_2 = t.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]])
    tokens_2 = t.tensor([[0, 47, 82, 9, 13, 47, 82, 9, 13]])
    tokens_2 = tokens_2.to(device)
    model.cache_all(cache_2)
    logits_2 = model(tokens_2)
    # This line turns off cacheing, so future runs of the model won't overwrite the cache
    model.reset_hooks()


def to_numpy(tensor):
    """Helper function to convert things to numpy before plotting with Plotly."""
    return tensor.detach().cpu().numpy()


def plot_logit_attribution(logit_attr, tokens):
    # Remove dummy batch dimension
    tokens = tokens.squeeze()
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(cfg["n_layers"]) for h in range(cfg["n_heads"])]
    px.imshow(
        to_numpy(logit_attr),
        x=x_labels,
        y=tokens[:-1].cpu(),
        labels={"x": "Term", "y": "Position", "color": "logit"},
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
    ).show()


def logit_attribution(
    embed,
    l1_results,
    l2_results,
    W_U,
    tokens,
):
    """
    We have provided 'W_U_to_logits' which is a (position, d_model) tensor where each row is the unembed for the correct NEXT token at the current position.
    Inputs:
        embed: (position, d_model)
        l1_results: (position, head_index, d_model)
        l2_results: (position, head_index, d_model)
        W_U: (d_vocab, d_model)
    Returns:
        Tensor representing the concatenation (along dim=-1) of logit attributions from the direct path (position-1,1), layer 0 logits (position-1, n_heads) and layer 1 logits (position-1, n_heads).
    """
    W_U_to_logits = W_U[tokens[1:], :]
    direct_path_logits = t.einsum("pm,pm->p", W_U_to_logits, embed[:-1, :])
    l1_logits = t.einsum("pm,pim->pi", W_U_to_logits, l1_results[:-1])
    l2_logits = t.einsum("pm,pim->pi", W_U_to_logits, l2_results[:-1])
    logit_attribution = t.concat([direct_path_logits[:, None], l1_logits, l2_logits], dim=-1)
    return logit_attribution


"""
Now we can visualise the logit attributions for each path through the model.
"""
if MAIN:
    batch_index = 0
    embed = cache_2["hook_embed"][batch_index]
    l1_results = cache_2["blocks.0.attn.hook_result"][batch_index]
    l2_results = cache_2["blocks.1.attn.hook_result"][batch_index]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens_2[batch_index])
    plot_logit_attribution(logit_attr, tokens_2)


def plot_attn_pattern(patterns, tokens, title=None):
    if len(patterns.shape) == 3:
        tokens = tokens[0].cpu()
        px.imshow(
            to_numpy(patterns),
            animation_frame=0,
            y=tokens,
            x=tokens,
            labels={"x": "Key", "y": "Query"},
            color_continuous_scale="Blues",
            title=title,
        ).show()
    else:
        px.imshow(
            to_numpy(patterns),
            y=tokens,
            x=tokens,
            labels={"x": "Key", "y": "Query"},
            color_continuous_scale="Blues",
            title=title,
        ).show()


if MAIN:
    for layer in range(cfg["n_layers"]):
        plot_attn_pattern(cache_2[f"blocks.{layer}.attn.hook_attn"][0], tokens_2, f"Layer {layer} attention patterns")


def current_attn_detector(cache):
    current_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
    for layer in range(cfg["n_layers"]):
        attn = cache[f"blocks.{layer}.attn.hook_attn"]
        current_attn_score[layer] = reduce(
            attn.diagonal(dim1=-2, dim2=-1), "batch head_index pos -> head_index", "mean"
        )
    return current_attn_score


def prev_attn_detector(cache):
    prev_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
    for layer in range(cfg["n_layers"]):
        attn = cache[f"blocks.{layer}.attn.hook_attn"]
        prev_attn_score[layer] = reduce(
            attn.diagonal(dim1=-2, dim2=-1, offset=-1), "batch head_index pos -> head_index", "mean"
        )
    return prev_attn_score


def first_attn_detector(cache):
    first_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
    for layer in range(cfg["n_layers"]):
        attn = cache[f"blocks.{layer}.attn.hook_attn"]
        first_attn_score[layer] = reduce(attn[:, :, :, 0], "batch head_index pos -> head_index", "mean")
    return first_attn_score


def plot_head_scores(scores_tensor, title=""):
    px.imshow(
        to_numpy(scores_tensor),
        labels={"y": "Layer", "x": "Head"},
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    ).show()


if MAIN:
    current_attn_scores = current_attn_detector(cache_2)
    plot_head_scores(current_attn_scores, "Current Token Heads")
    prev_attn_scores = prev_attn_detector(cache_2)
    plot_head_scores(prev_attn_scores, "Prev Token Heads")
    first_attn_scores = first_attn_detector(cache_2)
    plot_head_scores(first_attn_scores, "First Token Heads")


def ablate_residual_stream_hook(resid_post, hook):
    # resid_post.shape is [batch, position, d_model]
    resid_post[:, 3:] = 0.0
    return resid_post


def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    # Negate so high loss is bad, and index by zero to remove the trivial batch dimension
    return -pred_log_probs[0]


if MAIN:
    corrupted_logits = model.run_with_hooks(
        tokens_2, fwd_hooks=[("blocks.1.hook_resid_post", ablate_residual_stream_hook)]
    )
    clean_per_token_losses = per_token_losses(logits_2, tokens_2)
    corrupted_per_token_losses = per_token_losses(corrupted_logits, tokens_2)
    px.line(
        to_numpy((corrupted_per_token_losses - clean_per_token_losses)), title="Difference in per token loss"
    ).show()


def ablated_head_run(model: EasyTransformer, tokens: t.Tensor, layer: int, head_index: int):
    def ablate_head_hook(value, hook):
        value[:, :, head_index, :] = 0.0
        return value

    logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_v", ablate_head_hook)])
    return logits


def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -(pred_log_probs.mean())


if MAIN:
    original_loss = cross_entropy_loss(logits_2, tokens_2)
    ablation_scores = t.zeros((cfg["n_layers"], cfg["n_heads"]))
    for layer in range(cfg["n_layers"]):
        for head_index in range(cfg["n_heads"]):
            ablation_scores[layer, head_index] = (
                cross_entropy_loss(ablated_head_run(model, tokens_2, layer, head_index), tokens_2) - original_loss
            )
    plot_head_scores(ablation_scores)


def run_and_cache_model_repeated_tokens(model, seq_len, batch=1) -> tuple[t.Tensor, t.Tensor, dict]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache
    Add a prefix token, since the model was always trained to have one.
    Outputs are:
    rep_logits: [batch, 1+2*seq_len, d_vocab]
    rep_tokens: [batch, 1+2*seq_len]
    rep_cache: {} The cache of the model run on rep_tokens
    """
    prefix = t.ones((batch, 1), dtype=t.int64) * 0
    rep_cache = {}
    model.cache_all(rep_cache)
    rand_tokens = t.randint(1, 1000, (batch, seq_len))
    rep_tokens = t.concat([prefix, rand_tokens, rand_tokens], dim=1).cuda()
    rep_logits = model(rep_tokens)
    return rep_logits, rep_tokens, rep_cache


if MAIN:
    """
    These are small numbers, since the results are very obvious and this makes it easier to visualise - in practice we'd obviously use larger ones on more subtle tasks. But it's often easiest to iterate and debug on small tasks.
    """
    seq_len = 49
    batch = 1
    rep_logits, rep_tokens, rep_cache = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print("Performance on the first half:", ptl[:seq_len].mean())
    print("Performance on the second half:", ptl[seq_len:].mean())
    px.line(
        to_numpy(ptl),
        hover_name=to_numpy(rep_tokens[0, :-1]),
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
    ).show()


def induction_attn_detector(cache):
    induction_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
    for layer in range(cfg["n_layers"]):
        attn = cache[f"blocks.{layer}.attn.hook_attn"]
        induction_attn_score[layer] = reduce(
            attn.diagonal(dim1=-2, dim2=-1, offset=-(seq_len - 1)), "batch head_index pos -> head_index", "mean"
        )
    return induction_attn_score


if MAIN:
    induction_attn_scores = induction_attn_detector(rep_cache)
    plot_head_scores(induction_attn_scores)


def get_q_comp_scores(W_QK, W_OV):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the Q-Composition score from head L0Hj to L1Hi
    """
    q_full = t.einsum("Imn,imM->IinM", W_QK, W_OV)
    comp_scores = (
        t.linalg.matrix_norm(q_full) / t.linalg.matrix_norm(W_QK)[:, None] / t.linalg.matrix_norm(W_OV)[None, :]
    )
    return comp_scores


def get_k_comp_scores(W_QK, W_OV):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the K-Composition score from head L0Hj to L1Hi
    """
    k_full = t.einsum("Inm,imM->IinM", W_QK, W_OV)
    comp_scores = (
        t.linalg.matrix_norm(k_full) / t.linalg.matrix_norm(W_QK)[:, None] / t.linalg.matrix_norm(W_OV)[None, :]
    )
    return comp_scores


def get_v_comp_scores(W_OV_1, W_OV_0):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the V-Composition score from head L0Hj to L1Hi
    """
    v_full = t.einsum("Inm,imM->IinM", W_OV_1, W_OV_0)
    comp_scores = (
        t.linalg.matrix_norm(v_full) / t.linalg.matrix_norm(W_OV_1)[:, None] / t.linalg.matrix_norm(W_OV_0)[None, :]
    )
    return comp_scores


# Carefully mathed out by generating some samples and then seeing that this seems correct
# (d_head, d_model) -> stats
RANDOM_FROB = {(32, 64): {"mean": 0.125, "std": 0.004}}


def get_comp_scores(
    model: EasyTransformer, use_svd=False, sub_baseline=False, ord: Any="fro"
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    W_O_0 = model.blocks[0].attn.W_O
    W_V_0 = model.blocks[0].attn.W_V
    W_OV_0 = t.einsum("imh,ihM->imM", W_O_0, W_V_0)
    W_Q = model.blocks[1].attn.W_Q
    W_K = model.blocks[1].attn.W_K
    W_V_1 = model.blocks[1].attn.W_V
    W_O_1 = model.blocks[1].attn.W_O
    W_QK = t.einsum("ihm,ihM->imM", W_Q, W_K)
    W_OV_1 = t.einsum("imh,ihM->imM", W_O_1, W_V_1)

    if not use_svd:
        q_comp_scores = get_q_comp_scores(W_QK, W_OV_0)
        k_comp_scores = get_k_comp_scores(W_QK, W_OV_0)
        v_comp_scores = get_v_comp_scores(W_OV_1, W_OV_0)
    else:
        q_comp_list = []
        k_comp_list = []
        v_comp_list = []
        num_heads = model.cfg["n_heads"]
        for to_head in range(num_heads):
            for from_head in range(num_heads):
                q_comp_list.append(
                    stranded_composition_score(
                        W_K[to_head].T, W_Q[to_head], W_O_0[from_head], W_V_0[from_head], ord=ord
                    )
                )
                k_comp_list.append(
                    stranded_composition_score(
                        W_Q[to_head].T, W_K[to_head], W_O_0[from_head], W_V_0[from_head], ord=ord
                    )
                )
                v_comp_list.append(
                    stranded_composition_score(
                        W_O_1[to_head], W_V_1[to_head], W_O_0[from_head], W_V_0[from_head], ord=ord
                    )
                )
        q_comp_scores = t.tensor(q_comp_list).reshape((num_heads, num_heads))
        k_comp_scores = t.tensor(k_comp_list).reshape((num_heads, num_heads))
        v_comp_scores = t.tensor(v_comp_list).reshape((num_heads, num_heads))

    if sub_baseline:
        comp_scores_baseline = RANDOM_FROB.get((model.cfg["d_head"], model.cfg["d_model"]), None)
        if comp_scores_baseline is None:
            raise ValueError(f"Go generate a RANDOM_FROB for ({model.cfg['d_head']}, {model.cfg['d_model']})")
        q_comp_scores -= comp_scores_baseline["mean"]
        k_comp_scores -= comp_scores_baseline["mean"]
        v_comp_scores -= comp_scores_baseline["mean"]

    return q_comp_scores, k_comp_scores, v_comp_scores


if MAIN:
    q_comp_scores, k_comp_scores, v_comp_scores = get_comp_scores(model)
    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()


def generate_single_random_comp_score(d_head: int, d_model: int) -> float:
    """
    Write a function which generates a single composition score for random matrices
    """
    matrices = [t.empty((d_head, d_model)) for i in range(4)]
    for mat in matrices:
        t.nn.init.kaiming_uniform_(mat, a=np.sqrt(5))
    W1 = matrices[0].T @ matrices[1]
    W2 = matrices[2].T @ matrices[3]
    W3 = W1 @ W2
    return (t.linalg.matrix_norm(W3) / t.linalg.matrix_norm(W1) / t.linalg.matrix_norm(W2)).item()


if MAIN:
    comp_scores_baseline = np.array([generate_single_random_comp_score() for i in range(200)])
    print("Mean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    px.histogram(comp_scores_baseline, nbins=50).show()
"""
We can re-plot our above graphs with this baseline set to white. Look for interesting things in this graph!
"""


def plot_comp_scores_baselined(model):
    q_comp_scores, k_comp_scores, v_comp_scores = get_comp_scores(model)

    comp_scores_baseline = RANDOM_FROB.get((model.cfg["d_head"], model.cfg["d_model"]), None)
    if comp_scores_baseline is None:
        raise ValueError(f"Go generate a RANDOM_FROB for ({model.cfg['d_head']}, {model.cfg['d_model']})")

    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(model.cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline["mean"],
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(model.cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline["mean"],
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(model.cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(model.cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline["mean"],
    ).show()


def stranded_svd(A: t.Tensor, B: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Returns the SVD of AB in the torch format (ie (U, S, V^T))
    """
    UA, SA, VhA = t.svd(A)
    UB, SB, VhB = t.svd(B)
    intermed = SA.diag() @ VhA.T @ UB @ SB.diag()
    UC, SC, VhC = t.svd(intermed)
    return (UA @ UC), SC, (VhB @ VhC).T


def stranded_composition_score(W_A1: t.Tensor, W_A2: t.Tensor, W_B1: t.Tensor, W_B2: t.Tensor, ord: Any="fro"):
    """
    Returns the composition score for W_A = W_A1 @ W_A2 and W_B = W_B1 @ W_B2, with the entries in a low-rank factored form
    """
    UA, SA, VhA = stranded_svd(W_A1, W_A2)
    UB, SB, VhB = stranded_svd(W_B1, W_B2)

    normA = t.linalg.matrix_norm(SA.diag(), ord=ord)
    normB = t.linalg.matrix_norm(SB.diag(), ord=ord)
    intermed = SA.diag() @ VhA @ UB @ SB.diag()
    SC = t.linalg.svdvals(intermed)
    normC = t.linalg.matrix_norm(SC.diag(), ord=ord)
    return normC / (normA * normB)


if MAIN:
    A = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = t.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    u, s, v = stranded_svd(A, B)
    print(u, s, v)
    print((A @ B).svd())

# %%
import torch as t


def test_norm_stuff(A, B, ord="fro"):
    num = t.linalg.matrix_norm(A @ B, ord=ord)
    denom_1 = t.linalg.matrix_norm(A, ord=ord)
    denom_2 = t.linalg.matrix_norm(B, ord=ord)

    print(num, denom_1, denom_2)
    return num / denom_1 / denom_2

test_norm_stuff(t.ones(5, 5), t.ones(5, 5))

# %%