# %%
import dataclasses

from easy_transformer import EasyTransformer, EasyTransformerConfig
from typing import List, Dict, Tuple

import torch as t
import numpy as np
import torch.nn.functional as F
import tqdm
import random
import wandb
import time
import plotly
import plotly.express as px

from einops import rearrange
from neel_interp_stuff import get_comp_scores, get_k_comp_scores, get_q_comp_scores, get_v_comp_scores

cfg = EasyTransformerConfig(
    d_model=64,
    d_vocab=1000,
    n_ctx=100,
    n_layers=2,
    n_heads=2,
    d_head=32,
    attn_only=True,
    positional_embedding_type="shortformer"  # May want to enable this
)
train_cfg = {
    "lr": 1e-2,
    "batch_size": 1000,
}

# %%


def generate_data(batch_size: int, seq_len: int, vocab_size: int) -> t.Tensor:
    """Returns (batch_size, seq_len) tensor of tokens and same-shaped 'is this part of a repeated bit'"""
    i = 0
    tokens_needed = batch_size * seq_len
    tokens = t.randint(1, vocab_size, (tokens_needed,))
    token_copy = tokens.clone()  # So we can copy from it into the same memory space
    # learnable = t.zeros_like(tokens).bool()
    while i < tokens_needed:
        # Generate sentences up to half our context length (don't be cruel)
        sent_len = random.randint(1, seq_len // 2)
        gap_len = random.randint(0, seq_len // 2)

        # Start of sequence token
        tokens[i] = 0
        i += 1

        start_of_copy_from = i
        end_of_copy_from = i + sent_len + 1
        start_of_copy_to = end_of_copy_from + gap_len
        end_of_copy_to = start_of_copy_to + sent_len + 1
        # If we've run off the end, nothing else is learnable, and call it a day
        if end_of_copy_to >= tokens_needed:
            # learnable[i:] = False
            break

        tokens[start_of_copy_to:end_of_copy_to] = token_copy[start_of_copy_from:end_of_copy_from]
        # learnable[start_of_copy_to:end_of_copy_to] = True
        i = end_of_copy_to
    return rearrange(tokens, "(b s) -> b s", b=batch_size)


start = time.time()
foo = generate_data(2000, 100, 10000)
print(time.time() - start)
print(foo)

# %%


def get_test_loss(model: t.nn.Module, test_batch_size: int) -> float:
    with t.no_grad():
        test_sentences = []
        learnable_list = []
        for sent_i in range(test_batch_size):
            test_len = random.randint(1, cfg.n_ctx // 2 - 3)
            test_seq = t.randint(1, cfg.d_vocab, (test_len,)).repeat(2).to(device)
            test_seq = t.cat((t.tensor([0]).to(device), test_seq)).to(device)
            learnable = t.cat(
                (t.tensor([False]), t.zeros((test_len,)), t.tensor([False]), t.ones((test_len - 1,)))
            ).bool()
            test_sentences.append(test_seq)
            learnable_list.append(learnable)

        test_input = t.nn.utils.rnn.pad_sequence(test_sentences, batch_first=True)
        learnable = t.nn.utils.rnn.pad_sequence(learnable_list, batch_first=True, padding_value=False)
        test_out = model(test_input)

        learnable_in = test_input[learnable]
        learnable_out = test_out[learnable]
        loss = F.cross_entropy(learnable_out[:-1], learnable_in[1:])
        # print("LEARNABLE")
        # print(learnable_out[:-1].argmax(-1)[:100])
        # print(learnable_in[1:101])
        # print(learnable_out[:-1].argmax(-1)[:100] == learnable_in[1:101])

        # def color_when_true(arr):
        #     i = 0

        #     def _inner(x):
        #         nonlocal i
        #         color = arr.flatten()[i]
        #         i += 1
        #         return f"\u001b[31m{x}\u001b[0m" if color else str(x)

        #     return _inner

        # match = test_out[0].argmax(-1)[:-1] != test_input[0, 1:]
        # np.set_printoptions(formatter={"all": color_when_true(match)})
        # print("ALL")
        # print(test_out[0].argmax(-1)[:-1].cpu().numpy())
        # np.set_printoptions(formatter={"all": color_when_true(match)})
        # print(test_input[0, 1:].cpu().numpy())
        # np.set_printoptions(formatter={"all": color_when_true(match)})
        # print(test_out[0].argmax(-1)[:-1] == test_input[0, 1:])
        # np.set_printoptions()

        return loss.item()


# %%


def animate(x: List[np.ndarray], head_idx: int, fix_scale: bool = False, title: str = None):
    """Plot `x` as an animation.

    Args:
        x (List[np.ndarray]): _description_
        head_idx (int): What head index to look at
        fix_scale (bool, optional): Whether to pin the heatmap to the max range over all frames. Defaults to False.

    Returns:
        _type_: _description_
    """
    # x = [epoch, head, x, y]
    to_plot = np.stack(x)[:, head_idx, ...]
    zmin = to_plot.min() if fix_scale else None
    zmax = to_plot.max() if fix_scale else None
    fig = px.imshow(
        to_plot,
        animation_frame=0,
        title=title,
        zmin=zmin,
        zmax=zmax,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )
    # Transition every 50ms
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
    return fig


# %%
def make_model_and_optimizer(cfg: EasyTransformerConfig) -> Tuple[t.nn.Module, t.optim.Optimizer]:
    model = EasyTransformer(cfg)
    model.to(device)

    optim = t.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    return model, optim

def checkpoint(model: t.nn.Module, optim: t.optim.Optimizer, path: str) -> None:
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        },
        path,
    )


def load(config: EasyTransformerConfig, path: str) -> Tuple[t.nn.Module, t.optim.Optimizer]:
    model, optim = make_model_and_optimizer(config)

    checkpoint = t.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optim


device = "cuda"

if __name__ == "__main__":
    model, optim = make_model_and_optimizer(cfg)

    # scheduler = t.optim.lr_scheduler.MultiStepLR(optim, milestones=[1000, 1500], gamma=0.1)
    scheduler = t.optim.lr_scheduler.ConstantLR(optim, factor=1)  # Easy no-op schedule
# %%

# Out of function for now so I can get variables
log = True
num_epochs = 2000
resume_run = False
if True:
    # def train(model, num_epochs: int = 2000, log: bool = True, resume_run: bool = False):
    if log and not resume_run:
        # Save off both our model config and train config
        cfg_to_upload = dataclasses.asdict(cfg)
        cfg_to_upload.update(train_cfg)
        wandb.init(project="alex_resumes_overparameterizing", config=cfg_to_upload)

    batch_size = train_cfg["batch_size"]

    start_time = time.time()
    tot_tokens = 0

    w_q = []
    w_k = []
    w_o_0 = []
    w_v_0 = []
    w_o_1 = []
    w_v_1 = []
    w_qk = []
    w_ov_0 = []
    w_ov_1 = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        optim.zero_grad()

        tokens = generate_data(batch_size, cfg.n_ctx, cfg.d_vocab)
        # From that time I tried totally random data
        # tokens = t.randint(1, cfg["d_vocab"], (batch_size, cfg["n_ctx"]))
        tokens = tokens.to(device)

        out = model(tokens)

        flat_out = rearrange(out[:, :-1], "b s ... -> (b s) ...")
        flat_tokens = rearrange(tokens[:, 1:], "b s ... -> (b s) ...")
        l = F.cross_entropy(flat_out, flat_tokens)
        l.backward()

        optim.step()
        scheduler.step()

        comp_scores = {}
        with t.no_grad():
            # In-lined now to get access to everything
            # q_comp, k_comp, v_comp = get_comp_scores(model)
            W_O_0 = model.blocks[0].attn.W_O
            W_V_0 = model.blocks[0].attn.W_V
            W_OV_0 = t.einsum("imh,ihM->imM", W_O_0, W_V_0)
            W_Q = model.blocks[1].attn.W_Q
            W_K = model.blocks[1].attn.W_K
            W_V_1 = model.blocks[1].attn.W_V
            W_O_1 = model.blocks[1].attn.W_O
            W_QK = t.einsum("ihm,ihM->imM", W_Q, W_K)
            W_OV_1 = t.einsum("imh,ihM->imM", W_O_1, W_V_1)

            w_q.append(W_Q.detach().cpu().numpy())
            w_k.append(W_K.detach().cpu().numpy())
            w_o_0.append(W_O_0.detach().cpu().numpy())
            w_v_0.append(W_V_0.detach().cpu().numpy())
            w_o_1.append(W_O_1.detach().cpu().numpy())
            w_v_1.append(W_V_1.detach().cpu().numpy())
            w_qk.append(W_QK.detach().cpu().numpy())
            w_ov_0.append(W_OV_0.detach().cpu().numpy())
            w_ov_1.append(W_OV_1.detach().cpu().numpy())

            q_comp = get_q_comp_scores(W_QK, W_OV_0)
            k_comp = get_k_comp_scores(W_QK, W_OV_0)
            v_comp = get_v_comp_scores(W_OV_1, W_OV_0)

            for to_head in range(len(q_comp)):
                for from_head in range(len(q_comp[to_head])):
                    comp_scores[f"q_L1H{from_head}->L0H{to_head}"] = q_comp[to_head][from_head]
                    comp_scores[f"k_L1H{from_head}->L0H{to_head}"] = k_comp[to_head][from_head]
                    comp_scores[f"v_L1H{from_head}->L0H{to_head}"] = v_comp[to_head][from_head]

        test_loss = get_test_loss(model, 100) if epoch % 10 == 0 else None

        tot_tokens += tokens.nelement()
        if log:
            wandb.log(
                dict(
                    train_loss=l,
                    test_loss=test_loss,
                    elapsed=time.time() - start_time,
                    tokens=tot_tokens,
                    **comp_scores,
                )
            )
            # Could do this, but the scrobbler on wandb kinda sucks
            if epoch % 10 == 0 and False:
                head_idx = 0
                to_plot = w_q[-1][head_idx]
                fig = px.imshow(
                    to_plot,
                    title="test upload",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                )
                wandb.log({"w_qk_test": fig})

    if log:
        # Try to use up the wandb free tier as fast as possible with these 5MB HTML files
        # Uploading the plotly plots directly doesn't work for animations. See:
        # https://github.com/wandb/wandb/issues/2014
        # https://github.com/wandb/wandb/issues/2191
        wandb.log({"W_QK_0": wandb.Html(plotly.io.to_html(animate(w_qk[::50], head_idx=0, title="W_QK L1H0")))})
        wandb.log({"W_QK_1": wandb.Html(plotly.io.to_html(animate(w_qk[::50], head_idx=1, title="W_QK L1H1")))})

    if log:
        wandb.finish()

# %%


def predict(model, tokens: List[int]):
    return model(t.tensor([tokens])).argmax(-1)[:, -1]


def line_up_induction(model, tokens: List[int]):
    ret = model(t.tensor([tokens])).argmax(-1)
    print("actual:", "\t".join(str(i) for i in np.array(tokens)[1:]))
    print("pred:  ", "\t".join(str(i) for i in ret[0].cpu().numpy()[:-1]))


# TODO: Want to extra individual heads later
def copy_layer(layer: int, from_model: t.nn.Module, to_model: t.nn.Module, freeze: bool = False) -> t.nn.Module:
    from_dict = from_model.state_dict()
    old_to_dict = to_model.state_dict()
    match = f"blocks.{layer}"
    patched_dict = {k: (from_dict[k] if match in k else v) for k, v in old_to_dict.items()}
    to_model.load_state_dict(patched_dict)
    if freeze:
        for name, p in to_model.named_parameters():
            if match in name:
                p.requires_grad = False
    return to_model


# %%