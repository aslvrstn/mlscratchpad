# %%

from dataclasses import dataclass
from turtle import forward
from typing import TYPE_CHECKING, NewType
from typing_extensions import reveal_type

import torch as t
from fancy_einsum import einsum
from phantom_tensors import parse
from phantom_tensors.torch import Tensor

A = NewType("A", int)
B = NewType("B", int)

Bat = NewType("Bat", int)
Pos = NewType("Pos", int)
Tok = NewType("Tok", int)
Mod = NewType("Mod", int)
Vocab = NewType("Vocab", int)
HeadN = NewType("HeadN", int)


@dataclass
class Config:
    vocab_size: int
    d_model: int
    n_heads: int
    d_head: int
    n_ctx: int


cfg = Config(10000, 32, 4, 16, 512)


class Embed(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_E = t.nn.Parameter(t.empty(cfg.vocab_size, cfg.d_model))
        t.nn.init.kaiming_uniform_(self.W_E)

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Mod]:
        return parse(self.W_E[x, :], Tensor[Bat, Pos, Mod])

    if TYPE_CHECKING:
        def __call__(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Mod]: return self.forward(x)


class Unembed(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_U = t.nn.Parameter(t.empty(cfg.d_model, cfg.vocab_size))
        t.nn.init.kaiming_uniform_(self.W_U)

    def forward(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Vocab]:
        foo = einsum("mod voc, bat pos mod -> bat pos voc", self.W_U, x)
        return parse(foo, Tensor[Bat, Pos, Vocab])

    if TYPE_CHECKING:
        def __call__(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Vocab]: return self.forward(x)


class Attention(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_Q = t.nn.Parameter(t.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        t.nn.init.kaiming_uniform_(self.W_Q)
        self.W_K = t.nn.Parameter(t.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        t.nn.init.kaiming_uniform_(self.W_K)
        self.W_V = t.nn.Parameter(t.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        t.nn.init.kaiming_uniform_(self.W_V)
        self.W_O = t.nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        t.nn.init.kaiming_uniform_(self.W_O)

        self.causal_mask = t.tril(t.ones((cfg.n_ctx, cfg.n_ctx)).bool())

        self.attn_scale = t.sqrt(t.tensor(cfg.d_head))

    def forward(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Mod]:
        q = parse(einsum("headn head mod, bat pos mod -> bat pos headn mod", self.W_Q, x), Tensor[Bat, Pos, HeadN, Mod])
        k = parse(einsum("headn head mod, bat pos mod -> bat pos headn mod", self.W_K, x), Tensor[Bat, Pos, HeadN, Mod])
        v = parse(einsum("headn head mod, bat pos mod -> bat pos headn mod", self.W_V, x), Tensor[Bat, Pos, HeadN, Mod])

        attn = parse(einsum("bat posq headn mod, bat posk headn mod -> bat headn posq posk", q, k), Tensor[Bat, HeadN, Pos, Pos]) / self.attn_scale

        masked_attn = t.where(self.causal_mask, attn, -1e4)
        attn_probs = t.softmax(masked_attn, -1)

        attn_paid = parse(einsum("bat headn posq posk, bat posk headn mod -> bat posq headn mod", attn_probs, v), Tensor[Bat, Pos, HeadN, Mod])

        prereduce = parse(einsum("bat headn mod head, bat pos headn mod -> bat pos headn mod", v, attn_paid), Tensor[Bat, Pos, HeadN, Mod])
        result = parse(einsum("imh,bqih->bqim", self.W_O, prereduce), Tensor[Bat, Pos, HeadN, Mod])

        out = parse(result.sum(-2), Tensor[Bat, Pos, Mod])
        return out


    if TYPE_CHECKING:
        def __call__(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Mod]: return self.forward(x)

# %%


class EmbedUnembedModel(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = Embed(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]:
        return self.unembed(self.embed(x))

    if TYPE_CHECKING:
        def __call__(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]: return self.forward(x)


testEmbedUnembed = EmbedUnembedModel(cfg)
typed_input = parse(t.tensor([[50, 999]]), Tensor[Bat, Pos])
bargle = testEmbedUnembed(typed_input)


# %%


def func_on_good_output(x: Tensor[Bat, Pos, Vocab]):
    ...


def func_on_bad_output(x: Tensor[Vocab, Pos, Vocab]):
    ...


func_on_good_output(bargle)  # Happy
func_on_bad_output(bargle)  # Unhappy
# %%

class EmbedAttendUnembedModel(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = Embed(cfg)
        self.one_attn = Attention(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]:
        return self.unembed(self.one_attn(self.embed(x)))

    if TYPE_CHECKING:
        def __call__(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]: return self.forward(x)


testEmbedAttendUnembed = EmbedUnembedModel(cfg)
typed_input = parse(t.tensor([[50, 999]]), Tensor[Bat, Pos])
bargle = testEmbedAttendUnembed(typed_input)