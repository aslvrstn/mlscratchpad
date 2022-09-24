# %%

from dataclasses import dataclass
from turtle import forward
from typing import TYPE_CHECKING, NewType
from typing_extensions import reveal_type
from phantom import Predicate

import torch as t
from fancy_einsum import einsum
from typed_einsum import einsum as typed_einsum
from phantom_tensors import parse
from phantom_tensors.torch import Tensor

A = NewType("A", int)
B = NewType("B", int)

Bat = NewType("Bat", int)
Pos = NewType("Pos", int)
Tok = NewType("Tok", int)
Mod = NewType("Mod", int)
Vocab = NewType("Vocab", int)
Head = NewType("Head", int)
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

        def __call__(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Mod]:
            return self.forward(x)


class Unembed(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_U = t.nn.Parameter(t.empty(cfg.d_model, cfg.vocab_size))
        t.nn.init.kaiming_uniform_(self.W_U)

    def forward(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Vocab]:
        return typed_einsum(
            Tensor[Mod, Vocab],
            Tensor[Bat, Pos, Mod],
            out_type=Tensor[Bat, Pos, Vocab],
            tensors=[self.W_U, x],
        )

    if TYPE_CHECKING:

        def __call__(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Vocab]:
            return self.forward(x)


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
        q = typed_einsum(
            Tensor[HeadN, Head, Mod],
            Tensor[Bat, Pos, Mod],
            out_type=Tensor[Bat, Pos, HeadN, Mod],
            tensors=[self.W_Q, x],
        )
        k = typed_einsum(
            Tensor[HeadN, Head, Mod],
            Tensor[Bat, Pos, Mod],
            out_type=Tensor[Bat, Pos, HeadN, Mod],
            tensors=[self.W_K, x],
        )
        v = typed_einsum(
            Tensor[HeadN, Head, Mod],
            Tensor[Bat, Pos, Mod],
            out_type=Tensor[Bat, Pos, HeadN, Mod],
            tensors=[self.W_V, x],
        )

        PosQ = NewType("PosQ", int)
        PosK = NewType("PosK", int)
        attn = (
            typed_einsum(
                Tensor[Bat, PosQ, HeadN, Mod],
                Tensor[Bat, PosK, HeadN, Mod],
                out_type=Tensor[Bat, HeadN, PosQ, PosK],
                tensors=[q, k],
            )
            / self.attn_scale
        )

        masked_attn = t.where(self.causal_mask, attn, -1e4)
        attn_probs = t.softmax(masked_attn, -1)

        attn_paid = typed_einsum(
            Tensor[Bat, HeadN, PosQ, PosK],
            Tensor[Bat, PosK, HeadN, Mod],
            out_type=Tensor[Bat, PosQ, HeadN, Mod],
            tensors=[attn_probs, v],
        )

        prereduce = typed_einsum(
            Tensor[Bat, HeadN, Mod, Head],
            Tensor[Bat, Pos, HeadN, Mod],
            out_type=Tensor[Bat, Pos, HeadN, Mod],
            tensors=[v, attn_paid],
        )
        result = typed_einsum(
            Tensor[HeadN, Mod, Head],
            Tensor[Bat, PosQ, HeadN, Head],
            out_type=Tensor[Bat, Pos, HeadN, Mod],
            tensors=[self.W_O, prereduce],
        )

        out = parse(result.sum(-2), Tensor[Bat, Pos, Mod])
        return out

    if TYPE_CHECKING:

        def __call__(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Mod]:
            return self.forward(x)


# %%


class EmbedUnembedModel(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = Embed(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]:
        return self.unembed(self.embed(x))

    if TYPE_CHECKING:

        def __call__(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]:
            return self.forward(x)


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

        def __call__(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]:
            return self.forward(x)


testEmbedAttendUnembed = EmbedUnembedModel(cfg)
typed_input = parse(t.tensor([[50, 999]]), Tensor[Bat, Pos])
bargle = testEmbedAttendUnembed(typed_input)
# %%
