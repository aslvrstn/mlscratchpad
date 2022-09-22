# %%

from dataclasses import dataclass
from typing import TYPE_CHECKING, NewType

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


@dataclass
class Config:
    vocab_size: int
    d_model: int


cfg = Config(10000, 32)


class Embed(t.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_E = t.nn.Parameter(t.empty(cfg.vocab_size, cfg.d_model))
        t.nn.init.kaiming_uniform_(self.W_E)

    if TYPE_CHECKING:
        def __call__(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Mod]: ...

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Mod]:
        return parse(self.W_E[x, :], Tensor[Bat, Pos, Mod])


class Unembed(t.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_U = t.nn.Parameter(t.empty(cfg.d_model, cfg.vocab_size))
        t.nn.init.kaiming_uniform_(self.W_U)

    if TYPE_CHECKING:
        def __call__(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Vocab]: ...

    def forward(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Vocab]:
        foo = einsum("mod voc, bat pos mod -> bat pos voc", self.W_U, x)
        return parse(foo, Tensor[Bat, Pos, Vocab])


# %%


class EmbedUnembedModel(t.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = Embed(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]:
        return self.unembed(self.embed(x))


testEmbedUnembed = EmbedUnembedModel(cfg)
typed_input = parse(t.tensor([[50, 999]]), Tensor[Bat, Pos])
bargle = testEmbedUnembed.forward(typed_input)


def func_on_good_output(x: Tensor[Bat, Pos, Vocab]):
    ...


def func_on_bad_output(x: Tensor[Vocab, Pos, Vocab]):
    ...


func_on_good_output(bargle)  # Happy
func_on_bad_output(bargle)  # Unhappy
# %%
