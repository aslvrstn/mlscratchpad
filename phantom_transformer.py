# %%

from dataclasses import dataclass
import nntplib
import this
from turtle import forward
import torch as t

from typing import NewType, Any

import numpy as np

from phantom_tensors import parse
from phantom_tensors._parse import HasShape
from phantom_module import PhantomModule, make_typed

from phantom_tensors.numpy import NDArray
from phantom_tensors.torch import Tensor

from fancy_einsum import einsum


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

class Embed(PhantomModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_E = t.nn.Parameter(t.empty(cfg.vocab_size, cfg.d_model))
        t.nn.init.kaiming_uniform_(self.W_E)

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Mod]:
        return parse(self.W_E[x, :], Tensor[Bat, Pos, Mod])


class Unembed(PhantomModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.W_U = t.nn.Parameter(t.empty(cfg.d_model, cfg.vocab_size))
        t.nn.init.kaiming_uniform_(self.W_U)

    def forward(self, x: Tensor[Bat, Pos, Mod]) -> Tensor[Bat, Pos, Vocab]:
        foo = einsum("mod voc, bat pos mod -> bat pos voc", self.W_U, x)
        return parse(foo, Tensor[Bat, Pos, Vocab])

# %%

class EmbedUnembedModel(PhantomModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = make_typed(Embed(cfg))
        self.unembed = make_typed(Unembed(cfg))

    def forward(self, x: Tensor[Bat, Pos]) -> Tensor[Bat, Pos, Vocab]:
        return self.unembed(self.embed(x))

testEmbedUnembed = EmbedUnembedModel(cfg)
typed_input = parse(t.tensor([[50, 999]]), Tensor[Bat, Pos])
bargle = testEmbedUnembed.forward(typed_input)

def func_on_3d(x: Tensor[Bat, Pos, Vocab]):
    print(x)

func_on_3d(bargle)
# %%
