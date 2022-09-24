from fancy_einsum import einsum as _einsum

from typing import TypeVar, Type, Tuple, Any
from phantom_tensors import parse
from phantom_tensors._parse import HasShape

from phantom_tensors.torch import Tensor
from phantom_tensors.alphabet import A, B, C

T = TypeVar("T", bound=HasShape)

def _type_to_einstr(x): return ' '.join(v.__name__ for v in x.__args__)

def einsum(*in_types, out_type: Type[T], tensors) -> T:
    """
    Examples
    --------
    import torch as tr

    x, y = parse(
        (tr.ones(2, 3), Tensor[A, B]),
        (tr.ones(3, 4), Tensor[B, C]),
    )

    out = einsum(
        Tensor[A, B],
        Tensor[B, C],
        out_type=Tensor[A, C],
        tensors=(x, y),
    )

    out  # type checker sees: Tensor[A, C]"""
    assert len(in_types) == len(tensors)
    in_str = ", ".join(_type_to_einstr(tp) for tp in in_types)
    out_str = _type_to_einstr(out_type)

    out = _einsum(f"{in_str} -> {out_str}", *tensors)
    in_types += (out_type,)
    tensors += (out, )
    parse(*zip(tensors, in_types))  # check all types
    parse(out, out_type)  # casting
    return parse(out, out_type)
