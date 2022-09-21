from typing import TYPE_CHECKING, Generic, TypeVar, cast

from torch.nn import Module
from typing_extensions import ParamSpec, Protocol

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class HasForward(Protocol[P, R]):
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> R:
        ...


class PhantomModule(Generic[P, R], Module):
    def __init__(self) -> None:
        super().__init__()

    if TYPE_CHECKING:

        def __call__(self, *args: P.args, **kwds: P.kwargs) -> R:
            return super().__call__(*args, **kwds)


def make_typed(cls: HasForward[P, R]) -> PhantomModule[P, R]:
    return cast(PhantomModule[P, R], cls)


# This is non-critical and just an example
class FunkyNN(PhantomModule):
    def __init__(self, arg: int) -> None:
        super().__init__()
        self.params = ["moooo"]

    def forward(self, x: int) -> int:
        return x


model = make_typed(FunkyNN(arg=3))

out = model(x=1)  # Happy
out = model(x="str")  # Sad
print(out)
