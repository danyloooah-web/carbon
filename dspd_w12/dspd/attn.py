"""model.layers.*.self_attn: scope for lever MXT (dspd/mxt.py). The node runs the module class's stock forward inside
MXT's scope, so the engine's CuTe-DSL MXFP8 projections of this node take MXT's prefill tactic (bit-identical,
checked at first use per band); the audit's stock references run outside the context on the engine's tactic."""
from dspd import mxt


class _State:
    __slots__ = ("module", "stock")

    def __init__(self, module):
        self.module = module
        self.stock = type(module).forward


def prepare(module):
    mxt.install()
    return _State(module)


def forward(state, x, positions, forward_batch, x_quant=None, defer_all_reduce=False):
    with mxt.scope():
        return state.stock(state.module, x, positions, forward_batch, x_quant=x_quant, defer_all_reduce=defer_all_reduce)
