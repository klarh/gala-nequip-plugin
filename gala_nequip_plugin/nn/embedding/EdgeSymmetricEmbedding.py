from typing import Union

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


@compile_mode("script")
class EdgeSymmetricEmbedding(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as a concatenation of a sum and difference of node-level representations.

    Args:
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        num_types,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.num_types = num_types
        self.out_field = out_field

        out_irrep = o3.Irreps([(2*num_types, (0, 1))])

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: out_irrep},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        one_hot = data[AtomicDataDict.NODE_ATTRS_KEY]
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]

        ti = one_hot[edge_index[0]]
        tj = one_hot[edge_index[1]]
        plus = ti + tj
        minus = ti - tj

        data[self.out_field] = torch.cat([plus, minus], axis=-1)
        return data
