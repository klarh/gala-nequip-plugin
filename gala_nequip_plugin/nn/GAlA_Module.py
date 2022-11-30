
import torch

from e3nn.util.jit import compile_mode

from nequip.nn import GraphModuleMixin

@compile_mode('script')
class GAlA_Module(GraphModuleMixin, torch.nn.Module):
    pass

    def forward(data):
        import pdb;pdb.set_trace()
