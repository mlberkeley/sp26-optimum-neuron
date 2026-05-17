import sys
sys.path.insert(0, '..')   # so 'wan' is importable
import torch
import torch_neuronx
import wan                  # this triggers the device registration
from nkilib.core.mlp.mlp import mlp
from nkilib.core.utils.common_types import ActFnType

H, I, S = 2048, 8192, 4096
x = torch.randn(1, S, H, dtype=torch.bfloat16).to('neuron')
up_w = torch.randn(H, I, dtype=torch.bfloat16).to('neuron')
down_w = torch.randn(I, H, dtype=torch.bfloat16).to('neuron')
gate_w = torch.zeros(H, I, dtype=torch.bfloat16).to('neuron')

out = mlp(
    hidden_tensor=x,
    gate_proj_weights_tensor=gate_w,
    up_proj_weights_tensor=up_w,
    down_proj_weights_tensor=down_w,
    activation_fn=ActFnType.GELU_Tanh_Approx,
    skip_gate_proj=True,
)
print(type(out), out[0].shape)