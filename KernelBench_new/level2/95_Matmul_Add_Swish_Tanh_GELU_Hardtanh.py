import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['matmul', 'add_value']:
                attr = getattr(self, name, None)
                if attr is None:
                    continue
                if hasattr(attr, 'weight'):
                    attr.weight.copy_(torch.randn_like(attr.weight))
                    attr.weight.requires_grad = False
                    bias_attr = getattr(attr, 'bias', None)
                    if bias_attr is not None:
                        bias_attr.zero_()
                        bias_attr.requires_grad = False
                elif isinstance(attr, torch.nn.Parameter):
                    attr.copy_(torch.randn_like(attr))
                    attr.requires_grad = False
                elif isinstance(attr, torch.Tensor):
                    attr.copy_(torch.randn_like(attr))
                    attr.requires_grad = False

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = torch.sigmoid(x) * x # Swish
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x) # GELU
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1) # Hardtanh
        return x

batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]