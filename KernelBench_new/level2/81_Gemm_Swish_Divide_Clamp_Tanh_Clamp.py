import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['gemm']:
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
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = x / 2.0
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        x = torch.tanh(x)  # Tanh activation
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
        return x

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]