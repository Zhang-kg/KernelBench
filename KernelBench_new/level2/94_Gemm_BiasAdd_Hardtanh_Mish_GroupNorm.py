import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['gemm', 'bias', 'hardtanh', 'mish', 'groupnorm']:
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
        x = x + self.bias
        x = self.hardtanh(x)
        x = self.mish(x)
        x = self.groupnorm(x)
        return x


batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]