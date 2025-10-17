import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['linear']:
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
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x

batch_size = 128
input_size = 10
hidden_size = 20

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]