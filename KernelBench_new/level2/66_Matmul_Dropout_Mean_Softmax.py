import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, calculates the mean, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        # Deterministic initialization for benchmarking
        with torch.no_grad():
            torch.manual_seed(42)
            for name in ['matmul', 'dropout']:
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
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.softmax(x, dim=1)
        return x

batch_size = 128
in_features = 100
out_features = 50
dropout_p = 0.2

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]