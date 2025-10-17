import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
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
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)  # (batch_size, out_features)
        x = torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0] # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        return x

batch_size = 128
in_features = 10
out_features = 5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]