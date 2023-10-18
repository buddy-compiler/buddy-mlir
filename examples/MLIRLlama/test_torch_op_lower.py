import torch
import torch._dynamo as dynamo
from buddy.LlamaCompiler import DynamoCompiler

class OnesModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return torch.ones(a, dtype=torch.bool)

class FullModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.full(a, b, dtype=torch.float32)

class AddModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.add(a, b)

class LtModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.lt(a, b)

class MaskedfillModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b, c):
        return torch.masked_fill(a, b, c)

class SliceModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return a[:]

class ExpandModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return a.expand(1, 1, 13, 13)

class TocopyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return a.float()

class RsubModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return b-a

class PowModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.pow(a, b)

class MeanModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.mean(a, b, True)

class RsqrtModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return torch.rsqrt(a)

class MulModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.mul(a, b)

class TModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return a.t()

class TransposeModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return torch.transpose(a, 1, 2)

class SqueezeModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return torch.squeeze(a, 1)

class IndexModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return a[[b]]

class NegModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return torch.neg(a)

class CatModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.cat([a, b], -1)

class BmmModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.bmm(a, b)

class DivModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a, b):
        return torch.div(a, b)

class SoftmaxModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, a):
        return torch.softmax(a, -1)

model = SoftmaxModule()
model_opt = dynamo.optimize(DynamoCompiler)(model)
arg0 = torch.full([1, 32, 13, 13], 2, dtype=torch.float)
arg1 = torch.full([32, 128, 13], 1, dtype=torch.float)
print(model_opt(arg0))
