import torch


def fold_batch_norms(model: torch.nn.Module) -> None:
    if model.training:
        raise ValueError("BatchNorm folding requires an eval-mode model")
    for parent in model.modules():
        names = list(parent._modules)
        for index in range(len(names) - 1):
            conv_name, bn_name = names[index : index + 2]
            conv = parent._modules[conv_name]
            bn = parent._modules[bn_name]
            if not isinstance(conv, torch.nn.Conv2d) or not isinstance(
                bn, torch.nn.BatchNorm2d
            ):
                continue
            parent._modules[conv_name] = (
                torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            )
            parent._modules[bn_name] = torch.nn.Identity()
