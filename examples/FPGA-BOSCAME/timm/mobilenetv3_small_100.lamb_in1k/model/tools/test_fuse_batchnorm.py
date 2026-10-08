#!/usr/bin/env python3
"""Regression checks for folding math, activation preservation and graph safety."""

import unittest

import torch
from torch import nn
from timm.layers import BatchNormAct2d

from fuse_batchnorm import fold_batchnorm, inspect_graph


class FoldingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def check_fold(self, *, bias, groups, affine=True, activation=False):
        # Non-square, odd spatial dimensions, stride, padding and dilation check
        # that folding scales OIHW output channels without changing convolution.
        conv = nn.Conv2d(6, 6, (3, 5), stride=(2, 1), padding=(2, 2),
                         dilation=(2, 1), groups=groups, bias=bias).double()
        if activation:
            bn = BatchNormAct2d(6, eps=0.037, act_layer=nn.Hardswish,
                                drop_layer=nn.Dropout2d).double()
        else:
            bn = nn.BatchNorm2d(6, eps=0.037, affine=affine).double()
        with torch.no_grad():
            bn.running_mean.copy_(torch.linspace(-2, 3, 6))
            bn.running_var.copy_(torch.tensor([0.001, 0.2, 0.7, 1.2, 2.0, 4.0]))
            if bn.affine:
                # Negative and zero gamma expose incorrect activation ordering.
                bn.weight.copy_(torch.tensor([-1.5, 0.0, 0.5, 1.0, -0.2, 2.0]))
                bn.bias.copy_(torch.linspace(-1, 1, 6))
        model = nn.Sequential(conv, bn).eval()
        original_state = {name: value.clone() for name, value in model.state_dict().items()}
        x = torch.randn(2, 6, 13, 17, dtype=torch.float64)
        with torch.no_grad():
            reference = model(x)
        fused, inventory = fold_batchnorm(model)
        with torch.no_grad():
            actual = fused(x)
        torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
        self.assertEqual(len(inventory), 1)
        self.assertIsNotNone(fused[0].bias)
        self.assertEqual(fused[0].weight.shape, conv.weight.shape)
        for attr in ("stride", "padding", "dilation", "groups", "padding_mode"):
            self.assertEqual(getattr(fused[0], attr), getattr(conv, attr))
        self.assertEqual(inspect_graph(fused)[1], [])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        self.assertEqual(model[0].bias is not None, bias)
        if activation:
            self.assertIsInstance(fused[1].act, nn.Hardswish)
            self.assertIsInstance(fused[1].drop, nn.Dropout2d)
            self.assertFalse(fused[1].drop.training)

    def test_bias_none_grouped_and_depthwise(self):
        for groups in (1, 2, 6):
            with self.subTest(groups=groups):
                self.check_fold(bias=False, groups=groups)

    def test_existing_bias_grouped_and_depthwise(self):
        for groups in (1, 2, 6):
            with self.subTest(groups=groups):
                self.check_fold(bias=True, groups=groups)

    def test_non_affine_bn(self):
        self.check_fold(bias=True, groups=2, affine=False)

    def test_timm_activation_and_dropout_preserved(self):
        self.check_fold(bias=False, groups=6, activation=True)

    def test_training_rejected(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        with self.assertRaisesRegex(ValueError, "model.eval"):
            fold_batchnorm(model)
        model.eval()
        model[1].train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            fold_batchnorm(model)

    def test_missing_running_statistics_rejected(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1),
                              nn.BatchNorm2d(4, track_running_stats=False)).eval()
        with self.assertRaisesRegex(ValueError, "running statistics"):
            fold_batchnorm(model)

    def test_other_conv_consumers_rejected(self):
        class Branched(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                y = self.conv(x)
                return self.bn(y) + y

        with self.assertRaisesRegex(ValueError, "other consumers"):
            fold_batchnorm(Branched().eval())

    def test_reused_conv_rejected(self):
        class Reused(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.bn = nn.BatchNorm2d(4)

            def forward(self, x):
                return self.bn(self.conv(x)) + self.conv(x + 1)

        with self.assertRaisesRegex(ValueError, "Shared Conv/BN invocation"):
            fold_batchnorm(Reused().eval())


if __name__ == "__main__":
    unittest.main()
