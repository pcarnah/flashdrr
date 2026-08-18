"""Tests for the per-channel piecewise-linear transfer function helper.

The helper is a small pure-tensor function, so these tests can be strict
about numerical accuracy.
"""
from __future__ import annotations

import pytest
import torch

from flashdrr.rendering import piecewise_linear_channelwise


class TestPiecewiseLinearChannelwise:
    def test_shape_preserved_5d(self):
        x = torch.rand(2, 3, 4, 5, 6)
        xp = torch.linspace(0, 1, 5).unsqueeze(0).expand(3, -1).contiguous()
        yp = torch.linspace(0, 1, 5).unsqueeze(0).expand(3, -1).contiguous()
        y = piecewise_linear_channelwise(x, xp, yp)
        assert y.shape == x.shape

    def test_shape_preserved_4d(self):
        x = torch.rand(2, 3, 5, 7)
        xp = torch.linspace(0, 1, 4).unsqueeze(0).expand(3, -1).contiguous()
        yp = torch.linspace(0, 1, 4).unsqueeze(0).expand(3, -1).contiguous()
        y = piecewise_linear_channelwise(x, xp, yp)
        assert y.shape == x.shape

    def test_identity_when_xp_yp_match(self):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 4, 5)
        xp = torch.linspace(0, 1, 4).unsqueeze(0).expand(3, -1).contiguous()
        yp = xp.clone()  # f(x) = x
        y = piecewise_linear_channelwise(x, xp, yp)
        assert torch.allclose(y, x, atol=1e-6)

    def test_constant_output(self):
        x = torch.rand(2, 2, 4, 5)
        xp = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        yp = torch.tensor([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]])
        y = piecewise_linear_channelwise(x, xp, yp)
        # All channel 0 outputs are 0.3
        assert torch.allclose(y[:, 0], torch.full_like(y[:, 0], 0.3), atol=1e-6)
        # All channel 1 outputs are 0.7
        assert torch.allclose(y[:, 1], torch.full_like(y[:, 1], 0.7), atol=1e-6)

    def test_clamps_below_first_keypoint(self):
        # x has shape (B=1, C=1, D=1, H=1, W=3): three scalars going through
        # the piecewise-linear mapping, so we can index the single result
        # values with [0, 0, 0, 0, :].
        x = torch.tensor([[[[[-0.5, 0.5, 1.5]]]]])
        xp = torch.tensor([[0.0, 1.0]])
        yp = torch.tensor([[0.0, 1.0]])  # identity mapping
        y = piecewise_linear_channelwise(x, xp, yp)
        # Below the first keypoint the function extends the first segment
        # (slope 1), so y(-0.5) = -0.5.
        assert y[0, 0, 0, 0, 0].item() == pytest.approx(-0.5, abs=1e-6)
        # Midpoint is interpolated
        assert y[0, 0, 0, 0, 1].item() == pytest.approx(0.5, abs=1e-6)
        # Above the last keypoint the function extends the last segment
        # (slope 1), so y(1.5) = 1.5.
        assert y[0, 0, 0, 0, 2].item() == pytest.approx(1.5, abs=1e-6)

    def test_per_channel_independence(self):
        x = torch.zeros(1, 2, 1, 1)
        x[0, 0, 0, 0] = 0.5
        x[0, 1, 0, 0] = 0.5
        xp = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        yp = torch.tensor([[0.0, 1.0], [0.5, 0.5]])  # channel 0 = identity, channel 1 = constant
        y = piecewise_linear_channelwise(x, xp, yp)
        assert y[0, 0, 0, 0].item() == pytest.approx(0.5, abs=1e-6)
        assert y[0, 1, 0, 0].item() == pytest.approx(0.5, abs=1e-6)

    def test_input_validation(self):
        x = torch.rand(1, 2, 4, 4)
        with pytest.raises(ValueError):
            # xp and yp with mismatched shapes
            piecewise_linear_channelwise(
                x,
                torch.zeros(2, 3),
                torch.zeros(2, 4),
            )
