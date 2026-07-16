import unittest

import _pathfix  # noqa: F401
import torch

from unet import UNet


class TestUNetShapes(unittest.TestCase):
    def _build(self, image_size=16, base_channels=8, channel_mult=(1, 2), attn_resolutions=(8,)):
        return UNet(
            image_size=image_size,
            image_channels=3,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=1,
            attn_resolutions=attn_resolutions,
            dropout=0.0,
        )

    def test_forward_shape_matches_input(self):
        model = self._build()
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        self.assertEqual(out.shape, x.shape)

    def test_forward_no_nan(self):
        model = self._build()
        x = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_timestep_embedding_differs_across_t(self):
        # The residual blocks' last conv (and every other output-facing
        # conv) is zero-initialized by design (see unet.py / METHODOLOGY.md),
        # so a freshly-initialized model's *output* is t-invariant until it
        # has trained -- that's intentional, not a bug. What must vary with
        # t from the start is the embedding fed into each block.
        from unet import timestep_embedding
        emb0 = timestep_embedding(torch.tensor([0]), dim=32)
        emb1 = timestep_embedding(torch.tensor([999]), dim=32)
        self.assertFalse(torch.allclose(emb0, emb1))

    def test_output_becomes_t_dependent_once_zero_init_layers_move(self):
        # Every output-facing conv (residual blocks' conv2, attention's
        # proj_out, the network's final conv_out) is zero-initialized by
        # design, so t-dependence is gated behind *all* of them at init
        # (backprop through a zero-weight layer zeroes the gradient to
        # everything upstream of it -- one optimizer step only unfreezes
        # the outermost zero-init layer, not the temb path feeding it).
        # Injecting noise directly simulates "a few steps in", where every
        # layer has moved off zero and the temb path can actually reach
        # the output.
        model = self._build()
        torch.manual_seed(0)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))

        x = torch.randn(1, 3, 16, 16)
        t0 = torch.tensor([0])
        t1 = torch.tensor([999])
        out0 = model(x, t0)
        out1 = model(x, t1)
        self.assertFalse(torch.allclose(out0, out1))

    def test_single_resolution_level_no_attention(self):
        # smallest possible config: one level, no downsampling at all.
        model = self._build(image_size=8, base_channels=4, channel_mult=(1,), attn_resolutions=())
        x = torch.randn(2, 3, 8, 8)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        self.assertEqual(out.shape, x.shape)

    def test_odd_batch_size(self):
        model = self._build()
        x = torch.randn(3, 3, 16, 16)
        t = torch.randint(0, 1000, (3,))
        out = model(x, t)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
