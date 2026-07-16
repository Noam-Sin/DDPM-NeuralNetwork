import unittest

import _pathfix  # noqa: F401
import torch

from diffusion import (
    GaussianDiffusion,
    EMA,
    linear_beta_schedule,
    cosine_beta_schedule,
)
from unet import UNet


class TestBetaSchedules(unittest.TestCase):
    def test_linear_endpoints(self):
        betas = linear_beta_schedule(1000, beta_start=1e-4, beta_end=0.02)
        self.assertEqual(betas.shape, (1000,))
        self.assertAlmostEqual(betas[0].item(), 1e-4, places=6)
        self.assertAlmostEqual(betas[-1].item(), 0.02, places=6)

    def test_linear_monotonic(self):
        betas = linear_beta_schedule(50)
        self.assertTrue(torch.all(betas[1:] >= betas[:-1]))

    def test_cosine_in_bounds(self):
        betas = cosine_beta_schedule(1000)
        self.assertEqual(betas.shape, (1000,))
        self.assertTrue(torch.all(betas > 0))
        self.assertTrue(torch.all(betas <= 0.999))

    def test_unknown_schedule_raises(self):
        with self.assertRaises(ValueError):
            GaussianDiffusion(timesteps=10, schedule="quadratic")


class TestGaussianDiffusion(unittest.TestCase):
    def _diffusion(self, schedule="linear", T=50):
        return GaussianDiffusion(timesteps=T, beta_start=1e-4, beta_end=0.02, schedule=schedule)

    def test_configurable_T_changes_schedule_length(self):
        d1 = self._diffusion(T=10)
        d2 = self._diffusion(T=100)
        self.assertEqual(d1.betas.shape[0], 10)
        self.assertEqual(d2.betas.shape[0], 100)

    def test_q_sample_shape_and_range_sanity(self):
        d = self._diffusion()
        x0 = torch.zeros(4, 3, 8, 8)  # zero image -> x_t should equal sqrt(1-abar)*noise
        t = torch.tensor([0, 10, 25, 49])
        noise = torch.randn_like(x0)
        x_t = d.q_sample(x0, t, noise)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertFalse(torch.isnan(x_t).any())

    def test_q_sample_at_t0_is_close_to_x0(self):
        d = self._diffusion()
        x0 = torch.randn(2, 3, 8, 8)
        t = torch.zeros(2, dtype=torch.long)
        noise = torch.zeros_like(x0)
        x_t = d.q_sample(x0, t, noise)
        self.assertTrue(torch.allclose(x_t, x0 * d.sqrt_alphas_cumprod[0], atol=1e-5))

    def test_training_loss_is_finite_scalar_and_differentiable(self):
        d = self._diffusion()
        model = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        x0 = torch.randn(2, 3, 8, 8)
        loss = d.training_loss(model, x0)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad_norm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0)

    def test_p_sample_loop_shape(self):
        d = self._diffusion(T=5)  # tiny T so the loop is fast
        model = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        out = d.p_sample_loop(model, (2, 3, 8, 8), device="cpu")
        self.assertEqual(out.shape, (2, 3, 8, 8))
        self.assertFalse(torch.isnan(out).any())

    def test_sample_convenience_method(self):
        d = self._diffusion(T=5)
        model = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        out = d.sample(model, batch_size=3, image_size=8, channels=3, device="cpu")
        self.assertEqual(out.shape, (3, 3, 8, 8))

    def test_sample_uses_model_image_size_when_not_given(self):
        d = self._diffusion(T=5)
        model = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        model.image_size = 8
        out = d.sample(model, batch_size=2)
        self.assertEqual(out.shape, (2, 3, 8, 8))

    def test_sample_restores_training_mode(self):
        d = self._diffusion(T=2)
        model = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        model.train()
        d.sample(model, batch_size=1, image_size=8)
        self.assertTrue(model.training)


class TestEMA(unittest.TestCase):
    def test_shadow_moves_toward_updated_weights(self):
        model = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        ema = EMA(model, decay=0.5)
        key = next(iter(model.state_dict()))
        initial = ema.shadow[key].clone()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        updated_shadow = ema.shadow[key]
        self.assertFalse(torch.allclose(initial, updated_shadow))

    def test_copy_to_loads_shadow_weights(self):
        model = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        ema = EMA(model, decay=0.9)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        target = UNet(image_size=8, base_channels=4, channel_mult=(1,), num_res_blocks=1, attn_resolutions=())
        ema.copy_to(target)
        for k in model.state_dict():
            self.assertTrue(torch.allclose(target.state_dict()[k], ema.shadow[k]))


if __name__ == "__main__":
    unittest.main()
