"""Run with: python -m unittest test_transformer_regressions -v (from A5)."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from a5_helper import inference, train, val
from transformers import (
    CrossEntropyLoss, DecoderBlock, EncoderBlock, get_subsequent_mask,
    position_encoding_sinusoid, position_encoding_sinusoid_multiscale,
)


def reference_attention(module, query, key, value, mask=None):
    outputs = []
    for head in module.heads:
        q = F.linear(query, head.q.weight, head.q.bias)
        k = F.linear(key, head.k.weight, head.k.bias)
        v = F.linear(value, head.v.weight, head.v.bias)
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        outputs.append(scores.softmax(-1) @ v)
    return F.linear(
        torch.cat(outputs, -1), module.proj_down.weight, module.proj_down.bias
    )


def reference_norm(module, x):
    return F.layer_norm(
        x, (x.shape[-1],), module.gamma, module.beta, module.epsilon
    )


def reference_ffn(module, x):
    return F.linear(
        F.relu(F.linear(x, module.proj_up.weight, module.proj_up.bias)),
        module.proj_down.weight, module.proj_down.bias,
    )


def reference_block(module, x, enc=None, mask=None):
    def drop(y):
        return F.dropout(y, module.dropout.p, module.training)

    if enc is None:
        z = drop(reference_norm(
            module.LayerNorm1, x + reference_attention(module.MHA, x, x, x)
        ))
        return drop(reference_norm(module.LayerNorm2, z + reference_ffn(module.FFN, z)))
    z = drop(reference_norm(module.norm1, x + reference_attention(
        module.attention_self, x, x, x, mask
    )))
    u = drop(reference_norm(module.norm2, z + reference_attention(
        module.attention_cross, z, enc, enc
    )))
    return drop(reference_norm(module.norm3, u + reference_ffn(module.feed_forward, u)))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inp, inp_pos, ans, ans_pos):
        return F.one_hot(inp[:, 0], 16).float() * 10 + self.anchor * 0


class GenerationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_layer = nn.Embedding(16, 4)
        self.encoder = nn.Identity()
        self.masks = []

    def decoder(self, target, encoded, mask):
        self.masks.append(mask.detach().cpu())
        out = target.new_zeros(1, target.shape[1], 16)
        out[0, -1, [10, 2, 0, 15][target.shape[1] - 1]] = 10
        return out


class TransformerRegressions(unittest.TestCase):
    def test_block_formulas_and_gradients_with_shared_rng(self):
        for kind in [EncoderBlock, DecoderBlock]:
            for training in [False, True]:
                with self.subTest(kind=kind.__name__, training=training):
                    torch.manual_seed(7)
                    module = kind(2, 8, 16, .2).double().train(training)
                    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
                    enc = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)
                    mask = get_subsequent_mask(torch.zeros(2, 3))
                    state = torch.get_rng_state()
                    if kind is EncoderBlock:
                        actual = module(x)
                    else:
                        actual = module(x, enc, mask)
                    torch.set_rng_state(state)
                    expected = reference_block(
                        module, x, enc if kind is DecoderBlock else None, mask
                    )
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    variables = [x] + list(module.parameters())
                    if kind is DecoderBlock:
                        variables.append(enc)
                    upstream = torch.randn_like(actual)
                    ga = torch.autograd.grad(actual, variables, upstream, retain_graph=True)
                    ge = torch.autograd.grad(expected, variables, upstream)
                    for a, b in zip(ga, ge):
                        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)

    def test_validation_accumulates_all_batches_and_partial_batch(self):
        samples = []
        for target in [0, 0, 0, 1, 1]:
            samples.append((torch.tensor([0]), torch.zeros(1, 4),
                            torch.tensor([14, target]), torch.zeros(2, 4)))
        model = DummyModel()
        results = [val(model, DataLoader(samples, batch_size=b), CrossEntropyLoss, b)
                   for b in [2, 3, 5]]
        for loss, acc in results:
            self.assertAlmostEqual(acc, .6)
            self.assertAlmostEqual(loss, results[0][0], places=5)
        self.assertIsNone(model.anchor.grad)
        with self.assertRaises(ValueError):
            val(model, [], CrossEntropyLoss, 2)

    def test_inference_device_causal_mask_and_eos(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            model = GenerationModel().to(device)
            result, returned = inference(
                model, torch.tensor([14, 10, 2, 15]), torch.zeros(1, 4, 4),
                torch.zeros(1, 8, 4), 8,
            )
            self.assertIs(returned, model)
            self.assertEqual(result.tolist(), [[14, 10, 2, 0, 15]])
            self.assertEqual(result.device.type, device)
            self.assertEqual(len(model.masks), 4)
            for i, mask in enumerate(model.masks, 1):
                expected = torch.ones(1, i, i, dtype=torch.bool).triu(1)
                self.assertTrue(torch.equal(mask, expected))

    def test_position_encodings_keep_legacy_and_support_odd_width(self):
        legacy = position_encoding_sinusoid(4, 4)
        self.assertTrue(torch.equal(legacy[:, :, 0], legacy[:, :, 2]))
        for width in [1, 3, 4, 64]:
            actual = position_encoding_sinusoid_multiscale(9, width)
            self.assertEqual(actual.shape, (1, 9, width))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(actual[0, 0, 0].item(), 0)
        actual = position_encoding_sinusoid_multiscale(9, 4)
        self.assertFalse(torch.equal(actual[:, :, 0], actual[:, :, 2]))

    def test_training_restores_best_validation_weights(self):
        class LearnableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(16))

            def forward(self, inp, inp_pos, ans, ans_pos):
                return self.logits.expand(inp.shape[0], -1)

        model = LearnableModel()
        states = []

        def validation(*args, **kwargs):
            states.append(model.logits.detach().clone())
            return (1.0, .8) if len(states) == 1 else (.5, .7)

        batch = (torch.tensor([[0]]), torch.zeros(1, 1, 4),
                 torch.tensor([[14, 0]]), torch.zeros(1, 2, 4))
        with patch('a5_helper.val', side_effect=validation):
            returned = train(
                model, [batch], [batch], CrossEntropyLoss, num_epochs=2,
                batch_size=1, warmup_interval=None, lr=.1, restore_best=True,
            )
        self.assertIs(returned, model)
        self.assertFalse(torch.equal(states[0], states[1]))
        torch.testing.assert_close(model.logits, states[0])


if __name__ == '__main__':
    unittest.main()
