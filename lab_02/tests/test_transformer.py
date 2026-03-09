import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from config import TransformerConfig
from data_pipeline import DataPipeline
from transformer.utils.activations import softmax, relu
from transformer.utils.normalization import layer_norm
from transformer.layers.attention import ScaledDotProductAttention
from transformer.layers.feed_forward import FeedForwardNetwork
from transformer.layers.encoder_layer import EncoderLayer
from transformer.encoder import TransformerEncoder



@pytest.fixture
def cfg() -> TransformerConfig:
    return TransformerConfig(d_model=16, d_ff=32, n_layers=2, seed=0)


@pytest.fixture
def dummy_input(cfg) -> np.ndarray:
    """Tensor (1, 5, d_model) com valores aleatórios."""
    np.random.seed(0)
    return np.random.randn(1, 5, cfg.d_model)



class TestSoftmax:
    def test_output_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = softmax(x)
        assert np.isclose(out.sum(), 1.0), "Softmax deve somar 1."

    def test_shape_preserved(self):
        x = np.random.randn(2, 4, 6)
        assert softmax(x).shape == x.shape

    def test_numerical_stability(self):
        x = np.array([[1e10, 1e10, 1e10]])
        out = softmax(x)
        assert not np.any(np.isnan(out)), "Softmax não deve gerar NaN."


class TestRelu:
    def test_negative_values_zeroed(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        out = relu(x)
        assert np.all(out >= 0), "ReLU não deve produzir negativos."

    def test_positive_values_unchanged(self):
        x = np.array([1.0, 2.0, 5.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_shape_preserved(self):
        x = np.random.randn(3, 4, 5)
        assert relu(x).shape == x.shape


class TestLayerNorm:
    def test_mean_near_zero(self):
        x = np.random.randn(2, 5, 16)
        out = layer_norm(x)
        assert np.allclose(out.mean(axis=-1), 0, atol=1e-5), \
            "Média pós-LayerNorm deve ser ~0."

    def test_std_near_one(self):
        x = np.random.randn(2, 5, 16)
        out = layer_norm(x)
        assert np.allclose(out.std(axis=-1), 1, atol=1e-4), \
            "Desvio padrão pós-LayerNorm deve ser ~1."

    def test_shape_preserved(self):
        x = np.random.randn(1, 7, 64)
        assert layer_norm(x).shape == x.shape



class TestScaledDotProductAttention:
    def test_output_shape(self, cfg, dummy_input):
        attn = ScaledDotProductAttention(cfg.d_model)
        out  = attn.forward(dummy_input)
        assert out.shape == dummy_input.shape, \
            "Atenção deve preservar o shape do tensor."

    def test_output_finite(self, cfg, dummy_input):
        attn = ScaledDotProductAttention(cfg.d_model)
        out  = attn.forward(dummy_input)
        assert np.all(np.isfinite(out)), "Saída da atenção contém Inf/NaN."


class TestFeedForwardNetwork:
    def test_output_shape(self, cfg, dummy_input):
        ffn = FeedForwardNetwork(cfg.d_model, cfg.d_ff)
        out = ffn.forward(dummy_input)
        assert out.shape == dummy_input.shape, \
            "FFN deve preservar o shape do tensor."

    def test_output_finite(self, cfg, dummy_input):
        ffn = FeedForwardNetwork(cfg.d_model, cfg.d_ff)
        out = ffn.forward(dummy_input)
        assert np.all(np.isfinite(out)), "Saída da FFN contém Inf/NaN."


class TestEncoderLayer:
    def test_output_shape(self, cfg, dummy_input):
        layer = EncoderLayer(cfg.d_model, cfg.d_ff, cfg.epsilon)
        out   = layer.forward(dummy_input)
        assert out.shape == dummy_input.shape

    def test_residual_changes_values(self, cfg, dummy_input):
        layer = EncoderLayer(cfg.d_model, cfg.d_ff, cfg.epsilon)
        out   = layer.forward(dummy_input)
        assert not np.allclose(out, dummy_input), \
            "EncoderLayer não deve retornar a entrada inalterada."



class TestTransformerEncoder:
    def test_output_shape_preserved(self, cfg, dummy_input):
        encoder = TransformerEncoder(cfg)
        Z = encoder.forward(dummy_input)
        assert Z.shape == dummy_input.shape, \
            "O shape (batch, seq, d_model) deve ser preservado após N camadas."

    def test_invalid_input_ndim(self, cfg):
        encoder = TransformerEncoder(cfg)
        with pytest.raises(ValueError, match="3 dimensões"):
            encoder.forward(np.random.randn(5, cfg.d_model))

    def test_invalid_d_model(self, cfg):
        encoder = TransformerEncoder(cfg)
        wrong_input = np.random.randn(1, 5, cfg.d_model + 1)
        with pytest.raises(ValueError, match="d_model"):
            encoder.forward(wrong_input)

    def test_all_layers_executed(self, cfg, dummy_input):
        call_count = 0
        encoder = TransformerEncoder(cfg)
        original_forward = EncoderLayer.forward

        def counting_forward(self_layer, X):
            nonlocal call_count
            call_count += 1
            return original_forward(self_layer, X)

        EncoderLayer.forward = counting_forward
        encoder.forward(dummy_input)
        EncoderLayer.forward = original_forward

        assert call_count == cfg.n_layers, \
            f"Esperado {cfg.n_layers} chamadas, obteve {call_count}."



class TestDataPipeline:
    def test_tokenize_known_words(self):
        pipeline = DataPipeline()
        ids = pipeline.tokenize("o banco bloqueou")
        assert ids == [0, 1, 2]

    def test_tokenize_ignores_unknown(self):
        pipeline = DataPipeline()
        ids = pipeline.tokenize("o xpto banco")
        assert ids == [0, 1], "Palavras desconhecidas devem ser ignoradas."

    def test_input_tensor_shape(self):
        cfg      = TransformerConfig(d_model=16)
        pipeline = DataPipeline(config=cfg)
        ids      = pipeline.tokenize("o banco bloqueou")
        X        = pipeline.build_input_tensor(ids)
        assert X.shape == (1, 3, 16)

    def test_empty_ids_raises(self):
        pipeline = DataPipeline()
        with pytest.raises(ValueError, match="vazia"):
            pipeline.build_input_tensor([])
