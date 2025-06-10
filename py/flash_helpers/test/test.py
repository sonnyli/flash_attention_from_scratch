import unittest

import flash_attention
import torch
from parameterized import parameterized

from flash_helpers.kernel_configs import DType, get_kernels_to_build
from flash_helpers.test.utils import (
    BATCH_SIZE_FOR_SEQ_LEN,
    BENCHMARK_N_HEADS,
    QKVConfig,
    generate_qkv,
    py_flash_attention,
)


class BaseFlashAttentionTest:
    @classmethod
    def setUpClass(cls):
        seq_len = 2048
        batch_size = BATCH_SIZE_FOR_SEQ_LEN[seq_len]
        n_heads = BENCHMARK_N_HEADS
        dtype = cls.dtype()
        device = "cuda:0"

        cls.d_heads = [128]

        cls.data = {}
        cls.pt_b16_results = {}
        cls.pt_f32_results = {}
        for d_head in cls.d_heads:
            cfg = QKVConfig(
                n_heads=n_heads,
                d_head=d_head,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=dtype,
                device=device,
            )

            q, k, v = generate_qkv(cfg)
            cls.data[d_head] = (q, k, v)

            cls.pt_b16_results[d_head] = py_flash_attention(
                q, k, v, upcast=False
            )
            cls.pt_f32_results[d_head] = py_flash_attention(
                q, k, v, upcast=True
            )

    def _test_standard(self, name, cfg):
        q, k, v = self.__class__.data[cfg.d_head]
        result = flash_attention.forward(cfg, q, k, v)
        fp16_result = self.__class__.pt_b16_results[cfg.d_head]
        fp32_result = self.__class__.pt_f32_results[cfg.d_head]

        # Based on https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
        diff_fp16 = (result - fp16_result).abs().max().item()
        diff_fp32 = (fp16_result - fp32_result).abs().max().item()

        self.assertLessEqual(diff_fp16, diff_fp32 * 2)

    @classmethod
    def dtype(self):
        raise NotImplementedError("Subclasses must implement this method")


class FlashAttentionTestFP16(BaseFlashAttentionTest, unittest.TestCase):
    @classmethod
    def dtype(self):
        return torch.float16

    @parameterized.expand(
        [
            (str(cfg), cfg)
            for cfg in get_kernels_to_build()
            if cfg.dtype == DType.FP16
        ],
        skip_on_empty=True,
    )
    def test_fp16(self, name, cfg):
        super()._test_standard(name, cfg)


class FlashAttentionTestBF16(BaseFlashAttentionTest, unittest.TestCase):
    @classmethod
    def dtype(self):
        return torch.bfloat16

    @parameterized.expand(
        [
            (str(cfg), cfg)
            for cfg in get_kernels_to_build()
            if cfg.dtype == DType.BF16
        ],
        skip_on_empty=True,
    )
    def test_bf16(self, name, cfg):
        super()._test_standard(name, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
