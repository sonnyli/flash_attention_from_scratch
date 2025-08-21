# Flash Attention from Scratch

## Introduction

This repository accompanies my series on [implementing Flash Attention from Scratch](https://lubits.ch/flash/) targeting Nvidia's Ampere architecture (A100, RTX 3090). Through 16 iterative kernel optimizations, the final implementation achieves **99.2% of the [official Flash Attention 2](https://github.com/Dao-AILab/flash-attention) performance on A100 and 102.9% on RTX 3090** (measured at sequence length 4096, $d_{\text{head}} = 128$).

The kernels implement these features:

- **[Flash Attention 2](https://arxiv.org/abs/2307.08691)**
- **forward pass of non-causal attention**
- **head dimension = 128**
- **no dropout or KV caching**
- **equal query/key/value sequence lengths**
- **sequence lengths divisible by block sizes (typically 64-128 in our implementation, as defined in the paper)**
- **16-bit (bf16/fp16) input and output tensors, softmax calculation in fp32**

The source code for kernels 1-15 is in `./previous_kernels`, while the source for kernel 16 is in `./src`.

## Installation

This builds the `flash_attention` (CUDA kernels) and `flash_helpers` (utils and kernel configuration) python packages.

```bash
pip install --no-build-isolation .
pip install ./py
```

## Testing

To test the kernels, you'll need a CUDA GPU with compute capability 8.x (A100, RTX 3090/4090, etc).

```bash
python py/flash_helpers/test/test.py
```

## Benchmarks

Here are the performances of each kernel relative to the official implementation.

| Kernel Iteration                                                        |           A100 |       A100 |       RTX 3090 |   RTX 3090 |
| :---------------------------------------------------------------------- | -------------: | ---------: | -------------: | ---------: |
|                                                                         | seq_len = 4096 | harm. mean* | seq_len = 4096 | harm. mean* |
| 1. Base Implementation                                                  |          15.8% |      16.6% |          49.5% |      49.8% |
| 2. Swizzling                                                            |          72.6% |      72.4% |          98.3% |      98.6% |
| 3. Eagerly Loading K & V Blocks                                         |          77.6% |      79.9% |          99.4% |     100.0% |
| 4. Interleaving On-Chip LD/ST with Computation                          |          77.6% |      80.0% |         100.0% |     100.4% |
| 5. Double Buffering Shared Memory to Register File Loads                |          76.8% |      79.1% |          99.7% |     100.3% |
| 6. Improving FP32 Throughput                                            |          78.1% |      80.4% |          99.9% |     100.4% |
| 7. Auto-Tuning                                                          |          80.3% |      82.3% |         101.5% |     101.8% |
| 8. Reducing `IADD3`, `LOP3`, and `SHF` instructions                     |          87.8% |      88.9% |         101.7% |     101.2% |
| 9. Reducing `IMAD.MOV.U32` and `MOV` instructions                       |          95.3% |      96.3% |          97.5% |      97.4% |
| 10. Removing `CSRZ` Instructions + Optimizing Initial Softmax Iteration |          93.9% |      95.0% |         102.9% |     102.3% |
| 11. Encoded Swizzling from the RF to SMEM                               |          95.2% |      96.7% |         102.8% |     102.3% |
| 12. Miscellaneous Code Changes                                          |          95.3% |      97.0% |         102.8% |     102.3% |
| 13. Iterating Backwards                                                 |          97.6% |      98.8% |         101.5% |     101.2% |
| 14. Cache Configuration                                                 |          97.7% |      99.1% |         101.5% |     101.2% |
| 15. Tiling along `d_head`                                               |          97.9% |      99.5% |         101.5% |     101.3% |
| 16. Static GMEM Stride                                                  |          99.2% |     100.4% |         100.9% |     100.7% |
|                                                                         |                |            |                |            |
| 0. Reference (TFLOPs)                                                   |          94.28 |      88.01 |          34.04 |      33.48 |

> *The harmonic mean is taken over sequence lengths 512, 1024, 2048, 4096, 8192, 16384.
