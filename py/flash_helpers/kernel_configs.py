import itertools
import os
import re
from dataclasses import dataclass, replace
from enum import IntEnum


# This is a hack to avoid importing torch.
class DType(IntEnum):
    # https://github.com/pytorch/pytorch/blob/c37ddcaefbe9b877e1816ce97dedb8ad26d09450/c10/core/ScalarType.h
    # These are the enum values for the torch types
    FP16 = 5
    BF16 = 15

    def to_cpp_str(self) -> str:
        if self == DType.FP16:
            return "torch::kFloat16"
        elif self == DType.BF16:
            return "torch::kBFloat16"
        else:
            raise ValueError(f"Invalid DType: {self}")

    def to_torch_dtype(self):
        import torch

        if self == DType.FP16:
            return torch.float16
        elif self == DType.BF16:
            return torch.bfloat16
        else:
            raise ValueError(f"Invalid DType: {self}")

    @classmethod
    def from_string(cls, dtype_str: str) -> "DType":
        """Parse DType from string. Case-insensitive. Handles both names ('FP16', 'BF16') and integers ('0', '1')."""
        dtype_str = dtype_str.strip()

        # Try parsing as integer first
        try:
            dtype_int = int(dtype_str)
            return cls(dtype_int)
        except ValueError:
            pass

        # Try parsing as enum name
        dtype_str = dtype_str.upper()
        try:
            return cls[dtype_str]
        except KeyError:
            valid_options = [
                f"{member.name} ({member.value})" for member in cls
            ]
            raise ValueError(
                f"Invalid dtype string '{dtype_str}'. Valid options: {valid_options}"
            )


ELEM_SIZE = 2  # bytes


def tile_softmax_flop(B_r, B_c, d_head) -> int:
    # Kernel 6-16
    return B_r * (4 * B_c + d_head + 4)
    # Kernel 1-5
    return B_r * (5 * B_c + d_head + 2)


def kv_tile_flop(B_r, B_c, d_head) -> int:
    QK_flops = 2 * B_r * d_head * B_c
    PV_flops = 2 * B_r * B_c * d_head

    softmax_flops = tile_softmax_flop(B_r, B_c, d_head)

    return QK_flops + PV_flops + softmax_flops


def gmem_transfer_size(B_r, B_c, d_head) -> int:
    return d_head * 2 * (B_r + B_c) * ELEM_SIZE


def arithmetic_intensity(B_r, B_c, kv_seq_len, d_head) -> float:
    return (
        kv_tile_flop(B_r, B_c, d_head) * (kv_seq_len // B_c)
    ) / gmem_transfer_size(B_r, kv_seq_len, d_head)


def calc_total_flop(n_samples, n_heads, seq_len, B_r, B_c, d_head):
    assert seq_len % B_r == 0
    assert seq_len % B_c == 0

    T_r = seq_len // B_r
    T_c = seq_len // B_c

    epilogue_flops = B_r * d_head
    head_sample_flops = T_r * (
        T_c * kv_tile_flop(B_r, B_c, d_head) + epilogue_flops
    )

    return head_sample_flops * n_samples * n_heads


def calc_self_attn_flop(n_samples, n_heads, seq_len, d_head):
    return n_samples * n_heads * (4 * seq_len**2 * d_head + 6 * seq_len**2)


@dataclass(frozen=True, order=True)
class FlashForwardKernelConfig:
    dtype: DType
    d_head: int
    B_r: int
    B_c: int
    n_warps: int
    async_copy: bool
    eager_load_blocks: bool
    swizzled: bool
    Q_mma_load_K_tiles: int
    K_mma_load_K_tiles: int
    V_mma_load_K_tiles: int
    mma_double_buffer_loads: bool
    optimized_softmax: bool

    def __str__(self):
        return self.short_form()

    def short_form(self, include_d_head=True, include_tup=True):
        d_head_str = f"{self.d_head}, " if include_d_head else ""
        base = f"({self.dtype.name}, {d_head_str}{self.B_r}, {self.B_c}, {self.n_warps}): "
        if not include_tup:
            base = ""
        strs = []
        if self.async_copy:
            strs.append("async")
        if self.eager_load_blocks:
            strs.append("eager")
        if self.swizzled:
            strs.append("swizzled")

        strs.append(
            f"load_{self.Q_mma_load_K_tiles}_{self.K_mma_load_K_tiles}_{self.V_mma_load_K_tiles}_tiles"
        )
        if self.mma_double_buffer_loads:
            strs.append("buffer")
        if self.optimized_softmax:
            strs.append("opt_softmax")

        return base + "+".join(strs)

    def to_cpp_struct(self) -> str:
        def vstr(v):
            if isinstance(v, bool):
                return str(v).lower()
            else:
                return str(v)

        return (
            f"FlashForwardKernelConfig{{"
            f"{self.dtype.to_cpp_str()}, {self.d_head}, {self.B_r}, {self.B_c}, {self.n_warps}, "
            f"{vstr(self.async_copy)}, {vstr(self.eager_load_blocks)}, "
            f"{vstr(self.swizzled)}, {self.Q_mma_load_K_tiles}, {self.K_mma_load_K_tiles}, "
            f"{self.V_mma_load_K_tiles}, {vstr(self.mma_double_buffer_loads)}, "
            f"{vstr(self.optimized_softmax)}"
            f"}}"
        )

    def kernel_name(self) -> str:
        return "flash_forward_kernel"

    def total_flop(self, n_samples: int, n_heads: int, seq_len: int) -> int:
        return calc_total_flop(
            n_samples, n_heads, seq_len, self.B_r, self.B_c, self.d_head
        )

    def attn_flop(self, n_samples: int, n_heads: int, seq_len: int) -> int:
        return calc_self_attn_flop(n_samples, n_heads, seq_len, self.d_head)


def _parse_flash_forward_demanged_name(line) -> FlashForwardKernelConfig:
    # Regular expression to extract the kernel configuration from the line
    # void flash_forward_kernel<FlashForwardKernelConfig{5, 128, 64, 64, 4, 1, 1, 1, 0, 2, 0, 1, 1}>(FAForwardArgs)
    kernel_config_pattern = r"FlashForwardKernelConfig\{([^}]+)\}"
    match = re.search(kernel_config_pattern, line)

    if not match:
        raise ValueError(
            "Invalid line format: FlashForwardKernelConfig not found"
        )

    def convert_v(v):
        if v == "true":
            return 1
        elif v == "false":
            return 0
        else:
            return int(v)

    # Extract the configuration values and split them into a list
    config_values = list(map(convert_v, match.group(1).split(", ")))

    # Create and return a FlashForwardKernelConfig object
    return FlashForwardKernelConfig(
        dtype=DType(config_values[0]),
        d_head=config_values[1],
        B_r=config_values[2],
        B_c=config_values[3],
        n_warps=config_values[4],
        async_copy=config_values[5],
        eager_load_blocks=config_values[6],
        swizzled=config_values[7],
        Q_mma_load_K_tiles=config_values[8],
        K_mma_load_K_tiles=config_values[9],
        V_mma_load_K_tiles=config_values[10],
        mma_double_buffer_loads=config_values[11],
        optimized_softmax=config_values[12],
    )


def _parse_flash_forward_demanged_name_with_types(
    line: str,
) -> FlashForwardKernelConfig:
    # Regular expression to match the configuration values inside FlashForwardKernelConfig
    match = re.search(r"FlashForwardKernelConfig\{(.*?)\}", line)
    if not match:
        raise ValueError(
            "Invalid line format: FlashForwardKernelConfig block not found"
        )

    # Extract the parameters as a comma-separated string
    params_str = match.group(1)

    # Parse values while handling int and bool
    params = []
    for value in params_str.split(","):
        value = value.strip()
        if value.startswith("(c10::ScalarType)"):
            params.append(DType.from_string(value[len("(c10::ScalarType)") :]))
        elif value.startswith("(int)"):
            params.append(int(value[5:]))
        elif value.startswith("(bool)"):
            params.append(bool(int(value[6:])))
        else:
            raise ValueError(f"Unexpected parameter format: {value}")

    # Ensure we have exactly 12 parameters
    if len(params) != 13:
        raise ValueError("Incorrect number of parameters parsed")

    # Return the dataclass instance
    return FlashForwardKernelConfig(*params)


def _parse_short_form_flash_forward_kernel_config(
    line: str,
) -> FlashForwardKernelConfig:
    """
    Extract the kernel configuration (the 'short_form' portion) from a line like:
    | (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax |  8311200.00  | ...

    and return a FlashForwardKernelConfig instance.
    """

    # Split by '|' and strip each piece. We expect the config to be in the 2nd token.
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if not parts:
        raise ValueError(f"Cannot parse line (empty after splitting): {line}")

    # The first non-empty part should be something like:
    # "(FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax"
    config_str = parts[0]

    # 1) Extract the numeric tuple in parentheses
    #    We expect something like "(FP16, 128, 64, 64, 4)" at the start
    match = re.match(r"\(([^)]*)\):\s*(.*)", config_str)
    if not match:
        raise ValueError(
            f"Cannot parse config string (no matching pattern): {config_str}"
        )

    tuple_str, features_str = match.groups()

    dtype_str, rest = tuple_str.split(",", 1)
    dtype = DType.from_string(dtype_str)
    d_head, B_r, B_c, n_warps = map(int, rest.split(","))

    # 2) Split the features by '+'
    features = features_str.split("+")

    # 3) Extract booleans
    async_copy = "async" in features
    eager_load_blocks = "eager" in features
    swizzled = "swizzled" in features
    mma_double_buffer_loads = "buffer" in features
    optimized_softmax = "opt_softmax" in features

    # 4) Extract Q/K/V load tile counts from the segment that starts with 'load_'
    #    e.g. "load_0_2_2_tiles"
    load_segment = next((f for f in features if f.startswith("load_")), None)
    if not load_segment:
        raise ValueError(
            f"Cannot find load segment in features: {features_str}"
        )

    # Remove "load_" prefix and "_tiles" suffix
    load_values_str = load_segment[len("load_") : -len("_tiles")]
    q_tiles, k_tiles, v_tiles = map(int, load_values_str.split("_"))

    return FlashForwardKernelConfig(
        dtype=dtype,
        d_head=d_head,
        B_r=B_r,
        B_c=B_c,
        n_warps=n_warps,
        async_copy=async_copy,
        eager_load_blocks=eager_load_blocks,
        swizzled=swizzled,
        Q_mma_load_K_tiles=q_tiles,
        K_mma_load_K_tiles=k_tiles,
        V_mma_load_K_tiles=v_tiles,
        mma_double_buffer_loads=mma_double_buffer_loads,
        optimized_softmax=optimized_softmax,
    )


def parse_kernel_name_into_config(kernel_name: str) -> FlashForwardKernelConfig:
    try:
        return _parse_flash_forward_demanged_name(kernel_name)
    except ValueError:
        pass
    try:
        return _parse_flash_forward_demanged_name_with_types(kernel_name)
    except ValueError:
        pass
    try:
        return _parse_short_form_flash_forward_kernel_config(kernel_name)
    except ValueError:
        raise ValueError(f"Invalid kernel name: {kernel_name}")


REF_KERNEL_NAME_MAP = {
    "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, 0, 0, half_t, Flash_kernel_traits<128, 128, 64, 4, half_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(Flash_fwd_params)": "Reference",
    "void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)32, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, (int)128, (int)32, (int)4, cutlass::half_t>>, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0>(flash::Flash_fwd_params)": "Reference",
    "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 32, 4, 0, 0, half_t, Flash_kernel_traits<128, 128, 32, 4, half_t>>, 0, 0, 0, 0, 1, 1, 0, 0>(Flash_fwd_params)": "Reference",
    "void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, (int)128, (int)64, (int)4, cutlass::half_t>>, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)0>(flash::Flash_fwd_params)": "Reference",
    "void device_kernel<enable_sm80_to_sm89<FlashAttnFwdSm80<CollectiveMainloopFwdSm80<4, 1, 0, cute::tuple<cute::C<128>, cute::C<64>, cute::C<128>>, 128, half_t, float, arch::Sm80, 0, 0, 0, 0, 0, 0, 1, 0>, CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<64>>, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>, half_t, arch::Sm80, 128, 0, 1, 0>, SingleTileScheduler<0, 0, 1, 128>>>>(T1::Params)": "V3",
    "void cutlass::device_kernel<flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<flash::CollectiveMainloopFwdSm80<(int)4, (int)1, (bool)0, cute::tuple<cute::C<(int)128>, cute::C<(int)64>, cute::C<(int)128>>, (int)128, cutlass::half_t, float, cutlass::arch::Sm80, (bool)0, (bool)0, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<(int)128>, cute::C<(int)128>, cute::C<(int)64>>, cute::tuple<cute::C<(int)1>, cute::C<(int)1>, cute::C<(int)1>>, cutlass::half_t, cutlass::arch::Sm80, (int)128, (bool)0, (bool)1, (bool)0>, flash::SingleTileScheduler<(bool)0, (bool)0, (bool)1, (int)128>>>>(T1::Params)": "V3",
    "void device_kernel<enable_sm80_to_sm89<FlashAttnFwdSm80<CollectiveMainloopFwdSm80<8, 1, 1, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128>>, 128, half_t, float, arch::Sm80, 0, 0, 0, 0, 0, 0, 1, 0>, CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<128>, cute::C<128>>, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>, half_t, arch::Sm80, 256, 0, 1, 0>, SingleTileScheduler<0, 0, 1, 128>>>>(T1::Params)": "V3",
}


def transform_kernel_name_to_short_form(kernel_name: str) -> str:
    if kernel_name in REF_KERNEL_NAME_MAP:
        return REF_KERNEL_NAME_MAP[kernel_name]
    short_form = parse_kernel_name_into_config(kernel_name).short_form()

    return short_form


def transform_kernel_name(kernel_name: str) -> str:
    try:
        return parse_kernel_name_into_config(kernel_name).short_form()
    except ValueError:
        return kernel_name


def should_autotune_config(cfg: FlashForwardKernelConfig) -> bool:
    if not cfg.async_copy and cfg.eager_load_blocks:
        return False
    if (
        cfg.Q_mma_load_K_tiles != cfg.K_mma_load_K_tiles
        and cfg.Q_mma_load_K_tiles != 0
    ):
        return False

    if cfg.B_r == 64:
        if cfg.n_warps == 8:
            return False
        elif (
            cfg.B_c == 32 and cfg.Q_mma_load_K_tiles == 0
        ):  # over threshold of # registers for 3 CTA
            return False
        elif cfg.B_c == 64 and cfg.Q_mma_load_K_tiles != 0:
            return False
    elif cfg.B_r == 128:
        if cfg.Q_mma_load_K_tiles == 0:
            return False

    return True


def get_autotuning_kernel_configs(dtypes=[DType.BF16, DType.FP16]):
    d_heads = [128]
    B_rs = [64, 128]
    B_cs = [32, 64]
    n_warps_cfgs = [4]
    async_copy = [True]
    eager_load_blocks = [True]
    swizzleds = [True]
    Q_mma_load_K_tiles = [0, 2]
    K_mma_load_K_tiles = [0, 2]
    V_mma_load_K_tiles = [0, 2]
    mma_double_buffer_loads = [False, True]
    optimized_softmax = [False, True]

    params = [
        dtypes,
        d_heads,
        B_rs,
        B_cs,
        n_warps_cfgs,
        async_copy,
        eager_load_blocks,
        swizzleds,
        Q_mma_load_K_tiles,
        K_mma_load_K_tiles,
        V_mma_load_K_tiles,
        mma_double_buffer_loads,
        optimized_softmax,
    ]

    return [
        FlashForwardKernelConfig(*cfg)
        for cfg in itertools.product(*params)
        if should_autotune_config(FlashForwardKernelConfig(*cfg))
    ]


def get_kernel_progression_configs(all_block_sizes=False):
    base_progression = [
        "(FP16, 128, 64, 64, 4): async+load_0_0_0_tiles",
        "(FP16, 128, 64, 64, 4): async+swizzled+load_0_0_0_tiles",
        "(FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles",
        "(FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles",
        "(FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer",
        "(FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax",
        "(FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax",
    ]

    base_progression = [
        _parse_short_form_flash_forward_kernel_config(cfg)
        for cfg in base_progression
    ]
    if not all_block_sizes:
        return base_progression

    kernels = []
    for kernel in base_progression:
        for B_r, B_c, n_warps in itertools.product(
            [128, 64], [128, 64, 32], [4, 8]
        ):
            if (B_r < 128 and n_warps == 8) or (B_r < B_c):
                continue
            updated = replace(kernel, B_r=B_r, B_c=B_c, n_warps=n_warps)
            kernels.append(updated)

    return kernels


def get_kernels_to_build():
    cfgs = set()
    # cfgs.update(get_kernel_progression_configs())
    cfgs.update(get_autotuning_kernel_configs())

    return sorted(cfgs)


def get_kernel_configs(kernels_key=""):
    # This is a hack to set kernels to test globally without using distributed flags
    if kernels_key == "":
        kernels_key = os.environ.get("KERNELS", "")

    if kernels_key.startswith("prog"):
        return get_kernel_progression_configs(
            all_block_sizes="all" in kernels_key
        )
    elif kernels_key == "all":
        return get_kernels_to_build()
    elif kernels_key == "tune":
        return get_autotuning_kernel_configs()
    elif "," in kernels_key:
        B_r, B_c = map(int, kernels_key.split(","))
        autotune_cfgs = get_autotuning_kernel_configs()
        return [
            cfg for cfg in autotune_cfgs if cfg.B_r == B_r and cfg.B_c == B_c
        ]
    else:
        raise ValueError(f"Invalid kernels env key: {kernels_key}")
