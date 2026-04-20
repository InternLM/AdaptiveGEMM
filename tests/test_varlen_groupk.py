import random
from pathlib import Path
import sys
import torch
from typing import Tuple
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import adaptive_gemm
from adaptive_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
from adaptive_gemm.jit_kernels.k_grouped_gemm_dw import get_best_configs, get_bfloat16_ref, quant_input


REPLAY_INPUT_PATH = REPO_ROOT / "dw_kernel_replay_inputs_rank0_0.pt"


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("This kernel requires sm90+")


def _run_k_grouped_gemm_dw(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    k_indices: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty((k_indices.numel(), lhs.shape[0], rhs.shape[0]), device="cuda", dtype=torch.bfloat16)
    adaptive_gemm.k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(
        lhs, lhs_scales, rhs, rhs_scales, out, k_indices
    )
    torch.cuda.synchronize()
    return out


def test_k_grouped_gemm_dw_replay_input_is_deterministic() -> None:
    _require_sm90()
    if not REPLAY_INPUT_PATH.exists():
        pytest.skip(f"Missing replay input: {REPLAY_INPUT_PATH}")

    replay = torch.load(REPLAY_INPUT_PATH, map_location="cuda")
    lhs = replay["grad_out_trans_fp8"].contiguous()
    lhs_scales = replay["grad_out_trans_scale"].contiguous()
    rhs = replay["x_trans_quant_fp8"].contiguous()
    rhs_scales = replay["x_trans_quant_scale"].contiguous()
    k_indices = replay["tokens_per_expert_expand"].contiguous()

    # Warm up once so both measured calls reuse the same compiled kernel.
    _run_k_grouped_gemm_dw(lhs, lhs_scales, rhs, rhs_scales, k_indices)

    out_0 = _run_k_grouped_gemm_dw(lhs, lhs_scales, rhs, rhs_scales, k_indices)
    out_1 = _run_k_grouped_gemm_dw(lhs, lhs_scales, rhs, rhs_scales, k_indices)
    diff = (out_0.float() - out_1.float()).abs()
    assert torch.equal(out_0, out_1), (
        f"Replay input is not deterministic: max_diff={diff.max().item()}, "
        f"num_diff={diff.ne(0).sum().item()}"
    )


def test_k_grouped_gemm_dw_zero_groups_match_reference() -> None:
    _require_sm90()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    m, n = 256, 256
    k_indices = torch.tensor([0, 128, 256, 0, 128, 0, 512], device="cuda", dtype=torch.int32)
    total_k = int(k_indices.sum().item())
    lhs = torch.randn((m, total_k), device="cuda", dtype=torch.bfloat16)
    rhs = torch.randn((n, total_k), device="cuda", dtype=torch.bfloat16)

    (lhs_quant, lhs_scales), (rhs_quant, rhs_scales), k_indices = quant_input(lhs, rhs, k_indices)
    ref = get_bfloat16_ref((lhs_quant, lhs_scales), (rhs_quant, rhs_scales), k_indices)

    out_0 = _run_k_grouped_gemm_dw(lhs_quant, lhs_scales, rhs_quant, rhs_scales, k_indices)
    out_1 = _run_k_grouped_gemm_dw(lhs_quant, lhs_scales, rhs_quant, rhs_scales, k_indices)

    assert torch.equal(out_0, out_1), "Zero-group input should be deterministic"
    assert torch.allclose(out_0, ref, atol=1, rtol=1e-1)
    assert bool((out_0[k_indices == 0] == 0).all().item())


def test_k_grouped_gemm_dw_tuner_avoids_nonuniform_rhs_scale_path() -> None:
    _require_sm90()

    m, n = 256, 256
    num_sms = adaptive_gemm.get_num_sms()
    for num_groups in (4, 7, 9, 11, 13):
        _, block_n, _, _, _ = get_best_configs(m, n, 128 * num_groups, num_groups, num_sms, True)
        assert 128 % block_n == 0, f"Autotuner selected unsupported BLOCK_N={block_n}"


def generate_random_list(length, total_sum):
    # 生成一个长度为length的列表，元素之和为total_sum
    # 先生成一个平均分配的列表
    avg = total_sum // length
    remainder = total_sum % length
    lst = [avg] * length

    # 随机调整数值，确保总和不变
    for i in range(length):
        # 随机选择两个不同的位置
        lst[i] = random.randint(0, int(avg))
    return lst

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

def per_channel_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, n)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    if dtype == torch.float8_e4m3fn:
        fmax = torch.finfo(torch.float8_e4m3fn).max
        return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / fmax).view(m, -1)
    else:
        fmax = torch.finfo(torch.float8_e5m2).max
        return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e5m2).view(m, n), (x_amax / fmax).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def per_expert_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    num_groups, m, n = x.shape
    x_padded = torch.zeros((num_groups, ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(num_groups, m, 1, n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    if dtype == torch.float8_e4m3fn:
        fmax = torch.finfo(torch.float8_e4m3fn).max
        x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))
    else:
        fmax = torch.finfo(torch.float8_e5m2).max
        x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e5m2)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))


def gen_data(M, N, tokens_per_expert, dtype_a = 1, dtype_b = 1):
    ref_dw = torch.empty(len(tokens_per_expert), M, N, device = "cuda", dtype = dtype_out)
    
    x_fp8_list, x_scale_list, y_fp8_list, y_scale_list = [], [], [], []
    tokens_per_expert_pad = []
    # prepare data
    for i, tokens in enumerate(tokens_per_expert):
        tokens = int(tokens)
        tokens_padding = int((tokens + 127) / 128) * 128
        tokens_per_expert_pad.append(tokens_padding)
        x = torch.randn(M, tokens, device='cuda', dtype=torch.bfloat16)
        y = torch.randn(N, tokens, device='cuda', dtype=torch.bfloat16)

        ref_dw[i,:,:] = (x @ y.T)

        x_padding = torch.zeros(M, tokens_padding, device='cuda', dtype=torch.bfloat16)
        y_padding = torch.zeros(N, tokens_padding, device='cuda', dtype=torch.bfloat16)
        x_padding[:,:tokens] = x
        y_padding[:,:tokens] = y

        x_fp8, x_scale = per_token_cast_to_fp8(x_padding)
        y_fp8, y_scale = per_block_cast_to_fp8(y_padding)

        x_fp8_list.append(x_fp8); x_scale_list.append(x_scale)
        y_fp8_list.append(y_fp8); y_scale_list.append(y_scale)

    # input 都按照 token 连续，scale 都 按照 token stride
    x_fp8 = torch.cat(x_fp8_list, -1).to(dtype=x_fp8.dtype).contiguous()
    x_scale = torch.cat(x_scale_list, -1).contiguous().transpose(0, 1).contiguous().transpose(0, 1)
    y_fp8 = torch.cat(y_fp8_list, -1).to(dtype=y_fp8.dtype).contiguous()
    y_scale = torch.cat(y_scale_list, -1).contiguous()

    return x_fp8, x_scale, y_fp8, y_scale, ref_dw, tokens_per_expert_pad


if __name__=='__main__':
    from typing import Tuple
    import random
    
    print('Testing grouped contiguous GEMM:')

    dtype_a = torch.float8_e5m2
    dtype_b = torch.float8_e4m3fn

    dtype_out = torch.bfloat16

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # for (M, N) in ((6144, 5120), (5120, 3072), (7168, 4096), (7168, 2048), (7168, 16384)):
    for (M, N) in ((5120, 6144), ):
        # tokens_per_expert = generate_random_list(8, 16384)
        tokens_per_expert = [3840] * 8
        # tokens_per_expert = [3840, 1024] * 8
        tokens_per_expert = torch.tensor(tokens_per_expert, device='cuda', dtype=torch.long)
        num_groups = len(tokens_per_expert)
        # print(tokens_per_expert)

        x_fp8, x_scale, weights_fp8, weights_scale, ref_fwd, tokens_per_expert = gen_data(M, N, tokens_per_expert, dtype_a = dtype_a, dtype_b = dtype_b)
        size_per_group = torch.tensor(tokens_per_expert, device='cuda', dtype=torch.int32)
        output_tensor = torch.empty((num_groups, M, N), device = "cuda", dtype = torch.bfloat16)
        # import pdb; pdb.set_trace()
        
        for i in range(3):
            adaptive_gemm.k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(x_fp8, x_scale, weights_fp8, weights_scale, output_tensor, size_per_group)
        
        def test_func():
            adaptive_gemm.k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(x_fp8, x_scale, weights_fp8, weights_scale, output_tensor, size_per_group)
        
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        K = sum(tokens_per_expert)
        print(f' > Performance ({num_groups=}, {M=}, {N=}, {K=}, {t * 1e6:4.0f} us | '
                f'throughput: {2 * M * N * K / t / 1e12:4.0f} TFLOPS, '
                f'{(num_groups * K * N + M * K +  M * N * 2) / 1e9 / t:4.0f} GB/s')
        amax = max(output_tensor.abs().max(), ref_fwd.abs().max())
        adiffmax = (output_tensor - ref_fwd).abs().max()
        rdiffmax = adiffmax / amax
        print(f"    max relative difference of the layer is {rdiffmax}")


    # from torch.profiler import ProfilerActivity, profile, record_function
    # with profile(
    #         activities=[
    #                 ProfilerActivity.CPU, ProfilerActivity.CUDA
    #         ],
    #         with_stack = True,
    #         with_modules = True,
    #         record_shapes=True,) as prof:
    #     output_tensor = adaptive_gemm.m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous((x_fp8, x_scale), (weights_fp8, weights_scale), size_per_group)
    
    # trace = f'./grouped_m_gemm.json'
    # prof.export_chrome_trace(trace)
    # # Get time from trace
    # import json
    # with open(trace, "r") as file:
    #     data = json.load(file)
    
    # kernel_time = 0
    # for event in data["traceEvents"]:
    #     if "fp8_gemm_kernel" in event["name"]:
    #         kernel_time += event["dur"] / 1000
    # print(f"\nPure kernel Elapsed time {round((kernel_time), 1)} ms, {round((2 * M.item() * N * K)/(kernel_time)/10**9, 0)} tflops")