# import taichi.math as tm

import time

import numpy as np
import taichi as ti
import torch
from einops import rearrange

# ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.cpu, offline_cache=True)
ti.init(arch=ti.cuda, offline_cache=True, debug=True)
ndarray_f32 = ti.types.ndarray(dtype=ti.f32)
ndarray_i32 = ti.types.ndarray(dtype=ti.i32)


@ti.kernel
def _ti_generate_matrix(
    indices: ndarray_i32,
    intervals: ndarray_f32,
    matrix: ndarray_f32,
):
    """修改输入的全零矩阵为对齐矩阵

    Args:
        indices (ndarray_i32): (B, N)，数值范围是[0, C]
        intervals (ndarray_f32): (B, N, 2)，数值范围是[0, T]
        matrix (ndarray_f32): (B, C, T)
    """
    for b, n in indices:
        index = indices[b, n]
        if index < 0:
            continue

        left = intervals[b, n, 0]
        left_int = int(left)
        right = intervals[b, n, 1]
        right_int = int(right)

        for k in range(left_int, right_int):
            matrix[b, index, k] = 1
        matrix[b, index, left_int] = left_int - left + 1
        matrix[b, index, right_int] = right - right_int


@ti.kernel
def _ti_normalize_alignment_with_blank(
    matrix: ndarray_f32,
    matrix_sum: ndarray_f32,
    blank: int,
):
    """

    Args:
        matrix (ndarray_f32): (B, C, T)
        matrix_sum (ndarray_f32): (B, T)
        blank: int
    """
    for b, t in matrix_sum:
        if matrix_sum[b, t] < 1:
            rest = 1 - matrix_sum[b, t]
            matrix[b, blank, t] = rest
        else:
            for c in range(matrix.shape[1]):
                matrix[b, c, t] /= matrix_sum[b, t] + 1e-6


@ti.kernel
def _ti_normalize_alignment_without_blank(
    matrix: ndarray_f32,
    matrix_sum: ndarray_f32,
):
    """

    Args:
        matrix (ndarray_f32): (B, C, T)
        matrix_sum (ndarray_f32): (B, T)
    """
    for b, c, t in matrix:
        matrix[b, c, t] /= matrix_sum[b, t] + 1e-6


def generate_matrix(indices, intervals, matrix_shape, normalize=False, blank=0):
    """
    生成一个批次的对齐矩阵。

    参数:
        indices (torch.Tensor): 形状为 (B, N)，其中 B 是批次大小，N 是序列长度。
                                 指定每个元素在输出矩阵中的位置索引。索引为负数时忽略此索引。
        intervals (torch.Tensor): 形状为 (B, N, 2)，定义了每个元素对应的区间范围。
                                  第三个维度的第一个元素是区间的开始，第二个元素是结束。
        matrix_shape (tuple): 输出矩阵的形状，形式为 (B, C, T)，其中 B 是批次大小，其中 C 是通道数，T 是时间步长。
        normalize (bool, optional): 是否对输出矩阵沿着通道轴（dim=1）进行归一化，使得各通道的和等于 1。
                                     默认为 False。
        blank (int, optional): 当 `normalize=True` 且输出矩阵沿着通道轴的和小于 1 时，
                               可以指定此参数来填充不足的部分。如果 `blank >= 0`，则输出矩阵的
                               `blank` 通道会被调整以满足总和为 1；若 `blank < 0`，则整个通道轴按比例缩放。
                               默认为 0。

    返回:
        torch.Tensor: 形状为 (B, C, T) 的对齐矩阵。
    """
    matrix = torch.zeros(matrix_shape, device=indices.device, dtype=torch.float32)
    indices = indices.type(torch.int32)
    intervals = intervals.type(torch.float32)
    _ti_generate_matrix(indices, intervals, matrix)

    if normalize:
        if blank >= 0:
            _ti_normalize_alignment_with_blank(matrix, matrix.sum(1), blank)
        else:
            _ti_normalize_alignment_without_blank(matrix, matrix.sum(1))
    return matrix


@ti.kernel
def _ti_decode_matrix(
    matrix_logprobs: ndarray_f32,
    t_lengths: ndarray_i32,
    l_lengths: ndarray_i32,
    dp: ndarray_f32,
    backtrack: ndarray_i32,
    need_confidence: bool,
    log_confidence: ndarray_f32,
    result: ndarray_i32,
):
    """

    Args:
        matrix_logprobs (ndarray_f32): (B, max_T, max_L) 注意T和L的顺序是反的，L放在末尾速度更快
        t_lengths (ndarray_i32): (B,)
        l_lengths (ndarray_i32): (B,)
        dp (ndarray_f32): (B, max_T, max_L) 全负无穷矩阵，用于储存动态规划的中间状态
        backtrack (ndarray_i32): (B, max_T, max_L) 全零矩阵，用于储存回溯路径
        need_confidence (bool): 是否需要计算帧级置信度
        log_confidence (ndarray_f32): (B, max(l_lengths)) 用于储存音素置信度
        result (ndarray_i32): (B, max(l_lengths),2) 全零矩阵，用于储存结果
    """

    for b in range(matrix_logprobs.shape[0]):
        T = t_lengths[b]
        L = l_lengths[b]

        # forward
        dp[b, 0, 0] = matrix_logprobs[b, 0, 0]
        for t in range(1, T):
            last_t = t - 1

            dp[b, t, 0] = dp[b, last_t, 0] + matrix_logprobs[b, t, 0]
            backtrack[b, t, 0] = backtrack[b, last_t, 0]

            for i in range(1, L):
                retention = dp[b, last_t, i]
                transition = dp[b, last_t, i - 1]

                if transition > retention:
                    dp[b, t, i] = transition + matrix_logprobs[b, t, i]
                    backtrack[b, t, i] = t
                else:
                    dp[b, t, i] = retention + matrix_logprobs[b, t, i]
                    backtrack[b, t, i] = backtrack[b, last_t, i]

        # backward
        t = T - 1
        i = L - 1
        while i > 0:
            start_pos = backtrack[b, t, i]
            result[b, i - 1, 1] = result[b, i, 0] = start_pos
            t = start_pos - 1
            i -= 1
        result[b, L - 1, 1] = T

        if need_confidence:
            for i_ in range(1, L):
                start = result[b, i_, 0] - 1
                end = result[b, i_, 1] - 1
                log_confidence[b, i_] = (dp[b, end, i_] - dp[b, start, i_ - 1]) / (
                    end - start
                )
            log_confidence[b, 0] = dp[b, result[b, 0, 1] - 1, 0] / (result[b, 0, 1] - 1)


def decode_matrix(matrix_logprobs, t_lengths, l_lengths, return_confidence=False):
    """解码对齐矩阵。

    Args:
        matrix_logprobs (torch.Tensor): 形状为 (B, max_L, max_T)，其中 B 是批次大小，max_T 是最大的时间步长，max_L 是最大的序列长度。
        t_lengths (torch.Tensor): 形状为 (B,)，包含每个序列的实际长度。
        l_lengths (torch.Tensor): 形状为 (B,)，包含每个序列的实际长度。
        return_confidence (bool, optional): 是否返回音素级置信度。默认为 False。

    Returns:
        result (torch.Tensor): 形状为 (B, max(l_lengths), 2) 的对齐结果。
        log_confidence (torch.Tensor): 形状为 (B, max(l_lengths)) 的音素级置信度。
    """
    device = matrix_logprobs.device
    B, L, T = matrix_logprobs.shape

    matrix_logprobs = rearrange(matrix_logprobs.detach(), "b c t -> b t c").contiguous()
    t_lengths = t_lengths.type(torch.int32)
    l_lengths = l_lengths.type(torch.int32)
    dp = torch.full_like(matrix_logprobs, -1e6, dtype=torch.float32, device=device)
    backtrack = torch.zeros_like(matrix_logprobs, dtype=torch.int32, device=device)
    log_confidence = torch.full(
        (B, L),
        fill_value=-1e6,
        dtype=torch.float32,
        device=device,
    )
    result = torch.zeros(
        (B, L, 2),
        dtype=torch.int32,
        device=device,
    )

    _ti_decode_matrix(
        matrix_logprobs,
        t_lengths,
        l_lengths,
        dp,
        backtrack,
        return_confidence,
        log_confidence,
        result,
    )

    if return_confidence:
        return result, log_confidence
    return result


@ti.func
def _ti_comb_log(n, k):
    res = 0.0
    for i in range(k):
        res += ti.log((n - i) / (i + 1))
    return res


@ti.kernel
def _ti_generate_prior(
    matrix: ndarray_f32,
    t_lengths: ndarray_i32,
    l_lengths: ndarray_i32,
):
    for b, i, t in matrix:
        if not t < t_lengths[b]:
            continue
        if not i < l_lengths[b]:
            continue

        t_div_T = (t + 0.5) / t_lengths[b]

        matrix[b, i, t] = (
            # 利用组合数的对称性减少计算量
            _ti_comb_log(
                l_lengths[b] - 1,
                (l_lengths[b] - 1 - i if i > l_lengths[b] - 1 - i else i),
            )
            + i * ti.log(t_div_T)
            + (l_lengths[b] - 1 - i) * ti.log(1 - t_div_T)
        )
        if matrix[b, i, t] < -14.0:
            matrix[b, i, t] = -14.0


def generate_prior(matrix_shape, t_lengths, l_lengths, device):
    t_lengths = t_lengths.type(torch.int32)
    l_lengths = l_lengths.type(torch.int32)
    matrix = torch.full(matrix_shape, -1e6, device=device)
    _ti_generate_prior(matrix, t_lengths, l_lengths)
    return matrix


if __name__ == "__main__":

    def test_generate_matrix():
        matrix_shape = (30, 100, 1000)
        L = 20
        indices = np.random.randint(
            1, matrix_shape[1], size=L * matrix_shape[0]
        ).reshape(matrix_shape[0], L)
        intervals = (np.random.rand(L * matrix_shape[0], 2) * matrix_shape[2]).reshape(
            matrix_shape[0], L, 2
        )
        intervals.sort(axis=2)

        (indices, intervals) = (
            torch.tensor(indices, device="cuda", dtype=torch.int32),
            torch.tensor(intervals, device="cuda", dtype=torch.float32),
        )

        matrix = generate_matrix(
            indices, intervals, matrix_shape, normalize=False, blank=0
        )
        # test 2nd run time
        start_time = time.time()
        matrix = generate_matrix(
            indices, intervals, matrix_shape, normalize=False, blank=0
        )
        time_teken = time.time() - start_time
        print(f"Time taken: {time_teken:.4f}s")
        print(intervals[0])

        import matplotlib.pyplot as plt

        plt.imshow(
            matrix[0].cpu(), vmin=0, vmax=1, cmap="gray", origin="lower", aspect="auto"
        )

        plt.show()

    def test_decode_matrix():
        L = 20
        matrix_shape = (30, L, 10000)
        rand_pos = (
            np.random.rand((L + 1) * matrix_shape[0]).reshape(matrix_shape[0], L + 1)
            * (matrix_shape[-1] - 1)
            + 1
        )
        rand_pos.sort(axis=1)
        intervals = np.stack([rand_pos[:, :-1], rand_pos[:, 1:]], axis=2)
        indices = np.arange(L)
        indices = np.tile(indices, matrix_shape[0]).reshape(matrix_shape[0], L)

        (indices, intervals) = (
            torch.tensor(indices, device="cuda", dtype=torch.int32),
            torch.tensor(intervals, device="cuda", dtype=torch.float32),
        )

        matrix = generate_matrix(
            indices, intervals, matrix_shape, normalize=True, blank=0
        )

        import matplotlib.pyplot as plt

        plt.imshow(
            matrix[0].cpu(), vmin=0, vmax=1, cmap="gray", origin="lower", aspect="auto"
        )

        start_time = time.time()
        result, log_confidence = decode_matrix(
            torch.log(matrix + 1e-6),
            torch.full((matrix_shape[0],), matrix_shape[-1], dtype=torch.int32),
            torch.full((matrix_shape[0],), L, dtype=torch.int32),
            True,
            # False,
        )
        # test 2nd run time
        start_time = time.time()
        result, log_confidence = decode_matrix(
            torch.log(matrix + 1e-6),
            torch.full((matrix_shape[0],), matrix_shape[-1], dtype=torch.int32),
            torch.full((matrix_shape[0],), L, dtype=torch.int32),
            True,
            # False,
        )
        time_taken = time.time() - start_time
        print(f"Time taken: {time_taken:.4f}s")
        print(result[0])

        frame_idx = torch.zeros(matrix.shape[-1], dtype=torch.int32)
        for i, res in enumerate(result[0]):
            frame_idx[res[0] : res[1]] = i
        plt.plot(frame_idx.cpu())

        frame_confidence = torch.zeros(matrix.shape[-1], dtype=torch.float32)
        for i, res in enumerate(result[0]):
            frame_confidence[res[0] : res[1]] = log_confidence[0, i]
        plt.plot((torch.exp(frame_confidence) * 19).cpu())

        print(torch.exp(log_confidence[0]))
        print(torch.exp(torch.mean(log_confidence[0])))

        plt.show()

    def test_generate_prior():
        matrix_shape = (30, 150, 10000)
        t_lengths = torch.full((matrix_shape[0],), matrix_shape[-1], dtype=torch.int32)
        l_lengths = torch.full((matrix_shape[0],), matrix_shape[1], dtype=torch.int32)

        # 2nd run time
        generate_prior(matrix_shape, t_lengths, l_lengths, device="cuda")

        start_time = time.time()
        matrix = generate_prior(matrix_shape, t_lengths, l_lengths, device="cuda")
        print(f"Time taken: {time.time() - start_time:.4f}s")
        print(matrix[0], matrix[0].min())
        import matplotlib.pyplot as plt

        plt.imshow(
            matrix[0].cpu(),
            origin="lower",
            aspect="auto",  # , interpolation="nearest"
            vmin=-14,
            vmax=0,
        )
        plt.colorbar()
        plt.show()

        # log_softmax对于prior来说是一个恒等变换，prior可以加在logits上或者logprob上
        matrix = torch.nn.functional.log_softmax(matrix, dim=1)
        plt.imshow(
            matrix[0].cpu(),
            origin="lower",
            aspect="auto",  # , interpolation="nearest"
            vmin=-14,
            vmax=0,
        )
        plt.colorbar()
        plt.show()

    test_decode_matrix()
