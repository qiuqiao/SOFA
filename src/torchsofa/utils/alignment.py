# import taichi.math as tm

import time

import numpy as np
import taichi as ti
import torch

try:
    ti.init(arch=ti.cuda)
except Exception:
    ti.init(arch=ti.cpu)
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
    _ti_generate_matrix(indices, intervals, matrix)

    if normalize:
        if blank >= 0:
            _ti_normalize_alignment_with_blank(matrix, matrix.sum(1), blank)
        else:
            _ti_normalize_alignment_without_blank(matrix, matrix.sum(1))
    return matrix


if __name__ == "__main__":

    def test_generate_alignment_matrix():
        matrix_shape = (30, 100, 10000)
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

        start_time = time.time()
        matrix = generate_matrix(
            indices, intervals, matrix_shape, normalize=False, blank=0
        )
        time_teken = time.time() - start_time
        print(f"Time taken: {time_teken:.4f}s")

        import matplotlib.pyplot as plt

        print(matrix)
        plt.imshow(
            matrix[0].cpu(), vmin=0, vmax=1, cmap="gray", origin="lower", aspect="auto"
        )
        plt.show()
