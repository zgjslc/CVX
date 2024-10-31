import numpy as np
import cvxpy as cp


class BSpline:
    def __init__(self, control_points, degree):
        """
        初始化 B 样条。

        参数:
        control_points : cvxpy.Variable
            控制点，格式为 (n, control_dim) 的 cvxpy.Variable 矩阵。
        degree : int
            B 样条的阶数。
        """
        self.control_points = control_points
        self.degree = degree
        self.knot_vector = self._generate_knot_vector(control_points.shape[0], degree)

    def _generate_knot_vector(self, n, p):
        """
        生成准均匀 B 样条的节点矢量。

        参数:
        n : int
            控制点数量。
        p : int
            B 样条的阶数。

        返回:
        knot_vector : numpy array
            生成的节点矢量。
        """
        k = n + p + 1  # 节点矢量的长度
        knot_vector = np.zeros(k + 1)
        knot_vector[: p + 1] = 0
        knot_vector[-(p + 1) :] = 1
        for i in range(1, k - 2 * p):
            knot_vector[i + p] = i / (k - 2 * p)
        return knot_vector

    def evaluate(self, u_values):
        """
        计算曲线在一组参数 u 处的坐标。

        参数:
        u_values : list or numpy array
            一组参数 u 值，每个 u 值应位于 [0, 1]。

        返回:
        points : list of cvxpy.Expression
            曲线在各个参数 u 处的坐标，以表达式形式返回。
        """
        u_values = np.array(u_values)
        points = []

        for u in u_values:
            n = self.control_points.shape[0] - 1
            p = self.degree
            point = cp.Constant(np.zeros(self.control_points.shape[1]))
            denominator = 0.0

            # 构建表达式
            for i in range(n + 1):
                basis = self._basis_function(i, p, u)
                point += basis * self.control_points[i]
                denominator += basis

            # 归一化处理
            points.append(point / denominator if denominator != 0 else point)

        return points

    def _basis_function(self, i, p, u):
        """
        递归计算 B 样条基函数。

        参数:
        i : int
            控制点的索引。
        p : int
            B 样条的阶数。
        u : float
            参数 u 值。

        返回:
        basis : float
            基函数 N_{i,p}(u) 的值。
        """
        if p == 0:
            return 1.0 if self.knot_vector[i] <= u < self.knot_vector[i + 1] else 0.0
        left_term = (
            (u - self.knot_vector[i])
            / (self.knot_vector[i + p] - self.knot_vector[i])
            * self._basis_function(i, p - 1, u)
            if self.knot_vector[i + p] != self.knot_vector[i]
            else 0
        )
        right_term = (
            (self.knot_vector[i + p + 1] - u)
            / (self.knot_vector[i + p + 1] - self.knot_vector[i + 1])
            * self._basis_function(i + 1, p - 1, u)
            if self.knot_vector[i + p + 1] != self.knot_vector[i + 1]
            else 0
        )
        return left_term + right_term
