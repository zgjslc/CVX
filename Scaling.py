import numpy as np


class TrajectoryScaling:
    # Sx, iSx, sx 分别为状态量的缩放矩阵、还原矩阵、偏移矩阵
    # Su, iSu, su 分别为控制量的缩放矩阵、还原矩阵、偏移矩阵

    def __init__(self, x_min=None, x_max=None, u_min=None, u_max=None, tf=None):
        self.Sx, self.iSx, self.sx = None, None, None
        self.Su, self.iSu, self.su = None, None, None
        self.S_tf = tf

        if (
            x_min is not None
            and x_max is not None
            and u_min is not None
            and u_max is not None
        ):
            # 初始化缩放参数
            self.Sx, self.iSx, self.sx, self.Su, self.iSu, self.su = (
                self.compute_scaling(x_min, x_max, u_min, u_max)
            )

    def compute_scaling(self, x_min, x_max, u_min, u_max):
        # 计算状态和控制量的缩放矩阵及偏移量
        tol_zero = np.sqrt(np.finfo(float).eps)  # 数值稳定性阈值
        x_intrvl, u_intrvl = [0, 1], [0, 1]  # 目标缩放区间
        x_width, u_width = x_intrvl[1] - x_intrvl[0], u_intrvl[1] - u_intrvl[0]

        # 计算状态量的缩放矩阵 Sx
        Sx = np.maximum((x_max - x_min) / x_width, tol_zero)
        Sx = np.diag(Sx)  # 转换为对角矩阵
        iSx = np.linalg.inv(Sx)  # 计算 Sx 的逆矩阵
        sx = x_min - x_intrvl[0] * np.diag(Sx)  # 计算偏移量 sx

        # 计算控制量的缩放矩阵 Su
        Su = np.maximum((u_max - u_min) / u_width, tol_zero)
        Su = np.diag(Su)
        iSu = np.linalg.inv(Su)
        su = u_min - u_intrvl[0] * np.diag(Su)

        return Sx, iSx, sx.reshape(-1, 1), Su, iSu, su.reshape(-1, 1)

    def get_scaling(self):
        # 返回缩放矩阵、逆矩阵和偏移量
        return self.Sx, self.iSx, self.sx, self.Su, self.iSu, self.su

    def update_scaling_from_traj(self, x, u):
        # 从提供的状态和控制量轨迹中更新缩放范围
        if not (isinstance(x, np.ndarray) and isinstance(u, np.ndarray)):
            raise ValueError("x and u must be numpy arrays")
        if x.shape[1] != u.shape[1]:
            raise ValueError("x and u must have the same number of columns")

        x_max, x_min = np.max(x, axis=1), np.min(x, axis=1)
        u_max, u_min = np.max(u, axis=1), np.min(u, axis=1)
        self.Sx, self.iSx, self.sx, self.Su, self.iSu, self.su = self.compute_scaling(
            x_min, x_max, u_min, u_max
        )

    def scale(self, x, u, tf):
        # 对状态量和控制量进行缩放
        scaled_x = self.iSx @ (x - self.sx)
        scaled_u = self.iSu @ (u - self.su)
        scaled_tf = tf / self.S_tf
        return scaled_x, scaled_u, scaled_tf

    def unscale(self, scaled_x, scaled_u, scaled_tf):
        # 对缩放后的状态量和控制量进行还原
        x = self.Sx @ scaled_x + self.sx
        u = self.Su @ scaled_u + self.su
        tf = scaled_tf * self.S_tf
        return x, u, tf


if __name__ == "__main__":
    # 执行测试

    pass
