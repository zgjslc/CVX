import numpy as np


class TrajectoryScaling:
    # Sx, iSx, sx 分别为状态量的缩放矩阵、还原矩阵、偏移矩阵
    # Su, iSu, su 分别为控制量的缩放矩阵、还原矩阵、偏移矩阵

    def __init__(self, x_min=None, x_max=None, u_min=None, u_max=None, tf=None):
        self.Sx, self.iSx, self.sx = None, None, None
        self.Su, self.iSu, self.su = None, None, None
        self.S_sigma = tf

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

        x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
        u_max, u_min = np.max(u, axis=0), np.min(u, axis=0)
        self.Sx, self.iSx, self.sx, self.Su, self.iSu, self.su = self.compute_scaling(
            x_min, x_max, u_min, u_max
        )

    def scale(self, x, u, N):
        # 对状态量和控制量进行缩放
        x = x.reshape(N, -1, 1)
        u = u.reshape(N, -1, 1)
        scaled_x = self.iSx.T @ (x - self.sx).T
        scaled_u = self.iSu.T @ (u - self.su).T
        return scaled_x.T, scaled_u.T

    def unscale(self, scaled_x, scaled_u, N):
        # 对缩放后的状态量和控制量进行还原
        scaled_x = scaled_x.reshape(N, -1, 1)
        scaled_u = scaled_u.reshape(N, -1, 1)
        x = self.Sx @ scaled_x + self.sx
        u = self.Su @ scaled_u + self.su
        return x, u


# 测试用例
def test_trajectory_scaling():
    x_min, x_max = np.array([-1.0, -2.0]), np.array([1.0, 2.0])
    u_min, u_max = np.array([-0.5, -1.0]), np.array([0.5, 1.0])

    scaling = TrajectoryScaling(x_min, x_max, u_min, u_max)

    # 测试缩放和逆缩放
    x_test = np.array([[0.0, 1.0], [-1.0, 2.0]]).flatten()
    u_test = np.array([[0.0, 0.5], [-0.5, 1.0]]).flatten()

    scaled_x, scaled_u = scaling.scale(x_test, u_test, 2)
    print("Scaled x:", scaled_x)
    print("Scaled u:", scaled_u)

    # 验证逆缩放
    unscaled_x, unscaled_u = scaling.unscale(scaled_x, scaled_u, 2)
    print("Unscaled x:", unscaled_x)
    print("Unscaled u:", unscaled_u)


if __name__ == "__main__":
    # 执行测试

    a = np.random.randn(5, 5, 1)
    b = np.random.randn(5, 5)
    d = np.random.randn(5, 1)
    c = b @ a + d
    print(b @ a)
    test_trajectory_scaling()
