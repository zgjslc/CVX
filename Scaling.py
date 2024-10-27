import numpy as np


class TrajectoryScaling(object):
    # Sx iSx sx 分别为 状态量的缩放矩阵、还原矩阵、偏移矩阵
    # Su iSu su 分别为 控制量的缩放矩阵、还原矩阵、偏移矩阵
    def __init__(self, x_min=None, x_max=None, u_min=None, u_max=None, tf=None):
        # 验证输入参数
        if (
            x_min is not None
            and x_max is not None
            and u_min is not None
            and u_max is not None
        ):
            # 计算缩放参数
            self.Sx, self.iSx, self.sx, self.Su, self.iSu, self.su = (
                self.compute_scaling(x_min, x_max, u_min, u_max)
            )
        self.S_sigma = tf

    def get_scaling(self):
        # 返回缩放矩阵、逆矩阵和偏移量，用于后续缩放和逆缩放

        return self.Sx, self.iSx, self.sx, self.Su, self.iSu, self.su

    def update_scaling_from_traj(self, x, u):
        # 验证输入参数
        if not (isinstance(x, np.ndarray) and isinstance(u, np.ndarray)):
            raise ValueError("x and u must be numpy arrays")
        if x.shape[1] != u.shape[1]:
            raise ValueError("x and u must have the same number of columns")

        # 从提供的状态和控制量轨迹中更新缩放范围
        x_max = np.max(x, axis=0)  # 状态最大值
        x_min = np.min(x, axis=0)  # 状态最小值
        u_max = np.max(u, axis=0)  # 控制量最大值
        u_min = np.min(u, axis=0)  # 控制量最小值
        # 重新计算缩放参数
        self.Sx, self.iSx, self.sx, self.Su, self.iSu, self.su = self.compute_scaling(
            x_min, x_max, u_min, u_max
        )

    def compute_scaling(self, x_min, x_max, u_min, u_max):
        # 计算状态和控制量的缩放矩阵及偏移量
        tol_zero = np.sqrt(np.finfo(float).eps)  # 数值稳定性阈值
        x_intrvl = [0, 1]  # 状态量目标缩放区间
        u_intrvl = [0, 1]  # 控制量目标缩放区间
        x_width = x_intrvl[1] - x_intrvl[0]  # 状态量缩放区间宽度
        u_width = u_intrvl[1] - u_intrvl[0]  # 控制量缩放区间宽度

        # 计算状态量的缩放矩阵 Sx
        Sx = (x_max - x_min) / x_width
        Sx = np.maximum(Sx, tol_zero)  # 避免除零错误
        Sx = np.diag(Sx)  # 转换为对角矩阵
        try:
            iSx = np.linalg.inv(Sx)  # 计算 Sx 的逆矩阵
        except np.linalg.LinAlgError:
            raise ValueError(
                "The state scaling matrix is singular and cannot be inverted"
            )
        sx = x_min - x_intrvl[0] * np.diag(Sx)  # 计算偏移量 sx

        # 计算控制量的缩放矩阵 Su
        Su = (u_max - u_min) / u_width
        Su = np.maximum(Su, tol_zero)  # 避免除零错误
        Su = np.diag(Su)  # 转换为对角矩阵
        try:
            iSu = np.linalg.inv(Su)  # 计算 Su 的逆矩阵
        except np.linalg.LinAlgError:
            raise ValueError(
                "The control scaling matrix is singular and cannot be inverted"
            )
        su = u_min - u_intrvl[0] * np.diag(Su)  # 计算偏移量 su

        # 返回状态和控制量的缩放矩阵、逆矩阵及偏移量
        return Sx, iSx, sx, Su, iSu, su
