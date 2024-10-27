import numpy as np
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor


class DiscreteSystem:
    def __init__(self, ix, iu, delT, tf, A_func, B_func, f_func):
        """
        初始化类，指定状态维度ix、输入维度iu、时间步长delT、总时间tf，
        以及系统矩阵A、B和状态方程f的外部传入表达式。
        """
        self.ix = ix
        self.iu = iu
        self.delT = delT
        self.tf = tf
        self.A_func = A_func
        self.B_func = B_func
        self.f_func = f_func

    def _initialize_conditions(self, x):
        # 生成初始条件
        A0 = np.eye(self.ix).flatten()
        B0 = np.zeros(self.ix * self.iu)
        s0, z0 = np.zeros(self.ix), np.zeros(self.ix)
        result = [
            np.hstack((item, A0, B0, B0, s0, z0)).T.flatten(order="F") for item in x
        ]
        # 将结果堆叠成二维数组，方便后续处理
        return np.hstack(result[:-1])

    def _reshape_solution(self, sol):
        # 将求解结果按照 ix 维度切片
        idx = {
            "state": slice(0, self.ix),
            "A": slice(self.ix, self.ix + self.ix**2),
            "Bm": slice(self.ix + self.ix**2, self.ix + self.ix**2 + self.ix * self.iu),
            "Bp": slice(
                self.ix + self.ix**2 + self.ix * self.iu,
                self.ix + self.ix**2 + 2 * self.ix * self.iu,
            ),
            "s": slice(
                self.ix + self.ix**2 + 2 * self.ix * self.iu,
                self.ix + self.ix**2 + 2 * self.ix * self.iu + self.ix,
            ),
            "z": slice(
                self.ix + self.ix**2 + 2 * self.ix * self.iu + self.ix,
                self.ix + self.ix**2 + 2 * self.ix * self.iu + 2 * self.ix,
            ),
        }

        sol_final = sol.y[:, -1].reshape(
            -1, self.ix + self.ix**2 + 2 * self.ix * self.iu + 2 * self.ix
        )
        x_prop = sol_final[:, idx["state"]].reshape(-1, self.ix)
        A = sol_final[:, idx["A"]].reshape(-1, self.ix, self.ix)
        Bm = A @ sol_final[:, idx["Bm"]].reshape(-1, self.ix, self.iu)
        Bp = A @ sol_final[:, idx["Bp"]].reshape(-1, self.ix, self.iu)
        s = (A @ sol_final[:, idx["s"]].reshape(-1, self.ix, 1)).squeeze()
        z = (A @ sol_final[:, idx["z"]].reshape(-1, self.ix, 1)).squeeze()

        return A, Bm, Bp, s, z, x_prop

    def _dvdt_foh(self, t, V, u_m, u_p, N):
        # FOH 的 dvdt 方程
        alpha, beta = (self.delT - t) / self.delT, t / self.delT
        u_interp = alpha * u_m + beta * u_p

        return self._dvdt_common(V, u_interp, N, alpha, beta)

    def _dvdt_zoh(self, t, V, u_k, N):
        # ZOH 的 dvdt 方程
        return self._dvdt_common(V, u_k, N, alpha=1, beta=0)

    def _dvdt_common(self, V, u, N, alpha, beta):
        # 通用 dvdt 方程，用于计算 FOH 和 ZOH
        V = V.reshape(N, self.ix + self.ix**2 + 2 * self.ix * self.iu + 2 * self.ix)
        x = V[:, : self.ix]
        Phi_flat = V[:, self.ix : self.ix + self.ix**2]
        Phi = Phi_flat.reshape(N, self.ix, self.ix)
        Phi_inv = np.linalg.inv(Phi)

        A, B, f = (
            self.A_func(*(x.T), *u.T).transpose(2, 0, 1),
            self.B_func(*x.T, *u.T).transpose(2, 0, 1),
            self.f_func(*x.T, *u.T).transpose(2, 0, 1),
        )

        dpdt = (A @ Phi).reshape(N, -1)
        dbmdt = (Phi_inv @ B).reshape(N, self.ix * self.iu) * alpha
        dbpdt = (Phi_inv @ B).reshape(N, self.ix * self.iu) * beta
        dsdt = (Phi_inv @ f).squeeze() / self.tf
        dzdt = (
            Phi_inv @ (-A @ x[:, :, np.newaxis] - B @ u[:, :, np.newaxis])
        ).squeeze()

        dv = np.hstack((f.squeeze(), dpdt, dbmdt, dbpdt, dsdt, dzdt))
        return dv.flatten()

    def _solve_interval(self, x, u, mode="foh"):
        V0 = self._initialize_conditions(x)
        if mode == "foh":
            sol = solve_ivp(
                self._dvdt_foh, (0, self.delT), V0, args=(u[:-1], u[1:], len(x) - 1)
            )
        elif mode == "zoh":
            sol = solve_ivp(
                self._dvdt_zoh, (0, self.delT), V0, args=(u[:-1], len(x) - 1)
            )
        else:
            raise ValueError("Unknown mode specified.")
        return self._reshape_solution(sol)

    def diff_discrete(self, x, u, mode="foh"):

        results = self._solve_interval(x, u, mode)

        # 将结果解包
        A_list, Bm_list, Bp_list, s_list, z_list, x_prop_list = results
        # 返回结果数组
        return (
            np.array(A_list),
            np.array(Bm_list),
            np.array(Bp_list),
            np.array(s_list),
            np.array(z_list),
            np.array(x_prop_list),
        )


if __name__ == "__main__":
    # 定义测试用的 A, B, f 符号表达式
    import sympy as sp

    x1, x2, u1 = sp.symbols("x1 x2 u1")
    x = sp.Matrix([x1, x2])
    u = sp.Matrix([u1])

    # 示例线性系统矩阵和状态方程
    A_sym = sp.Matrix([[x1, x2], [x1 + x2, x2 * x1]])
    B_sym = sp.Matrix([[u1], [2 * u1]])
    f_sym = A_sym * x + B_sym * u

    # 将符号表达式转为数值函数
    A_func = sp.lambdify((*x, *u), A_sym, "numpy")
    B_func = sp.lambdify((*x, *u), B_sym, "numpy")
    f_func = sp.lambdify((*x, *u), f_sym, "numpy")

    # 测试系统参数
    ix, iu = 2, 1  # 状态和输入的维度
    delT, tf = 0.1, 1  # 时间步和总时间

    # 初始化 DiscreteSystem 类
    system = DiscreteSystem(ix, iu, delT, tf, A_func, B_func, f_func)

    # 定义测试的初始状态 x 和控制输入序列 u
    x_initial = np.array(
        [
            [1, 0],
            [0.8, 0.1],
            [0.5, 0.2],
            [0.2, 0.3],
            [1, 0],
            [0.8, 0.1],
            [0.5, 0.2],
            [0.2, 0.3],
            [1, 0],
            [0.8, 0.1],
            [0.5, 0.2],
            [0.2, 0.3],
        ]
    )
    u_sequence = np.array(
        [[1], [0.5], [0.2], [0.1], [1], [0.5], [0.2], [0.1], [1], [0.5], [0.2], [0.1]]
    )  # 示例控制输入序列

    def test_diff_discrete_foh():
        # 使用 FOH 方法离散化
        print("Testing FOH Discretization:")
        A_foh, Bm_foh, Bp_foh, s_foh, z_foh, x_prop_foh = system.diff_discrete(
            x_initial, u_sequence, mode="foh"
        )

        # 输出 FOH 结果
        print("FOH A Matrix:", A_foh)
        print("FOH B Matrix:", Bm_foh)
        print("FOH B Matrix:", Bp_foh)
        print("FOH s Vector:", s_foh)
        print("FOH z Vector:", z_foh)
        print("FOH x Propagation:", x_prop_foh)

        # 检查 FOH 输出维度
        assert A_foh.shape == (
            len(x_initial) - 1,
            ix,
            ix,
        ), "FOH A Matrix dimensions incorrect."
        assert Bm_foh.shape == (
            len(x_initial) - 1,
            ix,
            iu,
        ), "FOH B Matrix dimensions incorrect."
        assert s_foh.shape == (
            len(x_initial) - 1,
            ix,
        ), "FOH s Vector dimensions incorrect."
        assert z_foh.shape == (
            len(x_initial) - 1,
            ix,
        ), "FOH z Vector dimensions incorrect."
        assert x_prop_foh.shape == (
            len(x_initial) - 1,
            ix,
        ), "FOH x Propagation dimensions incorrect."
        print("FOH test passed successfully.\n")

    def test_diff_discrete_zoh():
        # 使用 ZOH 方法离散化
        print("Testing ZOH Discretization:")
        import time

        start = time.time()
        A_zoh, Bm_zoh, Bp_zoh, s_zoh, z_zoh, x_prop_zoh = system.diff_discrete(
            x_initial, u_sequence, mode="zoh"
        )
        print(time.time() - start)
        # 输出 ZOH 结果
        print("ZOH A Matrix:", A_zoh)
        print("ZOH B Matrix:", Bm_zoh)
        print("ZOH s Vector:", s_zoh)
        print("ZOH z Vector:", z_zoh)
        print("ZOH x Propagation:", x_prop_zoh)

        # 检查 ZOH 输出维度
        assert A_zoh.shape == (
            len(x_initial) - 1,
            ix,
            ix,
        ), "ZOH A Matrix dimensions incorrect."
        assert Bm_zoh.shape == (
            len(x_initial) - 1,
            ix,
            iu,
        ), "ZOH B Matrix dimensions incorrect."
        assert s_zoh.shape == (
            len(x_initial) - 1,
            ix,
        ), "ZOH s Vector dimensions incorrect."
        assert z_zoh.shape == (
            len(x_initial) - 1,
            ix,
        ), "ZOH z Vector dimensions incorrect."
        assert x_prop_zoh.shape == (
            len(x_initial) - 1,
            ix,
        ), "ZOH x Propagation dimensions incorrect."
        print("ZOH test passed successfully.\n")

    # 运行测试用例
    test_diff_discrete_foh()
    test_diff_discrete_zoh()
