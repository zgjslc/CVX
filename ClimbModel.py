import numpy as np
from GuidanceOptimizer import GuidanceOptimizer
import sympy as sy
from sympy import cos, sin, exp
from JacobianLinearizer import JacobianLinearizer
import cvxpy as cp

g0 = 9.81
hs, rho0 = 7110.0, 1.225


class ClimbModel(GuidanceOptimizer):
    def __init__(self):
        super().__init__()

    def runge_kutta(self, state, P, alpha, dt):
        # RK4积分
        K1 = self.dynamic(state, P, alpha)
        K2 = self.dynamic(state + dt * K1 / 2, P, alpha)
        K3 = self.dynamic(state + dt * K2 / 2, P, alpha)
        K4 = self.dynamic(state + dt * K3, P, alpha)
        return state + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6

    def dynamic(self, state, P, alpha):
        # 动力学计算
        return self.Ja.GetF(*state, P, alpha).flatten()

    def form_dynamic(self):
        # 定义动力学方程及雅可比矩阵
        y, V, theta, m, P, alpha = sy.symbols("y, V, theta, m, P, alpha")
        aL, bL, cL = self.CL
        aD, bD, cD = self.CD

        s = self.s
        ma = V / 300
        aL, bL, cL = self.CL
        aD, bD, cD = self.CD
        CL = aL + bL * alpha + cL * alpha**2
        CD = aD + bD * alpha + cD * alpha**2

        rho = rho0 * exp(-y / hs)
        L = 0.5 * rho * V**2 * s * CL
        D = 0.5 * rho * V**2 * s * CD

        # 定义状态方程
        f1 = V * sin(theta)
        f2 = (P * cos(alpha) - D) / m - g0 * sin(theta)
        f3 = (P * sin(alpha) + L) / m / V - g0 / V * cos(theta)
        f4 = -P / self.Isp

        funcs = sy.Matrix([f1, f2, f3, f4])
        state_vars = sy.Matrix([y, V, theta, m])
        control_vars = sy.Matrix([P, alpha])
        all_vars = sy.Matrix([y, V, theta, m, P, alpha])
        self.state_dim = len(state_vars)
        self.control_dim = len(control_vars)
        self.Ja = JacobianLinearizer(funcs, state_vars, control_vars, all_vars)

    def build_base_problem(self):
        # 构建基础优化模型

        # 定义轨迹缩放参数
        Sx, iSx, sx, Su, iSu, su = self.traj_scaling.get_scaling()
        X = cp.Variable((self.N, self.state_dim))
        U = cp.Variable((self.N, self.control_dim))
        virtual_control = cp.Variable((self.N - 1, self.state_dim))
        tf = cp.Variable(nonneg=True)
        A = [cp.Parameter((self.state_dim, self.state_dim)) for _ in range(self.N - 1)]
        s = [cp.Parameter(self.state_dim) for _ in range(self.N - 1)]
        z = [cp.Parameter(self.state_dim) for _ in range(self.N - 1)]
        if self.mode == "zoh":
            B = [
                cp.Parameter((self.state_dim, self.control_dim))
                for _ in range(self.N - 1)
            ]
            Bm, Bp = None, None
        elif self.mode == "foh":
            Bm = [
                cp.Parameter((self.state_dim, self.control_dim))
                for _ in range(self.N - 1)
            ]
            Bp = [
                cp.Parameter((self.state_dim, self.control_dim))
                for _ in range(self.N - 1)
            ]
            B = None
        else:
            raise ValueError("type discretization should be zoh or foh")

        x_ref = cp.Parameter((self.N, self.state_dim))
        u_ref = cp.Parameter((self.N, self.control_dim))
        tf_ref = cp.Parameter(nonneg=True)
        X0 = cp.Parameter(4)

        X_unscaled = (Sx @ X.T + sx).T
        U_unscaled = (Su @ U.T + su).T
        tf_unscaled = tf * self.traj_scaling.S_tf
        virtual_control_unscaled = virtual_control
        P_unscaled = U_unscaled[:, 0]
        alpha_unscaled = U_unscaled[:, 1]

        # 初值约束
        constraints = [X_unscaled[0] == X0]
        # 状态量过程约束
        constraints += [X_unscaled[:, 0] <= 50000]

        # 控制变量幅值约束
        constraints += [cp.abs(alpha_unscaled) <= self.alpha_max]
        constraints += [P_unscaled <= self.p_max, P_unscaled >= self.p_min]

        # 动力学方程约束
        for i in range(self.N - 1):
            if self.mode == "zoh":
                constraints += [
                    X_unscaled[i + 1]
                    == A[i] @ X_unscaled[i]
                    + B[i] @ U_unscaled[i]
                    + s[i] * tf_unscaled
                    + z[i]
                    + virtual_control_unscaled[i]
                ]
            else:
                constraints += [
                    X_unscaled[i + 1]
                    == A[i] @ X_unscaled[i]
                    + Bm[i] @ U_unscaled[i]
                    + Bp[i] @ U_unscaled[i + 1]
                    + s[i] * tf_unscaled
                    + z[i]
                    + virtual_control_unscaled[i]
                ]

        # 时间尽可能短的目标函数
        cost_tf = tf_unscaled

        # 虚拟控制量尽可能稀疏
        cost_vc = cp.sum([cp.norm(virtual_control[i], 1) for i in range(self.N - 1)])

        # 与参考轨迹尽可能接近
        cost_tr = (
            cp.sum(
                [
                    cp.quad_form((X[i] - x_ref[i]), np.eye(self.state_dim))
                    + cp.quad_form((U[i] - u_ref[i]), np.eye(self.control_dim))
                    for i in range(self.N)
                ]
            )
            + (tf - tf_ref) ** 2
        )

        cost = self.w_tf * cost_tf + self.w_vc * cost_vc + self.w_tr * cost_tr

        # 构建返回的变量和参数
        params = {
            "X0": X0,
            "A": A,
            "B": B,
            "Bm": Bm,
            "Bp": Bp,
            "s": s,
            "z": z,
            "XRef": x_ref,
            "URef": u_ref,
            "tfRef": tf_ref,
        }
        variables = {"X": X, "U": U, "tf": tf, "vc": virtual_control}
        return (X_unscaled, variables, params, constraints, cost)


if __name__ == "__main__":
    pass
