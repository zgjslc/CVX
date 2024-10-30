import numpy as np
from GuidanceOptimizer import GuidanceOptimizer
import sympy as sy
from sympy import cos, sin, exp
from JacobianLinearizer import JacM
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
        return self.Ja.GetF(*state, P, alpha)

    def form_dynamic(self):
        # 定义动力学方程及雅可比矩阵
        y, V, theta, m, P, alpha = sy.symbols("y, V, theta, m, P, alpha")
        cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9 = self.CL
        cd0, cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, cd9 = self.CD

        s = self.s
        ma = V / 300

        CD = (
            cd0
            + cd1 * ma
            + cd4 * alpha
            + cd5 * alpha**2
            + cd6 * alpha**3
            + cd7 * ma * alpha
            + cd9 * ma * alpha**2
        )  # 阻力系数
        CL = (
            cl0
            + cl1 * ma
            + cl4 * alpha
            + cl5 * alpha**2
            + cl6 * alpha**3
            + cl7 * ma * alpha
            + cl9 * ma * alpha**2
        )  # 升力系数

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
        self.Ja = JacM(funcs, state_vars, control_vars, all_vars)

    def build_base_problem(self):
        # 构建基础优化模型

        # 定义轨迹缩放参数
        Sx, iSx, sx, Su, iSu, su = self.traj_scaling.get_scaling()
        X = cp.Variable((self.N, self.stateDim, 1))
        U = cp.Variable((self.N, self.controlDim, 1))
        virtual_control = cp.Variable((self.N - 1, self.stateDim, 1))
        tf = cp.Variable(nonneg=True)

        A, s, z = (
            cp.Parameter((self.N - 1, self.stateDim, self.stateDim)),
            cp.Parameter((self.N - 1, self.stateDim, 1)),
            cp.Parameter((self.N - 1, self.stateDim, 1)),
        )

        if self.mode == "zoh":
            B = cp.Parameter((self.N - 1, self.stateDim, self.controlDim))
        elif self.mode == "foh":
            Bm = cp.Parameter((self.N - 1, self.stateDim, self.controlDim))
            Bp = cp.Parameter((self.N - 1, self.stateDim, self.controlDim))
        else:
            raise ValueError("type discretization should be zoh or foh")

        x_ref = cp.Parameter((self.N, self.stateDim, 1))
        u_ref = cp.Parameter((self.N, self.controlDim, 1))
        tf_ref = cp.Parameter(nonneg=True)

        x0 = np.array([self.y0, self.V0, self.theta0, self.m0]).reshape(-1, 1)

        X_unscaled = Sx @ X + sx
        U_unscaled = Su @ U + su
        tf_unscaled = tf * self.traj_scaling.S_tf
        virtual_control_unscaled = Sx @ virtual_control + sx
        P_unscaled = U_unscaled[:, 0]
        alpha_unscaled = U_unscaled[:, 1]

        # 初值约束
        constraints = [X_unscaled[0] == x0]

        # 控制变量幅值约束
        constraints += [cp.abs(alpha_unscaled) <= self.alphaMax]
        constraints += [P_unscaled <= self.PMax, P_unscaled >= self.PMin]

        # 动力学方程约束
        if self.mode == "zoh":
            constraints += [
                X[1:]
                == A @ X_unscaled[:-1]
                + B @ U_unscaled[:-1]
                + s * tf_unscaled
                + z
                + virtual_control_unscaled
            ]
        else:
            constraints += [
                X[1:]
                == A @ X_unscaled[:-1]
                + Bm @ U_unscaled[:-1]
                + Bp @ U_unscaled[1:]
                + s * tf_unscaled
                + z
                + virtual_control_unscaled
            ]

        # 时间尽可能短的目标函数
        cost_tf = tf_unscaled

        # 虚拟控制量尽可能稀疏
        cost_vc = cp.sum([cp.norm(vc, 1) for vc in virtual_control])

        # 与参考轨迹尽可能接近
        cost_tr = cp.sum(
            [
                cp.quad_form((X[i] - x_ref[i]), np.eye(self.state_dim))
                + cp.quad_form((U[i] - u_ref[i]), np.eye(self.control_dim))
                for i in range(self.N)
            ]
        ) + cp.quad_form(tf - tf_ref, np.eye(1))

        cost = self.w_tf * cost_tf + self.w_vc * cost_vc + self.w_tr * cost_tr

        # 构建返回的变量和参数
        params = {"A": A, "s": s, "z": z, "XRef": x_ref, "URef": u_ref, "tfRef": tf_ref}
        if self.mode == "zoh":
            params["B"] = B
        else:
            params["Bm"] = Bm
            params["Bp"] = Bp
        variables = {"X": X, "U": U, "tf": tf, "vc": virtual_control}
        return (variables, params, constraints, cost)


if __name__ == "__main__":
    pass
