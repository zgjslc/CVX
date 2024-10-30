import numpy as np
from numpy import zeros, linspace
from Core import Core
import sympy as sy
from sympy import cos, sin, tan, sqrt, exp
from JacobianLinearizer import JacM
import cvxpy as cp

R0 = 6371000
g0 = 9.81
hs, rho0 = 7110.0, 1.225
KQ = 9.4369e-5


class ClimbModel(Core):
    def __init__(self):
        super().__init__()
        self.alphaL = None
        self.sigmaL = None

    def rungeKutta(self, state, P, alpha, dt):
        # RK4积分
        K1 = self.dynamic(state, P, alpha)
        K2 = self.dynamic(state + dt * K1 / 2, P, alpha)
        K3 = self.dynamic(state + dt * K2 / 2, P, alpha)
        K4 = self.dynamic(state + dt * K3, P, alpha)
        state = state + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6
        return state

    def dynamic(self, state, P, alpha):
        # 动力学计算
        return self.Ja.GetF(*state, P, alpha)

    def formDynamic(self):
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

        f1 = V * sin(theta)
        f2 = (P * cos(alpha) - D) / m - g0 * sin(theta)
        f3 = (P * sin(alpha) + L) / m / V - g0 / V * cos(theta)
        f4 = -P / self.Isp

        funcs = sy.Matrix([f1, f2, f3, f4])
        Func = sy.Matrix([f1, f2, f3, f4])
        args = sy.Matrix([y, V, theta, m])
        allVars = sy.Matrix([y, V, theta, m, P, alpha])
        self.stateDim = 4
        self.controlDim = 2
        self.Ja = JacM(funcs, args, sy.Matrix([P, alpha]), Func, allVars)

    def buildBaseProblem(self):
        Sx, iSx, sx, Su, iSu, su = self.traj_scaling.get_scaling()
        X = cp.Variable((self.N, self.stateDim, 1))
        U = cp.Variable((self.N, self.controlDim, 1))
        H = cp.Variable((self.N - 1, self.stateDim, 1))
        tf = cp.Variable(nonneg=True)
        A, s, z = (
            cp.Parameter((self.N - 1, self.stateDim, self.stateDim)),
            cp.Parameter((self.N - 1, self.stateDim)),
            cp.Parameter((self.N - 1, self.stateDim)),
        )
        if self.mode == "zoh":
            B = cp.Parameter((self.N - 1, self.stateDim, self.controlDim))
        elif self.mode == "foh":
            Bm = cp.Parameter((self.N - 1, self.stateDim, self.controlDim))
            Bp = cp.Parameter((self.N - 1, self.stateDim, self.controlDim))
        else:
            raise ValueError("type discretization should be zoh or foh")

        XRef = cp.Parameter((self.N, self.stateDim, 1))
        URef = cp.Parameter((self.N, self.controlDim, 1))
        tfRef = cp.Parameter(nonneg=True)

        x0 = np.array([self.y0, self.V0, self.theta0, self.m0]).reshape(-1, 1)

        X_unscaled = Sx @ X + sx
        U_unscaled = Su @ U + su

        alpha = U_unscaled[:, 0]
        P = U_unscaled[:, 1]

        # 初值约束
        F = [X_unscaled[0] == x0]

        # 控制变量幅值约束
        F += [cp.abs(alpha) <= self.alphaMax]
        F += [P <= self.PMax, P >= self.PMin]

        # 动力学方程约束
        if self.mode == "zoh":
            F += [
                X[1:]
                == A @ X[:-1] + B @ U[:-1] + s * tf * self.traj_scaling.S_sigma + z + H
            ]
        else:
            F += [
                X[1:]
                == A @ X[:-1]
                + Bm @ U[:-1]
                + Bp @ U[1:]
                + s * tf * self.traj_scaling.S_sigma
                + z
                + H
            ]

        # 时间尽可能短的目标函数
        cost_tf = [tf * self.traj_scaling.S_sigma]

        # 虚拟控制量尽可能稀疏
        cost_vc = [cp.norm(vc, 1) for vc in H]

        # 与参考轨迹尽可能接近
        cost_tr = [
            cp.quad_form((X[i] - XRef[i]), np.eye(self.stateDim)) for i in range(self.N)
        ]
        cost_tr.append(
            [
                cp.quad_form((U[i] - URef[i]), np.eye(self.stateDim))
                for i in range(self.N)
            ]
        )
        cost_tr.append(cp.quad_form(tf - tfRef, np.eye(1)))

        cost = cp.sum(cost_tf) + cp.sum(cost_vc) + cp.sum(cost_tr)
        params = {"A": A, "s": s, "z": z, "XRef": XRef, "URef": URef, "tfRef": tfRef}
        if self.mode == "zoh":
            params["B"] = B
        else:
            params["Bm"] = Bm
            params["Bp"] = Bp
        variables = {"X": X, "U": U, "tf": tf, "vc": H}
        return (F, X, cost, variables, params)


if __name__ == "__main__":
    pass
