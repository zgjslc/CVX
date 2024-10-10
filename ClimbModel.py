import numpy as np
from numpy import zeros, linspace
from Core import Core
import sympy as sy
from sympy import cos, sin, tan, sqrt, exp
from ConvCore import JacM
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
        return JacM(funcs, args, sy.Matrix([P, alpha]), Func, allVars)

    def buildRefTraj(self):
        tK = linspace(0, 1, self.N)
        yK = linspace(self.y0, self.yf, self.N)
        VK = linspace(self.V0, self.Vf, self.N)
        thetaK = linspace(self.theta0, 0 / 57.3, self.N)
        mK = linspace(self.m0, self.mf, self.N)
        alphaK = linspace(2 / 57.3, -2 / 57.3, self.N)
        PK = linspace(self.PMin, self.PMax, self.N)
        refTraj = np.concatenate(
            (
                yK.reshape(1, -1),
                VK.reshape(1, -1),
                thetaK.reshape(1, -1),
                mK.reshape(1, -1),
            ),
            axis=0,
        )
        addParams = np.concatenate((PK.reshape(1, -1), alphaK.reshape(1, -1)))
        return tK, refTraj, addParams

    def buildBaseProblem(self):
        nP, nalpha = (
            np.arange(0, (self.N - 1) * self.controlDim, 1)
            .reshape(-1, self.controlDim)
            .T
        )
        X = cp.Variable(self.stateDim * self.N)
        U = cp.Variable(self.controlDim * (self.N - 1))
        H = cp.Variable(self.stateDim * (self.N - 1))
        tf = cp.Variable()
        AKk, BKk, FKk, CKk = (
            cp.Parameter(((self.N - 1) * self.stateDim, self.N * self.stateDim)),
            cp.Parameter(
                ((self.N - 1) * self.stateDim, (self.N - 1) * self.controlDim)
            ),
            cp.Parameter(((self.N - 1) * self.stateDim,)),
            cp.Parameter(((self.N - 1) * self.stateDim,)),
        )
        XRefP = cp.Parameter(self.N * self.stateDim)
        DeltaP = cp.Parameter(self.N * self.stateDim)

        x0 = np.array([self.y0, self.V0, self.theta0, self.m0])
        alpha = X[nalpha]
        P = X[nP]
        # 初值约束
        F = [X[: self.stateDim] == x0[: self.stateDim]]

        # 动力学方程约束
        F += [AKk @ X + BKk @ U + FKk * tf + CKk == 0]

        # 控制变量幅值约束
        F += [cp.abs(alpha) <= self.alphaMax]
        F += [P <= self.PMax]
        F += [P >= self.PMin]

        # 信赖域约束
        F += [cp.abs(X - XRefP) <= DeltaP]

        return F, X, U, H, AKk, BKk, FKk, CKk, XRefP, DeltaP, tf


if __name__ == "__main__":
    pass
