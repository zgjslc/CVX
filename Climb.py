from scipy.interpolate import interp1d
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from ClimbModel import ClimbModel

R0 = 6371000
g0 = 9.81
hs, rho0 = 7110.0, 1.225
KQ = 9.4369e-5


class Climb(ClimbModel):
    def __init__(self, Cl, CD):
        super().__init__()
        self.CL = Cl
        self.CD = CD

    # 构建特有约束与目标函数
    def buildProblem(self):
        ny, nV, ntheta, nm = (
            np.arange(0, self.N * self.stateDim, 1).reshape(-1, self.stateDim).T
        )

        F, X, U, H, AKk, BKk, FKk, CKk, XRefP, DeltaP, tf = self.buildBaseProblem()
<<<<<<< HEAD
        F = F + [X[ntheta[-1]] <= 0.0 / 57.3]
        F = F + [X[ntheta[-1]] >= 0.0 / 57.3]

        F = F + [X[ny[-1]] >= self.yf]
        obj = cp.Minimize((X[nV[-1]] - self.Vf) ** 2)
=======
        F = F + [X[ntheta[-1]] <= 10.0 / 57.3]
        F = F + [X[ntheta[-1]] >= -20.0 / 57.3]

        F = F + [X[nm[-1]] >= self.mf]

        F = F + [X[ny[-1]] >= self.yf]
        obj = cp.Minimize(X[ny[-1]])
>>>>>>> 12d81d4 (feat: 添加爬升模型及求解算法)
        problem = cp.Problem(obj, F)
        return problem, X, AKk, BKk, FKk, CKk, XRefP, DeltaP, tf

    def rungeKutta(self, state, alphaL=None, dt=0.1):

        t = 0
        result = []
        while t < self.T:
            alpha = alphaL(t)
            K1 = self.dynamic(state, alpha)
            K2 = self.dynamic(state + dt * K1 / 2, alpha)
            K3 = self.dynamic(state + dt * K2 / 2, alpha)
            K4 = self.dynamic(state + dt * K3, alpha)
            state = state + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6
            t += dt
            result.append([t, *state, float(alpha), 0, self.P])
        return result

    def dynamic(self, state, alpha):
        lon, y, lat, V, theta, psi, m = state
        cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9 = self.CL
        cd0, cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, cd9 = self.CD

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

        rho = rho0 * np.exp(-1 / hs * y)
        L = 0.5 * rho * V**2 * self.s * CL
        D = 0.5 * rho * V**2 * self.s * CD
        nx = (self.P * np.cos(alpha) - D) / m / g0
        ny = (self.P * np.sin(alpha) + L) / m / g0
        nz = 0
        dstates = np.zeros(7)
        dstates[0] = V * np.cos(theta) * np.sin(psi) / R0 / np.cos(lat)
        dstates[1] = V * np.sin(theta)
        dstates[2] = V * np.cos(theta) * np.cos(psi) / R0

        dstates[3] = nx * g0 - g0 * np.sin(theta)
        dstates[4] = ny * g0 / V - g0 * np.cos(theta) / V
        dstates[5] = (
            nz * g0 / V / np.cos(theta)
            + V * np.tan(lat) * np.cos(theta) * np.sin(psi) / R0
        )
        dstates[6] = -self.P / self.Isp
        return dstates


def main(
    X0,
    CL,
    CD,
    heightf=40000,
    Vf=3000,
    s=1.0,
    P=60000.0,
    Isp=4000.0,
    Mdry=1400,
    alphaMax=15 / 180 * np.pi,
):
    # print(X0, CL, CD, heightf, Vf, s, P, Isp, Mdry, alphaMax)
    X0 = [float("%.2f" % x) for x in X0]
    Xf = [0, heightf, 0, Vf, 0 / 57.3, 90 / 180.0 * np.pi, Mdry]

    Mis = Climb(CL, CD)
    Mis.setParams(
        alphaMax=alphaMax,
        s=s,
        P=P,
        Isp=Isp,
    )
<<<<<<< HEAD
    Mis.setSimuParams(N=200, iterMax=10)
=======
    Mis.setSimuParams(N=100, iterMax=10)
>>>>>>> 12d81d4 (feat: 添加爬升模型及求解算法)
    Mis.setIniState(*X0)
    Mis.setEndState(*Xf)
    result = Mis.solve()
    # alpha = result[-1, :]
    # tk = np.linspace(0, Mis.T, Mis.N)
    # alphaL = interp1d(tk, alpha)
    # # Traj = Mis.rungeKutta(X0, alphaL)
    return np.array(result)


if __name__ == "__main__":

    from Climb import main

    X0 = [0.0, 10000, 0.0, 600, -2.0 / 57.3, 0.0, 3000]

    CL = [
        1.00109221197850,
        -0.0411367290224851,
        4.02011176775144e-15,
        -1.91809495030897e-16,
        11.0381117491401,
        3.01237602001089,
        10.4572754224508,
        -0.347693889114020,
        3.41829136434955e-16,
        0.530980546888677,
    ]

    CD = [
        0.287987287293311,
        -0.0156744689692840,
        1.03874411609473e-14,
        -4.92572154181991e-16,
        1.88800131149693,
        14.5618281371450,
        11.8895002040662,
        -0.0414927163080754,
        -3.04321936908916e-17,
        -0.222188925678902,
    ]
    result = main(X0, CL, CD)
    plt.figure(1)
    plt.plot(result[:, 0], result[:, 2])

    plt.figure(2)
    plt.plot(result[:, 0], result[:, 4])
    plt.grid()

    plt.show()
