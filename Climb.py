import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from ClimbModel import ClimbModel


class Climb(ClimbModel):
    def __init__(self, Cl, CD):
        super().__init__()
        self.CL = Cl
        self.CD = CD

    # 构建特有约束与目标函数
    def buildProblem(self):
        F, X, cost, variables, params = self.buildBaseProblem()
        F += [X[-1, 2, 0] <= 2.0 / 57.3]
        F += [X[-1, 2, 0] >= -2.0 / 57.3]

        F += F + [X[-1, 3, 0] >= self.mf]

        F += F + [X[-1, 0, 0] >= self.yf]
        F += F + [X[-1, 1, 0] >= self.Vf]
        obj = cp.Minimize(cost)
        problem = cp.Problem(obj, F)
        return problem, variables, params

    def buildRefTrajectory(self):
        P = (self.PMax + self.PMin) / 2
        alpha = 5.0 / 180 * np.pi
        state = np.array([self.y0, self.V0, self.theta0, self.m0])
        dt = 0.1
        XRef, URef, tfRef = [], [], 0
        XRef.append(state)
        URef.append([P, alpha])
        while True:
            state = self.rungeKutta(state, P, alpha, dt)
            if state[-1] <= self.mDry:
                break
            XRef.append(state)
            URef.append([P, alpha])
            tfRef += dt
        return np.array(XRef), np.array(URef), tfRef


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
    Mis.setSimuParams(N=100, iterMax=10)
    Mis.setIniState(*X0)
    Mis.setEndState(*Xf)
    result = Mis.solve()
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
