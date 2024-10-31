import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from ClimbModel import ClimbModel


class Climb(ClimbModel):
    def __init__(self, CL, CD):
        super().__init__()
        self.CL = CL
        self.CD = CD

    # 构建特有约束与目标函数
    def build_problem(self):
        X, variables, params, constraints, cost = self.build_base_problem()
        U = variables["U"]
        # 终点约束条件
        constraints += [X[-1, 0] == self.yf]
        constraints += [X[-1, 1] >= self.Vf]
        constraints += [cp.abs(X[-1, 2]) <= 2.0 / 180 * np.pi]
        constraints += [X[-1, 3] >= self.m_dry]

        obj = cp.Minimize(
            cost + 0.01 * cp.sum(U[:, 0] ** 2) + 0.001 * cp.sum(U[:, 1] ** 2)
        )
        problem = cp.Problem(obj, constraints)
        return problem, variables, params

    def build_reference_trajectory(self, mode="linear"):
        if mode == "linear":
            y_ref = np.linspace(self.y0, self.yf, self.N).reshape(-1, 1)
            v_ref = np.linspace(self.V0, self.Vf, self.N).reshape(-1, 1)
            theta_ref = np.linspace(-2 / 57.3, 2 / 57.3, self.N).reshape(-1, 1)
            m_ref = np.linspace(self.m0, self.m_dry, self.N).reshape(-1, 1)
            x_ref = np.hstack((y_ref, v_ref, theta_ref, m_ref))
            P_ref = np.linspace(self.p_max, self.p_min, self.N).reshape(-1, 1)
            alpha_ref = np.full_like(P_ref, 2 / 57.3)
            u_ref = np.hstack((P_ref, alpha_ref))
            tf_ref = (self.m0 - self.m_dry) / ((self.p_max + self.p_min) / 2 / self.Isp)
        else:
            P = (self.p_max + self.p_min) / 2
            alpha = 2.0 / 180 * np.pi
            state = np.array([self.y0, self.V0, self.theta0, self.m0])
            dt = 0.1
            x_ref, u_ref, tf_ref = [], [], 0
            x_ref.append(state)
            u_ref.append([P, alpha])
            while state[-1] > self.m_dry:
                state = self.runge_kutta(state, P, alpha, dt)
                x_ref.append(state)
                u_ref.append([P, alpha])
                tf_ref += dt

        return np.array(x_ref), np.array(u_ref), tf_ref


def main(
    X0,
    CL,
    CD,
    heightf=40000,
    Vf=3000,
    s=1.0,
    Isp=4000.0,
    m_dry=1400,
    alpha_max=15 / 180 * np.pi,
):
    # print(X0, CL, CD, heightf, Vf, s, P, Isp, Mdry, alphaMax)
    X0 = [float("%.2f" % x) for x in X0]
    Xf = [heightf, 2 / 57.3, 0, Vf, 0 / 57.3, 90 / 180.0 * np.pi, m_dry]

    Mis = Climb(CL, CD)
    Mis.setParams(
        alpha_max=alpha_max,
        s=s,
        Isp=Isp,
    )
    Mis.setSimuParams(N=200, iter_max=10)
    Mis.setIniState(*X0)
    Mis.setEndState(*Xf)
    Mis.form_dynamic()
    X, U, tf = Mis.solve()


if __name__ == "__main__":

    from Climb import main

    X0 = [10000, 0.0, 0.0, 600.0, 0 / 57.3, 0.0, 3000]

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
    main(X0, CL, CD)
