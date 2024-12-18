import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from ClimbModel import ClimbModel
from scipy.interpolate import interp1d


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

    def build_reference_trajectory(
        self, mode="linear", t=None, P=None, alpha=None, tspan=4
    ):
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
            PL = interp1d(t, P)
            alphaL = interp1d(t, alpha)
            state = np.array([self.y0, self.V0, self.theta0, self.m0])
            dt = 0.1
            x_ref, u_ref, tf_ref = [], [], 0
            x_ref.append(state)
            u_ref.append([PL(0), alphaL(0)])
            while tf_ref < tspan:
                P = PL(tf_ref)
                alpha = alphaL(tf_ref)
                state = self.runge_kutta(state, P, alpha, dt)
                x_ref.append(state)
                u_ref.append([P, alpha])
                tf_ref += dt

        return np.array(x_ref), np.array(u_ref), tf_ref

    def update_reference(self, X, U, tf, tspan):
        tL = np.linspace(0, tf, self.N)
        XL = interp1d(tL, X.T)
        UL = interp1d(tL, U.T)
        t_inter = np.linspace(tspan, tf, self.N)
        return XL(t_inter).T, UL(t_inter).T, tf - tspan


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
    Mis.setSimuParams(N=300, iter_max=6)
    Mis.setIniState(*X0)
    Mis.setEndState(*Xf)
    Mis.form_dynamic()
    x_ref, u_ref, tf_ref = Mis.prePrcess()
    t_now = 0
    t_sapn = 100
    for i in range(1):
        X, U, tf = Mis.solve(x_ref, u_ref, tf_ref)
        tL = np.linspace(0, tf, 300)
        x_rel, u_rel, tf_rel = Mis.build_reference_trajectory(
            "test", tL, U[:, 0], U[:, 1], t_sapn
        )
        Mis.setIniState(
            y=x_rel[-1, 0], V=x_rel[-1, 1], theta=x_rel[-1, 2], m=x_rel[-1, 3]
        )
        tf_ref_plot = np.linspace(0, tf_rel, len(x_rel)) + t_now
        x_ref, u_ref, tf_ref = Mis.update_reference(X, U, tf, t_sapn)

        t = np.linspace(t_now, tf + t_now, 300)
        plt.figure(1)
        plt.plot(t, X[:, 0])

        plt.figure(2)
        plt.plot(t, X[:, 1])

        plt.figure(3)
        plt.plot(t, U[:, 0])
        plt.figure(4)
        plt.plot(t, U[:, 1])

        plt.figure(1)
        plt.plot(tf_ref_plot, x_rel[:, 0])
        plt.figure(2)
        plt.plot(tf_ref_plot, x_rel[:, 1])

        t_now += t_sapn
    plt.show()


if __name__ == "__main__":

    from Climb import main

    X0 = [10000, 0.0, 0.0, 600.0, 0 / 57.3, 0.0, 3000]

    CL = [0.5811, 7.981, 8.457]

    CD = [0.1288, 1.99, 12.38]
    main(X0, CL, CD)
