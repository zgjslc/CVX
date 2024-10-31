from numpy import pi
import numpy as np
from Discrete import DiscreteSystem
from Scaling import TrajectoryScaling
import matplotlib.pyplot as plt
import time


# 针对时间自由的问题
class GuidanceOptimizer:
    def __init__(self, epsilon=1e-2, vctrol=1e-2):
        """
        初始化Core类
        """
        self.epsilon = epsilon
        self.vctrol = vctrol
        self.state_dim = None
        self.control_dim = None
        self.N = 100
        self.iter_max = 6
        self.mode = "foh"

    def form_dynamic(self):
        """
        构造动力学模型
        """
        raise NotImplementedError("请在子类中实现此方法")

    def build_problem(self):
        """
        构造优化问题（需要在子类中实现）
        """
        raise NotImplementedError("请在子类中实现此方法")

    def build_reference_trajectory(self):
        """
        构造参考轨迹（需要在子类中实现）
        """
        raise NotImplementedError("请在子类中实现此方法")

    def setSimuParams(self, N=100, iter_max=6, mode="foh", w_tf=1, w_vc=1e4, w_tr=1e-3):
        """
        设置仿真基本参数
        """
        self.N = N
        self.iter_max = iter_max
        self.mode = mode
        self.w_tf = w_tf
        self.w_vc = w_vc
        self.w_tr = w_tr

    def setParams(
        self,
        s=1,
        alpha_max=15 / 180 * pi,
        sigma_max=60 / 180 * pi,
        p_min=40000,
        p_max=60000,
        Qs_max=1e6,
        q_ax=9e5,
        n_max=50,
        Isp=4000,
    ):
        """
        设置飞行器的基本制导参数
        """

        self.s = s
        self.alpha_max = alpha_max
        self.sigma_max = sigma_max
        self.p_max = p_max
        self.p_min = p_min
        self.Qs_max = Qs_max
        self.q_max = q_ax
        self.n_max = n_max
        self.Isp = Isp

        return True

    def setIniState(self, y, lon, lat, V, theta, psi, m):
        """
        设置初始状态
        """
        self.lon0 = lon
        self.y0 = y
        self.lat0 = lat
        self.V0 = V
        self.theta0 = theta / 180 * pi
        self.psi0 = psi / 180 * pi
        self.m0 = m

    def setEndState(self, y, lon, lat, V, theta, psi, m):
        """
        设置终点状态
        """
        self.lonf = lon
        self.yf = y
        self.latf = lat
        self.Vf = V
        self.thetaf = theta / 180 * pi
        self.psif = psi / 180 * pi
        self.m_dry = m

    def build_discrete_system(self, dt, tf, A, B, f):
        """
        构造离散系统
        """
        return DiscreteSystem(self.state_dim, self.control_dim, dt, tf, A, B, f)

    def form_abc(self, jacobian, x_ref, u_ref, tf_ref):
        """
        根据雅可比矩阵生成线性化模型的离散化矩阵
        """
        dt = tf_ref / self.N
        self.state_dim = jacobian.state_dim
        self.control_dim = jacobian.control_dim
        self.discrete_sys = self.build_discrete_system(
            dt, tf_ref, jacobian.GetA, jacobian.GetB, jacobian.GetF
        )
        return self.discrete_sys.diff_discrete(x_ref, u_ref, mode=self.mode)

    def tarjectory_scaling(self):
        """
        初始化轨迹缩放器
        """
        x_ref, u_ref, tf_ref = self.build_reference_trajectory()
        x_min = np.array([10000.0, 600.0, -2 / 180.0 * pi, 1400.0])
        x_max = np.array([40000.0, 3000.0, 10 / 180.0 * pi, 3000.0])
        u_min = np.array([self.p_min, -self.alpha_max])
        u_max = np.array([self.p_max, self.alpha_max])
        self.traj_scaling = TrajectoryScaling(x_min, x_max, u_min, u_max, tf=tf_ref)
        return x_ref, u_ref, tf_ref

    def solve(self):
        """
        主求解流程
        """
        x_ref, u_ref, tf_ref = self.tarjectory_scaling()
        x_ref_scaled, u_ref_scaled, tf_ref_scaled = self.traj_scaling.scale(
            x_ref, u_ref, tf_ref
        )
        problem, variables, params = self.build_problem()
        flag = True
        iterNum = 0
        while flag:

            A, Bm, Bp, s, z, xProb = self.form_abc(self.Ja, x_ref, u_ref, tf_ref)
            params["XRef"].value,
            params["URef"].value,
            params["tfRef"].value,
            for i in range(self.N - 1):
                params["A"][i].value = A[i]
                params["s"][i].value = s[i]
                params["z"][i].value = z[i]
                # 根据模式设置B矩阵
                if self.mode == "zoh":
                    params["B"][i].value = Bm[i]
                else:
                    params["Bm"][i].value = Bm[i]
                    params["Bp"][i].value = Bp[i]
            params["XRef"].value = x_ref
            params["URef"].value = u_ref
            params["tfRef"].value = tf_ref
            tic = time.time()
            # 求解优化问题
            problem.solve("MOSEK", verbose=False)
            print(
                f"第{iterNum+1}次求解，迭代时间: {time.time()-tic}，状态: {problem.status}"
            )
            if problem.status != "optimal":
                print(f"求解状态: {problem.status}")
                return [], [], []
            if np.max(variables["vc"].value) < self.vctrol:
                print(np.max(variables["X"].value[:, 0] - x_ref_scaled[:, 0]))
                if (
                    np.max(variables["X"].value - x_ref_scaled) < self.epsilon
                    or iterNum >= self.iter_max
                ):
                    flag = False
            # 更新参考轨迹
            x_ref_scaled, u_ref_scaled, tf_ref_scaled = (
                variables["X"].value,
                variables["U"].value,
                variables["tf"].value,
            )
            x_ref, u_ref, tf_ref = self.traj_scaling.unscale(
                x_ref_scaled, u_ref_scaled, tf_ref_scaled
            )
            iterNum += 1
            t = np.linspace(0, tf_ref, self.N)
        plt.figure(1)
        plt.plot(t, x_ref[:, 0])
        plt.figure(2)
        plt.plot(t, x_ref[:, 1])
        plt.figure(3)
        plt.plot(t, u_ref[:, 0])
        plt.figure(4)
        plt.plot(t, u_ref[:, 1])
        plt.show()
        return x_ref, u_ref, tf_ref


if __name__ == "__main__":
    pass
