import numpy as np
from numpy import sin, cos, exp, zeros, linspace, sqrt, pi
from Discrete import DiscreteSystem
from Scaling import TrajectoryScaling

g0 = 9.81
R0 = 6378137


# 针对时间自由的问题
class Core:
    def __init__(self):
        pass

    def setEpsilon(self, epsilon):
        # 设置收敛判定参数
        self.epsilon = epsilon

    def setParams(
        self,
        s=1,
        mDry=0,
        alphaMax=15 / 180 * pi,
        sigmaMax=60 / 180 * pi,
        PMax=60000,
        PMin=60000,
        QsMax=1e6,
        qMax=9e5,
        nMax=50,
        Isp=4000,
    ):
        # 设置飞行器基本制导参数，这里只有参考面积, 攻角限幅与倾侧角限幅

        self.s = s
        self.mDry = mDry
        self.alphaMax = alphaMax
        self.sigmaMax = sigmaMax
        self.PMax = PMax
        self.PMin = PMin
        self.Qsmax = QsMax
        self.qMax = qMax
        self.nmax = nMax
        self.Isp = Isp

        return True

    def setSimuParams(self, N=100, iterMax=6, mode="foh", wTf=1, wVc=1e4, wtr=1e-3):
        # 设置仿真基本参数 轨迹离散点数N 序列凸优化最大迭代次数iterMax

        self.N = N
        self.iterMax = iterMax
        self.mode = mode
        self.wtf = wTf
        self.wvc = wVc
        self.wtr = wtr
        return True

    def setIniState(self, y, lon, lat, V, theta, psi, m):
        # 设置期望初始状态
        self.lon0 = lon
        self.y0 = y
        self.lat0 = lat
        self.V0 = V
        self.theta0 = theta / 180 * pi
        self.psi0 = psi / 180 * pi
        self.m0 = m

    def setEndState(self, y, lon, lat, V, theta, psi, m):
        # 设置期望终点状态
        self.lonf = lon
        self.yf = y
        self.latf = lat
        self.Vf = V
        self.thetaf = theta / 180 * pi
        self.psif = psi / 180 * pi
        self.mf = m

    def buildDiscrete(self, dt, tf, A, B, f):
        # 线性插值的矩阵离散化方法（FOH） 矢量化
        return DiscreteSystem(
            self.stateDim,
            self.controlDim,
            dt,
            tf,
            A,
            B,
            f,
        )

    def formABCK(self, Ja, XRef, URef, tfRef):
        dt = tfRef / self.N
        self.stateDim = Ja.stateDim
        self.controlDim = Ja.controlDim
        self.discreteSys = self.buildDiscrete(dt, tfRef, Ja.GetA, Ja.GetB, Ja.GetF)
        return self.discreteSys.diff_discrete(XRef, URef, mode=self.mode)

    def tarjectory_scaling(self, x_min, x_max, u_min, u_max, tf):
        self.traj_scaling = TrajectoryScaling(x_min, x_max, u_min, u_max, tf)

    def buildProblem(self):
        # 构造最优化问题，父类基函数
        pass

    def buildRefTrajectory(self):
        # 构造参考轨迹 父类基函数
        pass

    def solve(self):
        self.formDynamic()
        _epsilon = self._epsilon.copy()
        _epsilon = np.repeat(_epsilon, self.N, axis=1).flatten("F")
        XRef, URef, tfRef = self.buildRefTraj()

        XRef_scaled, URef_scaled, tfRef_scaled = self.traj_scaling.scale(
            XRef, URef, tfRef
        )
        problem, variables, params = self.buildProblem()
        for _ in range(self.iterMax):

            A, Bm, Bp, s, z, xProb = self.formABCK(self.Ja, XRef, URef, tfRef)

            (
                params["A"],
                params["s"],
                params["z"],
                params["XRef"],
                params["URef"],
                params["tfRef"],
            ) = (A, s, z, XRef_scaled, URef_scaled, tfRef_scaled)
            if self.mode == "zoh":
                params["B"] = Bm
            else:
                params["Bm"] = Bm
                params["Bp"] = Bp
            problem.solve("MOSEK", verbose=True, ignore_dpp=True)
            if problem.status != "optimal":
                print(problem.status)
                return [], False

            XRef_scaled, URef_scaled, tfRef_scaled = (
                variables["X"].value,
                variables["U"].value,
                variables["tf"].value,
            )
            XRef, URef, tfRef = self.traj_scaling.unscale(
                XRef_scaled, URef_scaled, tfRef_scaled
            )

        return XRef, URef, tfRef


if __name__ == "__main__":
    pass
