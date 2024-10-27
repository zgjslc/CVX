import numpy as np
from numpy import sin, cos, exp, zeros, linspace, sqrt, pi
from scipy.linalg import block_diag
from Discrete import DiscreteSystem
from Scaling import TrajectoryScaling

g0 = 9.81
R0 = 6378137


# 针对时间自由的问题
class Core:
    def __init__(self):
        pass

    def setDelta(self, delta):
        # 设置信赖域参数
        self._delta = delta
        self._delta = np.array(self._delta).reshape(-1, 1)

    def setEpsilon(self, epsilon):
        # 设置收敛判定参数
        self._epsilon = epsilon
        self._epsilon = np.array(self._epsilon).reshape(-1, 1)

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

    def setSimuParams(self, N=100, iterMax=6, mode="foh"):
        # 设置仿真基本参数 轨迹离散点数N 序列凸优化最大迭代次数iterMax

        self.N = N
        self.iterMax = iterMax
        self.mode = mode
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

    def formABCK(self, Ja, refTraj, dt, tf):
        self.stateDim = Ja.stateDim
        self.controlDim = Ja.controlDim
        self.discreteSys = self.buildDiscrete(dt, tf, Ja.GetA, Ja.GetB, Ja.GetF)
        return self.discreteSys.diff_discrete(
            refTraj[:, : self.stateDim], refTraj[:, self.stateDim :], mode=self.mode
        )

    def solve(self):
        _delta = self._delta.copy()
        Ja = self.formDynamic()
        _epsilon = self._epsilon.copy()
        _epsilon = np.repeat(_epsilon, self.N, axis=1).flatten("F")
        tK, XRefTraj, addParm = self.buildRefTraj()
        dt = tK[1] - tK[0]
        self.DQFlag = True
        TF = 106.0
        for _ in range(self.iterMax):
            if self.DQFlag:
                _delta *= 0.8
            prob, X, AKk, BKk, FKk, CKk, XRefP, DeltaP, tf = self.buildProblem()
            AK, BK, FK, CK = self.formABCK(Ja, XRefTraj, addParm)
            AKk.value, BKk.value, FKk.value, CKk.value = self.formDynamicConstraints(
                AK, BK, FK, CK, self.N, dt, TF
            )
            XRefP.value = np.array(XRefTraj).flatten("F")
            DeltaP.value = np.repeat(_delta, self.N, axis=1).flatten("F")
            prob.solve("MOSEK", verbose=True, ignore_dpp=True)
            if prob.status != "optimal":
                print(prob.status)
                return [], False
            if X.value is None:
                self.DQFlag = False
                continue
            XRefTraj = X.value.reshape(-1, self.stateDim).T
            TF = tf.value
            if np.all(np.abs(X.value - XRefP.value) <= _epsilon):
                break
        return XRefTraj

    def diff_discrete_foh(self, x, u, delT, tf, A_func, B_func, f_func):
        # 线性插值的矩阵离散化方法（FOH） 矢量化
        ix, iu = self.ix, self.iu

        if x.ndim == 1:  # 单步状态和输入
            N = 1
            x, u = x[np.newaxis, :], u[np.newaxis, :]
        else:
            N = x.shape[0]

        def dvdt(t, V, u_m, u_p, N):
            alpha, beta = (delT - t) / delT, t / delT
            u_interp = alpha * u_m + beta * u_p

            V = V.reshape(N, ix + ix**2 + 2 * ix * iu + 2 * ix)
            x, Phi_flat = V[:, :ix], V[:, ix : ix + ix**2]
            Phi = Phi_flat.reshape(N, ix, ix)
            Phi_inv = np.linalg.inv(Phi)

            f = f_func(x, u_interp)
            A, B = A_func(x, u_interp), B_func(x, u_interp)

            dpdt = (A @ Phi).reshape(N, -1).T
            dbmdt = (Phi_inv @ B).reshape(N, ix * iu).T * alpha
            dbpdt = (Phi_inv @ B).reshape(N, ix * iu).T * beta
            dsdt = (Phi_inv @ f[:, :, np.newaxis]).squeeze().T / tf
            dzdt = (
                (Phi_inv @ (-A @ x[:, :, np.newaxis] - B @ u_interp[:, :, np.newaxis]))
                .squeeze()
                .T
            )

            dv = np.vstack((f.T, dpdt, dbmdt, dbpdt, dsdt, dzdt))
            return dv.flatten(order="F")

        # 初始条件设置
        A0, B0 = np.eye(ix).flatten(), np.zeros(ix * iu)
        V0 = np.hstack((x, A0, B0, B0, np.zeros(ix), np.zeros(ix))).T.flatten(order="F")

        sol = solve_ivp(dvdt, (0, delT), V0, args=(u[:-1], u[1:], N))

        # 提取各项结果
        idx = {
            "state": slice(0, ix),
            "A": slice(ix, ix + ix**2),
            "Bm": slice(ix + ix**2, ix + ix**2 + ix * iu),
            "Bp": slice(ix + ix**2 + ix * iu, ix + ix**2 + 2 * ix * iu),
            "s": slice(ix + ix**2 + 2 * ix * iu, ix + ix**2 + 2 * ix * iu + ix),
            "z": slice(
                ix + ix**2 + 2 * ix * iu + ix, ix + ix**2 + 2 * ix * iu + 2 * ix
            ),
        }

        sol_final = sol.y[:, -1].reshape(N, -1)
        x_prop = sol_final[:, idx["state"]].reshape(N, ix)
        A = sol_final[:, idx["A"]].reshape(N, ix, ix)
        Bm = A @ sol_final[:, idx["Bm"]].reshape(N, ix, iu)
        Bp = A @ sol_final[:, idx["Bp"]].reshape(N, ix, iu)
        s = A @ sol_final[:, idx["s"]].reshape(N, ix, 1).squeeze()
        z = A @ sol_final[:, idx["z"]].reshape(N, ix, 1).squeeze()

        return A, Bm, Bp, s, z, x_prop


if __name__ == "__main__":
    pass
