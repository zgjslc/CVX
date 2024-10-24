import numpy as np
from numpy import sin, cos, exp, zeros, linspace, sqrt, pi
from scipy.linalg import block_diag

g0 = 9.81
R0 = 6378137


# 针对时间自由的问题
class Core:
    def __init__(self):
        self.obstacleR = None
        self.obstacleCenter = None
        self.psif = None
        self.thetaF = None
        self.zf = None
        self.yf = None
        self.xf = None
        self.P0 = None
        self.sigma0 = None
        self.alpha0 = 0
        self.m0 = None
        self.psi0 = None
        self.theta0 = None
        self.V0 = None
        self.z0 = None
        self.y0 = None
        self.x0 = None
        self.iterMax = None
        self.N = None
        self.mMin = None
        self.Isp = None
        self.s = None
        self.alphaMax = None
        self.sigmaMax = None
        self.PMax = None
        self.PMin = None
        self.problem = None
        self.CLdown = 1
        # 信赖域
        self._delta = np.array(
            [
                2000,
                1000,
                40 * pi / 180,
                1000,
            ]
        ).reshape(-1, 1)
        # 收敛判据
        self._epsilon = np.array(
            [
                100,
                20,
                1 * pi / 180,
                20,
            ]
        ).reshape(-1, 1)

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
        mMin=0,
        alphaMax=15 / 180 * pi,
        sigmaMax=60 * pi / 180,
        PMax=60000,
        PMin=60000,
        QsMax=1e6,
        qMax=9e5,
        nMax=50,
        P=160000,
        Isp=4000,
    ):
        # 设置飞行器基本制导参数，这里只有参考面积, 攻角限幅与倾侧角限幅

        self.s = s
        self.mMin = mMin
        self.alphaMax = alphaMax
        self.sigmaMax = sigmaMax
        self.PMax = PMax
        self.PMin = PMin
        self.Qsmax = QsMax
        self.qMax = qMax
        self.nmax = nMax
        self.P = P
        self.Isp = Isp

        return True

    def setSimuParams(self, N=100, iterMax=6):
        # 设置仿真基本参数 轨迹离散点数N 序列凸优化最大迭代次数iterMax

        self.N = N
        self.iterMax = iterMax
        return True

    def setIniState(self, lon, y, lat, V, theta, psi, m):
        # 设置期望初始状态
        self.lon0 = lon
        self.y0 = y
        self.lat0 = lat
        self.V0 = V
        self.theta0 = theta / 180 * pi
        self.psi0 = psi / 180 * pi
        self.m0 = m

    def setEndState(self, lon, y, lat, V, theta, psi, m):
        # 设置期望终点状态
        self.lonf = lon
        self.yf = y
        self.latf = lat
        self.Vf = V
        self.thetaf = theta / 180 * pi
        self.psif = psi / 180 * pi
        self.mf = m

    def formDynamicConstraints(self, A, B, F, C, N, dt, tf):
        # 针对时间自由构造问题
        stateDim = self.stateDim
        I = np.eye(self.stateDim)
        dt = dt * tf
        AK = np.zeros(((N - 1) * stateDim, N * stateDim))
        BK = dt * block_diag(*B[:-1])
        FK = (F[:-1]).reshape(
            -1,
        )
        CK = dt * (C[:-1]).reshape(
            -1,
        )
        for i in range(N - 1):
            start_idx = stateDim * i
            end_idx = stateDim * (i + 1)
            AK[start_idx:end_idx, start_idx:end_idx] = dt * A[i] + I
            AK[
                start_idx:end_idx,
                end_idx : end_idx + stateDim,
            ] = -I

        return (AK, BK, FK, CK)

    def formABCK(self, Ja, refTraj, addParameters=None):
        num = refTraj.shape[1]
        self.stateDim = Ja.stateDim
        self.controlDim = Ja.controlDim
        if addParameters is not None:
            A = Ja.GetA(*refTraj, *addParameters).transpose(2, 0, 1)
            F = Ja.GetF(*refTraj, *addParameters).transpose(2, 0, 1)
            B = Ja.GetB(*refTraj, *addParameters).transpose(2, 0, 1)
        else:
            A = Ja.GetA(*refTraj).transpose(2, 0, 1)
            F = Ja.GetF(*refTraj).transpose(2, 0, 1)
            B = Ja.GetB(*refTraj).transpose(2, 0, 1)
        X = np.array(refTraj.T).reshape(num, self.stateDim, 1)
        U = np.array(addParameters.T).reshape(num, self.controlDim, 1)
        return A, B, F, -np.matmul(A, X) - np.matmul(B, U)

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


if __name__ == "__main__":
    pass
