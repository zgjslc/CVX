import sympy as sy
import numpy as np


class JacM:

    def __init__(self, funcs, StateVars, ControlStates, Func, AllVars):
        self.F = sy.symbols("F")
        self.funcs = funcs
        self.StateVars = StateVars
        self.ControlVars = ControlStates
        self.stateDim = len(StateVars)
        self.controlDim = len(ControlStates)
        self.jac_s = sy.simplify(
            self.funcs.jacobian(StateVars)
            + sy.ones(len(StateVars), len(StateVars)) * self.F
        )
        self.jac_c = sy.simplify(
            self.funcs.jacobian(ControlStates)
            + sy.ones(len(StateVars), len(ControlStates)) * self.F
        )
        self.Func = sy.simplify(Func + sy.ones(len(StateVars), 1) * self.F)
        self.allVars = AllVars.row_insert(len(AllVars), sy.Matrix([self.F]))
        self.A = sy.lambdify(self.allVars, self.jac_s, "numpy")
        self.B = sy.lambdify(self.allVars, self.jac_c, "numpy")
        self.Func = sy.lambdify(self.allVars, self.Func, "numpy")

    def GetA(self, *Vars):
        FF = np.full_like(Vars[0], 0)
        J_subs = self.A(*Vars, FF)
        return np.array(J_subs).astype(np.float64)

    def GetB(self, *Vars):
        FF = np.full_like(Vars[0], 0)
        J_subs = self.B(*Vars, FF)
        return np.array(J_subs).astype(np.float64)

    def GetF(self, *Vars):
        FF = np.full_like(Vars[0], 0)
        Func = self.Func(*Vars, FF)
        return np.array(Func).astype(np.float64)


if __name__ == "__main__":
    pass
