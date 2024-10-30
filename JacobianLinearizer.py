import sympy as sy
import numpy as np


class JacobianLinearizer:
    """
    计算状态变量和控制变量的雅可比矩阵（Jacobian），并包含一个符号变量 F 的依赖项。
    """

    def __init__(self, funcs, state_vars, control_vars, all_vars):
        """
        初始化 JacM 类

        参数:
        - funcs (sy.Matrix): 系统的符号函数（方程组矩阵）。
        - state_vars (list): 状态变量符号列表。
        - control_vars (list): 控制变量符号列表。
        - func (sy.Matrix): 额外包含在计算中的函数。
        - all_vars (sy.Matrix): 用于 lambdify 函数的所有变量的矩阵。
        """
        # 初始化符号变量 F
        self.F = sy.symbols("F")

        # 获取状态变量和控制变量的维度
        self.state_dim = len(state_vars)
        self.control_dim = len(control_vars)

        # 计算状态变量和控制变量的雅可比矩阵，并添加 F 依赖
        self.jac_s = sy.simplify(
            funcs.jacobian(state_vars)
            + sy.ones(self.state_dim, self.state_dim) * self.F
        )
        self.jac_c = sy.simplify(
            funcs.jacobian(control_vars)
            + sy.ones(self.state_dim, self.control_dim) * self.F
        )

        # 简化主函数表达式，添加 F 依赖
        self.func_simplified = sy.simplify(funcs + sy.ones(self.state_dim, 1) * self.F)

        # 将 F 添加到所有变量的列表中
        self.all_vars = all_vars.row_insert(len(all_vars), sy.Matrix([self.F]))

        # 使用 numpy 库创建雅可比矩阵和函数的 lambdified 函数
        self.A_func = sy.lambdify(self.all_vars, self.jac_s, "numpy")
        self.B_func = sy.lambdify(self.all_vars, self.jac_c, "numpy")
        self.Func_func = sy.lambdify(self.all_vars, self.func_simplified, "numpy")

    def GetA(self, *vars):
        """
        计算相对于状态变量的雅可比矩阵，默认将 F 设置为零。

        参数:
        - vars: 替代到雅可比矩阵中的变量值。

        返回:
        - numpy.ndarray: 计算后的状态变量雅可比矩阵。
        """
        J_subs = self.A_func(*vars, np.zeros_like(vars[0]))
        return np.array(J_subs, dtype=np.float64)

    def GetB(self, *vars):
        """
        计算相对于控制变量的雅可比矩阵，默认将 F 设置为零。

        参数:
        - vars: 替代到雅可比矩阵中的变量值。

        返回:
        - numpy.ndarray: 计算后的控制变量雅可比矩阵。
        """
        J_subs = self.B_func(*vars, np.zeros_like(vars[0]))
        return np.array(J_subs, dtype=np.float64)

    def GetF(self, *vars):
        """
        计算主函数值，并考虑符号变量 F 的依赖。

        参数:
        - vars: 替代到函数中的变量值。

        返回:
        - numpy.ndarray: 计算后的函数值。
        """
        Func = self.Func_func(*vars, np.zeros_like(vars[0]))
        return np.array(Func, dtype=np.float64)


if __name__ == "__main__":
    pass
