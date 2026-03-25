class GMVIProblem:
    def __init__(self, operator_func, g_func):
        self.d = operator_func.d + operator_func.n
        self.operator_func = operator_func
        self.g_func = g_func

    def func_value(self, x):
        return self.operator_func.func_value(x) + self.g_func.func_value(x)