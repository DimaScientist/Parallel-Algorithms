import numpy as np


def get_step(max: float, min: float, num: int) -> float:
    """Вычисление шага дискретизации."""
    return (max - min) / (num - 1)


def get_linespace(max: float, min: float, num: int) -> np.array:
    """Равномерное заполнение по промежутку min - max."""
    array = np.zeros(num)
    for i in range(num):
        array[i] = ((num - i - 1) * min + i * max) / (num - 1)
    return array


def get_lambda_value(tau: float, h: float, mu: float) -> float:
    """Вычисление лямбды."""
    return mu * tau / h ** 2


def get_system_matrix(num: int, lambda_value: float) -> np.array:
    """Формирует трехдиагональную системную матрицу."""
    matrix = np.zeros(num * num)

    for i in range(num):
        matrix[i + num * i] = 1 + 2 * lambda_value
        if i > 0:
            matrix[i - 1 + i * num] = - lambda_value
        if i < num - 1:
            matrix[i + 1 + i * num] = - lambda_value

    return matrix


def gauss_seidel_solver(A: np.array, b: np.array, size: int, eps: float = 0.00001) -> np.array:
    """Метод Гаусса-Зейделя для решения системы A * x = b"""
    diag = np.zeros(size)
    x = np.zeros(size)
    step_error = eps * 2

    while step_error > eps:
        step_error = 0

        for i in range(size):
            diag[i] = x[i]
            x[i] = b[i]

            for j in range(size):
                if i != j:
                    x[i] -= A[j + i * size] * x[j]

            x[i] = x[i] / A[i + i * size]
            diag[i] = np.abs(diag[i] - x[i])

        step_error = diag[0]
        for i in range(1, size):
            if diag[i] > step_error:
                step_error = diag[i]

    return x
