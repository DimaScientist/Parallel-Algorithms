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
    x = np.zeros(size)
    x_prev = np.zeros(size)
    diag = np.zeros(size * size)
    step_error = eps * 2

    for i in range(size):
        for j in range(size):
            if i == j:
                diag[j + i * size] = 0
            else:
                diag[j + i * size] = -A[j + i * size] / A[i + i * size]

    while step_error > eps:
        x_prev = np.copy(x)

        for i in range(size):
            sum = 0

            for j in range(0, i):
                sum += diag[j + i * size] * x[j]
            for j in range(i, size):
                sum += diag[j + i * size] * x_prev[j]

            x[i] = sum + b[i] / A[i + i * size]

        step_error = 0
        for i in range(size):
            step_error += (x_prev[i] - x[i]) * (x_prev[i] - x[i])

        step_error = np.sqrt(step_error)

    return x
