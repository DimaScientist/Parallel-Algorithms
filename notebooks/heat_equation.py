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


def jacobi_solver(A: np.array, b: np.array, size: int, eps: float = 0.00001, max_iterations: int = 1e10) -> np.array:
    """Метод Якоби для решения системы A * x = b."""
    x = np.zeros(size)
    step_error = eps * 2
    iteration = 0

    while step_error > eps and iteration < max_iterations:
        x_prev = np.copy(x)

        for i in range(size):
            sum = 0
            for j in range(0, i):
                sum += - A[j + i * size] / A[i + i * size] * x_prev[j]
            for j in range(i + 1, size):
                sum += - A[j + i * size] / A[i + i * size] * x_prev[j]

            x[i] = sum + b[i] / A[i + i * size]

        step_error = 0
        for i in range(size):
            step_error += (x_prev[i] - x[i]) * (x_prev[i] - x[i])

        step_error = np.sqrt(step_error)
        iteration += 1

    return x
