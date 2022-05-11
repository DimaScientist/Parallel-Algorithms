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


def get_courant_friedrichs_loewy_criterion(tau: float, h: float, mu: float) -> float:
    """Вычисление критерия Куранта — Фридрихса — Леви."""
    return mu * tau / h ** 2


def get_system_matrix(num: int, clf_coefficient: float) -> np.array:
    """Формирует системную матрицу."""
    matrix = np.zeros(3 * num)

    matrix[0 + 3 * 0] = 0
    matrix[1 + 3 * 0] = 1
    matrix[0 + 3 * 1] = 0

    for i in range(1, num - 1):
        matrix[2 + 3 * (i - 1)] = -clf_coefficient
        matrix[1 + 3 * i] = 1 + 2 * clf_coefficient
        matrix[0 + 3 * (i + 1)] = -clf_coefficient

    matrix[2 + 3 * (num - 2)] = 0
    matrix[1 + 3 * (num - 1)] = 1
    matrix[2 + 3 * (num - 1)] = 0

    return matrix


def to_triangular(array: np.array, size: int) -> bool:
    """Возвращает матрицу в триангулярной форме."""
    for i in range(1, size - 1):

        if array[1 + 3 * (i - 1)] == 0:
            return False

        array[2 + 3 * (i - 1)] = array[2 + 3 * (i - 1)] / array[1 + 3 * (i - 1)]
        array[1 + 3 * i] = array[1 + 3 * i] - array[2 + 3 * (i - 1)] * array[3 * i]

        if array[1 + 3 * (size - 1)] == 0:
            return False

    return True


def solve_system(A: np.array, b: np.array, size: int) -> np.array:
    """Решаем систему вида A * x = b."""
    result = b.copy()

    for i in range(1, size):
        result[i] -= A[2 + 3 * (i - 1)] * result[i - 1]

    for i in range(size, 0, -1):
        result[i - 1] /= A[1 + 3 * (i - 1)]
        if i > 1:
            result[i - 2] -= A[3 * (i - 1)] * result[i - 1]

    return result


def get_right_dirichlet_condition(
        a: float,
        b: float,
        t0: float,
        t: float,
        func=lambda *args: 0
) -> float:
    """Задает правое граничное условие Дирихле."""
    value = func(a, b, t0, t)
    return value


def get_left_dirichlet_condition(
        a: float,
        b: float,
        t0: float,
        t: float,
        func=lambda *args: 0
) -> float:
    """Задает левое граничное условие Дирихле."""
    value = func(a, b, t0, t)
    return value


def function(x: np.array, values: np.array, t: float, size: int, func=lambda *args: 0) -> None:
    """Задает значения функции переноса в данный момент времени."""
    for i in range(size):
        values[i] = func(x, t)
