import numpy as np
from typing import Optional


def get_step(max: float, min: float, num: int) -> float:
    """Вычисление шага дискретизации."""
    return (max - min) / (num - 1)


def get_courant_friedrichs_loewy_criterion(tau: float, h: float, mu: float) -> float:
    """
    Вычисление критерия Куранта — Фридрихса — Леви.
    :param tau: шаг дискретизации по времени
    :param h: шаг дискретизации по оси X
    :param mu: коэффициент теплопроводности
    :return: коэффициент Куранта — Фридрихса — Леви
    """
    return mu * tau / h ** 2


def get_system_matrix(x_num: int, cfl_coefficient: float) -> np.array:
    """
    Формирует системную матрицу.
    :param x_num: количество узлов по Оx
    :param cfl_coefficient: коэффициент Куранта — Фридрихса — Леви
    :return: системная матрица
    """
    matrix = np.zeros((x_num, x_num))
    matrix[0, 0] = 1

    for i in range(1, x_num - 1):
        matrix[i, i - 1] = - cfl_coefficient
        matrix[i, i] = 1 + 2 * cfl_coefficient
        matrix[i, i + 1] = - cfl_coefficient

    matrix[x_num - 1, x_num - 1] = 1
    return matrix


def implicit_heat_equation_finite_method(
        u_old: np.array,
        array: np.array,
        x_num: int,
        x: np.array,
        t: float,
        tau: float,
        function=None,
        dirichlet_conditions=None,
) -> np.array:
    """
    Вычисление нового значения для функции U.
    Решается система вида A * x = b.
    :param array: матрица системы
    :param x_num: количество узлов по Оx
    :param x: координаты узлов
    :param t: текущее время
    :param tau: шаг дискретизации по фремени
    :param function: возвращает значение функции переноса f(x, t + tau)
    :param dirichlet_conditions: функция для граничных условий Дирихле
    :param u_old: старые значения U
    :return: новые значения для функции U
    """
    f = np.zeros((1, x_num))
    if function:
        f = function(x_num, x, t)

    b = u_old.copy()
    for i in range(1, x_num - 1):
        b[i] = b[i] + tau * f[i]

    u = np.linalg.solve(array, b)

    if dirichlet_conditions:
        u = dirichlet_conditions(x_num, x, t, u)

    return u


