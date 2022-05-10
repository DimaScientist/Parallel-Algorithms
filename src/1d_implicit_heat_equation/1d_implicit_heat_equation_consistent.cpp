#include <iostream>
#include <string.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cmath>


using namespace std;


const double MU = 0.01;

const double START_U_VALUE = 0;
const double START_U_RIGHT_VALUE = 90;
const double START_U_LEFT_VALUE = 40;

const double X_MAX = 1;
const double X_MIN = 0;
const int X_NUM = 100;

const double T_MAX = 100;
const double T_MIN = 0;
const int T_NUM = 200;


double get_step(double max_value, double min_value, int num) {
	// Вычисление шага дискретизации.
	return (max_value - min_value) / (num - 1);
}


double* get_linespace(double max_value, double min_value, int num) {
	// Равномерное заполнение в промежутке min_value - max_value
	double* array = new double[num];

	for (int i = 0; i < num; i++) {
		array[i] = ((double)(num - i - 1) * min_value
			+ (double)(i)*max_value)
			/ (double)(num - 1);
	}

	return array;
}


double get_courant_friederichs_loewy_criterion(double tau, double h, double mu) {
	// Вычисление критерия Куранта — Фридрихса — Леви.
	return mu * tau / (h * h);
}


double* get_system_matrix(int x_num, double cfl_coefficient) {
	// Задает системную матрицу.
	double *matrix = new double[3 * x_num];
	
	matrix[0 + 0 * 3] = 0;
	matrix[1 + 0 * 3] = 1;
	matrix[0 + 1 * 3] = 0;

	for (int i = 1; i < x_num - 1; i++) {
		matrix[2 + 3 * (i - 1)] = - cfl_coefficient;
		matrix[1 + 3 * i] = 1 + 2 * cfl_coefficient;
		matrix[0 + 3 * (i + 1)] = -cfl_coefficient;
	}

	matrix[2 + 3 * (x_num- 2)] = 0;
	matrix[1 + 3 * (x_num - 1)] = 1;
	matrix[2 + 3 * (x_num - 1)] = 0;
	
	return matrix;
}


bool to_triangular(double* array, int size) {
	// Приведение матрицы к триангулярному формату.
	for (int i = 1; i <= size - 1; i++) {
		if (array[1 + 3 * (i - 1)] == 0) return false;

		array[2 + 3 * (i - 1)] = array[2 + 3 * (i - 1)] / array[1 + 3 * (i - 1)];

		array[1 + 3 * i] = array[1 + 3 * i] - array[2 + 3 * (i - 1)] * array[3 * i];


		if (array[1 + 3 * (size - 1)] == 0) return false;
	}

	return true;
}


double* solve_system(double* a, double* b, int size, int job) {
	// Решение системы A * x = b.
	double* result = new double[size];

	for (int i = 0; i < size; i++) {
		result[i] = b[i];
	}

	if (job == 0) {
		for (int i = 1; i < size; i++) {
			result[i] -= a[2 + 3 * (i - 1)] * result[i - 1];
		}
		for (int i = size; i >= 1; i--) {
			result[i - 1] /= a[1 + 3 * (i - 1)];
			if (i > 1) {
				result[i - 2] -= a[3 * (i - 1)] * result[i - 1];
			}
		}
	}
	else {
		for (int i = 1; i <= size; i++) {
			result[i] /= a[1 + 3 * (i - 1)];
			if (i < size) {
				result[i] -= a[3 * i] * result[i - 1];
			}
		}
		for (int i = size - 1; i >= size; i--) {
			result[i - 1] -= a[2 + 3 * (i - 1)] * result[i];
		}
	}
	return result;
}


void set_starting_values(double* u, int size){
	// Заполняем U начальными значениями.
	for (int i = 0; i < size; i++) {
		u[i] = START_U_VALUE;
	}
}


double get_right_dirichlet_condition(double a, double b, double t0, double t) {
	// Установка граничных условий Дирихле справа.
	double value = START_U_RIGHT_VALUE;
	return value;
}


double get_left_dirichlet_condition(double a, double b, double t0, double t) {
	// Установка граничных условий Дирихле слева.
	double value = START_U_LEFT_VALUE;
	return value;
}


void function(double* x, double* values, double t, int size) {
	// Функция переноса.
	for (int i = 0; i < size; i++) {
		values[i] = 0;
	}
}


void save_to_file(string file_name, double* matrix, int n, int m) {
	ofstream output;
	
	output.open(file_name.c_str());

	if (!output) {
		cout << "Error: Could not open the file" << endl;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			output << setw(10) << matrix[j + i * m] << " ";
		}
		output << "\n";
	}

	output.close();
	cout << "Data has written in file " << file_name << endl;
}


int main(int argc, char* argv[], char* envp[]) {
	setlocale(LC_ALL, "Russian");

	double x_step = get_step(X_MAX, X_MIN, X_NUM);
	double* x = get_linespace(X_MAX, X_MIN, X_NUM);
	
	double t_step = get_step(T_MAX, T_MIN, T_NUM);
	double* t = get_linespace(T_MAX, T_MIN, T_NUM);

	double* u = new double[X_NUM * T_NUM];
	set_starting_values(u, X_NUM * T_NUM);

	double clf_coefficient = get_courant_friederichs_loewy_criterion(t_step, x_step, MU);

	double* matrix = get_system_matrix(X_NUM, clf_coefficient);
	to_triangular(matrix, X_NUM);

	double* b = new double[X_NUM];
	double* f_vec = new double[X_NUM];

	for (int i = 1; i < T_NUM; i++) {
		b[0] = get_left_dirichlet_condition(X_MIN, X_MAX, T_MIN, t[i]);
		function(x, f_vec, t[i], X_NUM);

		for (int j = 1; j < X_NUM - 1; j++) {
			b[j] = u[j + X_NUM * (i - 1)] + t_step * f_vec[j];
		}

		b[X_NUM - 1] = get_right_dirichlet_condition(X_MIN, X_MAX, T_MIN, t[i]);

		double* solution = solve_system(matrix, b, X_NUM, 0);

		for (int j = 0; j < X_NUM; j++) {
			u[j + X_NUM * i] = solution[j];
		}

		delete solution;
	}

	string file_name = "./solutions/u.txt";
	save_to_file(file_name, u, X_NUM, T_NUM);

	delete x;
	delete t;
	delete u;
	delete matrix;
	delete b;
	delete f_vec;

	return 0;
}