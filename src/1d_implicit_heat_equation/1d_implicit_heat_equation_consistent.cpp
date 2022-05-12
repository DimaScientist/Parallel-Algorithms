#include <iostream>
#include <string.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cmath>


using namespace std;

const string FILE_PATH = "solutions/u.txt";


const double MU = 0.02;

const double START_U_VALUE = 0;
const double START_U_RIGHT_VALUE = 90;
const double START_U_LEFT_VALUE = 40;

const double X_MAX = 1.0;
const double X_MIN = 0;
const int X_NUM = 20;

const double T_MAX = 20;
const double T_MIN = 0;
const int T_NUM = 100;


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


double get_lambda_value(double tau, double h, double mu) {
	// Вычисление лямбды.
	return mu * tau / (h * h);
}


void format_matrix(double* array, int size) {
	// Форматирование матрицы для решения СЛАУ.
	for (int i = 1; i <= size - 1; i++) {
		if (array[1 + 3 * (i - 1)] == 0) return;

		array[2 + 3 * (i - 1)] = array[2 + 3 * (i - 1)] / array[1 + 3 * (i - 1)];

		array[1 + 3 * i] = array[1 + 3 * i] - array[2 + 3 * (i - 1)] * array[3 * i];


		if (array[1 + 3 * (size - 1)] == 0) return;
	}
}


double* get_system_matrix(int x_num, double lambda_value) {
	// Задает трехдиагональную иатрицу матрицу.
	double *matrix = new double[3 * x_num];
	
	matrix[0 + 0 * 3] = 0;
	matrix[1 + 0 * 3] = 1;
	matrix[0 + 1 * 3] = 0;

	for (int i = 1; i < x_num - 1; i++) {
		matrix[2 + 3 * (i - 1)] = -lambda_value;
		matrix[1 + 3 * i] = 1 + 2 * lambda_value;
		matrix[0 + 3 * (i + 1)] = -lambda_value;
	}

	matrix[2 + 3 * (x_num- 2)] = 0;
	matrix[1 + 3 * (x_num - 1)] = 1;
	matrix[2 + 3 * (x_num - 1)] = 0;

	format_matrix(matrix, x_num);
	
	return matrix;
}


double* solve_system(double* a, double* b, int size) {
	// Решение системы A * x = b.
	double* result = new double[size];

	for (int i = 0; i < size; i++) {
		result[i] = b[i];
	}
	for (int i = 1; i < size; i++) {
		result[i] -= a[2 + 3 * (i - 1)] * result[i - 1];
	}
	for (int i = size; i >= 1; i--) {
		result[i - 1] /= a[1 + 3 * (i - 1)];
		if (i > 1) {
			result[i - 2] -= a[3 * (i - 1)] * result[i - 1];
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


void save_to_file(string file_name, double* matrix, int n, int m) {
	ofstream output;
	
	output.open(file_name.c_str());

	if (!output) {
		cout << "Error: Could not open the file" << endl;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			output << matrix[j + i * m] << ",";
		}
		output << "\n";
	}

	output.close();
	cout << "Data has written in file " << file_name << endl;
}


double exact_func(double x, double t) {
	// Точное решение.
	return exp(-t) * sin(sqrt(MU) * x);
}


int main(int argc, char* argv[], char* envp[]) {
	setlocale(LC_ALL, "Russian");

	double x_step = get_step(X_MAX, X_MIN, X_NUM);
	double* x = get_linespace(X_MAX, X_MIN, X_NUM);
	
	double t_step = get_step(T_MAX, T_MIN, T_NUM);
	double* t = get_linespace(T_MAX, T_MIN, T_NUM);

	double* u = new double[X_NUM * T_NUM];
	set_starting_values(u, X_NUM * T_NUM);

	double clf_coefficient = get_lambda_value(t_step, x_step, MU);

	double* matrix = get_system_matrix(X_NUM, clf_coefficient);

	double* b = new double[X_NUM];
	double* f_vec = new double[X_NUM];

	for (int i = 0; i < X_NUM; i++) {
		b[i] = 0;
		f_vec[i] = 0;
	}

	for (int i = 0; i < T_NUM; i++) {
		if (i == 0) {
			for (int j = 0; j < X_NUM; j++) {

				if (j == 0 || j == X_NUM - 1) {
					u[0] = exact_func(x[j], t[i]);
				}
				else {
					u[j] = exact_func(X_NUM, t[i]);
				}

			}
		}
		else {
			for (int j = 0; j < X_NUM; j++) {

				double u_current = u[j + X_NUM * (i - 1)];

				if (j == 0 || j == X_NUM - 1) {
					b[j] = exact_func(u_current, t[i]);
				}
				else {
					b[j] = u_current + t_step * f_vec[j];
				}
			}

			double* solution = solve_system(matrix, b, X_NUM);

			for (int j = 0; j < X_NUM; j++) {
				u[j + X_NUM * i] = solution[j];
			}

			delete solution;

		}
		
	}

	save_to_file(FILE_PATH, u, T_NUM, X_NUM);

	delete x;
	delete t;
	delete u;
	delete matrix;
	delete b;
	delete f_vec;

	return 0;
}