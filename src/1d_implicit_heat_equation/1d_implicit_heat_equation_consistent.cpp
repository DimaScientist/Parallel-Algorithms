#include <iostream>
#include <string>
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
const int X_NUM = 10;

const double T_MAX = 20;
const double T_MIN = 0;
const int T_NUM = 10;

const double EPS = 0.00001;


double get_step(double max_value, double min_value, int num) {
	// Compute sampling step.
	return (max_value - min_value) / (num - 1);
}

void fill_zeros(double* array, int size) {
	// Array is filled zeros.
	for (int i = 0; i < size; i++) {
		array[i] = 0;
	}
}


double* get_linespace(double max_value, double min_value, int num) {
	// Get line space in min_value - max_value
	double* array = new double[num];

	for (int i = 0; i < num; i++) {
		array[i] = ((double)(num - i - 1) * min_value
			+ (double)(i)*max_value)
			/ (double)(num - 1);
	}

	return array;
}


double get_lambda_value(double tau, double h, double mu) {
	// Compute lambda value.
	return mu * tau / (h * h);
}


double* get_system_matrix(int x_num, double lambda_value) {
	// Get tridiagonal matrix.
	double *matrix = new double[x_num * x_num];
	fill_zeros(matrix, x_num * x_num);

	for (int i = 0; i < x_num; i++) {
		matrix[i + x_num * i] = 1 + 2 * lambda_value;
		if (i > 0) {
			matrix[i - 1 + i * x_num] = -lambda_value;
		}
		if (i < x_num - 1) {
			matrix[i + 1 + i * x_num] = -lambda_value;
		}
	}
	
	return matrix;
}


void gauss_seidel_solver(double* A, double* b, double* X, int size) {
	// Gauss-Seidel method to solve system of equations.
	double* X_prev = new double[size];
	fill_zeros(X_prev, size);

	double* diag = new double[size * size];
	fill_zeros(diag, size * size);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i == j) {
				diag[j + i * size] = 0;
			}
			else {
				diag[j + i * size] = -A[j + i * size] / A[i + i * size];
			}
		}
	}

	double step_error = 0;


	do {
		for (int i = 0; i < size; i++) {
			X_prev[i] = X[i];
		}


		for (int i = 0; i < size; i++) {
			double sum = 0;

			for (int j = 0; j < i; j++) {
				sum += diag[j + i * size] * X[j];
			}
			for (int j = i; j < size; j++) {
				sum += diag[j + i * size] * X_prev[j];
			}
			X[i] = sum + b[i] / A[i + i * size];
		}

		step_error = 0;

		for (int i = 0; i < size; i++) {
			step_error += (X[i] - X_prev[i]) * (X[i] - X_prev[i]);
		}
		step_error = sqrt(step_error);

	} while (step_error > EPS);


}


void set_starting_values(double* u, int size){
	// U is filled starting values without borders.
	for (int i = 0; i < size; i++) {
		u[i] = START_U_VALUE;
	}
}


void save_to_file(string file_name, double* matrix, int n, int m) {
	// Save matrix in file.
	ofstream output;
	output.open(file_name.c_str());
	if (!output) {
		cout << "Error: Could not open the file" << endl;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			output << matrix[j + i * m] << " ";
		}
		output << "\n";
	}
	output.close();
	cout << "Data has written in file " << file_name << endl;
}


double exact_func(double x, double t) {
	// Exact function.
	return exp(-t) * sin(sqrt(MU) * x);
}

void print_matrix(double* matrix, double* x, double* t, int n, int m) {
	// Print matrix of heat equation.
	cout << setw(10) << "t";
	for (int i = 0; i < m; i++) {
		cout << setw(16) << "x[" << i + 1 << "]" << " ";
	}
	cout << endl;

	cout << setw(10) << " ";
	for (int i = 0; i < m; i++) {
		cout << setw(20) << x[i] << " ";
	}
	cout << endl;


	for (int i = 0; i < n; i++) {
		cout << setw(10) << t[i];
		for (int j = 0; j < m; j++) {
			cout << setw(20) << matrix[j + i * m] << " ";
		}
		cout << endl;
	}
}


int main(int argc, char* argv[], char* envp[]) {
	setlocale(LC_ALL, "Russian");
	cout << "Одномерная неявная схема теплопроводности" << endl;
	cout << "Студент: Баканов Д.С., группа 6132-010402D" << endl;
	cout << "------------------------------------------" << endl;

	double x_step = get_step(X_MAX, X_MIN, X_NUM);
	double* x = get_linespace(X_MAX, X_MIN, X_NUM);
	
	double t_step = get_step(T_MAX, T_MIN, T_NUM);
	double* t = get_linespace(T_MAX, T_MIN, T_NUM);

	double* u = new double[X_NUM * T_NUM];
	fill_zeros(u, X_NUM * T_NUM);
	set_starting_values(u, X_NUM * T_NUM);

	double clf_coefficient = get_lambda_value(t_step, x_step, MU);

	double* matrix = get_system_matrix(X_NUM, clf_coefficient);

	double* b = new double[X_NUM];
	double* f_vec = new double[X_NUM];
	double* step_result = new double[X_NUM];

	fill_zeros(b, X_NUM);
	fill_zeros(f_vec, X_NUM);

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

				double u_prev = u[j + X_NUM * (i - 1)];

				if (j == 0 || j == X_NUM - 1) {
					b[j] = exact_func(u_prev, t[i]);
				}
				else {
					b[j] = u_prev + t_step * f_vec[j];
				}
			}

			fill_zeros(step_result, X_NUM);
			gauss_seidel_solver(matrix, b, step_result, X_NUM);

			for (int j = 0; j < X_NUM; j++) {
				u[j + X_NUM * i] = step_result[j];
			}


		}
		
	}

	if (T_NUM * X_NUM <= 100) {
		print_matrix(u, x, t, T_NUM, X_NUM);
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