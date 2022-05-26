#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cmath>
#include "mpi.h"


using namespace std;

const char* FILE_PATH = "solutions/u.txt";


MPI_Status status;

const double MU = 0.02;

const double START_U_VALUE = 0;
const double START_U_RIGHT_VALUE = 90;
const double START_U_LEFT_VALUE = 40;

const double X_MAX = 1.0;
const double X_MIN = 0;
const int X_NUM = 75;

const double T_MAX = 20;
const double T_MIN = 0;
const int T_NUM = 75;


const double EPS = 0.00001;
const int JACOBI_MAX_ITERATIONS = pow(10, 10);


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
	// Get line space in min_value - max_value.
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
	double* matrix = new double[x_num * x_num];
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


void jacobi_solver(double* A, double* b, double* X, int size) {
	// Jacobi method to solve system of equations.
	int numtasks, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int count_rows_per_process = size / numtasks;

	double* X_prev = new double[size];
	fill_zeros(X_prev, size);
	double* X_block = new double[count_rows_per_process];
	fill_zeros(X_block, count_rows_per_process);

	double* A_recv = new double[count_rows_per_process * size];
	fill_zeros(A_recv, count_rows_per_process * size);
	double* b_recv = new double[count_rows_per_process];
	fill_zeros(b_recv, count_rows_per_process);

	MPI_Scatter(A, count_rows_per_process * size, MPI_DOUBLE, A_recv, count_rows_per_process * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(b, count_rows_per_process, MPI_DOUBLE, b_recv, count_rows_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int iteration = 0;
	double step_error = 0;

	do {
		step_error = 0;

		for (int i = 0; i < size; i++) {
			X_prev[i] = X[i];
		}

		for (int i = 0; i < count_rows_per_process; i++) {
			int global_row_index = (rank * count_rows_per_process) + i;
			int global_col_index = i * size;

			X_block[i] = b_recv[i] / A_recv[global_row_index + i * size];

			for (int j = 0; j < global_row_index; j++) {
				X_block[i] -= A_recv[global_col_index + j] / A_recv[global_row_index + i * size] * X_prev[j];
			}

			for (int j = global_row_index + 1; j < size; j++) {
				X_block[i] -= A_recv[global_col_index + j] / A_recv[global_row_index + i * size] * X_prev[j];
			}
		}

		MPI_Allgather(X_block, count_rows_per_process, MPI_DOUBLE, X, count_rows_per_process, MPI_DOUBLE, MPI_COMM_WORLD);

		for (int i = 0; i < size; i++) {
			step_error += (X_prev[i] - X[i]) * (X_prev[i] - X[i]);
		}
		step_error = sqrt(step_error);

		iteration++;
		
	} while (step_error > EPS);

	free(X_prev);
}


void set_starting_values(double* u, int size) {
	// U is filled starting values without borders.
	for (int i = 0; i < size; i++) {
		u[i] = START_U_VALUE;
	}
}


double exact_func(double x, double t) {
	// Exact function.
	return exp(-t) * sin(sqrt(MU) * x);
}

void print_matrix(double* matrix, double* x, double* t, int n, int m) {
	cout << "Result matrix of temperature: " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			cout << matrix[j + i * m] << " ";
		}
		cout << endl;
	}
}


void save_to_file(const char* file_name, double* matrix, int n, int m) {
	// Save matrix in file.

	MPI_File output;
	MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output);

	if (!output) {
		cout << "Error: Could not open the file" << endl;
	}
	else
	{
		for (int i = 0; i < n; i++) {
			string row = "";
			for (int j = 0; j < m; j++) {
				row += to_string(matrix[j + i * m]);
				row += " ";
			}
			row += "\n";
			MPI_File_write(output, row.c_str(), strlen(row.c_str()), MPI_CHAR, &status);
		}
		MPI_File_close(&output);
		cout << "Data has written in file " << file_name << endl;
	}
}


int main(int argc, char** argv) {
	srand(time(NULL));

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

	for (int i = 0; i < X_NUM; i++) {
		b[i] = 0;
		f_vec[i] = 0;
	}

	double time_start, time_end;
	int numtasks, rank;

	if (X_NUM < 2) {
		cout << "X_NUM should be more than 2" << endl;
		return 0;
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		time_start = MPI_Wtime();
	}

	for (int i = 0; i < T_NUM; i++) {

		MPI_Barrier(MPI_COMM_WORLD);

		if (i == 0) {
			if (rank == 0) {
				for (int j = 0; j < X_NUM; j++) {
					u[j] = exact_func(x[j], t[i]);
				}
			}
		}
		else {
			if (rank == 0) {
				for (int j = 0; j < X_NUM; j++) {

					double u_prev = u[j + X_NUM * (i - 1)];

					if (j == 0 || j == X_NUM - 1) {
						b[j] = exact_func(u_prev, t[i]);
					}
					else {
						b[j] = u_prev + t_step * f_vec[j];
					}
				}
			}


			MPI_Bcast(matrix, T_NUM * X_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(b, X_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			fill_zeros(step_result, X_NUM);
			MPI_Barrier(MPI_COMM_WORLD);

			jacobi_solver(matrix, b, step_result, X_NUM);

			if (rank == 0) {
				for (int j = 0; j < X_NUM; j++) {
					u[j + X_NUM * i] = step_result[j];
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}

	}

	if (rank == 0) {
		time_end = MPI_Wtime();
		cout << "1D implicit method of heat equation MPI" << endl;
		cout << "Student: Bakanov D.S., group 6132-010402D" << endl;
		cout << "------------------------------------------" << endl;
		cout << "Matrix size: " << X_NUM * X_NUM << ", process number: " << numtasks << endl;
		cout << "Time elapsed: " << (time_end - time_start) * 1000 << "ms" << endl;
		if (X_NUM * T_NUM <= 100) {
			print_matrix(u, x, t, T_NUM, X_NUM);
		}
		save_to_file(FILE_PATH, u, T_NUM, X_NUM);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	

	return 0;
}
