#include "lab3_io.h"

void read_matrix (const char* input_filename, int* M, int* N, double** D){
	FILE *fin = fopen(input_filename, "r");
	int i;

	fscanf(fin, "%d%d", M, N);
	
	int num_elements = (*M) * (*N);
	*D = (double*) malloc(sizeof(double)*(num_elements));
	
	for (i = 0; i < num_elements; i++){
		fscanf(fin, "%lf", (*D + i));
	}
	fclose(fin);
}

void printMatrix (double *M, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M[i*n + j]);
        }
        printf("\n");
    }
    printf("-----------\n");
}

void MatMulCk(double* A, double *B, double *C, int m, int n, int p){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i*p + j] += A[i*n + k] * B[k*p + j];
            }
        }
    }
}

void checkSVD (int M, 
	int N,
	double* D, 
	double* U, 
	double* SIGMA, 
	double* V_T,
	int SIGMAm, 
	int SIGMAn) {

	double *D_ours = (double *) malloc (SIGMAm * SIGMAn * sizeof(double));
	double *D_intermediate = (double *) malloc (SIGMAm * SIGMAn * sizeof(double));
	double *FULL_SIGMA = (double *) malloc (SIGMAm * SIGMAn * sizeof(double));
	for (int i = 0; i < SIGMAm; i++) {
		for (int j = 0; j < SIGMAn; j++) {
			FULL_SIGMA[i * SIGMAn + j] = (i == j) ? SIGMA[i] : 0;
		}
	}
	MatMulCk (U, FULL_SIGMA, D_intermediate, SIGMAm, SIGMAm, SIGMAn);
	MatMulCk (D_intermediate, V_T, D_ours, SIGMAm, SIGMAn, SIGMAn);

	double epsilon = 1e-2;
	bool incorrect = false;
	if (SIGMAm == M && SIGMAn == N) {
		printf ("Done SVD of D\n");
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (abs(D[i*N+j] - D_ours[i*N+j]) > epsilon) {
					incorrect = true;
				}
			}
		}
	} else {
		printf ("Done SVD of D.T\n");
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (abs(D[i*N+j] - D_ours[j*M+i]) > epsilon) {
					incorrect = true;
				}
			}
		}
	}
	printf ("Correctness status %d\n", incorrect);

	printMatrix (U, SIGMAm, SIGMAm);
	printMatrix (FULL_SIGMA, SIGMAm, SIGMAn);
	printMatrix (V_T, SIGMAn, SIGMAn);
}

void write_result (int M, 
		int N, 
		double* D, 
		double* U, 
		double* SIGMA, 
		double* V_T,
		int SIGMAm, 
		int SIGMAn, 
		int K, 
		double* D_HAT,
		double computation_time){
	// Will contain output code

	printf ("Time taken: %.3f\n", computation_time);
	checkSVD (M, N, D, U, SIGMA, V_T, SIGMAm, SIGMAn);
	// printMatrix (SIGMA, 1, N);
	printf("\n%d %d %d\n\n", M, N, K);
	// printMatrix (D_HAT, M, K);
}
