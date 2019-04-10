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

void write_result (int M, 
		int N, 
		double* D, 
		double* U, 
		double* SIGMA, 
		double* V_T,
		int K, 
		double* D_HAT,
		double computation_time){
	// Will contain output code
	// printMatrix (U, N, N);
    printMatrix (SIGMA, 1, N);
	printf("\n%d %d %d\n\n", M, N, K);
	printMatrix (D_HAT, M, K);
}
