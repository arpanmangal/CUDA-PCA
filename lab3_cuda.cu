#include "lab3_cuda.h"
#include <iostream>

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */

const int BLOCK_SIZE = 16;

// Function declarations
void MatMul (double *dev_A, double *dev_B, double *dev_C, int M, int N, int P);
void MatTranspose (double *dev_A, double *dev_B, int M, int N);
void PrintMatrix (double *A, int M, int N, int GPU=0);
void SimpleMatMul(double* A, double *B, double *C, int m, int n, int p);
void SimpleMatTrans (double *A, double *B, int m, int n);

// void Jacobi (double *S, int N, double *e, double **E);
void Jacobi(int N);
void SortEigenVals (double *SIGMA, double **E_rows, int N);

// JACOBI GLOBALS
#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001
// #define FILENAME "testcase_1000_100"
// #define samples 1000
// #define features 100

double **S; // Symmetric Matrix
double *e; // eigenvalues
double **E; // eigenvectors
int *ind;
bool *changed;
int state;

void SVD_and_PCA (int M, 
        int N, 
        double* D, 
        double** U, 
        double** SIGMA, 
        double** V_T, 
        double** D_HAT, 
        int *K,
        int retention) 
{
    double* Dt = (double *) malloc (sizeof(double) * N * M);
    SimpleMatTrans (D, Dt, M, N);

    double *DtD = (double *) malloc (sizeof(double) * N * N);
    SimpleMatMul (Dt, D, DtD, N, M, N);

    // PrintMatrix (Dt, N, M);
    // PrintMatrix (D, M, N);
    // PrintMatrix (DtD, N, N);
    // printf("\n");

    S = (double **) malloc (sizeof(double*) * N);
    for (int i = 0; i < N; i++) {
        S[i] = (double *) malloc (sizeof(double) * N);
        for (int j = 0; j < N; j++) {
            S[i][j] = DtD[i*N + j];
        }
    }
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf ("%.2f ", S[i][j]);
    //     }
    //     printf("\n");
    // }

    Jacobi (N);
    for (int i = 0; i < N; i++) {
        printf ("%f ", e[i]);
    }
    printf ("\n");

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf ("%.3f ", E[i][j]);
    //     }
    //     printf("\n");
    // }

    printf("-----------\n");
    double **Et = (double **) malloc (sizeof(double*) * N);
    for (int i = 0; i < N; i++) {
        Et[i] = (double *) malloc (sizeof(double) * N);
        for (int j = 0; j < N; j++) {
            Et[i][j] = DtD[i*N + j];
        }
    }
    SortEigenVals (*SIGMA, Et, N);
    for (int i = 0; i < N; i++) {
        printf ("%.3f ", (*SIGMA)[i]);
    }
    printf ("\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf ("%.3f ", Et[i][j]);
    //     }
    //     printf("\n");
    // }

    // Computing the SVD of D.T
    // Vt = SIGMA-1.T * E.T * D.T
    // U = E
    double *ET = (double *) malloc (sizeof(double) * N * N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ET[i*N+j] = Et[i][j];
        }
    }
    SimpleMatTrans (ET, *U, N, N);

    double *SIGMAINV = (double *) malloc (sizeof(double *) * M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            SIGMAINV[i*N+j] = (i == j) * (1 / (*SIGMA)[j]);
        }
    }

    double *ETDT = (double *) malloc (sizeof(double) * N * M);
    SimpleMatMul (ET, Dt, ETDT, N, N, M);

    SimpleMatMul (SIGMAINV, ETDT, *V_T, M, N, M);

    // printf("-----------\n");
    // PrintMatrix (*U, N, N);
    printf("-----------\n");
    PrintMatrix (*SIGMA, 1, N);
    printf("-----------\n");
    // PrintMatrix (*V_T, M, M);

    free (ET);
    free (SIGMAINV);
    free (ETDT);


    printf ("hey\n");
    // PCA COMPUTATION
    double totSigma = 0.0;
    for (int i = 0; i < N; i++) {
        totSigma += (*SIGMA)[i] * (*SIGMA)[i];
    }

    printf ("%f\n", totSigma);

    double cumSigma = 0.0;
    for (int i = 0; i < N; i++) {
        cumSigma += (*SIGMA)[i] * (*SIGMA)[i];
        printf ("%f\n", cumSigma);
        if (cumSigma / totSigma >= retention / 100.0) {
            *K = i + 1;
            break;
        }
    }

    printf ("%d\n", *K);

    double *W = (double *) malloc (sizeof(double) * N * (*K));
    *D_HAT = (double *) malloc (sizeof(double) * M * (*K));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < *K; j++) {
            W[i * (*K) + j] = (*U)[i * N + j];
        }
    }
    PrintMatrix (W, N, *K);
    SimpleMatMul (D, W, *D_HAT, M, N, *K);
    free (W);

    /*
    // Allocate the Memory on the device
    double *dev_D, *dev_Dt, *dev_U, *dev_SIGMA, *dev_V_T;
    cudaMalloc ((void **) &dev_D, M*N*sizeof(double));
    cudaMalloc ((void **) &dev_Dt, N*M*sizeof(double));
    // cudaMalloc ((void **) &dev_U, N*N*sizeof(double));
    // cudaMalloc ((void **) &dev_SIGMA, N*sizeof(double));
    // cudaMalloc ((void **) &dev_V_T, M*M*sizeof(double));

    cudaMemcpy (dev_D,     D,      M*N*sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy (dev_U,     *U,     N*N*sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy (dev_SIGMA, *SIGMA, N*sizeof(double),   cudaMemcpyHostToDevice);
    // cudaMemcpy (dev_V_T,   *V_T,   M*M*sizeof(double), cudaMemcpyHostToDevice);

    MatTranspose (dev_D, dev_Dt, M, N);
    // // PrintMatrix (D, M, N);
    // PrintMatrix (dev_D, M, N, 1);
    // printf ("\n");
    // PrintMatrix (dev_Dt, N, M, 1);

    // MatMul (dev_D, dev_Dt, dev_V_T, M, N, M);
    // PrintMatrix (dev_V_T, M, M, 1);
    // Call the SVD Kernel
    // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid(N / dimBlock.x, M / dimBlock.y);
    // SVDkernel<<<dimGrid, dimBlock>>> (M, N, dev_D, dev_U, dev_SIGMA, dev_V_T);


    // Clean-up
    // cudaMemcpy (*U,     dev_U,     N*N*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy (*SIGMA, dev_SIGMA, N*sizeof(double),   cudaMemcpyDeviceToHost);
    // cudaMemcpy (*V_T,   dev_V_T,   M*M*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_D);
    cudaFree(dev_Dt);
    // cudaFree(dev_U);
    // cudaFree(dev_SIGMA);
    // cudaFree(dev_V_T);

    */
}

// Jacobi Stuff
void init_jacobi(int N);
int maxind (int k, int N);
void update (int k, double t);
void rotate(int k, int l, int i, int j, double c, double s, bool eigenvectors);
double** mat_mul(double** A, int Am, int An, 
    double** B, int Bm, int Bn);
void Jacobi(int N) {
    // N = n;
    // S = input_matrix;

    init_jacobi(N);

    while(state != 0){
        int m = 0;

        for (int k=1; k<N-1; k++){
            if (fabs(S[k][ind[k]]) > fabs(S[m][ind[m]])){
                m = k;
            }
        }

        int k = m;
        int l = ind[m];
        double p = S[k][l];
        double y = (e[l] - e[k]) / 2.0;
        double d = fabs(y) + sqrt(p*p + y*y);
        double r = sqrt(p*p + d*d);
        double c = d / r;
        double s = p / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k][l] = 0.0;
        update(k, -t);
        update(l, t);

        for (int i=0; i<k; i++)  { rotate(i, k, i, l, c, s, false); }
        for (int i=k+1; i<l; i++){ rotate(k, i, i, l, c, s, false); }
        for (int i=l+1; i<N; i++)  { rotate(k, i, l, i, c, s, false); }

        for (int i=0; i<N; i++){
            rotate(k, l, i, i, c, s, true);
        }

        ind[k] = maxind(k, N);
        ind[l] = maxind(l, N);
    }
}

void init_jacobi(int N) {
    E = (double**) malloc(__SIZEOF_POINTER__*N);
    for (int i=0; i<N; i++){
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N);
        for (int j=0; j<N; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }

    state = N;

    e = (double*)malloc(__SIZEOF_DOUBLE__*N);
    ind = (int*)malloc(__SIZEOF_INT__*N);
    changed = (bool*)malloc(sizeof(bool)*N);

    for (int k=0; k<N; k++){
        ind[k]     = maxind(k, N);
        e[k]       = S[k][k];
        changed[k] = true;
    }
}

int maxind(int k, int N) {
    int m = k+1;
    for (int i = k+2; i < N; i++){
        if (fabs(S[k][i]) > fabs(S[k][m])){
            m = i;
        }
    }
    return m;
}

void update(int k, double t) {
    double ek_prev = e[k];
    e[k] = ek_prev + t;

    if (e[k] < 0) e[k] = 0;

    if (changed[k] && (ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        changed[k] = false;
        state = state - 1;
    }
    else if ((! changed[k]) && (ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        changed[k] = true;
        state = state + 1;
    }
}

void rotate(int k, int l, int i, int j, double c, double s, bool eigenvectors){
    double** mat1;
    double** mat2;
    double** mat3;

    mat1 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat1[0] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[1] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[0][0] = c; mat1[0][1] = -s;
    mat1[1][0] = s; mat1[1][1] = c;

    mat2 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat2[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat2[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    if (eigenvectors){
        mat2[0][0] = E[i][k];
        mat2[1][0] = E[i][l];
    }
    else {
        mat2[0][0] = S[k][l];
        mat2[1][0] = S[i][j];
    }

    mat3 = mat_mul(mat1, 2, 2, mat2, 2, 1);

    if (eigenvectors){
        E[i][k] = mat3[0][0];
        E[i][l] = mat3[1][0];
    }
    else{
        S[k][l] = mat3[0][0];
        S[i][j] = mat3[1][0];
    }

    free(mat1[0]);
    free(mat1[1]);
    free(mat1);
    free(mat2[0]);
    free(mat2[1]);
    free(mat2);
    free(mat3[0]);
    free(mat3[1]);
    free(mat3);
}

double** mat_mul(double** A, int Am, int An, 
                 double** B, int Bm, int Bn){
    double **C;
    C = (double**)malloc(__SIZEOF_POINTER__*Am);
    for (int i=0; i<Am; i++)
        C[i] = (double*)malloc(__SIZEOF_DOUBLE__*Bn);

    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i][j] = 0;
            for (int k=0; k<An; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

struct pair {
    double e;
    int idx;
};

void OddEvenSort (struct pair* A, int N)
{
    int exch = 1, start = 0;
    while (exch || start) {
        exch = 0;
        for (int i = start; i < N - 1; i += 2) {
            if (A[i].e < A[i+1].e) {
                // Swap them
                double tmp = A[i].e;
                A[i].e = A[i+1].e;
                A[i+1].e = tmp;

                int tmpIdx = A[i].idx;
                A[i].idx = A[i+1].idx;
                A[i+1].idx = tmpIdx;

                exch = 1;
            }
        }
        if (start == 0) start = 1;
        else start = 0;
    }
}

void SortEigenVals (double *SIGMA, double **E_rows, int N) {
    struct pair *EigenVals = (struct pair *) malloc (sizeof(struct pair) * N);
    for (int i = 0; i < N; i++) {
        EigenVals[i].e = sqrt(abs(e[i]));
        EigenVals[i].idx = i;
    }
    OddEvenSort (EigenVals, N);

    for (int i = 0; i < N; i++) {
        SIGMA[i] = EigenVals[i].e;
        int r = EigenVals[i].idx;
        for (int j = 0; j < N; j++) {
            E_rows[i][j] = E[j][r];
        }
    }
}


// CUDA Stuff
// Kernel declarations
__global__ void MatMulKernel (double *A, double *B, double *C, int M, int N, int P);
__global__ void MatTransKernel (double *A, double *B, int M, int N);
__global__ void PrintMatrixKernel (double *A, int M, int N);

void SimpleMatMul(double* A, double *B, double *C, int m, int n, int p){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i*p + j] += A[i*n + k] * B[k*p + j];
            }
        }
    }
}

void SimpleMatTrans (double *A, double *B, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            B[j*m + i] = A[i*n + j];
}

void MatMul (double *dev_A, double *dev_B, double *dev_C, int M, int N, int P)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(P / dimBlock.x + 1, M / dimBlock.y + 1);
    MatMulKernel<<<dimGrid, dimBlock>>> (dev_A, dev_B, dev_C, M, N, P);
}

void MatTranspose (double *dev_A, double *dev_B, int M, int N)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x + 1, M / dimBlock.y + 1);
    MatTransKernel<<<dimGrid, dimBlock>>> (dev_A, dev_B, M, N);
}

void PrintMatrix (double *A, int M, int N, int GPU) {
    if (GPU == 0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf ("%f ", A[i*N + j]);
            }
            printf("\n");
        }
    } else {
        PrintMatrixKernel<<<1,1>>> (A, M, N);
    }
}

__global__ void PrintMatrixKernel (double *A, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf ("%f ", A[i*N + j]);
        }
        printf("\n");
    }
}

__global__ void MatMulKernel (double *A, double *B, double *C, int M, int N, int P) {
    // Block row and column
    int I = blockIdx.y;
    int J = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    double Cval = 0.0;

    // Shared memory
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    int i = I * BLOCK_SIZE + row;
    int j = J * BLOCK_SIZE + col;
    for (int K = 0; K <= (N / BLOCK_SIZE); K++) {
        // Load the shared matrices
        int k = K * BLOCK_SIZE;
        As[row][col] = (i < M && (k+col) < N) ? A[N*i + (k + col)] : 0;
        Bs[row][col] = ((k+row) < N && j < P) ? B[P*(k+row) + j]   : 0;

        // Ensure loading
        __syncthreads();

        // Multiply the respective row and col of Asub and Bsub
        for (int e = 0; e < BLOCK_SIZE; e++) {
            Cval += As[row][e] * Bs[e][col];
        }

        // Complete this block
        __syncthreads();
    }

    // Update C matrix

    if (i < M && j < P)
        C[P*i + j] = Cval;
}

__global__ void MatTransKernel (double *A, double *B, int M, int N)
{
    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;
    if (I < M && J < N)
        B[M*J + I] = A[N*I + J];
}