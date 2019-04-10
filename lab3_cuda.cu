#include "lab3_cuda.h"

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */

const int BLOCK_SIZE = 16;

__global__ void SVDkernel (int M, int N, double *D, double *U, double *SIGMA, double *V_T)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double *A = (double *) malloc (100*100*sizeof(double));
}

// Function declarations
void MatMul (double *dev_A, double *dev_B, double *dev_C, int m, int n, int p);
void PrintMatrix (double *A, int M, int N);

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
    // Allocate the Memory on the device
    double *dev_D, *dev_Dt, *dev_U, *dev_SIGMA, *dev_V_T;
    cudaMalloc ((void **) &dev_D, M*N*sizeof(double));
    cudaMalloc ((void **) &dev_Dt, N*M*sizeof(double));
    cudaMalloc ((void **) &dev_U, N*N*sizeof(double));
    cudaMalloc ((void **) &dev_SIGMA, N*sizeof(double));
    cudaMalloc ((void **) &dev_V_T, M*M*sizeof(double));

    cudaMemcpy (dev_D,     D,      M*N*sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy (dev_U,     *U,     N*N*sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy (dev_SIGMA, *SIGMA, N*sizeof(double),   cudaMemcpyHostToDevice);
    // cudaMemcpy (dev_V_T,   *V_T,   M*M*sizeof(double), cudaMemcpyHostToDevice);

    PrintMatrix (D, M, N);
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
    cudaFree(dev_U);
    cudaFree(dev_SIGMA);
    cudaFree(dev_V_T);

}


// Kernel declarations
__global__ void MatMulKernel (double *A, double *B, double *C, int M, int N, int P);


void MatMul (double *dev_A, double *dev_B, double *dev_C, int M, int N, int P)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(P / dimBlock.x + 1, M / dimBlock.y + 1);
    MatMulKernel<<<dimGrid, dimBlock>>> (dev_A, dev_B, dev_C, M, N, P);
}

void PrintMatrix (double *A, int M, int N) {
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