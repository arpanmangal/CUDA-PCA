all:
	nvcc -lm -std=c++11 main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca
