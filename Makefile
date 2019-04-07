all:
	nvcc -lm main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca
