NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_75 -Xcompiler -Wall -Xcompiler -Wextra -m64

histo: main.cu reference_calc.o student.o Makefile
	nvcc -o HW5 main.cu reference_calc.o student.o radixSort.o compactArray.o $(NVCC_OPTS)

student.o: student.cu utils.h radixSort.o compactArray.o
	nvcc -c student.cu $(NVCC_OPTS)

radixSort.o: radixSort.cu utils.h
	nvcc -c radixSort.cu $(NVCC_OPTS)

compactArray.o: compactArray.cu utils.h
	nvcc -c compactArray.cu $(NVCC_OPTS)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o hw *.bin
