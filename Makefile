CUDA_HOME   = /Soft/cuda/11.2.1/

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -Wno-deprecated-gpu-targets -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	        = Hist-CPU.exe
OBJ	        = Hist-CPU.o

default: $(EXE)

Hist-CPU.o: Hist-CPU.cu
	$(NVCC) -c -o $@ Hist-CPU.cu $(NVCC_FLAGS) -I/Soft/stb/20200430

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
