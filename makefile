CC = g++
NVCC = nvcc
LDFLAGS_GPU = -l curand
OBJECTS_GPU=ga_gpu
OBJECTS_CPU=ga_cpu

clean: 
	$(RM) *~ *.o ga_gpu ga_cpu *.out


ga_gpu: $(OBJECTS_GPU).cu
	$(NVCC) $(OBJECTS_GPU).cu -o ga_gpu $(LDFLAGS_GPU)

ga_cpu: $(OBJECTS_CPU).cpp
	$(CC) $(OBJECTS_CPU).cpp -o ga_cpu 

all: $(OBJECTS_GPU) $(OBJECTS_CPU)