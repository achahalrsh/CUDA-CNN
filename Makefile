CUDA_PATH ?= /usr/local/cuda
SRC = *.cu

all::cnn_hardware cnn_sim

cnn_hardware: $(SRC)
	nvcc -o $@ $^ -lcuda -lcublas 

cnn_sim: $(SRC)
	nvcc -o $@ $^ -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lculibos 

clean:
	rm cnn_hardware cnn_sim
