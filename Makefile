IDIR=./src
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda --std=c++17 -O2

.PHONY: all build run clean

all: clean build run

build: src/signal_processor.cu src/signal_processor.h
	$(COMPILER) $(COMPILER_FLAGS) src/signal_processor.cu -o signal_processor.exe

run: build
	./run.sh

clean:
	rm -f signal_processor.exe
	rm -rf data/input/*.csv data/output/*.csv logs/*.log
