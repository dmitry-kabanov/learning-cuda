nvcc = /opt/cuda/bin/nvcc
includes = /opt/cuda/include

srcs = $(wildcard *.cu)
exes = $(patsubst %.cu,%.exe,$(srcs))

all: $(exes)

%.exe: %.cu
	$(nvcc) -I$(includes) $< -o $@
