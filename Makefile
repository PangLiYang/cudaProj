NVCC = nvcc
CFLAGS = -arch=sm_70

# Project files
SRCS = main.cu maxk_forward.cu
OBJS = $(SRCS:.cu=.o)
EXEC = test

# Default target
all: $(EXEC)

# Link the object files to create the executable
$(EXEC): $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^

# Compile each .cu file into an object file
%.o: %.cu data.h
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean target to remove build artifacts
clean:
	rm -f $(OBJS) $(EXEC)