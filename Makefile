NVCC = nvcc
CC = gcc

LIB_HOME = .
LIBS = -L$(LIB_HOME)/lib64
INCLUDE = -Isrc

MAIN = transpose.cu

BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/gpu_transpose

ifndef TILE_DIM
TILE_DIM = 32
endif

ifndef MATRIX_SIZE
MATRIX_SIZE = 1024
endif

ifndef DATA_TYPE
DATA_TYPE = float
else
$(info DATA_TYPE is $(DATA_TYPE))
endif

ifeq ($(DATA_TYPE), float)
DATA_TYPE_FLAG = -DDATA_TYPE_FLOAT
else ifeq ($(DATA_TYPE), double)
DATA_TYPE_FLAG = -DDATA_TYPE_DOUBLE
else
DATA_TYPE_FLAG = -DDATA_TYPE_FLOAT
DATA_TYPE=float
endif


ifeq ($(DATA_TYPE), int)
FORMAT_SPECIFIER = %d
else
FORMAT_SPECIFIER = %f
endif


$(TARGETDIR)/gpu_transpose: ${MAIN}
	mkdir -p $(@D)
	$(NVCC) $< -lcublas -rdc=true -DDATA_TYPE=$(DATA_TYPE) -DFORMAT_SPECIFIER='"$(FORMAT_SPECIFIER)"' -DTILE_DIM=$(TILE_DIM) -DMATRIX_SIZE=$(MATRIX_SIZE) $(DATA_TYPE_FLAG) --use_fast_math -o $@ $(INCLUDE) $(LIBS)


clean:
	rm -rf $(BUILDDIR)/*.o $(TARGETDIR)/*