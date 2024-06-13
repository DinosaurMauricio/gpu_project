NVCC = nvcc
CC = gcc

LIB_HOME = .
LIBS = -L$(LIB_HOME)/lib64
INCLUDE = -Isrc

MAIN = collab_groups.cu

BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/gpu_transpose

ifndef TILE_DIM
TILE_DIM = 32
endif

ifndef BLOCK_ROWS
BLOCK_ROWS = 8
endif


ifndef DATA_TYPE
DATA_TYPE = int
else
$(info DATA_TYPE is $(DATA_TYPE))
endif

ifeq ($(DATA_TYPE), int)
FORMAT_SPECIFIER = %d
else
FORMAT_SPECIFIER = %f
endif

ifndef UNROLL_FLAG
UNROLL_FLAG = false
endif

OBJECTS = $(BUILDDIR)/my_library.o

$(TARGETDIR)/gpu_transpose: ${MAIN}  $(OBJECTS)
	mkdir -p $(@D)
	$(NVCC) $< $(OBJECTS) -lcublas -DDATA_TYPE=$(DATA_TYPE) -DFORMAT_SPECIFIER='"$(FORMAT_SPECIFIER)"' -DTILE_DIM=$(TILE_DIM) -DBLOCK_ROWS=$(BLOCK_ROWS) -DUNROLL_FLAG=$(UNROLL_FLAG) --use_fast_math -o $@ $(INCLUDE) $(LIBS) $(if $(BANDWIDTH_PERFORMANCE),-Xptxas -$(BANDWIDTH_PERFORMANCE))

$(BUILDDIR)/my_library.o: my_library.c
	mkdir -p $(BUILDDIR) $(TARGETDIR)
	$(CC) -c -DDATA_TYPE=$(DATA_TYPE) -DFORMAT_SPECIFIER='"$(FORMAT_SPECIFIER)"' -o $@ $(INCLUDE) my_library.c $(if $(BANDWIDTH_PERFORMANCE),-$(BANDWIDTH_PERFORMANCE))


clean:
	rm -rf $(BUILDDIR)/*.o $(TARGETDIR)/*