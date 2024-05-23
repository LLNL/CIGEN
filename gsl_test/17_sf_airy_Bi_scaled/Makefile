ifndef $(CC)
CC = clang
endif

ifndef (LLVM_PASS)
LLVM_PASS = 0
endif

ifndef $(OPT_LEVEL)
OPT_LEVEL = 0
endif

ifndef $(FASTMATH)
FASTMATH = 0
endif

OPT = -O0

ifeq ($(OPT_LEVEL), 0)
OPT = -O0
endif

ifeq ($(OPT_LEVEL), 1)
OPT = -O1
endif

ifeq ($(OPT_LEVEL), 2)
OPT = -O2
endif

ifeq ($(OPT_LEVEL), 3)
OPT = -O3
endif

ifeq ($(FASTMATH), 1)
ifeq ($(CC), nvcc)
OPT += -use_fast_math
else
OPT += -ffast-math
endif
endif

ifndef $(LIBS)
LIBS += -static -lgsl_$(OPT_LEVEL)_$(FASTMATH)_$(LLVM_PASS) -L../ -lm 
endif

CFLAGS = -I/usr/local/include -I../ -I./ -I../../../gsl-2.7.1
LDFLAGS = $(LIBS)

ifeq ($(GLOBAL_TEXT_OUTPUT), 1)
CFLAGS += -DTEXT_OUTPUT
endif

ifeq ($(LLVM_PASS), 1)
CFLAGS += -fpass-plugin=libTraceDiffPass.so
endif

default: main

mk_workspace:
	mkdir -p workspace
	mkdir -p workspace/func_analysis
	mkdir -p workspace/original_files

run:
	cat input.txt | ./test `xargs`

main: mk_workspace
	$(CC) test.c -o test $(OPT) $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf *.o test *.out
	rm -rf workspace 0*/ output.map