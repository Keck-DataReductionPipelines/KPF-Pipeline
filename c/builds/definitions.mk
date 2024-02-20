KPF_BIN_PATH = $(KPF_SW)/c/bin
KPF_LIB_PATH = $(KPF_SW)/c/lib
KPF_INC_PATH = $(KPF_SW)/c/include
COMPILE.c = $(CC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
LINK.c = $(CC) $(LINKFLAGS)
CC = gcc -O2
CFLAGS = -fPIC -std=c99
OUTPUT_OPTION = -o $@
AR = ar
ARFLAGS = rv
RANLIB = ranlib
RM = rm
CP = cp
MV = mv
ifdef DYLD_LIBRARY_PATH
   SHLIB_SUFFIX =  .dylib 
   SHLIB_LD = gcc -dynamiclib
   SHLIB_LD_ALT = gcc
else
   SHLIB_SUFFIX =  .so
   SHLIB_LD = gcc -shared
   SHLIB_LD_ALT = gcc -shared
endif
