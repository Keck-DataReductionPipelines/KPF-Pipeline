include $(KPF_SW)/c/builds/definitions.mk
BIN_NAME = makeTestFitsFile

COMPILE.c = $(CC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
LINK.c = $(CC) $(LINKFLAGS)
CC = g++
CFLAGS = -fPIC

CPPFLAGS = -I $(KPF_INC_PATH) -I $(KPF_INC_PATH)/cfitsio -I $(KPF_INC_PATH)/nan
LINKFLAGS = -lm -lgsl -lgslcblas -lcfitsio -lnan -L$(KPF_LIB_PATH)  

OBJECTS = makeTestFitsFile.o 

all: $(BIN_NAME)

$(BIN_NAME): $(OBJECTS)
	$(LINK.c) $(OUTPUT_OPTION) $(OBJECTS)
	if [ -f $(BIN_NAME) ]; then \
		$(MV) $(BIN_NAME) $(KPF_BIN_PATH); \
	fi;

.PHONY: clean
clean:
	rm *.o
	if [ -f $(KPF_BIN_PATH)/$(BIN_NAME) ]; then \
		$(RM) $(KPF_BIN_PATH)/$(BIN_NAME); \
	fi;

%.o: %.c
	$(COMPILE.c) $<

