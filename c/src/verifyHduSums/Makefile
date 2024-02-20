include $(KPF_SW)/c/builds/definitions.mk
BIN_NAME = verifyHduSums
CPPFLAGS = -I $(KPF_INC_PATH)/cfitsio
LINKFLAGS = -lcfitsio -lm -L$(KPF_LIB_PATH)

OBJECTS = verifyHduSums.o

all: $(BIN_NAME)

$(BIN_NAME): $(OBJECTS)
	$(LINK.c) $(OUTPUT_OPTION) $(OBJECTS)
	if [ -f $(BIN_NAME) ]; then \
		$(MV) $(BIN_NAME) $(KPF_BIN_PATH); \
	fi;

.PHONY: clean
clean:
	$(RM) *.o
	if [ -f $(KPF_BIN_PATH)/$(BIN_NAME) ]; then \
		$(RM) $(KPF_BIN_PATH)/$(BIN_NAME); \
	fi;

%.o: %.c
	$(COMPILE.c) $<


