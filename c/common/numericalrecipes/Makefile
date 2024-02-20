include $(KPF_SW)/c/builds/definitions.mk
CPPFLAGS = -I $(KPF_INC_PATH)/nan
LIB_DIR_NAME = numericalrecipes
LIB_NAME = lib$(LIB_DIR_NAME)
LINKFLAGS = -L$(KPF_LIB_PATH) -lnan -lm

OBJECTS = computesum.o computeweightedsum.o computemedian.o computeaverage.o computeclippedmean.o computescale.o computestddev.o computeskew.o computekurtosis.o computejarquebera.o computepercentile.o selectkthvalue.o swapvaluesinarray.o

all: $(LIB_NAME).a $(LIB_NAME)$(SHLIB_SUFFIX)

$(LIB_NAME).a: numericalrecipes.h $(OBJECTS) $(KPF_LIB_PATH)/libnan.a 
	$(AR) $(ARFLAGS) $(LIB_NAME).a $(OBJECTS) $(KPF_LIB_PATH)/libnan.a 
	$(RANLIB) $(LIB_NAME).a
	if [ -f $(LIB_NAME).a ]; then \
		$(MV) $(LIB_NAME).a $(KPF_LIB_PATH)/$(LIB_NAME).a; \
		$(CP) numericalrecipes.h \
			$(KPF_INC_PATH)/$(LIB_DIR_NAME)/numericalrecipes.h; \
	fi;

$(LIB_NAME)$(SHLIB_SUFFIX): $(OBJECTS)  
	${SHLIB_LD} ${LINKFLAGS} -o $@ $(OBJECTS)  
	if [ -f $(LIB_NAME)$(SHLIB_SUFFIX) ]; then \
		$(MV) $(LIB_NAME)$(SHLIB_SUFFIX) $(KPF_LIB_PATH); \
	fi;

%.o: %.c
	$(COMPILE.c) $<

.PHONY: clean
clean:
	$(RM) *.o
	if [ -f $(KPF_LIB_PATH)/$(LIB_NAME).a ]; then \
		$(RM) $(KPF_LIB_PATH)/$(LIB_NAME).a; \
	fi;
	if [ -f $(KPF_LIB_PATH)/$(LIB_NAME)$(SHLIB_SUFFIX) ]; then \
		$(RM) $(KPF_LIB_PATH)/$(LIB_NAME)$(SHLIB_SUFFIX); \
	fi;
	if [ -f $(KPF_INC_PATH)/$(LIB_DIR_NAME)/numericalrecipes.h ]; then \
		$(RM) $(KPF_INC_PATH)/$(LIB_DIR_NAME)/numericalrecipes.h; \
	fi;

