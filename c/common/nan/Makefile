include $(KPF_SW)/c/builds/definitions.mk
LIB_DIR_NAME = nan
LIB_NAME = lib$(LIB_DIR_NAME)

all: $(LIB_NAME).a $(LIB_NAME)$(SHLIB_SUFFIX)

$(LIB_NAME).a: nanvalue.c nanvalue.h
	$(AR) $(ARFLAGS) $(LIB_NAME).a $<
	$(RANLIB) $(LIB_NAME).a
	if [ -f $(LIB_NAME).a ]; then \
		$(MV) $(LIB_NAME).a $(KPF_LIB_PATH)/$(LIB_NAME).a; \
		$(CP) nanvalue.h \
			$(KPF_INC_PATH)/$(LIB_DIR_NAME)/nanvalue.h; \
	fi;

$(LIB_NAME)$(SHLIB_SUFFIX): nanvalue.o
	${SHLIB_LD} -o $@ nanvalue.o
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
	if [ -f $(KPF_INC_PATH)/$(LIB_DIR_NAME)/nanvalue.h ]; then \
		$(RM) $(KPF_INC_PATH)/$(LIB_DIR_NAME)/nanvalue.h; \
	fi;

