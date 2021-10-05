CC = gcc
CFLAGS = -g -Wall -Wstrict-prototypes -pedantic  -ansi

CCF_3d_c: CCF_3d_c.o
	$(CC) CCF_3d_c.o -o CCF_3d_c

CCF_3d_c.o: CCF_3d_c.c
	$(CC) $(CFLAGS) -c CCF_3d_c.c

C: CCF_3d_c.c
	gcc -o CCF_3d_c.so -shared -fPIC -O2 CCF_3d_c.c

clean:
	rm -f CCF_3d_c *.o *.so

