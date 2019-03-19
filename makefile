CC=gcc
CCAVX512=icc
CFLAGS=-I.
DEPS = jar_sim.h jar_utils.h jar_type.h
OBJ = demo.o jar_utils.o jar_sim.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf *.o
	rm -rf jar_type.h
	rm -rf demo
	rm -rf demoavx512

jar_type.h : configure_jar.h 
	$(CC) -c generator.c
	$(CC) -o generate generator.o -lm
	./generate

%.o: %.c $(DEPS) 
	$(CC) -c -o $@ $< $(CFLAGS)

demo: $(OBJ) jar_type.h
	$(CC) -o $@ $^ $(CFLAGS) -lm
 
