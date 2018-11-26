CC = gcc-8
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: assignment3 assignment3_2 assignment3_3

assignment3: main.c linked-list.c
	$(CC) $(CFLAGS) $< -o $@

assignment3_2: main-2.c linked-list-2.c
	$(CC) $(CFLAGS) $< -o $@

assignment3_3: main-3.c linked-list-3.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f assignment3*
