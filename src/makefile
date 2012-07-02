OPT := -g -O0

all: clean
	mpicc $(OPT) -c -o lwgrp.o lwgrp.c
	mpicc $(OPT) -c -o lwgrp_util.o lwgrp_util.c
	mpicc $(OPT) -c -o lwgrp_chain_ops.o lwgrp_chain_ops.c
	mpicc $(OPT) -c -o lwgrp_ring_ops.o lwgrp_ring_ops.c
	mpicc $(OPT) -c -o lwgrp_logchain_ops.o lwgrp_logchain_ops.c
	mpicc $(OPT) -c -o lwgrp_logring_ops.o lwgrp_logring_ops.c
	mpicc $(OPT) -c -o lwgrp_comm_split.o lwgrp_comm_split.c
	mpicc $(OPT) -o testsplit testsplit.c *.o

clean:
	rm -f *.o testsplit
