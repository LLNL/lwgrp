MPICC = mpicc
OPT := -g -O0

OBJS = \
	lwgrp.o \
	lwgrp_util.o \
	lwgrp_chain_ops.o \
	lwgrp_ring_ops.o \
	lwgrp_logchain_ops.o \
	lwgrp_logring_ops.o \
	lwgrp_comm_split.o

all: clean liblwgrp.a testsplit

liblwgrp.a: $(OBJS)
	ar r $@ $(OBJS)
	ranlib $@

%.o: %.c *.h
	$(MPICC) $(OPT) $(CFLAGS) -c $< -o $@ -I.

testsplit: testsplit.c *h liblwgrp.a
	$(MPICC) $(OPT) -o testsplit testsplit.c liblwgrp.a

clean:
	rm -f *.o liblwgrp.a testsplit
