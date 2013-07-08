/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-568372.
 * All rights reserved.
 * This file is part of the LWGRP library.
 * For details, see https://github.com/hpc/lwgrp
 * Please also read this file: LICENSE.TXT. */

#ifndef _LWGRP_H
#define _LWGRP_H

/* TODO: move collectives to their own library, like lwgrp_colls,
 * and just provide simple get/set routines here to construct
 * groups and acquire member info:
 *
 * lwgrp_chain_set(chain, rank, size, left, right)
 * lwgrp_chain_get(chain, rank, size, left, right)
 * lwgrp_chain_get_size_rank(chain, size, rank)
 * lwgrp_chain_get_left_right(chain, left, right)
 *
 * size = lwgrp_logchain_get_left_size(logchain)
 * left = lwgrp_lochain_get_left(logchain, index)
 *
 * pt2pt library
 *   test whether address is NULL
 *   send(void* addr, buf, dt, req, flag_onetime)
 *   recv(void* addr, buf, dr, req)
 *   wait(req)
 * */

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LWGRP_SUCCESS (0)

extern int LWGRP_MSG_TAG_0;

/* We represent groups of processes using a doubly-linked list called
 * a "chain".  This is a very simple struct that records the number
 * of processes in the group, the rank of the local process within the
 * group, and the address of the local process and of the processes
 * having ranks one less (left) and one more (right) than the local
 * process.  We implement this version of the chain on MPI, so for
 * addresses we record a parent communicator and ranks within that
 * communicator.  To be lightweight, the reference to the communicator
 * is a literal copy of the handle value, not a full dup. */

typedef struct lwgrp_chain {
  MPI_Comm comm;  /* communicator to send messages to procs in group */
  int comm_rank;  /* address (rank) of current process in communicator */
  int comm_left;  /* address (rank) of process one less than current */
  int comm_right; /* address (rank) of process one more than current */
  int group_size; /* number of processes in our group */
  int group_rank; /* our rank within the group [0,group_size) */
} lwgrp_chain;

/* We define a "ring", which is just a chain where the endpoints wrap,
 * so we can use the same structure, but we define a different type
 * in case we want to extend this in the future. */

typedef lwgrp_chain lwgrp_ring;

/* A logchain is a data structure that records the number and addresses
 * of processes that are 2^d ranks away from the local process to the
 * left and right sides, for d = 0 to d = ceiling(log N)-1.
 *
 * When multiple collectives are to be issued on a given chain, one
 * can construct and cache the logchain as an optimization.
 *
 * A logchain can be constructed by executing a collective operation on
 * a chain, or it can be filled in locally given a communicator as the
 * initial group.  It must often be used in conjunction with a chain
 * in communication operations. */

/* structure to associate a skip list of process ranks for a group */
typedef struct lwgrp_logchain {
  int  left_size;  /* number of elements in our left list */
  int  right_size; /* number of elements in our right list */
  int* left_list;  /* process addresses for 2^d hops to the left */
  int* right_list; /* process addresses for 2^d hops to the right */
} lwgrp_logchain;

/* again, we define a ring variant of the logchain */
typedef lwgrp_logchain lwgrp_logring;

/* ---------------------------------
 * Methods to create and free chains
 * --------------------------------- */

int lwgrp_chain_set_null(lwgrp_chain* chain);
int lwgrp_chain_copy(const lwgrp_chain* in, lwgrp_chain* out);
int lwgrp_chain_build_from_ring(const lwgrp_ring* ring, lwgrp_chain* chain);
int lwgrp_chain_build_from_mpicomm(MPI_Comm comm, lwgrp_chain* chain);
int lwgrp_chain_build_from_vals(MPI_Comm comm, int left, int right, int size, int rank, lwgrp_chain* chain);
int lwgrp_chain_free(lwgrp_chain* chain);

/* ---------------------------------
 * Methods to create and free rings
 * --------------------------------- */

int lwgrp_ring_set_null(lwgrp_ring* ring);
int lwgrp_ring_copy(const lwgrp_ring* in, lwgrp_ring* out);
int lwgrp_ring_build_from_chain(const lwgrp_chain* chain, lwgrp_ring* ring);
int lwgrp_ring_build_from_mpicomm(MPI_Comm comm, lwgrp_ring* ring);
int lwgrp_ring_build_from_list(MPI_Comm comm, int size, const int ranklist[], lwgrp_ring* ring);
int lwgrp_ring_free(lwgrp_ring* ring);

/* ---------------------------------
 * Methods to create and free logchains
 * --------------------------------- */

int lwgrp_logchain_build_from_chain(const lwgrp_chain* chain, lwgrp_logchain* list);
int lwgrp_logchain_build_from_logring(const lwgrp_ring* ring, const lwgrp_logring* logring, lwgrp_logchain* list);
int lwgrp_logchain_build_from_mpicomm(MPI_Comm comm, lwgrp_logchain* list);
int lwgrp_logchain_free(lwgrp_logchain* list);

/* ---------------------------------
 * Methods to create and free logrings
 * --------------------------------- */

int lwgrp_logring_build_from_ring(const lwgrp_ring* ring, lwgrp_logring* list);
int lwgrp_logring_build_from_mpicomm(MPI_Comm comm, lwgrp_logring* list);
int lwgrp_logring_build_from_list(MPI_Comm, int size, const int ranklist[], lwgrp_logring* list);
int lwgrp_logring_free(lwgrp_logring* list);

/* ---------------------------------
 * Collectives using chains
 * --------------------------------- */

/* given a specified number of bins, an index into those bins, and a
 * input group, create and return a new group consisting of all ranks
 * belonging to the same bin, my_bin should be in range [0,bins-1]
 * if my_bin < 0, then the empty group is returned */
int lwgrp_chain_split_bin_scan(
  int bins,                /* IN  - number of bins (non-negative integer) */
  int bin,                 /* IN  - bin to which calling proc belongs (integer) */
  const lwgrp_chain* in,   /* IN  - group to be split (handle) */
  lwgrp_chain* out         /* OUT - group containing all procs in same
                            *       bin as calling process (handle) */
);

/* execute a barrier over the chain */
int lwgrp_chain_barrier_dissemination(
  const lwgrp_chain* group /* IN  - group (handle) */
);

#if (MPI_VERSION == 2 && MPI_SUBVERSION >= 2) || (MPI_VERSION >= 3)

int lwgrp_chain_double_exscan_recursive(
  const void* sendleft,    /* IN  - input buffer for right-to-left exscan */
  void* recvright,         /* OUT - output buffer for right-to-left exscan */
  const void* sendright,   /* IN  - input buffer for left-to-right excan */
  void* recvleft,          /* OUT - output buffer for left-to-right exscan */
  int count,               /* IN  - number of elements in buffer
                            *       (non-negative integer) */
  MPI_Datatype type,       /* IN  - buffer datatype (handle) */
  MPI_Op op,               /* IN  - reduction operation (handle) */
  const lwgrp_chain* group /* IN  - group (handle) */
);

/* execute an allreduce over the chain */
int lwgrp_chain_allreduce_recursive(
  const void* inbuf,       /* IN  - input buffer for reduction */
  void* outbuf,            /* OUT - output buffer for reduction */
  int count,               /* IN  - number of elements in buffer
                            *       (non-negative integer) */
  MPI_Datatype type,       /* IN  - buffer datatype (handle) */
  MPI_Op op,               /* IN  - reduction operation (handle) */
  const lwgrp_chain* group /* IN  - group (handle) */
);

#endif /* MPI >= v2.2 */

/* executes an allgather-like operation of a single integer */
int lwgrp_chain_allgather_brucks_int(
  int sendint,
  int recvbuf[],
  const lwgrp_chain* group
);

/* ---------------------------------
 * Collectives using logchains
 * --------------------------------- */

#if (MPI_VERSION == 2 && MPI_SUBVERSION >= 2) || (MPI_VERSION >= 3)

int lwgrp_logchain_reduce_recursive(
  const void* inbuf,         /* IN  - input buffer for reduction */
  void* outbuf,              /* OUT - output buffer for reduction */
  int count,                 /* IN  - number of elements in buffer
                              *       (non-negative integer) */
  MPI_Datatype type,         /* IN  - buffer datatype (handle) */
  MPI_Op op,                 /* IN  - reduction operation (handle) */
  int root,                  /* IN  - rank of root process (integer) */
  const lwgrp_chain* group,  /* IN  - group (handle) */
  const lwgrp_logchain* list /* IN  - list (handle) */
);

int lwgrp_logchain_allreduce_recursive(
  const void* inbuf,         /* IN  - input buffer for reduction */
  void* outbuf,              /* OUT - output buffer for reduction */
  int count,                 /* IN  - number of elements in buffer
                              *       (non-negative integer) */
  MPI_Datatype type,         /* IN  - buffer datatype (handle) */
  MPI_Op op,                 /* IN  - reduction operation (handle) */
  const lwgrp_chain* group,  /* IN  - group (handle) */
  const lwgrp_logchain* list /* IN  - list (handle) */
);

#endif /* MPI >= v2.2 */

/* ---------------------------------
 * Collectives using rings
 * --------------------------------- */

/* given a specified number of bins, an index into those bins, and a
 * input group, create and return a new group consisting of all ranks
 * belonging to the same bin, my_bin should be in range [0,bins-1]
 * if my_bin < 0, then the empty group is returned */
int lwgrp_ring_split_bin_scan(
  int bins,               /* IN  - number of bins (non-negative integer) */
  int bin,                /* IN  - bin to which calling proc belongs (integer) */
  const lwgrp_ring* in,   /* IN  - group to be split (handle) */
  lwgrp_ring* out         /* OUT - group containing all procs in same
                           *       bin as calling process (handle) */
);

/* chunks split operation into pieces */
int lwgrp_ring_split_bin_radix(
  int bins,               /* IN  - number of bins (non-negative integer) */
  int bin,                /* IN  - bin to which calling proc belongs (integer) */
  const lwgrp_ring* in,   /* IN  - group to be split (handle) */
  lwgrp_ring* out         /* OUT - group containing all procs in same
                           *       bin as calling process (handle) */
);

int lwgrp_ring_alltoallv_linear(
  const void* sendbuf,    /* IN  - starting address of send buffer */
  const int sendcounts[], /* IN  - non-negative integer array (of length group size) specifying
                                   the number of elements to send to each processor */
  const int senddispls[], /* IN  - integer array (of length group size).  Entry j specifies
                                   the displacement (relative to sendbuf) from which to take
                                   the outgoing data destined for process j */
  void* recvbuf,          /* OUT - address of receive buffer */
  const int recvcounts[], /* IN  - non-negative integer array (of length group size) specifying
                                   the number of elements that can be received from each processor */
  const int recvdispls[], /* IN  - integer array (of length group size).  Entry i specifies
                                   the displacement (relative to recvbuf) at which to palce the
                                   incoming data from process i */
  MPI_Datatype recvtype,  /* IN  - data type of receive buffer elements (handle) */
  const lwgrp_ring* group /* IN  - group (handle) */
);

/* ---------------------------------
 * Collectives using logrings
 * --------------------------------- */

/* execute a barrier over the chain */
int lwgrp_logring_barrier_dissemination(
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_bcast_binomial(
  void* buffer,             /* IN  - send buffer (on root), receive buffer otherwise */
  int count,                /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype datatype,    /* IN  - data type of buffer elements (handle) */
  int root,                 /* IN  - rank of root process (integer) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

/* TODO: problem here in MPI is that intermediate ranks may use
 * a different type for which we can't just append lots of datatypes
 * in a temporary buffer */
int lwgrp_logring_gather_brucks(
  const void* sendbuf,      /* IN  - send buffer */
  void* recvbuf,            /* OUT - recive buffer */
  int num,                  /* IN  - number of elements on each process (non-negative integer) */
  MPI_Datatype datatype,    /* IN  - element datatype (handle) */
  int root,                 /* IN  - rank of root process (integer) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_allgather_brucks(
  const void* sendbuf,      /* IN  - send buffer */
  void* recvbuf,            /* OUT - recive buffer */
  int num,                  /* IN  - number of elements on each process (non-negative integer) */
  MPI_Datatype datatype,    /* IN  - element datatype (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_allgatherv_binomial(
  const void* sendbuf,      /* IN  - send buffer */
  void* recvbuf,            /* OUT - recive buffer */
  int num,                  /* IN  - number of elements on each process (non-negative integer) */
  MPI_Datatype datatype,    /* IN  - element datatype (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_alltoall_brucks(
  const void* sendbuf,      /* IN  - starting address of send buffer */
  void* recvbuf,            /* OUT - address of receive buffer */
  int num,                  /* IN  - number of elements sent to each process (non-negative integer) */
  MPI_Datatype datatype,    /* IN  - data type of buffer elements (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_alltoallv_linear(
  const void* sendbuf,      /* IN  - starting address of send buffer */
  const int sendcounts[],   /* IN  - non-negative integer array (of length group size) specifying
                                     the number of elements to send to each processor */
  const int senddispls[],   /* IN  - integer array (of length group size).  Entry j specifies
                                     the displacement (relative to sendbuf) from which to take
                                     the outgoing data destined for process j */
  void* recvbuf,            /* OUT - address of receive buffer */
  const int recvcounts[],   /* IN  - non-negative integer array (of length group size) specifying
                                     the number of elements that can be received from each processor */
  const int recvdispls[],   /* IN  - integer array (of length group size).  Entry i specifies
                                     the displacement (relative to recvbuf) at which to palce the
                                     incoming data from process i */
  MPI_Datatype datatype,    /* IN  - data type of buffer elements (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

#if (MPI_VERSION == 2 && MPI_SUBVERSION >= 2) || (MPI_VERSION >= 3)

int lwgrp_logring_reduce_recursive(
  const void* inbuf,         /* IN  - input buffer for reduction */
  void* outbuf,              /* OUT - output buffer for reduction */
  int count,                 /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,         /* IN  - buffer datatype (handle) */
  MPI_Op op,                 /* IN  - reduction operation (handle) */
  int root,                  /* IN  - rank of root process (integer) */
  const lwgrp_ring* group,   /* IN  - group (handle) */
  const lwgrp_logring* list  /* IN  - list (handle) */
);

int lwgrp_logring_allreduce_recursive(
  const void* inbuf,        /* IN  - input buffer for reduction */
  void* outbuf,             /* OUT - output buffer for reduction */
  int count,                /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,        /* IN  - buffer datatype (handle) */
  MPI_Op op,                /* IN  - reduction operation (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_scan_recursive(
  const void* inbuf,        /* IN  - input buffer for reduction */
  void* outbuf,             /* OUT - output buffer for reduction */
  int count,                /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,        /* IN  - buffer datatype (handle) */
  MPI_Op op,                /* IN  - reduction operation (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_exscan_recursive(
  const void* inbuf,        /* IN  - input buffer for reduction */
  void* outbuf,             /* OUT - output buffer for reduction */
  int count,                /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,        /* IN  - buffer datatype (handle) */
  MPI_Op op,                /* IN  - reduction operation (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

int lwgrp_logring_double_exscan_recursive(
  const void* sendleft,     /* IN  - input buffer for right-to-left exscan */
  void* recvright,          /* OUT - output buffer for right-to-left exscan */
  const void* sendright,    /* IN  - input buffer for left-to-right excan */
  void* recvleft,           /* OUT - output buffer for left-to-right exscan */
  int count,                /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,        /* IN  - buffer datatype (handle) */
  MPI_Op op,                /* IN  - reduction operation (handle) */
  const lwgrp_ring* group,  /* IN  - group (handle) */
  const lwgrp_logring* list /* IN  - list (handle) */
);

#endif /* MPI >= v2.2 */

#ifdef __cplusplus
}
#endif
#endif /* _LWGRP_H */
