/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-568372.
 * All rights reserved.
 * This file is part of the LWGRP library.
 * For details, see https://github.com/hpc/lwgrp
 * Please also read this file: LICENSE.TXT. */

#ifndef _LWGRP_COMM_H
#define _LWGRP_COMM_H

/* TODO: defines higher level group concept
 * closer to an MPI communicator (still no
 * pt2pt) */

#include "mpi.h"
#include "lwgrp.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct lwgrp_comm {
  lwgrp_ring  ring;
  lwgrp_logring logring;
} lwgrp_comm;

/* ---------------------------------
 * Constructors / destructors
 * --------------------------------- */

/* create a lwgrp comm from an MPI communicator */
int lwgrp_comm_build_from_mpicomm(
  MPI_Comm comm,      /* IN  - MPI communicator (handle) */
  lwgrp_comm* newcomm /* OUT - lwgrp communicator (pointer to comm struct) */
);

/* create a lwgrp comm from a lwgrp chain */
int lwgrp_comm_build_from_chain(
  const lwgrp_chain* chain, /* IN  - lwgrp chain (pointer to chain struct) */
  lwgrp_comm* newcomm       /* OUT - lwgrp communicator (pointer to comm struct) */
);

/* copy a lwgrp comm */
int lwgrp_comm_copy(
  const lwgrp_comm* comm, /* IN  - lwgrp chain (pointer to comm struct) */
  lwgrp_comm* newcomm     /* OUT - copy of lwgrp chain (pointer to comm struct) */
);

/* split a lwgrp comm into subcomms, where each subcomm holds all procs
 * in the same bin, takes O(B) space and O(B*log(N)) time where B is number
 * of bins and N is the number of procs in comm */
int lwgrp_comm_split_bin(
  const lwgrp_comm* comm, /* IN  - lwgrp communicator (pointer to comm struct) */
  int bins,               /* IN  - number of bins (non-negative integer) */
  int bin,                /* IN  - bin number 0 to bins-1, or -1  (integer) */
  lwgrp_comm* newcomm     /* OUT - lwgrp communicator of all procs in
                           *       the same bin */
);

/* implements semantics of MPI_Comm_split */
int lwgrp_comm_split(
  const lwgrp_comm* comm, /* IN  - lwgrp communicator (pointer to comm struct) */
  int color,              /* IN  - non-negative color value or MPI_UNDEFINED (integer) */
  int key,                /* IN  - key value to order ranks (integer) */
  lwgrp_comm* newcomm     /* OUT - lwgrp communicator of all procs with same color,
                           *       ordered by key, then rank in comm */
);

/* assigns an id to each unique string in the union of all strings of procs in comm */
int lwgrp_comm_rank_str(
  const lwgrp_comm* comm, /* IN  - lwgrp communicator (pointer to comm struct) */
  const char* str,        /* IN  - string to be ranked (string) */
  int* groups,            /* OUT - number of unique strings (non-negative integer) */
  int* groupid            /* OUT - rank of input string (non-negative integer) */
);

/* frees memory associated with comm structure */
int lwgrp_comm_free(
  lwgrp_comm* comm /* INOUT - lwgrp comm (pointer to comm struct) */
);

/* ---------------------------------
 * Query routines
 * --------------------------------- */

/* get our rank within the communicator */
int lwgrp_comm_rank(
  const lwgrp_comm* comm, /* IN  - communicator (pointer to comm struct) */
  int* rank               /* OUT - rank within communicator (integer) */
);

/* get size of the communicator */
int lwgrp_comm_size(
  const lwgrp_comm* comm, /* IN  - communicator (pointer to comm struct) */
  int* size               /* OUT - size of communicator (integer) */
);

/* ---------------------------------
 * Collectives
 * --------------------------------- */

int lwgrp_comm_barrier(
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_bcast(
  void* buffer,          /* IN  - send buffer (on root), receive buffer otherwise */
  int count,             /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype datatype, /* IN  - data type of buffer elements (handle) */
  int root,              /* IN  - rank of root process (integer) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_gather(
  const void* sendbuf,   /* IN  - send buffer */
  void* recvbuf,         /* OUT - recive buffer */
  int num,               /* IN  - number of elements on each process (non-negative integer) */
  MPI_Datatype datatype, /* IN  - element datatype (handle) */
  int root,              /* IN  - rank of root process (integer) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_allgather(
  const void* sendbuf,   /* IN  - send buffer */
  void* recvbuf,         /* OUT - recive buffer */
  int num,               /* IN  - number of elements on each process (non-negative integer) */
  MPI_Datatype datatype, /* IN  - element datatype (handle) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_allgatherv(
  const void* sendbuf,   /* IN  - send buffer */
  void* recvbuf,         /* OUT - recive buffer */
  const int counts[],    /* IN  - non-negative integer array (of length group size) specifying
                                  the number of elements each processor has */
  const int displs[],    /* IN  - integer array (of length group size).  Entry i specifies
                                  the displacement (relative to recvbuf) at which to palce the
                                  incoming data from process i */
  MPI_Datatype datatype, /* IN  - element datatype (handle) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_alltoall(
  const void* sendbuf,   /* IN  - send buffer */
  void* recvbuf,         /* OUT - recive buffer */
  int num,               /* IN  - number of elements on each process (non-negative integer) */
  MPI_Datatype datatype, /* IN  - element datatype (handle) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_alltoallv(
  const void* sendbuf,    /* IN  - send buffer */
  const int sendcounts[], /* IN  - non-negative integer array (of length group size) specifying
                                   the number of elements to send to each processor */
  const int senddispls[], /* IN  - integer array (of length group size).  Entry j specifies
                                   the displacement (relative to sendbuf) from which to take
                                   the outgoing data destined for process j */
  void* recvbuf,          /* OUT - recive buffer */
  const int recvcounts[], /* IN  - non-negative integer array (of length group size) specifying
                                   the number of elements that can be received from each processor */
  const int recvdispls[], /* IN  - integer array (of length group size).  Entry i specifies
                                   the displacement (relative to recvbuf) at which to palce the
                                   incoming data from process i */
  MPI_Datatype datatype,  /* IN  - element datatype (handle) */
  const lwgrp_comm* comm  /* IN  - group (handle) */
);

int lwgrp_comm_reduce(
  const void* inbuf,     /* IN  - input buffer for reduction */
  void* outbuf,          /* OUT - output buffer for reduction */
  int count,             /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,     /* IN  - buffer datatype (handle) */
  MPI_Op op,             /* IN  - reduction operation (handle) */
  int root,              /* IN  - rank of root process (integer) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_allreduce(
  const void* inbuf,     /* IN  - input buffer for reduction */
  void* outbuf,          /* OUT - output buffer for reduction */
  int count,             /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,     /* IN  - buffer datatype (handle) */
  MPI_Op op,             /* IN  - reduction operation (handle) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_scan(
  const void* inbuf,     /* IN  - input buffer for reduction */
  void* outbuf,          /* OUT - output buffer for reduction */
  int count,             /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,     /* IN  - buffer datatype (handle) */
  MPI_Op op,             /* IN  - reduction operation (handle) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_exscan(
  const void* inbuf,     /* IN  - input buffer for reduction */
  void* outbuf,          /* OUT - output buffer for reduction */
  int count,             /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,     /* IN  - buffer datatype (handle) */
  MPI_Op op,             /* IN  - reduction operation (handle) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

int lwgrp_comm_double_exscan(
  const void* sendleft,  /* IN  - input buffer for right-to-left exscan */
  void* recvright,       /* OUT - output buffer for right-to-left exscan */
  const void* sendright, /* IN  - input buffer for left-to-right excan */
  void* recvleft,        /* OUT - output buffer for left-to-right exscan */
  int count,             /* IN  - number of elements in buffer (non-negative integer) */
  MPI_Datatype type,     /* IN  - buffer datatype (handle) */
  MPI_Op op,             /* IN  - reduction operation (handle) */
  const lwgrp_comm* comm /* IN  - group (handle) */
);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _LWGRP_COMM_H */
