/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-568372.
 * All rights reserved.
 * This file is part of the LWGRP library.
 * For details, see https://github.com/hpc/lwgrp
 * Please also read this file: LICENSE.TXT. */

#include "mpi.h"
#include "lwgrp.h"

void* lwgrp_malloc(size_t size, size_t alignment, const char* file, int line);

void lwgrp_free(void*);

/* given a datatype and a count, return the number of bytes required
 * to hold that number of items in one buffer */
size_t lwgrp_type_get_bufsize(int num, MPI_Datatype type);

/* given the start of a buffer in memory, a datatype, and a count n,
 * return the address that should be passed to an MPI function that
 * represents the start of the nth item, adjust for lower bound and
 * extent, we isolate the logic here since the math is simple but
 * not intuitive */
void* lwgrp_type_get_bufstart(const void* buf, int num, MPI_Datatype type);

void lwgrp_memcpy(void* dst, const void* src, int count, MPI_Datatype type, int rank, MPI_Comm comm);

/* find largest power of two that fits within ranks */
int lwgrp_largest_pow2_log2(int ranks, int* outpow2, int* outlog2);

/* find largest power strictly less than ranks */
int lwgrp_largest_pow2_log2_lessthan(int ranks, int* outpow2, int* outlog2);
