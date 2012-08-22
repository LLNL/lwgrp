/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-568372.
 * All rights reserved.
 * This file is part of the LWGRP library.
 * For details, see https://github.com/hpc/lwgrp
 * Please also read this file: LICENSE.TXT. */

#ifndef _LWGRP_INTERNAL_H
#define _LWGRP_INTERNAL_H

#include <stdlib.h>
#include "mpi.h"
#include "lwgrp.h"

void* lwgrp_type_dtbuf_from_dtbuf(const void* dtbuf, int count, MPI_Datatype type);

void* lwgrp_type_dtbuf_from_membuf(const void* membuf, int count, MPI_Datatype type);

void* lwgrp_type_dtbuf_alloc(int count, MPI_Datatype type, const char* file, int line);

int lwgrp_type_dtbuf_free(void** dtbuf_ptr, MPI_Datatype type, const char* file, int line);

void lwgrp_type_dtbuf_memcpy(void* dst, const void* src, int count, MPI_Datatype type);

void* lwgrp_malloc(size_t size, size_t alignment, const char* file, int line);

void lwgrp_free(void*);

/* find largest power of two that fits within ranks */
int lwgrp_largest_pow2_log2(int ranks, int* outpow2, int* outlog2);

/* find largest power strictly less than ranks */
int lwgrp_largest_pow2_log2_lessthan(int ranks, int* outpow2, int* outlog2);

#endif /* _LWGRP_INTERNAL_H */
