/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-568372.
 * All rights reserved.
 * This file is part of the LWGRP library.
 * For details, see https://github.com/hpc/lwgrp
 * Please also read this file: LICENSE.TXT. */

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "lwgrp.h"

/* malloc with some checks on size and the returned pointer, along with
 * future support for alignment, if size <= 0, malloc is not called and
 * NULL pointer is returned, error is printed with file name and line
 * number if size > 0 and malloc returns NULL pointer */
void* lwgrp_malloc(size_t size, size_t align, const char* file, int line)
{
  void* ptr = NULL;
  if (size > 0) {
/*
    if (align == 0) {
      ptr = malloc(size);
    } else {
      posix_memalign(&ptr, size, align);
    }
*/
    ptr = malloc(size);
    if (ptr == NULL) {
      printf("ERROR: Failed to allocate memory %lu bytes @ %s:%d\n", size, file, line);
      exit(1);
    }
  }
  return ptr;
}

/* careful to not call free if pointer is already NULL, sets pointer
 * value to NULL */
void lwgrp_free(void* p)
{
  /* we really receive a pointer to a pointer (void**), but it's typed
   * as void* so caller doesn't need to add casts all over the place */
  void** ptr = (void**) p;

  /* free associated memory and set pointer to NULL */
  if (ptr == NULL) {
    printf("ERROR: Expected address of pointer value, but got NULL instead @ %s:%d\n",
      __FILE__, __LINE__
    );
  } else if (*ptr != NULL) {
    free(*ptr);
    *ptr = NULL;
  }
}

/* given a datatype and a count, return the number of bytes required
 * to hold that number of items in one buffer */
size_t lwgrp_type_get_bufsize(int num, MPI_Datatype type)
{
  /* TODO: we could do this with type_contiguous, type_get_true_extent,
   * type_free, which would be more accurate but likely more expensive */

  /* get true extent of datatype */
  MPI_Aint true_lb, true_extent;
  MPI_Type_get_true_extent(type, &true_lb, &true_extent);

  /* compute memory needed to hold num copies */
  size_t size = num * true_extent;

  return size;
}

/* given the start of a buffer in memory, a datatype, and a count n,
 * return the address that should be passed to an MPI function that
 * represents the start of the nth item, adjust for lower bound and
 * extent, we isolate the logic here since the math is simple but
 * not intuitive */
void* lwgrp_type_get_bufstart(const void* buf, int num, MPI_Datatype type)
{
  /* nothing to do if the buffer is NULL */
  if (buf == NULL) {
    return NULL;
  }

  /* get extent of datatype (used as scaling factor from start) */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(type, &lb, &extent);

  /* get true lower bound of datatype (need to adjust starting offset) */
  MPI_Aint true_lb, true_extent;
  MPI_Type_get_true_extent(type, &true_lb, &true_extent);

  /* adjust for lower bounds */
  const char* newbuf = (const char*)buf - true_lb;

  /* now move num extents up */
  newbuf += extent * num;

  return (void*)newbuf;
}

void lwgrp_memcpy(void* dst, const void* src, int count, MPI_Datatype type, int rank, MPI_Comm comm)
{
    MPI_Status status[2];
    MPI_Sendrecv(
      (void*)src, count, type, rank, LWGRP_MSG_TAG_0,
             dst, count, type, rank, LWGRP_MSG_TAG_0,
      comm, status
    );
}

/* find largest power of two that fits within group_ranks */
int lwgrp_largest_pow2_log2_lte(int ranks, int* outpow2, int* outlog2)
{
  int pow2 = 1;
  int log2 = 0;
  while (pow2 <= ranks) {
    pow2 <<= 1;
    log2++;
  }
  pow2 >>= 1;
  log2--;

  *outpow2 = pow2;
  *outlog2 = log2;

  return LWGRP_SUCCESS;
}

/* find largest power of two strictly less than ranks */
int lwgrp_largest_pow2_log2_lessthan(int ranks, int* outpow2, int* outlog2)
{
  int pow2, log2;
  lwgrp_largest_pow2_log2_lte(ranks, &pow2, &log2);
  if (pow2 == ranks) {
    pow2 >>= 1;
    log2--;
  }

  *outpow2 = pow2;
  *outlog2 = log2;

  return LWGRP_SUCCESS;
}
