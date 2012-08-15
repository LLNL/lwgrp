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
#include "lwgrp_internal.h"

/* given a pointer to the start of a datatype buffer, return pointer to
 * datatype buffer for the start of the count-th element */
void* lwgrp_type_dtbuf_from_dtbuf(const void* dtbuf, int count, MPI_Datatype type)
{
  /* get lower bounds and extent of datatype */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(type, &lb, &extent);

  char* ptr = (char*)dtbuf + count * extent;
  return ptr;
}

/* given a pointer to the start of a memory buffer, return pointer to
 * be used as a datatype buffer for the start of the count-th element */
void* lwgrp_type_dtbuf_from_membuf(const void* membuf, int count, MPI_Datatype type)
{
  /* get lower bounds and extent of datatype */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(type, &lb, &extent);

  char* ptr = (char*)membuf -lb + count * extent;
  return ptr;
}

/* returns the size of a buffer needed to hold count consecutive items */
size_t lwgrp_type_membuf_size(int count, MPI_Datatype type)
{
}

/* allocate a buffer large enough to hold count consecutive items,
 * and align buf to type */
void* lwgrp_type_dtbuf_alloc(int count, MPI_Datatype type, const char* file, int line)
{
#if MPI_VERSION >=2 && MPI_SUBVERSION >=2
#endif
  /* get lower bounds and extent of datatype */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(type, &lb, &extent);

  size_t size  = count * extent;
  size_t align = 0;
  char* ptr = (char*) lwgrp_malloc(size, align, file, line);
  ptr -= lb;
  return ptr;
}

/* free buffer allocated with lwgrp_type_dtbuf_alloc */
int lwgrp_type_dtbuf_free(void** dtbuf_ptr, MPI_Datatype type, const char* file, int line)
{
  if (dtbuf_ptr != NULL) {
    void* dtbuf = *dtbuf_ptr;
    if (dtbuf != NULL) {
      /* get lower bounds and extent of datatype */
      MPI_Aint lb, extent;
      MPI_Type_get_extent(type, &lb, &extent);

      char* ptr = (char*)dtbuf + lb;
      if (ptr != NULL) {
        free(ptr);
      } else {
        /* ERROR: dtbuf should be NULL in this case */
      }
    } else {
      /* OK: user can pass a pointer whose value is NULL, ignore it */
    }
  } else {
    /* ERROR: user passed in a NULL value as the address of his pointer */
  }

  return LWGRP_SUCCESS;
}


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
