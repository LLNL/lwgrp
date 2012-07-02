#ifndef _LWGRP_COMM_SPLIT_H
#define _LWGRP_COMM_SPLIT_H

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* lwgrp_comm_split_members(comm, color, key, size, members)
 *
 * IN  comm    - MPI communicator on which to perform split (handle)
 * IN  color   - color value, same meaning as MPI_COMM_SPLIT (integer)
 * IN  key     - key value, same meaning as in MPI_COMM_SPLIT (integer)
 * OUT size    - size of output group after split (non-negative integer)
 * OUT members - array of ordered members in group after split (array of
 *               non-negative integers)
 *
 * The members array must be allocated and passed in by caller.
 * It should be big enough to store ranks from largest group after split,
 * so to be safe, it should be as large as size(comm). */

int lwgrp_comm_split_members(MPI_Comm comm, int color, int key, int* size, int members[]);

#if MPI_VERSION >= 2 && MPI_SUBVERSION >= 2
/* same as above but returns a newly created communicator via MPI_Comm_create,
 * can only use in MPI-2.2 and later and still not useful unless MPI_COMM_CREATE
 * is scalable */
int lwgrp_comm_split_create(MPI_Comm comm, int color, int key, MPI_Comm* newcomm);
#endif

#ifdef __cplusplus
}
#endif
#endif /* _LWGRP_COMM_SPLIT_H */
