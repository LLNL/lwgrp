/* Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Adam Moody <moody20@llnl.gov>.
 * LLNL-CODE-568372.
 * All rights reserved.
 * This file is part of the LWGRP library.
 * For details, see https://github.com/hpc/lwgrp
 * Please also read this file: LICENSE.TXT. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"
#include "lwgrp.h"
#include "lwgrp_comm.h"
#include "lwgrp_internal.h"

/* Based on "Exascale Algorithms for Generalized MPI_Comm_split",
 * EuroMPI 2011, Adam Moody, Dong H. Ahn, and Bronis R. de Supinkski
 *
 * Executes an MPI_Comm_split operation using bitonic sort, a double
 * inclusive scan to find color boundaries and left and right group
 * neighbors, a recv from ANY_SOURCE, and a barrier, returns the
 * output group as a chain.
 *
 * Optionally, gathers ordered list of members to build dense group
 * representation as a full array. */

enum chain_fields {
  CHAIN_SRC   = 0,
  CHAIN_LEFT  = 1,
  CHAIN_RIGHT = 2,
  CHAIN_RANK  = 3,
  CHAIN_SIZE  = 4,
};

enum ckr_fields {
  CKR_COLOR = 0,
  CKR_KEY   = 1,
  CKR_RANK  = 2,
  CKR_ADDR  = 3,
};

/* just compares the first int of an array, to send data
 * back to its originating rank */
static int lwgrp_cmp_one_int(const int a[], const int b[])
{
  if (a[CHAIN_SRC] > b[CHAIN_SRC]) {
    return 1;
  }
  return -1;
}

/* compares a (color,key,rank) integer tuple, first by color,
 * then key, then rank */
static int lwgrp_cmp_three_ints(const int a[], const int b[])
{
  /* compare color values first */
  if (a[CKR_COLOR] != b[CKR_COLOR]) {
    if (a[CKR_COLOR] > b[CKR_COLOR]) {
      return 1;
    }
    return -1;
  }

  /* then compare key values */
  if (a[CKR_KEY] != b[CKR_KEY]) {
    if (a[CKR_KEY] > b[CKR_KEY]) {
      return 1;
    }
    return -1;
  }

  /* finally compare ranks */
  if (a[CKR_RANK] != b[CKR_RANK]) {
    if (a[CKR_RANK] > b[CKR_RANK]) {
      return 1;
    }
    return -1;
  }

  /* all three are equal if we make it here */
  return 0;
}

static int lwgrp_sort_bitonic_merge(
  int num_ints,
  int value[],
  int scratch[],
  int start,
  int num,
  int direction,
  int (*compare)(const int*, const int*),
  const lwgrp_chain* group,
  const lwgrp_logchain* list,
  int tag)
{
  if (num > 1) {
    /* get our group communicator and our rank within the group */
    MPI_Comm comm = group->comm;
    int rank = group->group_rank;

    /* determine largest power of two that is smaller than num */
    int count = 1;
    int index = 0;
    while (count < num) {
      count <<= 1;
      index++;
    }
    count >>= 1;
    index--;

    /* divide range into two chunks, execute bitonic half-clean step,
     * then recursively merge each half */
    MPI_Status status[2];
    if (rank < start + count) {
      int dst_rank = rank + count;
      if (dst_rank < start + num) {
        /* exchange data with our partner rank */
        int partner = list->right_list[index];
        MPI_Sendrecv(
          value,   num_ints, MPI_INT, partner, tag,
          scratch, num_ints, MPI_INT, partner, tag,
          comm, status
        );

        /* select the appropriate value,
         * depedning on the sort direction */
        int cmp = (*compare)(scratch, value);
        if ((direction && cmp < 0) || (!direction && cmp > 0)) {
          memcpy(value, scratch, num_ints * sizeof(int));
        }
      }

      /* recursively merge our half */
      lwgrp_sort_bitonic_merge(
        num_ints, value, scratch, start, count, direction, compare,
        group, list, tag
      );
    } else {
      int dst_rank = rank - count;
      if (dst_rank >= start) {
        /* exchange data with our partner rank */
        int partner = list->left_list[index];
        MPI_Sendrecv(
          value,   num_ints, MPI_INT, partner, tag,
          scratch, num_ints, MPI_INT, partner, tag,
          comm, status
        );

        /* select the appropriate value,
         * depedning on the sort direction */
        int cmp = (*compare)(scratch, value);
        if ((direction && cmp > 0) || (!direction && cmp < 0)) {
          memcpy(value, scratch, num_ints * sizeof(int));
        }
      }

      /* recursively merge our half */
      int new_start = start + count;
      int new_num   = num - count;
      lwgrp_sort_bitonic_merge(
        num_ints, value, scratch, new_start, new_num, direction, compare,
        group, list, tag
      );
    }
  }

  return 0;
}

static int lwgrp_sort_bitonic_sort(
  int num_ints,
  int value[],
  int scratch[],
  int start,
  int num,
  int direction,
  int (*compare)(const int*, const int*),
  const lwgrp_chain* group,
  const lwgrp_logchain* list,
  int tag)
{
  if (num > 1) {
    /* get our rank in our group */
    int rank = group->group_rank;

    /* recursively divide and sort each half */
    int mid = num / 2;
    if (rank < start + mid) {
      lwgrp_sort_bitonic_sort(
        num_ints, value, scratch, start, mid, !direction, compare,
        group, list, tag
      );
    } else {
      int new_start = start + mid;
      int new_num   = num - mid;
      lwgrp_sort_bitonic_sort(
        num_ints, value, scratch, new_start, new_num, direction, compare,
        group, list, tag
      );
    }

    /* merge the two sorted halves */
    lwgrp_sort_bitonic_merge(
      num_ints, value, scratch, start, num, direction, compare,
      group, list, tag
    );
  }

  return 0;
}

/* globally sort (color,key,rank) items across processes in group,
 * each process provides its tuple as item on input,
 * on output item is overwritten with a new item
 * such that if rank_i < rank_j, item_i < item_j for all i and j */
static int lwgrp_logchain_sort_bitonic(
  int num_ints,
  int item[],
  int scratch[],
  int (*compare)(const int*, const int*),
  const lwgrp_chain* group,
  const lwgrp_logchain* list,
  int tag)
{
  /* conduct the bitonic sort on our values */
  int ranks = group->group_size;
  int rc = lwgrp_sort_bitonic_sort(
    num_ints, item, scratch, 0, ranks, 1, compare,
    group, list, tag
  );
  return rc;
}

static enum scan_fields {
  SCAN_FLAG  = 0, /* set flag to 1 when we should stop accumulating */
  SCAN_COUNT = 1, /* running count being accumulated */
  SCAN_NEXT  = 2, /* rank of next process to talk to */
};

/* assumes that color/key/rank tuples have been globally sorted
 * across ranks of in chain, computes corresponding group
 * information for val and passes that back to originating rank:
 *   1) determines group boundaries and left and right neighbors
 *      by sending pt2pt msgs to left and right neighbors and
 *      comparing color values
 *   2) executes left-to-right and right-to-left (double) inclusive
 *      segmented scan to compute number of ranks to left and right
 *      sides of host value */
static int lwgrp_chain_split_sorted(
  const int item[4],
  const lwgrp_chain* in,
  const lwgrp_logchain* list,
  int tag1,
  int tag2,
  lwgrp_chain* out)
{
  int k;
  MPI_Request request[4];
  MPI_Status  status[4];

  /* we will fill in five integer values (src, left, right, rank, size)
   * representing the chain data structure for the the globally
   * ordered color/key/rank tuple that we hold, which we'll later
   * send back to the rank that contributed our item */

  /* record address of process that contributed this item */
  int send_ints[5];
  send_ints[CHAIN_SRC] = item[CKR_ADDR];

  /* get the communicator to send our messages on */
  MPI_Comm comm = in->comm;

  /* exchange data with left and right neighbors to find
   * boundaries of group */
  k = 0;
  int left_rank  = in->comm_left;
  int right_rank = in->comm_right;
  int left_buf[4];
  int right_buf[4];
  if (left_rank != MPI_PROC_NULL) {
    MPI_Isend(
      (void*)item, 4, MPI_INT, left_rank, tag1, comm, &request[k]
    );
    k++;

    MPI_Irecv(
      left_buf, 4, MPI_INT, left_rank, tag1, comm, &request[k]
    );
    k++;
  }
  if (right_rank != MPI_PROC_NULL) {
    MPI_Isend(
      (void*)item, 4, MPI_INT, right_rank, tag1, comm, &request[k]
    );
    k++;

    MPI_Irecv(
      right_buf, 4, MPI_INT, right_rank, tag1, comm, &request[k]
    );
    k++;
  }
  if (k > 0) {
    MPI_Waitall(k, request, status);
  }

  /* if we have a left neighbor, and if his color value matches ours,
   * then our element is part of his group, otherwise we are the first
   * rank of a new group */
  int first_in_group = 0;
  if (left_rank != MPI_PROC_NULL &&
      left_buf[CKR_COLOR] == item[CKR_COLOR])
  {
    /* record the rank of the item from our left neighbor */
    send_ints[CHAIN_LEFT] = left_buf[CKR_ADDR];
  } else {
    first_in_group = 1;
    send_ints[CHAIN_LEFT] = MPI_PROC_NULL;
  }

  /* if we have a right neighbor, and if his color value matches ours,
   * then our element is part of his group, otherwise we are the last
   * rank of our group */
  int last_in_group = 0;
  if (right_rank != MPI_PROC_NULL &&
      right_buf[CKR_COLOR] == item[CKR_COLOR])
  {
    /* record the rank of the item from our right neighbor */
    send_ints[CHAIN_RIGHT] = right_buf[CKR_ADDR];
  } else {
    last_in_group = 1;
    send_ints[CHAIN_RIGHT] = MPI_PROC_NULL;
  }

  /* prepare buffers for our scan operations */
  int send_left_ints[3]  = {0,1,MPI_PROC_NULL}; /* flag, cnt, next */
  int send_right_ints[3] = {0,1,MPI_PROC_NULL};
  int recv_left_ints[3]  = {0,0,MPI_PROC_NULL};
  int recv_right_ints[3] = {0,0,MPI_PROC_NULL};
  if (first_in_group) {
    left_rank = MPI_PROC_NULL;
    send_right_ints[SCAN_FLAG] = 1;
  }
  if (last_in_group) {
    right_rank = MPI_PROC_NULL;
    send_left_ints[SCAN_FLAG] = 1;
  }

  /* execute inclusive scan in both directions to count number of
   * ranks in our group to our left and right sides */
  while (left_rank != MPI_PROC_NULL || right_rank != MPI_PROC_NULL) {
    /* select our left and right partners for this iteration */
    k = 0;

    /* send and receive data with left partner */
    if (left_rank != MPI_PROC_NULL) {
      MPI_Irecv(
        recv_left_ints, 3, MPI_INT, left_rank, tag1, comm, &request[k]
      );
      k++;

      /* send the rank of our right neighbor to our left,
       * since it will be his right neighbor in the next step */
      send_left_ints[SCAN_NEXT] = right_rank;
      MPI_Isend(
        send_left_ints, 3, MPI_INT, left_rank, tag1, comm, &request[k]
      );
      k++;
    }

    /* send and receive data with right partner */
    if (right_rank != MPI_PROC_NULL) {
      MPI_Irecv(
        recv_right_ints, 3, MPI_INT, right_rank, tag1, comm, &request[k]
      );
      k++;

      /* send the rank of our left neighbor to our right,
       * since it will be his left neighbor in the next step */
      send_right_ints[SCAN_NEXT] = left_rank;
      MPI_Isend(
        send_right_ints, 3, MPI_INT, right_rank, tag1, comm, &request[k]
      );
      k++;
    }

    /* wait for communication to finsih */
    if (k > 0) {
      MPI_Waitall(k, request, status);
    }

    /* reduce data from left partner */
    if (left_rank != MPI_PROC_NULL) {
      /* continue accumulating the count in our right-going data
       * if our flag has not already been set */
      if (send_right_ints[SCAN_FLAG] != 1) {
        send_right_ints[SCAN_FLAG]   = recv_left_ints[SCAN_FLAG];
        send_right_ints[SCAN_COUNT] += recv_left_ints[SCAN_COUNT];
      }

      /* get the next rank on our left */
      left_rank = recv_left_ints[SCAN_NEXT];
    }

    /* reduce data from right partner */
    if (right_rank != MPI_PROC_NULL) {
      /* continue accumulating the count in our left-going data
       * if our flag has not already been set */
      if (send_left_ints[SCAN_FLAG] != 1) {
        send_left_ints[SCAN_FLAG]   = recv_right_ints[SCAN_FLAG];
        send_left_ints[SCAN_COUNT] += recv_right_ints[SCAN_COUNT];
      }

      /* get the next rank on our right */
      right_rank = recv_right_ints[SCAN_NEXT];
    }
  }

  /* Now we can set our rank and the number of ranks in our group.
   * At this point, our right-going count is the number of ranks to our
   * left including ourself, and the left-going count is the number of
   * ranks to our right including ourself.
   * Our rank is the number of ranks to our left (right-going count
   * minus 1), and the group size is the sum of right-going and
   * left-going counts minus 1 so we don't double counts ourself. */
  send_ints[CHAIN_RANK] = send_right_ints[SCAN_COUNT] - 1;
  send_ints[CHAIN_SIZE] = send_right_ints[SCAN_COUNT] + 
                          send_left_ints[SCAN_COUNT] - 1;

  /* send group info back to originating rank */
  int recv_ints[5];
#ifdef LWGRP_USE_ANYSOURCE
  /* send group info back to originating rank,
   * receive our own from someone else
   * (don't know who so use ANY_SOURCE) */
  MPI_Isend(
    send_ints, 5, MPI_INT, send_ints[CHAIN_SRC], tag2,
    comm, &request[0]
  );
  MPI_Irecv(
    recv_ints, 5, MPI_INT, MPI_ANY_SOURCE, tag2,
    comm, &request[1]
  );
  MPI_Waitall(2, request, status);
#else
  /* if we can't use MPI_ANY_SOURCE, then sort item back to its
   * destination */
  int scratch_ints[5];
  lwgrp_logchain_sort_bitonic(
    5, send_ints, scratch_ints, lwgrp_cmp_one_int,
    in, list, tag1
  );
  memcpy(recv_ints, send_ints, 5 * sizeof(int));
#endif

  /* fill in info for our group */
  out->comm       = in->comm;
  out->comm_rank  = in->comm_rank;
  out->comm_left  = recv_ints[CHAIN_LEFT];
  out->comm_right = recv_ints[CHAIN_RIGHT];
  out->group_rank = recv_ints[CHAIN_RANK];
  out->group_size = recv_ints[CHAIN_SIZE];

  return LWGRP_SUCCESS;
}

int lwgrp_comm_split(
  const lwgrp_comm* comm,
  int color,
  int key,
  lwgrp_comm* newcomm)
{
  int tag1 = 0;
  int tag2 = 1;

  /* TODO: for small groups, fastest to do an allgather and
   * local sort */

  /* TODO: allreduce to determine whether keys are already ordered and
   * to compute min and max color values, if already ordered, reduce
   * problem to bin split using min/max colors to set number of bins */

  /* build a group representing our input communicator -- O(1) local */
  lwgrp_chain chain;
  lwgrp_chain_build_from_ring(&comm->ring, &chain);

  /* build a logchain for our input communicator -- O(log N) local */
  lwgrp_logchain logchain;
  lwgrp_logchain_build_from_logring(&comm->ring, &comm->logring, &logchain);

  /* allocate memory to hold item for sorting (color,key,rank) tuple
   * and prepare input -- O(1) local */
  int item[4];
  int scratch[4];
  item[CKR_COLOR] = color;
  item[CKR_KEY]   = key;
  item[CKR_RANK]  = chain.group_rank;
  item[CKR_ADDR]  = chain.comm_rank;

  /* sort our values using bitonic sort algorithm -- 
   * O(log^2 N) communication */
  lwgrp_logchain_sort_bitonic(
    4, item, scratch, lwgrp_cmp_three_ints,
    &chain, &logchain, tag1
  );

  /* now split our sorted values by comparing our value with our
   * left and right neighbors to determine group boundaries --
   * O(log N) communication */
  lwgrp_chain newchain;
  lwgrp_chain_split_sorted(item, &chain, &logchain, tag1, tag2, &newchain);

  /* if color is undefined, at this point we have the group of
   * processes that all set color == MPI_UNDEFINED, but we
   * really want the empty group -- O(1) local */
  if (color == MPI_UNDEFINED) {
    lwgrp_chain_set_null(&newchain);
  }

  /* build comm from newly created chain */
  lwgrp_comm_build_from_chain(&newchain, newcomm);

  /* free the chain representing the new group -- O(1) local */
  lwgrp_chain_free(&newchain);

  /* free the logchain -- O(1) local */
  lwgrp_logchain_free(&logchain);

  /* free our group -- O(1) local */
  lwgrp_chain_free(&chain);

  return LWGRP_SUCCESS;
}

#if 0
/* same as lwgrp_comm_split_members, but runs in O(log^2 N) time
 * instead of O(N) by building and returing a chain representation
 * of the output group */
int lwgrp_comm_split_from_mpicomm(
  MPI_Comm mpicomm,
  int color,
  int key,
  lwgrp_comm* newcomm)
{
  /* TODO: for small groups, fastest to do an allgather and
   * local sort */

  /* build a group representing our input communicator -- O(N) local */
  lwgrp_comm comm;
  lwgrp_comm_build_from_comm(mpicomm, &comm);

  /* sort our values using bitonic sort algorithm -- 
   * O(log^2 N) communication */
  lwgrp_comm_split(&comm, color, key, newcomm);

  /* free our group -- O(1) local */
  lwgrp_comm_free(&comm);

  return 0;
}

int lwgrp_comm_split_members(
  MPI_Comm comm,
  int color,
  int key,
  int tag1,
  int tag2,
  int* size,
  int members[])
{
  /* split comm with lwgrp_comm_split_chain to get the chain
   * corresponding to our group, this takes O(log^2 N) time due
   * to bitonic sort */
  lwgrp_chain out;
  lwgrp_comm_split_chain(comm, color, key, tag1, tag2, &out);

  /* fill in output parameters with group information --
   * O(N) communication */
  if (color != MPI_UNDEFINED) {
    /* gather members of group -- O(N) communication,
     * but if max group size is small, N is small */
    *size = out.group_size;
    lwgrp_chain_allgather_int(out.comm_rank, members, &out);
  } else {
    /* we don't have a group since color == MPI_UNDEFINED */
    *size = 0;
  }

  /* free our group -- O(1) local */
  lwgrp_chain_free(&out);

  return 0;
}

#if MPI_VERSION >= 2 && MPI_SUBVERSION >= 2
/* uses MPI_COMM_CREATE in a way that was only supported in
 * MPI-2.2 and later */
int lwgrp_comm_split_create(
  MPI_Comm comm,
  int color,
  int key,
  int tag1,
  int tag2,
  MPI_Comm* new_comm)
{
  /* get the size of comm */
  int ranks;
  MPI_Comm_size(comm, &ranks);

  /* allocate an array to hold all members of our output group,
   * could be as large as comm if all ranks specify the same color */
  int* members = (int*) malloc(ranks * sizeof(int));

  /* split the communicator and the get the list of members */
  int size;
  lwgrp_comm_split_members(
    comm, color, key, tag1, tag2, &size, members
  );

  if (size > 0) {
    /* construct an MPI group based on our list of processes */
    MPI_Group comm_group, new_group;
    MPI_Comm_group(comm, &comm_group);
    MPI_Group_incl(comm_group, size, members, &new_group);
    MPI_Comm_create(comm, new_group, new_comm);
    MPI_Group_free(&new_group);
    MPI_Group_free(&comm_group);
  } else {
    /* if we don't have any processes in our group,
     * call Comm_create with MPI_GROUP_EMPTY */
    MPI_Comm_create(comm, MPI_GROUP_EMPTY, new_comm);
  }

  /* free the list of group members */
  lwgrp_free(&members);

  return 0;
}
#endif

#endif
