#include <stdlib.h>
#include <string.h>

#include "mpi.h"
#include "lwgrp.h"
#include "lwgrp_internal.h"

int lwgrp_logring_barrier(
  const lwgrp_ring* group,
  const lwgrp_logring* list)
{
  /* get ring info */
  MPI_Comm comm  = group->comm;
  int ranks      = group->group_size;

  /* execute Bruck's dissemination algorithm */
  int index = 0;
  int dist  = 1;
  while (dist < ranks) {
    /* get source and destination ranks */
    int src = list->left_list[index];
    int dst = list->right_list[index];

    /* send empty messages as a signal */
    MPI_Status status[2];
    MPI_Sendrecv(
      NULL, 0, MPI_BYTE, dst, LWGRP_MSG_TAG_0,
      NULL, 0, MPI_BYTE, src, LWGRP_MSG_TAG_0,
      comm, status
    );

    /* prepare for next iteration */
    index++;
    dist <<= 1;
  } 

  return 0;
}

/* implements a binomail tree */
int lwgrp_logring_bcast(
  void* buffer,
  int count,
  MPI_Datatype datatype,
  int root,
  const lwgrp_ring* group,
  const lwgrp_logring* list)
{
  /* get ring info */
  MPI_Comm comm  = group->comm;
  int rank       = group->group_rank;
  int ranks      = group->group_size;

  /* adjust our rank by setting the root to be rank 0 */
  int treerank = rank - root;
  if (treerank < 0) {
    treerank += ranks;
  }

  /* get largest power-of-two strictly less than ranks */
  int pow2, log2;
  lwgrp_largest_pow2_log2_lessthan(ranks, &pow2, &log2);

  /* run through binomial tree */
  int parent = 0;
  int received = (rank == root) ? 1 : 0;
  while (pow2 > 0) {
    /* check whether we need to receive or send data */
    if (! received) {
      /* if we haven't received a message yet, see if the parent for
       * this step will send to us */
      int target = parent + pow2;
      if (treerank == target) {
        /* we're the target, get the real rank and receive data */
        MPI_Status status;
        int src = list->left_list[log2];
        MPI_Recv(buffer, count, datatype, src, LWGRP_MSG_TAG_0, comm, &status);
        received = 1;
      } else if (treerank > target) {
        /* if we are in the top half of the subtree set our new
         * potential parent */
        parent = target;
      }
    } else {
      /* we have received the data, so if we have a child, send data */
      if (treerank + pow2 < ranks) {
        int dst = list->right_list[log2];
        MPI_Send(buffer, count, datatype, dst, LWGRP_MSG_TAG_0, comm);
      }
    }

    /* cut the step size in half and keep going */
    log2--;
    pow2 >>= 1;
  }

  return 0;
}

/* issue an allgather using Bruck's algorithm */
int lwgrp_logring_allgather_brucks(
  const void* sendbuf,
  void* recvbuf,
  int num,
  MPI_Datatype datatype,
  const lwgrp_ring* group,
  const lwgrp_logring* list)
{
  /* get ring info */
  MPI_Comm comm  = group->comm;
  int rank       = group->group_rank;
  int ranks      = group->group_size;

  /* get extent of datatype so we can allocate space */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(datatype, &lb, &extent);

  /* get true extent of datatype so we can allocate space */
  MPI_Aint true_lb, true_extent;
  MPI_Type_get_true_extent(datatype, &true_lb, &true_extent);

  /* compute size of datatype */
  size_t size = (size_t) true_extent * num;
  if (size <= 0) {
    return 0;
  }

  /* free some temporary space to work with */
  int scratch_size = size * ranks;
  char* scratch = (char*) malloc(scratch_size);
  if (scratch == NULL) {
    /* TODO: fail */
  }

  /* adjust for lower bounds */
  char* tmpbuf = scratch - true_lb;

  /* copy our own data into the temporary buffer */
  const char* inputbuf = (const char*) sendbuf;
#if MPI_VERSION >= 2
  if (sendbuf == MPI_IN_PLACE) {
    inputbuf = (const char*)recvbuf + rank * extent * num;
  }
#endif
  lwgrp_memcpy(tmpbuf, inputbuf, num, datatype, rank, comm);

  /* execute the allgather operation */
  MPI_Request request[2];
  MPI_Status status[2];
  int index = 0;
  int step  = 1;
  int ranks_received = 1;
  while (step < ranks) {
    /* get ranks for left and right partners */
    int src = list->right_list[index];
    int dst = list->left_list[index];

    /* determine number of elements we'll be sending and receiving in this round */
    int ranks_incoming = step;
    if (ranks_received + ranks_incoming > ranks) {
      ranks_incoming = ranks - ranks_received;
    }
    int num_exchange = num * ranks_incoming;

    /* receive data from source */
    char* recv_pos = tmpbuf + ranks_received * extent * num;
    MPI_Irecv(recv_pos, num_exchange, datatype, src, LWGRP_MSG_TAG_0, comm, &request[0]); 

    /* send the data to destination */
    MPI_Isend(tmpbuf, num_exchange, datatype, dst, LWGRP_MSG_TAG_0, comm, &request[1]);

    /* wait for communication to complete */
    MPI_Waitall(2, request, status);

    /* add the count to the total number we've received */
    ranks_received += ranks_incoming;

    /* go on to next iteration */
    index++;
    step <<= 1;
  }

  /* shift our data back to the proper position in receive buffer */
  lwgrp_memcpy((char*)recvbuf + rank * extent * num, tmpbuf, (ranks - rank) * num, datatype, rank, comm);
  lwgrp_memcpy(recvbuf, (char*)tmpbuf + (ranks - rank) * extent * num, rank * num, datatype, rank, comm);

  /* free the temporary buffer */
  lwgrp_free(&scratch);

  return 0;
}

/* execute an alltoall using Bruck's algorithm */
int lwgrp_logring_alltoall_brucks(
  const void* sendbuf,
  void* recvbuf,
  int num,
  MPI_Datatype datatype,
  const lwgrp_ring* group,
  const lwgrp_logring* list)
{
  int i;

  /* get ring info */
  MPI_Comm comm  = group->comm;
  int rank       = group->group_rank;
  int ranks      = group->group_size;

  /* get extent of datatype so we can allocate space */
  MPI_Aint lb, extent;
  MPI_Type_get_extent(datatype, &lb, &extent);

  /* get true extent of datatype so we can allocate space */
  MPI_Aint true_lb, true_extent;
  MPI_Type_get_true_extent(datatype, &true_lb, &true_extent);

  /* compute size of datatype */
  size_t size = (size_t) true_extent * num;
  if (size <= 0) {
    return 0;
  }

  /* TODO: feels like some of these memory copies could be avoided */

  int bufsize = size * ranks;
  char* send_data = (char*) malloc(bufsize);
  char* recv_data = (char*) malloc(bufsize);
  char* tmp_data  = (char*) malloc(bufsize);

  char* tmp_ptr  = tmp_data  - true_lb;

  /* copy our send data to our receive buffer, and rotate it so our own rank is at the top */
  const char* inputbuf = (const char*) sendbuf;
#if MPI_VERSION >=2
  if (sendbuf == MPI_IN_PLACE) {
    inputbuf = (const char*) recvbuf;
  }
#endif
  lwgrp_memcpy(tmp_ptr, inputbuf + rank * extent * num, (ranks - rank) * num, datatype, rank, comm);
  lwgrp_memcpy(tmp_ptr + (ranks - rank) * extent * num, inputbuf, rank * num, datatype, rank, comm);

  /* now run through Bruck's index algorithm to exchange data */
  MPI_Request request[2];
  MPI_Status  status[2];
  int index = 0;
  int step = 1;
  while (step < ranks) {
    /* determine our source and destination ranks for this step */
    int dst = list->right_list[index];
    int src = list->left_list[index];

    /* pack our data to send and count number of bytes */
    int send_count = 0;
    char* send_ptr = send_data - true_lb;
    for (i = 0; i < ranks; i++) {
      int mask = (i & step);
      if (mask) {
        lwgrp_memcpy(send_ptr, tmp_ptr + i * num * extent, num, datatype, rank, comm);
        send_ptr += num * extent;
        send_count += num;
      }
    }

    /* exchange messages */
    MPI_Irecv(recv_data, send_count, datatype, src, LWGRP_MSG_TAG_0, comm, &request[0]);
    MPI_Isend(send_data, send_count, datatype, dst, LWGRP_MSG_TAG_0, comm, &request[1]);
    MPI_Waitall(2, request, status);

    /* unpack received data into our buffer */
    char* recv_ptr = recv_data - true_lb;
    for (i = 0; i < ranks; i++) {
      int mask = (i & step);
      if (mask) {
        lwgrp_memcpy(tmp_ptr + i * num * extent, recv_ptr, num, datatype, rank, comm);
        recv_ptr += num * extent;
      }
    }

    /* go on to the next phase of the exchange */
    index++;
    step <<= 1;
  }

  /* copy our data to our receive buffer, and rotate it back so our own rank is in its proper position */
  lwgrp_memcpy(send_data - true_lb, tmp_data - true_lb + (rank + 1) * num * extent, (ranks - rank - 1) * num, datatype, rank, comm);
  lwgrp_memcpy(send_data - true_lb + (ranks - rank - 1) * num * extent, tmp_data - true_lb, (rank + 1) * num, datatype, rank, comm);
  for (i = 0; i < ranks; i++) {
    lwgrp_memcpy((char*)recvbuf + i * num * extent, send_data - true_lb + (ranks - i - 1) * num * extent, num, datatype, rank, comm);
  }

  /* free off our internal data structures */
  lwgrp_free(&tmp_data);
  lwgrp_free(&recv_data);
  lwgrp_free(&send_data);

  return 0;
}

int lwgrp_logring_alltoallv_linear(
  const void* sendbuf, const int sendcounts[], const int senddispls[],
        void* recvbuf, const int recvcounts[], const int recvdispls[],
  MPI_Datatype datatype, const lwgrp_ring* group, const lwgrp_logring* list)
{
  /* this is really implemented with the ring, so just call that */
  int rc = lwgrp_ring_alltoallv_linear(sendbuf, sendcounts, senddispls, recvbuf, recvcounts, recvdispls, datatype, group);
  return rc;
}

#if MPI_VERSION >=2 && MPI_SUBVERSION >=2
int lwgrp_logring_double_exscan(
  const void* sendleft, void* recvright, const void* sendright, void* recvleft,
  int count, MPI_Datatype type, MPI_Op op, const lwgrp_ring* group, const lwgrp_logring* list)
{
  /* convert ring to chain and call chain's double exscan function */
  int rc = 0;
  lwgrp_chain chain;
  lwgrp_chain_build_from_ring(group, &chain);
  rc = lwgrp_chain_double_exscan(sendleft, recvright, sendright, recvleft, count, type, op, &chain);
  lwgrp_chain_free(&chain);
  return rc;
}

int lwgrp_logring_allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype type, MPI_Op op, const lwgrp_chain* group, const lwgrp_logring* list)
{
  int rc = 0;

  lwgrp_chain chain;
  lwgrp_chain_build_from_ring(group, &chain);

  lwgrp_logchain logchain;
  lwgrp_logchain_build_from_logring(group, list, &logchain);

  rc = lwgrp_logchain_allreduce(sendbuf, recvbuf, count, type, op, &chain, &logchain);

  lwgrp_logchain_free(&logchain);
  lwgrp_chain_free(&chain);
  return rc;
}
#endif
