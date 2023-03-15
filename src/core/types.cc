// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/types.hh"
#include "slate/internal/util.hh"

namespace slate {

//------------------------------------------------------------------------------
MPI_Datatype mpi_type<float >::value = MPI_FLOAT;
MPI_Datatype mpi_type<double>::value = MPI_DOUBLE;
MPI_Datatype mpi_type< std::complex<float>  >::value = MPI_C_COMPLEX;
MPI_Datatype mpi_type< std::complex<double> >::value = MPI_C_DOUBLE_COMPLEX;

MPI_Datatype mpi_type< max_loc_type<float>  >::value = MPI_FLOAT_INT;
MPI_Datatype mpi_type< max_loc_type<double> >::value = MPI_DOUBLE_INT;

//------------------------------------------------------------------------------
CallStack::CallStack( int mpi_rank, const char* format, ... )
{
    if (mpi_rank == -1)
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

    char buf[ 1024 ];
    int len = snprintf( buf, sizeof(buf), "%2d:%2d ",
                        mpi_rank, omp_get_thread_num() );
    for (int i = 0; i < s_depth; ++i) {
        len += snprintf( &buf[ len ], sizeof( buf ) - len, "    " );
    }

    va_list args;
    va_start( args, format );
    len += vsnprintf( &buf[ len ], sizeof( buf ) - len, format, args );
    va_end( args );

    snprintf( &buf[ len ], sizeof( buf ) - len, "\n" );

    s_msg += buf;

    ++s_depth;
}

//------------------------------------------------------------------------------
CallStack::~CallStack()
{
   --s_depth;
}

//------------------------------------------------------------------------------
/// [static]
/// Add a comment to the call stack output.
void CallStack::comment( const char* msg )
{
    s_msg += msg;
}

//------------------------------------------------------------------------------
/// [static]
/// MPI rank 0 prints accumulated messages from all other ranks.
/// The message buffer is reset on each rank.
///
void CallStack::print( MPI_Comm mpi_comm )
{
    int mpi_rank = -1;
    int mpi_size = -1;
    slate_mpi_call(
        MPI_Comm_rank( mpi_comm, &mpi_rank ) );
    slate_mpi_call(
        MPI_Comm_size( mpi_comm, &mpi_size ) );

    int tag_0 = 0;

    if (mpi_rank == 0) {
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (rank > 0) {
                int len;
                slate_mpi_call(
                    MPI_Recv( &len, 1, MPI_INT, rank, tag_0,
                              mpi_comm, MPI_STATUS_IGNORE ) );
                s_msg.resize( len );
                slate_mpi_call(
                    MPI_Recv( &s_msg[0], len, MPI_CHAR, rank, tag_0,
                              mpi_comm, MPI_STATUS_IGNORE ) );
            }
            printf( "======================================== rank %d\n", rank );
            printf( "%s", s_msg.c_str() );
        }
    }
    else {
        int len = s_msg.size();
        MPI_Send( &len, 1, MPI_INT, 0, tag_0, mpi_comm );
        MPI_Send( s_msg.c_str(), len, MPI_CHAR, 0, tag_0, mpi_comm );
    }
    s_msg.clear();
}

//------------------------------------------------------------------------------
/// [static]
std::string CallStack::s_msg;

int CallStack::s_depth = 0;

} // namespace slate
