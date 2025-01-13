// Copyright (c) 2017-2025, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack_slate.hh"

namespace slate {
namespace lapack_api {

//------------------------------------------------------------------------------
/// SLATE ScaLAPACK wrapper sets up SLATE matrices from ScaLAPACK descriptors
/// and calls SLATE.
template <typename scalar_t>
void slate_getri(
    blas_int n,
    scalar_t* A_data, blas_int lda,
    blas_int* ipiv,
    scalar_t* work, blas_int lwork,
    blas_int* info )
{
    using real_t = blas::real_type<scalar_t>;

    // Respond to workspace query with A_data minimal value (1); workspace
    // is allocated within the SLATE routine.
    if (lwork == -1) {
        work[0] = (real_t)1.0;
        *info = 0;
        return;
    }

    // Start timing
    int verbose = VerboseConfig::value();
    double timestart = 0.0;
    if (verbose)
        timestart = omp_get_wtime();

    // Check and initialize MPI, else SLATE calls to MPI will fail
    // Since this is an lapack wrapper there will be only one MPI process
    blas_int initialized=0, provided=0;
    MPI_Initialized( &initialized );
    if (! initialized)
        MPI_Init_thread( nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();

    // sizes
    int64_t nb = NBConfig::value();

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        n, n,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );

    // extract pivots from LAPACK ipiv to SLATES pivot structure
    slate::Pivots pivots; // std::vector< std::vector<Pivot> >
    {
        // allocate pivots
        int64_t min_mt_nt = std::min(A.mt(), A.nt());
        pivots.resize(min_mt_nt);
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);
        }
        // transfer ipiv to pivots
        int64_t p_count = 0;
        int64_t t_iter_add = 0;
        for (auto t_iter = pivots.begin(); t_iter != pivots.end(); ++t_iter) {
            for (auto p_iter = t_iter->begin(); p_iter != t_iter->end(); ++p_iter) {
                int64_t tileIndex = (ipiv[p_count] - 1 - t_iter_add) / nb;
                int64_t elementOffset = (ipiv[p_count] - 1 - t_iter_add) % nb;
                *p_iter = Pivot(tileIndex, elementOffset);
                ++p_count;
            }
            t_iter_add += nb;
        }
    }

    // inverse
    slate::getri( A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get A_data real value for info
    *info = 0;

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "getri( "
                  << n << ", "
                  << (void*)A_data << ", " << lda << ", " << (void*)ipiv << ", "
                  << (void*)work << ", " << lwork << ", "
                  << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_sgetri BLAS_FORTRAN_NAME( slate_sgetri, SLATE_SGETRI )
void slate_sgetri(
    blas_int const* n,
    float* A_data, blas_int const* lda,
    blas_int* ipiv,
    float* work, blas_int const* lwork,
    blas_int* info )
{
    slate_getri( *n, A_data, *lda, ipiv, work, *lwork, info );
}

#define slate_dgetri BLAS_FORTRAN_NAME( slate_dgetri, SLATE_DGETRI )
void slate_dgetri(
    blas_int const* n,
    double* A_data, blas_int const* lda,
    blas_int* ipiv,
    double* work, blas_int const* lwork,
    blas_int* info )
{
    slate_getri( *n, A_data, *lda, ipiv, work, *lwork, info );
}

#define slate_cgetri BLAS_FORTRAN_NAME( slate_cgetri, SLATE_CGETRI )
void slate_cgetri(
    blas_int const* n,
    std::complex<float>* A_data, blas_int const* lda,
    blas_int* ipiv,
    std::complex<float>* work, blas_int const* lwork,
    blas_int* info )
{
    slate_getri( *n, A_data, *lda, ipiv, work, *lwork, info );
}

#define slate_zgetri BLAS_FORTRAN_NAME( slate_zgetri, SLATE_ZGETRI )
void slate_zgetri(
    blas_int const* n,
    std::complex<double>* A_data, blas_int const* lda,
    blas_int* ipiv,
    std::complex<double>* work, blas_int const* lwork,
    blas_int* info )
{
    slate_getri( *n, A_data, *lda, ipiv, work, *lwork, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
