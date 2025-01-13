// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
void slate_gesv(
    blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int lda,
    blas_int* ipiv,
    scalar_t* B_data, blas_int ldb,
    blas_int* info )
{
    // Start timing
    int verbose = VerboseConfig::value();
    double timestart = 0.0;
    if (verbose)
        timestart = omp_get_wtime();

    // Check and initialize MPI, else SLATE calls to MPI will fail
    blas_int initialized, provided;
    MPI_Initialized( &initialized );
    if (! initialized)
        MPI_Init_thread( nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();

    // sizes
    int64_t Am = n, An = n;
    int64_t Bm = n, Bn = nrhs;
    int64_t nb = NBConfig::value();
    int64_t ib = IBConfig::value();
    slate::Pivots pivots;

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
        Bm, Bn,
        B_data, ldb,
        nb, p, q, MPI_COMM_SELF );

    // computes the solution to the system of linear equations with a
    // square coefficient matrix A and multiple right-hand sides.
    slate::gesv( A, pivots, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    // extract pivots from SLATE's Pivots structure into LAPACK ipiv array
    // todo: rewrite with C++ `for (p : pivots)`
    {
        int64_t p_count = 0;
        int64_t t_iter_add = 0;
        for (auto t_iter = pivots.begin(); t_iter != pivots.end(); ++t_iter) {
            for (auto p_iter = t_iter->begin(); p_iter != t_iter->end(); ++p_iter) {
                ipiv[p_count] = p_iter->tileIndex() * nb + p_iter->elementOffset() + 1 + t_iter_add;
                ++p_count;
            }
            t_iter_add += nb;
        }
    }

    // todo:  get A_data real value for info
    *info = 0;

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "gesv( "
                  << n << ", " << nrhs << ", "
                  << (void*)A_data << ", " << lda << ", " << (void*)ipiv << ", "
                  << (void*)B_data << ", " << ldb << ", "
                  << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_sgesv BLAS_FORTRAN_NAME( slate_sgesv, SLATE_SGESV )
void slate_sgesv(
    blas_int const* n, blas_int const* nrhs,
    float* A_data, blas_int const* lda,
    blas_int* ipiv,
    float* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_gesv( *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

#define slate_dgesv BLAS_FORTRAN_NAME( slate_dgesv, SLATE_DGESV )
void slate_dgesv(
    blas_int const* n, blas_int const* nrhs,
    double* A_data, blas_int const* lda,
    blas_int* ipiv,
    double* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_gesv( *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

#define slate_cgesv BLAS_FORTRAN_NAME( slate_cgesv, SLATE_CGESV )
void slate_cgesv(
    blas_int const* n, blas_int const* nrhs,
    std::complex<float>* A_data, blas_int const* lda,
    blas_int* ipiv,
    std::complex<float>* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_gesv( *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

#define slate_zgesv BLAS_FORTRAN_NAME( slate_zgesv, SLATE_ZGESV )
void slate_zgesv(
    blas_int const* n, blas_int const* nrhs,
    std::complex<double>* A_data, blas_int const* lda,
    blas_int* ipiv,
    std::complex<double>* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_gesv( *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
