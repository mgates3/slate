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
void slate_gesvd(
    const char* jobu_str, const char* jobvt_str,
    blas_int m, blas_int n,
    scalar_t* A_data, blas_int lda,
    blas::real_type<scalar_t>* Sigma,
    scalar_t* U_data, blas_int ldu,
    scalar_t* VT_data, blas_int ldvt,
    scalar_t* work, blas_int lwork,
    blas_int* info )
{
    // Start timing
    int verbose = VerboseConfig::value();
    double timestart = 0.0;
    if (verbose)
        timestart = omp_get_wtime();

    // sizes
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();
    int64_t min_mn = std::min( m, n );

    // TODO check args more carefully
    *info = 0;

    // todo: upcase strings
    if (lwork == -1) {
        if (jobu_str[0] == 'O') {
            work[0] = m * min_mn;
        }
        else if (jobvt_str[0] == 'O') {
            work[0] = min_mn * n;
        }
        else {
            work[0] = 0;
        }
    }
    else {
        // Check and initialize MPI, else SLATE calls to MPI will fail
        blas_int initialized, provided;
        MPI_Initialized( &initialized );
        if (! initialized)
            MPI_Init_thread( nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided );

        int64_t lookahead = 1;
        int64_t p = 1;
        int64_t q = 1;

        // create SLATE matrix from the LAPACK data
        auto A = slate::Matrix<scalar_t>::fromLAPACK(
        m, n,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
        std::vector< blas::real_type<scalar_t> > Sigma_( min_mn );

        slate::Matrix<scalar_t> U;
        switch (jobu_str[0]) {
            case 'A':
                U = slate::Matrix<scalar_t>::fromLAPACK(
                    m, m,
                    U_data, ldu,
                    nb, p, q, MPI_COMM_SELF );
                break;
            case 'S':
                U = slate::Matrix<scalar_t>::fromLAPACK(
                    m, min_mn,
                    U_data, ldu,
                    nb, p, q, MPI_COMM_SELF );
                break;
            case 'O':
                if (lwork >= m * min_mn) {
                    U = slate::Matrix<scalar_t>::fromLAPACK(
                        m, min_mn, work, m, nb, p, q, MPI_COMM_SELF );
                }
                else {
                    U = slate::Matrix<scalar_t>(
                        m, min_mn, nb, p, q, MPI_COMM_SELF );
                    U.insertLocalTiles( target );
                }
                break;
            case 'N':
                // Leave U empty
                break;
            default:
                *info = 1;
        }

        slate::Matrix<scalar_t> VT;
        switch (jobvt_str[0]) {
            case 'A':
                VT = slate::Matrix<scalar_t>::fromLAPACK(
                    n, n,
                    VT_data, ldvt,
                    nb, p, q, MPI_COMM_SELF );
                break;
            case 'S':
                VT = slate::Matrix<scalar_t>::fromLAPACK(
                    min_mn, n,
                    VT_data, ldvt,
                    nb, p, q, MPI_COMM_SELF );
                break;
            case 'O':
                if (lwork >= min_mn * n) {
                    VT = slate::Matrix<scalar_t>::fromLAPACK(
                        min_mn, n, work, m, nb, p, q, MPI_COMM_SELF );
                }
                else {
                    VT = slate::Matrix<scalar_t>(
                        min_mn, n, nb, p, q, MPI_COMM_SELF );
                    VT.insertLocalTiles( target );
                }
                break;
            case 'N':
                // Leave VT empty
                break;
            default:
                *info = 2;
        }

        if (*info == 0) {
            // solve
            slate::svd( A, Sigma_, U, VT, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            std::copy( Sigma_.begin(), Sigma_.end(), Sigma );

            if (jobu_str[0] == 'O') {
                auto A_slice = A.slice( 0, m-1, 0, min_mn-1 );
                slate::copy( U, A_slice, {
                    {slate::Option::Target, target}
                });
            }
            if (jobvt_str[0] == 'O') {
                auto A_slice = A.slice( 0, n-1, 0, min_mn-1 );
                slate::copy( VT, A_slice, {
                    {slate::Option::Target, target}
                });
            }
        }
    }

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "gesvd( "
                  << jobu_str[0] << ", " << jobvt_str[0] << ", "
                  << m << ", " << n << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << (void*)Sigma << ", "
                  << (void*)U_data << ", " << ldu << ", "
                  << (void*)VT_data << ", " << ldvt << ", "
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

#define slate_sgesvd BLAS_FORTRAN_NAME( slate_sgesvd, SLATE_SGESVD )
void slate_sgesvd(
    const char* jobu_str, const char* jobvt_str,
    blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* lda,
    float* Sigma,
    float* U_data, blas_int const* ldu,
    float* VT_data, blas_int const* ldvt,
    float* work, blas_int const* lwork,
    blas_int* info )
{
    slate_gesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *lda, Sigma,
        U_data, *ldu,
        VT_data, *ldvt,
        work, *lwork, info );
}

#define slate_dgesvd BLAS_FORTRAN_NAME( slate_dgesvd, SLATE_DGESVD )
void slate_dgesvd(
    const char* jobu_str, const char* jobvt_str,
    blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* lda,
    double* Sigma,
    double* U_data, blas_int const* ldu,
    double* VT_data, blas_int const* ldvt,
    double* work, blas_int const* lwork,
    blas_int* info )
{
    slate_gesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *lda, Sigma,
        U_data, *ldu,
        VT_data, *ldvt,
        work, *lwork, info );
}

#define slate_cgesvd BLAS_FORTRAN_NAME( slate_cgesvd, SLATE_CGESVD )
void slate_cgesvd(
    const char* jobu_str, const char* jobvt_str,
    blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* lda,
    float* Sigma,
    std::complex<float>* U_data, blas_int const* ldu,
    std::complex<float>* VT_data, blas_int const* ldvt,
    std::complex<float>* work, blas_int const* lwork,
    float* rwork,
    blas_int* info )
{
    slate_gesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *lda, Sigma,
        U_data, *ldu,
        VT_data, *ldvt,
        work, *lwork, info );
}

#define slate_zgesvd BLAS_FORTRAN_NAME( slate_zgesvd, SLATE_ZGESVD )
void slate_zgesvd(
    const char* jobu_str, const char* jobvt_str,
    blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* lda,
    double* Sigma,
    std::complex<double>* U_data, blas_int const* ldu,
    std::complex<double>* VT_data, blas_int const* ldvt,
    std::complex<double>* work, blas_int const* lwork,
    double* rwork,
    blas_int* info )
{
    slate_gesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *lda, Sigma,
        U_data, *ldu,
        VT_data, *ldvt,
        work, *lwork, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
