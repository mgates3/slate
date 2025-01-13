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
void slate_heev(
    const char* jobz_str, const char* uplo_str, blas_int n,
    scalar_t* A_data, blas_int lda,
    blas::real_type<scalar_t>* w,
    scalar_t* work, blas_int lwork, blas::real_type<scalar_t>* rwork,
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

    // TODO check args more carefully
    *info = 0;

    if (lwork == -1) {
        work[0] = n * n;
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

        Uplo uplo{};
        from_string( std::string( 1, uplo_str[0] ), &uplo );

        // create SLATE matrix from the LAPACK data
        auto A = slate::Matrix<scalar_t>::fromLAPACK( n, n,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
        slate::HermitianMatrix<scalar_t> AH( uplo, A );
        std::vector< blas::real_type<scalar_t> > Lambda_( n );

        slate::Matrix<scalar_t> Z;
        switch (jobz_str[0]) {
            case 'V':
                if (lwork >= n * n) {
                    Z = slate::Matrix<scalar_t>::fromLAPACK(
                        n, n, work, n, nb, p, q, MPI_COMM_SELF );
                }
                else {
                    Z = slate::Matrix<scalar_t>( n, n, nb, p, q, MPI_COMM_SELF );
                    Z.insertLocalTiles(target);
                }
                break;
            case 'N':
                // Leave Z empty
                break;
            default:
                *info = 1;
        }

        if (*info == 0) {
            // solve
            slate::heev( AH, Lambda_, Z, {
                {slate::Option::MethodEig, MethodEig::QR},
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            std::copy(Lambda_.begin(), Lambda_.end(), w);

            if (jobz_str[0] == 'V') {
                slate::copy( Z, A, {
                    {slate::Option::Target, target}
                });
            }
        }
    }

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "heev( "
                  << jobz_str[0] << ", " << uplo_str[0] << ", "
                  << n << ", "
                  << (void*)A_data << ", " << lda << ", " << (void*)w << ", "
                  << (void*)work << ", " << lwork << ", ";
        if (is_complex<scalar_t>::value) {
            std::cout << (void*)rwork << ", ";
        }
        std::cout << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_ssyev BLAS_FORTRAN_NAME( slate_ssyev, SLATE_SSYEV )
void slate_ssyev(
    const char* jobz_str, const char* uplo, blas_int const* n,
    float* A_data, blas_int const* lda,
    float* w,
    float* work, blas_int const* lwork,
    blas_int* info )
{
    float dummy;  // in place of rwork
    slate_heev(
        jobz_str, uplo, *n,
        A_data, *lda, w,
        work, *lwork,
        &dummy, info );
}

#define slate_dsyev BLAS_FORTRAN_NAME( slate_dsyev, SLATE_DSYEV )
void slate_dsyev(
    const char* jobz_str, const char* uplo, blas_int const* n,
    double* A_data, blas_int const* lda,
    double* w,
    double* work, blas_int const* lwork,
    blas_int* info )
{
    double dummy;  // in place of rwork
    slate_heev(
        jobz_str, uplo, *n,
        A_data, *lda, w,
        work, *lwork,
        &dummy, info );
}

#define slate_cheev BLAS_FORTRAN_NAME( slate_cheev, SLATE_CHEEV )
void slate_cheev(
    const char* jobz_str, const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* lda,
    float* w,
    std::complex<float>* work, blas_int const* lwork,
    float* rwork,
    blas_int* info )
{
    slate_heev(
        jobz_str, uplo, *n,
        A_data, *lda, w,
        work, *lwork,
        rwork, info );
}

#define slate_zheev BLAS_FORTRAN_NAME( slate_zheev, SLATE_ZHEEV )
void slate_zheev(
    const char* jobz_str, const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* lda,
    double* w,
    std::complex<double>* work, blas_int const* lwork,
    double* rwork,
    blas_int* info )
{
    slate_heev(
        jobz_str, uplo, *n,
        A_data, *lda, w,
        work, *lwork,
        rwork, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
