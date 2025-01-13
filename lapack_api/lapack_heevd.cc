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
void slate_heevd(
    const char* jobz_str, const char* uplo_str, blas_int n,
    scalar_t* A_data, blas_int lda,
    blas::real_type<scalar_t>* Lambda,
    scalar_t* work, blas_int lwork,
    blas::real_type<scalar_t>* rwork, blas_int lrwork,
    blas_int* iwork, blas_int liwork,
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

    if (lwork == -1 || lrwork == -1 || liwork == -1) {
        work[0] = n * n;
        rwork[0] = 0;
        iwork[0] = 0;
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
                    Z.insertLocalTiles( target );
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
                {slate::Option::MethodEig, MethodEig::DC},
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            std::copy( Lambda_.begin(), Lambda_.end(), Lambda );

            if (jobz_str[0] == 'V') {
                slate::copy( Z, A, {
                    {slate::Option::Target, target}
                });
            }
        }
    }

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "heevd( "
                  << jobz_str[0] << ", " << uplo_str[0] << ", "
                  << n << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << (void*)Lambda << ", "
                  << (void*)work << ", " << lwork << ", ";
        if constexpr (! is_complex<scalar_t>::value) {
            std::cout << (void*)rwork << ", " << lrwork << ", ";
        }
        std::cout << (void*)iwork << ", " << liwork << ", "
                  << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_ssyevd BLAS_FORTRAN_NAME( slate_ssyevd, SLATE_SSYEVD )
void slate_ssyevd(
    const char* jobz_str, const char* uplo, blas_int const* n,
    float* A_data, blas_int const* lda,
    float* Lambda,
    float* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    float dummy;
    slate_heevd(
        jobz_str, uplo, *n,
        A_data, *lda, Lambda,
        work, *lwork,
        &dummy, 1,
        iwork, *liwork, info );
}

#define slate_dsyevd BLAS_FORTRAN_NAME( slate_dsyevd, SLATE_DSYEVD )
void slate_dsyevd(
    const char* jobz_str, const char* uplo, blas_int const* n,
    double* A_data, blas_int const* lda,
    double* Lambda,
    double* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    double dummy;
    slate_heevd(
        jobz_str, uplo, *n,
        A_data, *lda, Lambda,
        work, *lwork,
        &dummy, 1,
        iwork, *liwork, info );
}

#define slate_cheevd BLAS_FORTRAN_NAME( slate_cheevd, SLATE_CHEEVD )
void slate_cheevd(
    const char* jobz_str, const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* lda,
    float* Lambda,
    std::complex<float>* work, blas_int const* lwork,
    float* rwork, blas_int const* lrwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_heevd(
        jobz_str, uplo, *n,
        A_data, *lda, Lambda,
        work, *lwork,
        rwork, *lrwork,
        iwork, *liwork, info );
}

#define slate_zheevd BLAS_FORTRAN_NAME( slate_zheevd, SLATE_ZHEEVD )
void slate_zheevd(
    const char* jobz_str, const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* lda,
    double* Lambda,
    std::complex<double>* work, blas_int const* lwork,
    double* rwork, blas_int const* lrwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_heevd(
        jobz_str, uplo, *n,
        A_data, *lda, Lambda,
        work, *lwork,
        rwork, *lrwork,
        iwork, *liwork, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
