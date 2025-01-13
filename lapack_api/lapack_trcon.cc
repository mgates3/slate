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
void slate_trcon(
    const char* norm_str, const char* uplo_str, const char* diag_str,
    blas_int const n,
    scalar_t* A_data, blas_int const lda,
    blas::real_type<scalar_t>* rcond,
    scalar_t* work,
    blas_int* iwork,
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

    Uplo uplo{};
    Diag diag{};
    Norm norm{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, diag_str[0] ), &diag );
    from_string( std::string( 1, norm_str[0] ), &norm );

    // sizes
    int64_t nb = NBConfig::value();

    // create SLATE matrix from the LAPACK data
    auto A = slate::TriangularMatrix<scalar_t>::fromLAPACK(
        uplo, diag, n,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );

    blas::real_type<scalar_t> Anorm = slate::norm( norm, A, {
        {slate::Option::Target, target}
    });

    // solve
    *rcond = slate::trcondest( norm, A, Anorm, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get A_data real value for info
    *info = 0;

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "trcon( "
                  << norm_str[0] << ", " << uplo_str[0] << ", "
                  << diag_str[0] << ", "
                  << n << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << Anorm << ", " << (void*)rcond << ", "
                  << (void*)work << ", " << (void*)iwork << ", "
                  << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_strcon BLAS_FORTRAN_NAME( slate_strcon, SLATE_STRCON )
void slate_strcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    float* A_data, blas_int const* lda,
    float* rcond,
    float* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_trcon(
        norm, uplo, diag, *n,
        A_data, *lda, rcond,
        work, iwork, info );
}

#define slate_dtrcon BLAS_FORTRAN_NAME( slate_dtrcon, SLATE_DTRCON )
void slate_dtrcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    double* A_data, blas_int const* lda,
    double* rcond,
    double* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_trcon(
        norm, uplo, diag, *n,
        A_data, *lda, rcond,
        work, iwork, info );
}

#define slate_ctrcon BLAS_FORTRAN_NAME( slate_ctrcon, SLATE_CTRCON )
void slate_ctrcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    std::complex<float>* A_data, blas_int const* lda,
    float* rcond,
    std::complex<float>* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_trcon(
        norm, uplo, diag, *n,
        A_data, *lda, rcond,
        work, iwork, info );
}

#define slate_ztrcon BLAS_FORTRAN_NAME( slate_ztrcon, SLATE_ZTRCON )
void slate_ztrcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    std::complex<double>* A_data, blas_int const* lda,
    double* rcond,
    std::complex<double>* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_trcon(
        norm, uplo, diag, *n,
        A_data, *lda, rcond,
        work, iwork, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
