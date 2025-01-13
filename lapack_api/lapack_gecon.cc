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
void slate_gecon(
    const char* norm_str, blas_int n,
    scalar_t* A_data, blas_int lda,
    blas::real_type<scalar_t> Anorm, blas::real_type<scalar_t>* rcond,
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

    Norm norm{};
    from_string( std::string( 1, norm_str[0] ), &norm );

    // sizes
    int64_t nb = NBConfig::value();

    // create SLATE matrix from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        n, n,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF);

    // solve
    *rcond = slate::gecondest( norm, A, Anorm, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get A_data real value for info
    *info = 0;

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "gecon( "
                  << norm_str[0] << ", "
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

#define slate_sgecon BLAS_FORTRAN_NAME( slate_sgecon, SLATE_SGECON )
void slate_sgecon(
    const char* norm, blas_int const* n,
    float* A_data, blas_int const* lda,
    float* Anorm, float* rcond,
    float* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_gecon( norm, *n, A_data, *lda, *Anorm, rcond, work, iwork, info );
}

#define slate_dgecon BLAS_FORTRAN_NAME( slate_dgecon, SLATE_DGECON )
void slate_dgecon(
    const char* norm, blas_int const* n,
    double* A_data, blas_int const* lda,
    double* Anorm, double* rcond,
    double* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_gecon( norm, *n, A_data, *lda, *Anorm, rcond, work, iwork, info );
}

#define slate_cgecon BLAS_FORTRAN_NAME( slate_cgecon, SLATE_CGECON )
void slate_cgecon(
    const char* norm, blas_int const* n,
    std::complex<float>* A_data, blas_int const* lda,
    float* Anorm, float* rcond,
    std::complex<float>* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_gecon( norm, *n, A_data, *lda, *Anorm, rcond, work, iwork, info );
}

#define slate_zgecon BLAS_FORTRAN_NAME( slate_zgecon, SLATE_ZGECON )
void slate_zgecon(
    const char* norm, blas_int const* n,
    std::complex<double>* A_data, blas_int const* lda,
    double* Anorm, double* rcond,
    std::complex<double>* work,
    blas_int* iwork,
    blas_int* info )
{
    slate_gecon( norm, *n, A_data, *lda, *Anorm, rcond, work, iwork, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
