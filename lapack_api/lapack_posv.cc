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
void slate_posv(
    const char* uplo_str, blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int lda,
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

    Uplo uplo{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();
    slate::Pivots pivots;

    // create SLATE matrices from the LAPACK data
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(
        uplo, n,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
        n, nrhs,
        B_data, ldb,
        nb, p, q, MPI_COMM_SELF );

    // computes the solution to the system of linear equations with a
    // square coefficient matrix A and multiple right-hand sides.
    slate::posv( A, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get A_data real value for info
    *info = 0;

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "posv( "
                  << uplo_str[0] << ", "
                  << n << ", " << nrhs << ", "
                  << (void*)A_data << ", " << lda << ", "
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

#define slate_sposv BLAS_FORTRAN_NAME( slate_sposv, SLATE_SPOSV )
void slate_sposv(
    const char* uplo, blas_int const* n, blas_int const* nrhs,
    float* A_data, blas_int const* lda,
    float* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_posv( uplo, *n, *nrhs, A_data, *lda, B_data, *ldb, info );
}

#define slate_dposv BLAS_FORTRAN_NAME( slate_dposv, SLATE_DPOSV )
void slate_dposv(
    const char* uplo, blas_int const* n, blas_int const* nrhs,
    double* A_data, blas_int const* lda,
    double* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_posv( uplo, *n, *nrhs, A_data, *lda, B_data, *ldb, info );
}

#define slate_cposv BLAS_FORTRAN_NAME( slate_cposv, SLATE_CPOSV )
void slate_cposv(
    const char* uplo, blas_int const* n, blas_int const* nrhs,
    std::complex<float>* A_data, blas_int const* lda,
    std::complex<float>* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_posv( uplo, *n, *nrhs, A_data, *lda, B_data, *ldb, info );
}

#define slate_zposv BLAS_FORTRAN_NAME( slate_zposv, SLATE_ZPOSV )
void slate_zposv(
    const char* uplo, blas_int const* n, blas_int const* nrhs,
    std::complex<double>* A_data, blas_int const* lda,
    std::complex<double>* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_posv( uplo, *n, *nrhs, A_data, *lda, B_data, *ldb, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
