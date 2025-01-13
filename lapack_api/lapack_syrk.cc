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
void slate_syrk(
    const char* uplo_str, const char* transa_str,
    blas_int n, blas_int k, scalar_t alpha,
    scalar_t* A_data, blas_int lda, scalar_t beta,
    scalar_t* C_data, blas_int ldc )
{
    // start timing
    int verbose = VerboseConfig::value();
    double timestart = 0.0;
    if (verbose)
        timestart = omp_get_wtime();

    // need A_data dummy MPI_Init for SLATE to proceed
    blas_int initialized, provided;
    MPI_Initialized( &initialized );
    if (! initialized)
        MPI_Init_thread( nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided );

    Uplo uplo{};
    Op transA{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, transa_str[0] ), &transA );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // setup so op(A) is n-by-k
    int64_t Am = (transA == blas::Op::NoTrans ? n : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : n);
    int64_t Cn = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto C = slate::SymmetricMatrix<scalar_t>::fromLAPACK(
        uplo, Cn,
        C_data, ldc,
        nb, p, q, MPI_COMM_SELF );

    if (transA == blas::Op::Trans)
        A = transpose( A );
    else if (transA == blas::Op::ConjTrans)
        A = conj_transpose( A );
    assert( A.mt() == C.mt() );

    slate::syrk( alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "syrk( "
                  << uplo_str[0] << ", " << transa_str[0] << ", "
                  << n << ", " << k << ", " << alpha << ", "
                  << (void*)A_data << ", " << lda << ", " << beta << ", "
                  << (void*)C_data << ", " << ldc << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_ssyrk BLAS_FORTRAN_NAME( slate_ssyrk, SLATE_SSYRK )
void slate_ssyrk(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    float const* alpha,
    float* A_data, blas_int const* lda,
    float const* beta,
    float* C_data, blas_int const* ldc )
{
    slate_syrk(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda, *beta,
        C_data, *ldc );
}

#define slate_dsyrk BLAS_FORTRAN_NAME( slate_dsyrk, SLATE_DSYRK )
void slate_dsyrk(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    double const* alpha,
    double* A_data, blas_int const* lda,
    double const* beta,
    double* C_data, blas_int const* ldc )
{
    slate_syrk(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda, *beta,
        C_data, *ldc );
}

#define slate_csyrk BLAS_FORTRAN_NAME( slate_csyrk, SLATE_CSYRK )
void slate_csyrk(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    std::complex<float> const* alpha,
    std::complex<float>* A_data, blas_int const* lda,
    std::complex<float> const* beta,
    std::complex<float>* C_data, blas_int const* ldc )
{
    slate_syrk(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda, *beta,
        C_data, *ldc );
}

#define slate_zsyrk BLAS_FORTRAN_NAME( slate_zsyrk, SLATE_ZSYRK )
void slate_zsyrk(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    std::complex<double> const* alpha,
    std::complex<double>* A_data, blas_int const* lda,
    std::complex<double> const* beta,
    std::complex<double>* C_data, blas_int const* ldc )
{
    slate_syrk(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda, *beta,
        C_data, *ldc );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
