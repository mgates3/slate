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
void slate_herk(
    const char* uplo_str, const char* transa_str,
    blas_int n, blas_int k, blas::real_type<scalar_t> alpha,
    scalar_t* A_data, blas_int lda,
    blas::real_type<scalar_t> beta,
    scalar_t* C_data, blas_int ldc )
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
    auto C = slate::HermitianMatrix<scalar_t>::fromLAPACK(
        uplo, Cn,
        C_data, ldc,
        nb, p, q, MPI_COMM_SELF );

    if (transA == blas::Op::Trans)
        A = transpose( A );
    else if (transA == blas::Op::ConjTrans)
        A = conj_transpose( A );
    assert( A.mt() == C.mt() );

    slate::herk( alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "herk( "
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

#define slate_cherk BLAS_FORTRAN_NAME( slate_cherk, SLATE_CHERK )
void slate_cherk(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    float const* alpha,
    std::complex<float>* A_data, blas_int const* lda,
    float const* beta,
    std::complex<float>* C_data, blas_int const* ldc )
{
    slate_herk(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda, *beta,
        C_data, *ldc );
}

#define slate_zherk BLAS_FORTRAN_NAME( slate_zherk, SLATE_ZHERK )
void slate_zherk(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    double const* alpha,
    std::complex<double>* A_data, blas_int const* lda,
    double const* beta,
    std::complex<double>* C_data, blas_int const* ldc )
{
    slate_herk(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda, *beta,
        C_data, *ldc );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
