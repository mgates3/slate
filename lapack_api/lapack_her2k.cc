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
void slate_her2k(
    const char* uplo_str, const char* transa_str,
    blas_int n, blas_int k, scalar_t alpha,
    scalar_t* A_data, blas_int lda,
    scalar_t* B_data, blas_int ldb, blas::real_type<scalar_t> beta,
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
    Op trans{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, transa_str[0] ), &trans );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // setup so op(A) and op(B) are n-by-k
    int64_t Am = (trans == blas::Op::NoTrans ? n : k);
    int64_t An = (trans == blas::Op::NoTrans ? k : n);
    int64_t Bm = Am;
    int64_t Bn = An;
    int64_t Cn = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
        Bm, Bn,
        B_data, ldb,
        nb, p, q, MPI_COMM_SELF );
    auto C = slate::HermitianMatrix<scalar_t>::fromLAPACK(
        uplo, Cn,
        C_data, ldc,
        nb, p, q, MPI_COMM_SELF );

    if (trans == blas::Op::Trans) {
        A = transpose( A );
        B = transpose( B );
    }
    else if (trans == blas::Op::ConjTrans) {
        A = conj_transpose( A );
        B = conj_transpose( B );
    }
    assert( A.mt() == C.mt() );
    assert( B.mt() == C.mt() );
    assert( A.nt() == B.nt() );

    slate::her2k( alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "her2k( "
                  << uplo_str[0] << ", " << transa_str[0] << ", "
                  << n << ", " << k << ", " << alpha << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << (void*)B_data << ", " << ldb << ", " << beta << ", "
                  << (void*)C_data << ", " << ldc << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_cher2k BLAS_FORTRAN_NAME( slate_cher2k, SLATE_CHER2K )
void slate_cher2k(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    std::complex<float> const* alpha,
    std::complex<float>* A_data, blas_int const* lda,
    std::complex<float>* B_data, blas_int const* ldb,
    float const* beta,
    std::complex<float>* C_data, blas_int const* ldc )
{
    slate_her2k(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

#define slate_zher2k BLAS_FORTRAN_NAME( slate_zher2k, SLATE_ZHER2K )
void slate_zher2k(
    const char* uplo, const char* transA,
    blas_int const* n, blas_int const* k,
    std::complex<double> const* alpha,
    std::complex<double>* A_data, blas_int const* lda,
    std::complex<double>* B_data, blas_int const* ldb,
    double const* beta,
    std::complex<double>* C_data, blas_int const* ldc )
{
    slate_her2k(
        uplo, transA, *n, *k, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
