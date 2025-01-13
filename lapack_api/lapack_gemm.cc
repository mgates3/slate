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
void slate_gemm(
    const char* transa_str, const char* transb_str,
    blas_int m, blas_int n, blas_int k, scalar_t alpha,
    scalar_t* A_data, blas_int lda,
    scalar_t* B_data, blas_int ldb, scalar_t beta,
    scalar_t* C_data, blas_int ldc )
{
    // Start timing
    int verbose = VerboseConfig::value();
    double timestart = 0.0;
    if (verbose)
        timestart = omp_get_wtime();

    // Need A_data dummy MPI_Init for SLATE to proceed
    blas_int initialized, provided;
    MPI_Initialized( &initialized );
    if (! initialized)
        MPI_Init_thread( nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided );

    int64_t p = 1;
    int64_t q = 1;
    int64_t lookahead = 1;
    slate::Target target = TargetConfig::value();

    Op transA{};
    Op transB{};
    from_string( std::string( 1, transa_str[0] ), &transA );
    from_string( std::string( 1, transb_str[0] ), &transB );

    // sizes
    int64_t Am = (transA == blas::Op::NoTrans ? m : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : m);
    int64_t Bm = (transB == blas::Op::NoTrans ? k : n);
    int64_t Bn = (transB == blas::Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;
    int64_t nb = NBConfig::value();

    // create SLATE matrices from the Lapack layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
        Bm, Bn,
        B_data, ldb,
        nb, p, q, MPI_COMM_SELF );
    auto C = slate::Matrix<scalar_t>::fromLAPACK(
        Cm, Cn,
        C_data, ldc,
        nb, p, q, MPI_COMM_SELF );

    if (transA == blas::Op::Trans)
        A = transpose( A );
    else if (transA == blas::Op::ConjTrans)
        A = conj_transpose( A );

    if (transB == blas::Op::Trans)
        B = transpose( B );
    else if (transB == blas::Op::ConjTrans)
        B = conj_transpose( B );

    slate::gemm( alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "gemm( "
                  << transa_str[0] << ", " << transb_str[0] << ", "
                  << m << ", " << n << ", " << k << ", " << alpha << ", "
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

#define slate_sgemm BLAS_FORTRAN_NAME( slate_sgemm, SLATE_SGEMM )
void slate_sgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    float* alpha,
    float* A_data, blas_int* lda,
    float* B_data, blas_int* ldb,
    float* beta,
    float* C_data, blas_int* ldc )
{
    slate_gemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

#define slate_dgemm BLAS_FORTRAN_NAME( slate_dgemm, SLATE_DGEMM )
void slate_dgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    double* alpha,
    double* A_data, blas_int* lda,
    double* B_data, blas_int* ldb,
    double* beta,
    double* C_data, blas_int* ldc )
{
    slate_gemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

#define slate_cgemm BLAS_FORTRAN_NAME( slate_cgemm, SLATE_CGEMM )
void slate_cgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    std::complex<float>* alpha,
    std::complex<float>* A_data, blas_int* lda,
    std::complex<float>* B_data, blas_int* ldb,
    std::complex<float>* beta,
    std::complex<float>* C_data, blas_int* ldc )
{
    slate_gemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

#define slate_zgemm BLAS_FORTRAN_NAME( slate_zgemm, SLATE_ZGEMM )
void slate_zgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    std::complex<double>* alpha,
    std::complex<double>* A_data, blas_int* lda,
    std::complex<double>* B_data, blas_int* ldb,
    std::complex<double>* beta,
    std::complex<double>* C_data, blas_int* ldc )
{
    slate_gemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
