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
void slate_trsm(
    const char* side_str, const char* uplo_str,
    const char* transa_str, const char* diag_str,
    blas_int m, blas_int n, scalar_t alpha,
    scalar_t* A_data, blas_int lda,
    scalar_t* B_data, blas_int ldb )
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

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    Side side{};
    Uplo uplo{};
    Op transA{};
    Diag diag{};
    from_string( std::string( 1, side_str[0] ), &side );
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, transa_str[0] ), &transA );
    from_string( std::string( 1, diag_str[0] ), &diag );

    // setup so op(B) is m-by-n
    int64_t An  = (side == blas::Side::Left ? m : n);
    int64_t Bm  = m;
    int64_t Bn  = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::TriangularMatrix<scalar_t>::fromLAPACK(
        uplo, diag, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
        Bm, Bn,
        B_data, ldb,
        nb, p, q, MPI_COMM_SELF );

    if (transA == Op::Trans)
        A = transpose( A );
    else if (transA == Op::ConjTrans)
        A = conj_transpose( A );

    slate::trsm( side, alpha, A, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "trsm( "
                  << side_str[0] << ", " << uplo_str[0] << ", "
                  << transa_str[0] << ", " << diag_str[0] << ", "
                  << m << ", " << n << ", " << alpha << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << (void*)B_data << ", " << ldb << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_strsm BLAS_FORTRAN_NAME( slate_strsm, SLATE_STRSM )
void slate_strsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n,
    float const* alpha,
    float* A_data, blas_int const* lda,
    float* B_data, blas_int const* ldb )
{
    slate_trsm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb );
}

#define slate_dtrsm BLAS_FORTRAN_NAME( slate_dtrsm, SLATE_DTRSM )
void slate_dtrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n,
    double const* alpha,
    double* A_data, blas_int const* lda,
    double* B_data, blas_int const* ldb )
{
    slate_trsm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb );
}

#define slate_ctrsm BLAS_FORTRAN_NAME( slate_ctrsm, SLATE_CTRSM )
void slate_ctrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n,
    std::complex<float> const* alpha,
    std::complex<float>* A_data, blas_int const* lda,
    std::complex<float>* B_data, blas_int const* ldb )
{
    slate_trsm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb );
}

#define slate_ztrsm BLAS_FORTRAN_NAME( slate_ztrsm, SLATE_ZTRSM )
void slate_ztrsm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n,
    std::complex<double> const* alpha,
    std::complex<double>* A_data, blas_int const* lda,
    std::complex<double>* B_data, blas_int const* ldb )
{
    slate_trsm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
