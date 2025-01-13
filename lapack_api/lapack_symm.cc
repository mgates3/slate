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
void slate_symm(
    const char* side_str, const char* uplo_str,
    blas_int m, blas_int n, scalar_t alpha,
    scalar_t* A_data, blas_int lda,
    scalar_t* B_data, blas_int ldb, scalar_t beta,
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

    Side side{};
    Uplo uplo{};
    from_string( std::string( 1, side_str[0] ), &side );
    from_string( std::string( 1, uplo_str[0] ), &uplo );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // sizes of data
    int64_t An = (side == blas::Side::Left ? m : n);
    int64_t Bm = m;
    int64_t Bn = n;
    int64_t Cm = m;
    int64_t Cn = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::SymmetricMatrix<scalar_t>::fromLAPACK(
        uplo, An,
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

    if (side == blas::Side::Left)
        assert( A.mt() == C.mt() );
    else
        assert( A.mt() == C.nt() );
    assert( B.mt() == C.mt() );
    assert( B.nt() == C.nt() );

    slate::symm( side, alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "symm( "
                  << side_str[0] << ", " << uplo_str[0] << ", "
                  << m << ", " << n << ", " << alpha << ", "
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

#define slate_ssymm BLAS_FORTRAN_NAME( slate_ssymm, SLATE_SSYMM )
void slate_ssymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n,
    float* alpha,
    float* A_data, blas_int const* lda,
    float* B_data, blas_int const* ldb,
    float* beta,
    float* C_data, blas_int const* ldc )
{
    slate_symm(
        side, uplo, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

#define slate_dsymm BLAS_FORTRAN_NAME( slate_dsymm, SLATE_DSYMM )
void slate_dsymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n,
    double* alpha,
    double* A_data, blas_int const* lda,
    double* B_data, blas_int const* ldb,
    double* beta,
    double* C_data, blas_int const* ldc )
{
    slate_symm(
        side, uplo, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

#define slate_csymm BLAS_FORTRAN_NAME( slate_csymm, SLATE_CSYMM )
void slate_csymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n,
    std::complex<float>* alpha,
    std::complex<float>* A_data, blas_int const* lda,
    std::complex<float>* B_data, blas_int const* ldb,
    std::complex<float>* beta,
    std::complex<float>* C_data, blas_int const* ldc )
{
    slate_symm(
        side, uplo, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

#define slate_zsymm BLAS_FORTRAN_NAME( slate_zsymm, SLATE_ZSYMM )
void slate_zsymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n,
    std::complex<double>* alpha,
    std::complex<double>* A_data, blas_int const* lda,
    std::complex<double>* B_data, blas_int const* ldb,
    std::complex<double>* beta,
    std::complex<double>* C_data, blas_int const* ldc )
{
    slate_symm(
        side, uplo, *m, *n, *alpha,
        A_data, *lda,
        B_data, *ldb, *beta,
        C_data, *ldc );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
