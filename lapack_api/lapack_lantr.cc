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
blas::real_type<scalar_t> slate_lantr(
    const char* norm_str, const char* uplo_str, const char* diag_str,
    blas_int m, blas_int n,
    scalar_t* A_data, blas_int lda,
    blas::real_type<scalar_t>* work)
{
    // quick return
    if (std::min(m, n) == 0)
        return 0;

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

    int64_t Am = m;
    int64_t An = n;

    Norm norm{};
    Uplo uplo{};
    Diag diag{};
    from_string( std::string( 1, norm_str[0] ), &norm );
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, diag_str[0] ), &diag );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // create SLATE matrix from the Lapack layouts
    auto A = slate::TrapezoidMatrix<scalar_t>::fromLAPACK(
        uplo, diag, Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm( norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "lantr( "
                  << norm_str[0] << ", " << uplo_str[0] << ", "
                  << diag_str[0] << ", "
                  << m << ", " << n << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << (void*)work << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }

    return A_norm;
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_slantr BLAS_FORTRAN_NAME( slate_slantr, SLATE_SLANTR )
float slate_slantr(
    const char* norm, const char* uplo, const char* diag,
    blas_int const* m, blas_int const* n,
    float* A_data, blas_int* lda,
    float* work )
{
    return slate_lantr( norm, uplo, diag, *m, *n, A_data, *lda, work );
}

#define slate_dlantr BLAS_FORTRAN_NAME( slate_dlantr, SLATE_DLANTR )
double slate_dlantr(
    const char* norm, const char* uplo, const char* diag,
    blas_int const* m, blas_int const* n,
    double* A_data, blas_int* lda,
    double* work )
{
    return slate_lantr( norm, uplo, diag, *m, *n, A_data, *lda, work );
}

#define slate_clantr BLAS_FORTRAN_NAME( slate_clantr, SLATE_CLANTR )
float slate_clantr(
    const char* norm, const char* uplo, const char* diag,
    blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int* lda,
    float* work )
{
    return slate_lantr( norm, uplo, diag, *m, *n, A_data, *lda, work );
}

#define slate_zlantr BLAS_FORTRAN_NAME( slate_zlantr, SLATE_ZLANTR )
double slate_zlantr(
    const char* norm, const char* uplo, const char* diag,
    blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int* lda,
    double* work )
{
    return slate_lantr( norm, uplo, diag, *m, *n, A_data, *lda, work );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
