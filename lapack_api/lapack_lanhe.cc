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
blas::real_type<scalar_t> slate_lanhe(
    const char* norm_str, const char* uplo_str,
    blas_int n,
    scalar_t* A_data, blas_int lda,
    blas::real_type<scalar_t>* work)
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

    Norm norm{};
    Uplo uplo{};
    from_string( std::string( 1, norm_str[0] ), &norm );
    from_string( std::string( 1, uplo_str[0] ), &uplo );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // sizes of matrices
    int64_t An = n;

    // create SLATE matrix from the Lapack layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(
        uplo, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm( norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "lanhe( "
                  << norm_str[0] << ", " << uplo_str[0] << ", "
                  << n << ", "
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

#define slate_clanhe BLAS_FORTRAN_NAME( slate_clanhe, SLATE_CLANHE )
float slate_clanhe(
    const char* norm, const char* uplo,
    blas_int const* n,
    std::complex<float>* A_data, blas_int* lda,
    float* work )
{
    return slate_lanhe( norm, uplo, *n, A_data, *lda, work );
}

#define slate_zlanhe BLAS_FORTRAN_NAME( slate_zlanhe, SLATE_ZLANHE )
double slate_zlanhe(
    const char* norm, const char* uplo,
    blas_int const* n,
    std::complex<double>* A_data, blas_int* lda,
    double* work )
{
    return slate_lanhe( norm, uplo, *n, A_data, *lda, work );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
