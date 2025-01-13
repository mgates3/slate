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
void slate_potri(
    const char* uplo_str, blas_int n,
    scalar_t* A_data, blas_int lda,
    blas_int* info )
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
    from_string( std::string( 1, uplo_str[0] ), &uplo );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // sizes of data
    int64_t An = n;

    // create SLATE matrices from the Lapack layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(
        uplo, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );

    slate::potri( A, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo get A_data real value for info
    *info = 0;

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "potri( "
                  << uplo_str[0] << ", "
                  << n << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_spotri BLAS_FORTRAN_NAME( slate_spotri, SLATE_SPOTRI )
void slate_spotri(
    const char* uplo, blas_int const* n,
    float* A_data, blas_int const* lda,
    blas_int* info )
{
    slate_potri( uplo, *n, A_data, *lda, info );
}

#define slate_dpotri BLAS_FORTRAN_NAME( slate_dpotri, SLATE_DPOTRI )
void slate_dpotri(
    const char* uplo, blas_int const* n,
    double* A_data, blas_int const* lda,
    blas_int* info )
{
    slate_potri( uplo, *n, A_data, *lda, info );
}

#define slate_cpotri BLAS_FORTRAN_NAME( slate_cpotri, SLATE_CPOTRI )
void slate_cpotri(
    const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* lda,
    blas_int* info )
{
    slate_potri( uplo, *n, A_data, *lda, info );
}

#define slate_zpotri BLAS_FORTRAN_NAME( slate_zpotri, SLATE_ZPOTRI )
void slate_zpotri(
    const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* lda,
    blas_int* info )
{
    slate_potri( uplo, *n, A_data, *lda, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
