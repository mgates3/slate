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
blas::real_type<scalar_t> slate_lange(
    const char* norm_str,
    blas_int m, blas_int n,
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
    from_string( std::string( 1, norm_str[0] ), &norm );

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // sizes of matrices
    int64_t Am = m;
    int64_t An = n;

    // create SLATE matrix from the Lapack layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm( norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "lange( "
                  << norm_str[0] << ", "
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

#define slate_slange BLAS_FORTRAN_NAME( slate_slange, SLATE_SLANGE )
float slate_slange(
    const char* norm,
    blas_int const* m, blas_int const* n,
    float* A_data, blas_int* lda,
    float* work )
{
    return slate_lange( norm, *m, *n, A_data, *lda, work );
}

#define slate_dlange BLAS_FORTRAN_NAME( slate_dlange, SLATE_DLANGE )
double slate_dlange(
    const char* norm,
    blas_int const* m, blas_int const* n,
    double* A_data, blas_int* lda,
    double* work )
{
    return slate_lange( norm, *m, *n, A_data, *lda, work );
}

#define slate_clange BLAS_FORTRAN_NAME( slate_clange, SLATE_CLANGE )
float slate_clange(
    const char* norm,
    blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int* lda,
    float* work )
{
    return slate_lange( norm, *m, *n, A_data, *lda, work );
}

#define slate_zlange BLAS_FORTRAN_NAME( slate_zlange, SLATE_ZLANGE )
double slate_zlange(
    const char* norm,
    blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int* lda,
    double* work )
{
    return slate_lange( norm, *m, *n, A_data, *lda, work );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
