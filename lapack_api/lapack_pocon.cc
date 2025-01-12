// Copyright (c) 2017-2025, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack_slate.hh"

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------
// Local function
template <typename scalar_t>
void slate_pocon(const char* uplostr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t> Anorm, blas::real_type<scalar_t>* rcond, scalar_t* work, int* iwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_spocon BLAS_FORTRAN_NAME( slate_spocon, SLATE_SPOCON )
#define slate_dpocon BLAS_FORTRAN_NAME( slate_dpocon, SLATE_DPOCON )
#define slate_cpocon BLAS_FORTRAN_NAME( slate_cpocon, SLATE_CPOCON )
#define slate_zpocon BLAS_FORTRAN_NAME( slate_zpocon, SLATE_ZPOCON )

extern "C" void slate_spocon(const char* uplostr, const int* n, float* a, const int* lda, float* Anorm, float* rcond, float* work, int* iwork, int* info)
{
    slate_pocon(uplostr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}
extern "C" void slate_dpocon(const char* uplostr, const int* n, double* a, const int* lda, double* Anorm, double* rcond, double* work, int* iwork, int* info)
{
    slate_pocon(uplostr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}
extern "C" void slate_cpocon(const char* uplostr, const int* n, std::complex<float>* a, const int* lda, float* Anorm, float* rcond, std::complex<float>* work, int* iwork, int* info)
{
    slate_pocon(uplostr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}
extern "C" void slate_zpocon(const char* uplostr, const int* n, std::complex<double>* a, const int* lda, double* Anorm, double* rcond, std::complex<double>* work, int* iwork, int* info)
{
    slate_pocon(uplostr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_pocon(const char* uplostr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t> Anorm, blas::real_type<scalar_t>* rcond, scalar_t* work, int* iwork, int* info)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // Check and initialize MPI, else SLATE calls to MPI will fail
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();

    Uplo uplo{};
    from_string( std::string( 1, uplostr[0] ), &uplo );

    // sizes
    static int64_t nb = slate_lapack_set_nb(target);

    // create SLATE matrix from the LAPACK data
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(uplo, n, a, lda, nb, p, q, MPI_COMM_WORLD);

    // solve
    *rcond = slate::pocondest( slate::Norm::One, A, Anorm, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get a real value for info
    *info = 0;

    if (verbose) std::cout << "slate_lapack_api: " << to_char(a) << "pocon(" <<  uplostr[0] << "," << n << "," << (void*)a << "," <<  lda << "," << Anorm << "," << (void*)rcond << "," << (void*)work << "," << (void*)iwork << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
