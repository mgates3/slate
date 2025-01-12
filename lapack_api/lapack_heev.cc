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
void slate_heev(const char* jobzstr, const char* uplostr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t>* w, scalar_t* work, const int lwork, blas::real_type<scalar_t>* rwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_ssyev BLAS_FORTRAN_NAME( slate_ssyev, SLATE_SSYEV )
#define slate_dsyev BLAS_FORTRAN_NAME( slate_dsyev, SLATE_DSYEV )
#define slate_cheev BLAS_FORTRAN_NAME( slate_cheev, SLATE_CHEEV )
#define slate_zheev BLAS_FORTRAN_NAME( slate_zheev, SLATE_ZHEEV )

extern "C" void slate_ssyev(const char* jobzstr, const char* uplostr, const int* n, float* a, const int* lda, float* w, float* work, const int* lwork, int* info)
{
    slate_heev(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, nullptr, info);
}
extern "C" void slate_dsyev(const char* jobzstr, const char* uplostr, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* info)
{
    slate_heev(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, nullptr, info);
}
extern "C" void slate_cheev(const char* jobzstr, const char* uplostr, const int* n, std::complex<float>* a, const int* lda, float* w, std::complex<float>* work, const int* lwork, float* rwork, int* info)
{
    slate_heev(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, rwork, info);
}
extern "C" void slate_zheev(const char* jobzstr, const char* uplostr, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, int* info)
{
    slate_heev(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, rwork, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_heev(const char* jobzstr, const char* uplostr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t>* w, scalar_t* work, const int lwork, blas::real_type<scalar_t>* rwork, int* info)
{
    // Start timing
    int verbose = VerboseConfig::value();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // sizes
    slate::Target target = TargetConfig::value();
    int64_t nb = NBConfig::value();

    // TODO check args more carefully
    *info = 0;

    if (lwork == -1) {
        work[0] = n * n;
    }
    else {
        // Check and initialize MPI, else SLATE calls to MPI will fail
        int initialized, provided;
        MPI_Initialized(&initialized);
        if (! initialized)
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

        int64_t lookahead = 1;
        int64_t p = 1;
        int64_t q = 1;

        Uplo uplo{};
        from_string( std::string( 1, uplostr[0] ), &uplo );

        // create SLATE matrix from the LAPACK data
        auto A = slate::Matrix<scalar_t>::fromLAPACK( n, n, a, lda, nb, p, q, MPI_COMM_WORLD );
        slate::HermitianMatrix<scalar_t> AH( uplo, A );
        std::vector< blas::real_type<scalar_t> > Lambda_( n );

        slate::Matrix<scalar_t> Z;
        switch (jobzstr[0]) {
            case 'V':
                if (lwork >= n * n) {
                    Z = slate::Matrix<scalar_t>::fromLAPACK( n, n, work, n, nb, p, q, MPI_COMM_WORLD );
                }
                else {
                    Z = slate::Matrix<scalar_t>( n, n, nb, p, q, MPI_COMM_WORLD );
                    Z.insertLocalTiles(target);
                }
                break;
            case 'N':
                // Leave Z empty
                break;
            default:
                *info = 1;
        }

        if (*info == 0) {
            // solve
            slate::heev( AH, Lambda_, Z, {
                {slate::Option::MethodEig, MethodEig::QR},
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            std::copy(Lambda_.begin(), Lambda_.end(), w);

            if (jobzstr[0] == 'V') {
                slate::copy( Z, A, {
                    {slate::Option::Target, target}
                });
            }
        }
    }

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(a) << "heev( "
                  << jobzstr[0] << ", " << uplostr[0] << ", "
                  << n << ", "
                  << (void*)a << ", " << lda << ", " << (void*)w << ", "
                  << (void*)work << ", " << lwork << ", ";
        if (is_complex<scalar_t>::value) {
            std::cout << (void*)rwork << ", ";
        }
        std::cout << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

} // namespace lapack_api
} // namespace slate
