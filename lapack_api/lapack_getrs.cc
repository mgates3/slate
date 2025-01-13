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
void slate_getrs(
    const char* trans_str, blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int lda,
    blas_int* ipiv,
    scalar_t* B_data, blas_int ldb,
    blas_int* info )
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

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    slate::Target target = TargetConfig::value();

    Op trans{};
    from_string( std::string( 1, trans_str[0] ), &trans );

    // sizes
    int64_t Am = n, An = n;
    int64_t Bm = n, Bn = nrhs;
    int64_t nb = NBConfig::value();

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
        Bm, Bn,
        B_data, ldb,
        nb, p, q, MPI_COMM_SELF );

    // extract pivots from LAPACK ipiv to SLATES pivot structure
    slate::Pivots pivots; // std::vector< std::vector<Pivot> >
    {
        // allocate pivots
        int64_t min_mt_nt = std::min(A.mt(), A.nt());
        pivots.resize(min_mt_nt);
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);
        }
        // transfer ipiv to pivots
        int64_t p_count = 0;
        int64_t t_iter_add = 0;
        for (auto t_iter = pivots.begin(); t_iter != pivots.end(); ++t_iter) {
            for (auto p_iter = t_iter->begin(); p_iter != t_iter->end(); ++p_iter) {
                int64_t tileIndex = (ipiv[p_count] - 1 - t_iter_add) / nb;
                int64_t elementOffset = (ipiv[p_count] - 1 - t_iter_add) % nb;
                *p_iter = Pivot(tileIndex, elementOffset);
                ++p_count;
            }
            t_iter_add += nb;
        }
    }

    // apply operator to A
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose( A );
    else if (trans == slate::Op::ConjTrans)
        opA = conj_transpose( A );

    // solve
    slate::getrs( opA, pivots, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get A_data real value for info
    *info = 0;

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "getrs( "
                  << trans_str[0] << ", "
                  << n << ", " << nrhs << ", "
                  << (void*)A_data << ", " << lda << ", " << (void*)ipiv << ", "
                  << (void*)B_data << ", " << ldb << ", "
                  << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_sgetrs BLAS_FORTRAN_NAME( slate_sgetrs, SLATE_SGETRS )
void slate_sgetrs(
    const char* trans, blas_int const* n, blas_int const* nrhs,
    float* A_data, blas_int const* lda,
    blas_int* ipiv,
    float* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_getrs( trans, *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

#define slate_dgetrs BLAS_FORTRAN_NAME( slate_dgetrs, SLATE_DGETRS )
void slate_dgetrs(
    const char* trans, blas_int const* n, blas_int const* nrhs,
    double* A_data, blas_int const* lda,
    blas_int* ipiv,
    double* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_getrs( trans, *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

#define slate_cgetrs BLAS_FORTRAN_NAME( slate_cgetrs, SLATE_CGETRS )
void slate_cgetrs(
    const char* trans, blas_int const* n, blas_int const* nrhs,
    std::complex<float>* A_data, blas_int const* lda,
    blas_int* ipiv,
    std::complex<float>* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_getrs( trans, *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

#define slate_zgetrs BLAS_FORTRAN_NAME( slate_zgetrs, SLATE_ZGETRS )
void slate_zgetrs(
    const char* trans, blas_int const* n, blas_int const* nrhs,
    std::complex<double>* A_data, blas_int const* lda,
    blas_int* ipiv,
    std::complex<double>* B_data, blas_int const* ldb,
    blas_int* info )
{
    slate_getrs(  trans, *n, *nrhs, A_data, *lda, ipiv, B_data, *ldb, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
