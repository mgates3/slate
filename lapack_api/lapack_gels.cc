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
void slate_pgels(
    const char* trans_str, blas_int m, blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int lda,
    scalar_t* B_data, blas_int ldb,
    scalar_t* work, blas_int lwork,
    blas_int* info )
{
    using real_t = blas::real_type<scalar_t>;

    // Respond to workspace query with A_data minimal value (1); workspace
    // is allocated within the SLATE routine.
    if (lwork == -1) {
        work[0] = (real_t)1.0;
        *info = 0;
        return;
    }

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
    int64_t nb = NBConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();

    Op trans{};
    from_string( std::string( 1, trans_str[0] ), &trans );

    // sizes
    // A is m-by-n, BX is max(m, n)-by-nrhs.
    // If op == NoTrans, op(A) is m-by-n, B is m-by-nrhs
    // otherwise,        op(A) is n-by-m, B is n-by-nrhs.
    int64_t Am = (trans == slate::Op::NoTrans ? m : n);
    int64_t An = (trans == slate::Op::NoTrans ? n : m);
    int64_t Bm = (trans == slate::Op::NoTrans ? m : n);
    int64_t Bn = nrhs;

    // create SLATE matrices from the LAPACK layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(
        Am, An,
        A_data, lda,
        nb, p, q, MPI_COMM_SELF );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
        Bm, Bn,
        B_data, ldb,
        nb, p, q, MPI_COMM_SELF );

    // Apply transpose
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose( A );
    else if (trans == slate::Op::ConjTrans)
        opA = conj_transpose( A );

    slate::gels( opA, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    } );

    if (verbose) {
        std::cout << "slate_lapack_api: " << to_char(A_data) << "gels( "
                  << trans_str[0] << ", "
                  << m << ", " << n << ", " << nrhs << ", "
                  << (void*)A_data << ", " << lda << ", "
                  << (void*)B_data << ", " << ldb << ", "
                  << (void*)work << ", " << lwork << ", "
                  << *info << " ) "
                  << (omp_get_wtime() - timestart) << " sec"
                  << " nb: " << nb
                  << " max_threads: " << omp_get_max_threads() << "\n";
    }

    // todo: extract the real info
    *info = 0;
}

//------------------------------------------------------------------------------
// Fortran interfaces

extern "C" {

#define slate_sgels BLAS_FORTRAN_NAME( slate_sgels, SLATE_SGELS )
void slate_sgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    float* A_data, blas_int* lda,
    float* B_data, blas_int* ldb,
    float* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *lda,
        B_data, *ldb,
        work, *lwork, info );
}

#define slate_dgels BLAS_FORTRAN_NAME( slate_dgels, SLATE_DGELS )
void slate_dgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    double* A_data, blas_int* lda,
    double* B_data, blas_int* ldb,
    double* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *lda,
        B_data, *ldb,
        work, *lwork, info );
}

#define slate_cgels BLAS_FORTRAN_NAME( slate_cgels, SLATE_CGELS )
void slate_cgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    std::complex<float>* A_data, blas_int* lda,
    std::complex<float>* B_data, blas_int* ldb,
    std::complex<float>* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *lda,
        B_data, *ldb,
        work, *lwork, info );
}

#define slate_zgels BLAS_FORTRAN_NAME( slate_zgels, SLATE_ZGELS )
void slate_zgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    std::complex<double>* A_data, blas_int* lda,
    std::complex<double>* B_data, blas_int* ldb,
    std::complex<double>* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *lda,
        B_data, *ldb,
        work, *lwork, info );
}

} // extern "C"

} // namespace lapack_api
} // namespace slate
