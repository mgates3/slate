//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_Matrix.hh"
#include "slate_internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::trmm from internal::specialization::trmm
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel triangular matrix multiplication.
/// Generic implementation for any target.
template <Target target, typename scalar_t>
void trmm(slate::internal::TargetType<target>,
          Side side, Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int64_t lookahead)
{
    assert(A.mt() == A.nt());

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = A.nt()-1; k >= 0; --k) {

        auto Ak = A.sub(k, k, 0, k);
        auto Tk = TrapezoidMatrix< scalar_t >( Uplo::Lower, Ak );

        internal::tzmm<Target::HostTask>(
            side, diag,
            alpha, std::move(Tk),
                   B.sub(0, k, 0, B.nt()-1));
    }
}

} // namespace specialization
} // namespace internal

///-----------------------------------------------------------------------------
/// \brief
///
/// Precision and target templated function.
template <Target target, typename scalar_t>
void trmm(blas::Side side, blas::Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::trmm(internal::TargetType<target>(),
                                   side, diag,
                                   alpha, A,
                                          B,
                                   lookahead);
}

//------------------------------------------------------------------------------
// Explicit instantiations for double precision and various targets.
template
void trmm< Target::HostTask, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trmm< Target::HostTask, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

// ----------------------------------------
template
void trmm< Target::HostTask, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostNest, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::HostBatch, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< Target::Devices, std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

} // namespace slate
