/*  This file is part of the Vc project
    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

*/

#include <Vc/Vc>
#include <Vc/IO>
#include <iostream>
#include <iomanip>
#include <valarray>
#include "../tsc.h"

static constexpr size_t UnrollOuterloop = 4;

template <typename T, size_t N> class MatrixValarray
{
    using V = std::valarray<T>;
    V data[N];

public:
    MatrixValarray()
    {
        for (auto &v : data) {
            v.resize(N);
        }
    }
    std::valarray<T> &operator[](std::size_t i) { return data[i]; }
    const std::valarray<T> &operator[](std::size_t i) const { return data[i]; }
};

template <typename T, size_t N>
inline MatrixValarray<T, N> operator*(const MatrixValarray<T, N> &a,
                                      const MatrixValarray<T, N> &b)
{
    MatrixValarray<T, N> c;
    for (size_t i = 0; i < N; ++i) {
        c[i] = a[i][0] * b[0];
        for (size_t k = 1; k < N; ++k) {
            c[i] += a[i][k] * b[k];
        }
    }
    return c;
}

template <typename T, size_t N> class Matrix
{
  using V = Vc::Vector<T>;

  // round up to the next multiple of V::size()
  static constexpr size_t NPadded = (N + V::size() - 1) / V::size() * V::size();

  // the inner array stores one row of values and is padded
  using RowArray = std::array<T, NPadded>;
  // The outer array stores N rows and does not require further padding. It must be aligned
  // correctly for Vc::Aligned loads and stores, though.
  alignas(V::MemoryAlignment) std::array<RowArray, N> data;

 public:
  // returns a reference to the i-th row
  RowArray &operator[](size_t i) { return data[i]; }
  // const overload of the above
  const RowArray &operator[](size_t i) const { return data[i]; }
};

// vectorized matrix multiplication
template <typename T, size_t N>
inline Matrix<T, N> operator*(const Matrix<T, N> &a, const Matrix<T, N> &b)
{
    using V = Vc::Vector<T>;
    // resulting matrix c
    Matrix<T, N> c;

    // The row index (for a and c) is unrolled using the UnrollOuterloop stride. Therefore the last
    // rows may need special treatment if N is not a multiple of UnrollOuterloop. N0 is the number
    // of rows that can safely be iterated with a stride of UnrollOuterloop.
    constexpr size_t N0 = N / UnrollOuterloop * UnrollOuterloop;
    for (size_t i = 0; i < N0; i += UnrollOuterloop) {
        // The iteration over the column index of b and c uses a stride of V::size(). This enables
        // row-vector loads (from b) and stores (to c). The matrix storage is padded accordingly,
        // ensuring correct bounds and alignment.
        for (size_t j = 0; j < N; j += V::size()) {
            // This temporary variables are used to accumulate the results of the products producing
            // the new values for the c matrix. This variable is necessary because we need a V
            // object for data-parallel accumulation. Storing to c directly stores to scalar objects
            // and thus would drop the ability for data-parallel (SIMD) addition.
            V c_ij[UnrollOuterloop];
            for (size_t n = 0; n < UnrollOuterloop; ++n) {
                c_ij[n] = a[i + n][0] * V(&b[0][j], Vc::Aligned);
            }
            for (size_t k = 1; k < N - 1; ++k) {
                for (size_t n = 0; n < UnrollOuterloop; ++n) {
                    c_ij[n] += a[i + n][k] * V(&b[k][j], Vc::Aligned);
                }
            }
            for (size_t n = 0; n < UnrollOuterloop; ++n) {
                c_ij[n] += a[i + n][N - 1] * V(&b[N - 1][j], Vc::Aligned);
                c_ij[n].store(&c[i + n][j], Vc::Aligned);
            }
        }
    }
    // This final loop treats the remaining N - N0 rows.
    for (size_t j = 0; j < N; j += V::size()) {
        V c_ij[UnrollOuterloop];
        for (size_t n = N0; n < N; ++n) {
            c_ij[n - N0] = a[n][0] * V(&b[0][j], Vc::Aligned);
        }
        for (size_t k = 1; k < N - 1; ++k) {
            for (size_t n = N0; n < N; ++n) {
                c_ij[n - N0] += a[n][k] * V(&b[k][j], Vc::Aligned);
            }
        }
        for (size_t n = N0; n < N; ++n) {
            c_ij[n - N0] += a[n][N - 1] * V(&b[N - 1][j], Vc::Aligned);
            c_ij[n - N0].store(&c[n][j], Vc::Aligned);
        }
    }
    return c;
}

// scalar matrix multiplication
template <typename T, size_t N>
Matrix<T, N> scalar_mul(const Matrix<T, N> &a, const Matrix<T, N> &b)
{
    Matrix<T, N> c;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            c[i][j] = a[i][0] * b[0][j];
            for (size_t k = 1; k < N; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

// valarray matrix multiplication
template <typename T, size_t N>
Matrix<T, N> valarray_mul(const Matrix<T, N> &a, const Matrix<T, N> &b)
{
    Matrix<T, N> c;
    using V = std::valarray<T>;
    for (size_t i = 0; i < N; ++i) {
        V c_i = a[i][0] * V(&b[0][0], N);
        for (size_t k = 1; k < N; ++k) {
            c_i += a[i][k] * V(&b[k][0], N);
        }
        std::copy(std::begin(c_i), std::end(c_i), &c[i][0]);
    }
    return c;
}

template <template <typename, size_t> class M, typename T, size_t N,
          typename = Vc::enable_if<(std::is_same<M<T, N>, Matrix<T, N>>::value ||
                                    std::is_same<M<T, N>, MatrixValarray<T, N>>::value)>>
std::ostream &operator<<(std::ostream &out, const M<T, N> &m)
{
    out.precision(3);
    auto &&w = std::setw(6);
    out << "⎡" << w << m[0][0];
    for (size_t j = 1; j < N; ++j) {
        out << ", " << w << m[0][j];
    }
    out << " ⎤\n";
    for (size_t i = 1; i + 1 < N; ++i) {
        out << "⎢" << w << m[i][0];
        for (size_t j = 1; j < N; ++j) {
            out << ", " << w << m[i][j];
        }
        out << " ⎥\n";
    }
    out << "⎣" << w << m[N - 1][0];
    for (size_t j = 1; j < N; ++j) {
        out << ", " << w << m[N - 1][j];
    }
    return out << " ⎦\n";
}

template <typename T> void unused(const T &) {}

int main()
{
    static constexpr size_t N = 23;
    Matrix<float, N> A;
    Matrix<float, N> B;
    MatrixValarray<float, N> AV;
    MatrixValarray<float, N> BV;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i][j] = 0.01 * (i + j);
            B[i][j] = 0.01 * (N + i - j);
            AV[i][j] = 0.01 * (i + j);
            BV[i][j] = 0.01 * (N + i - j);
        }
    }
    {
        auto C = A * B;
        std::cout << A << "times\n" << B << "=\n" << C;
        auto CS = scalar_mul(A, B);
        std::cout << "scalar=\n" << CS;
        auto CV = AV * BV;
        std::cout << "valarray=\n" << CV;
    }
    TimeStampCounter tsc;
    for (int i = 0; i < 10; ++i) {
        tsc.start();
        auto CS = scalar_mul(A, B);
        tsc.stop();
        unused(CS);
        std::cout << tsc << " Cycles for " << N * N * (N + N - 1) << " FLOP => ";
        std::cout << double(N * N * (N + N - 1)) / tsc.cycles() << " FLOP/cycle (scalar)\n";
    }
    for (int i = 0; i < 10; ++i) {
        tsc.start();
        auto C = A * B;
        tsc.stop();
        unused(C);
        std::cout << tsc << " Cycles for " << N * N * (N + N - 1) << " FLOP => ";
        std::cout << double(N * N * (N + N - 1)) / tsc.cycles() << " FLOP/cycle (vector)\n";
    }
    for (int i = 0; i < 10; ++i) {
        tsc.start();
        auto C = AV * BV;
        tsc.stop();
        unused(C);
        std::cout << tsc << " Cycles for " << N * N * (N + N - 1) << " FLOP => ";
        std::cout << double(N * N * (N + N - 1)) / tsc.cycles() << " FLOP/cycle (valarray)\n";
    }
    return 0;
}
