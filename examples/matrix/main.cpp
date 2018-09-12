/*  This file is part of the Vc project
    Copyright (C) 2009-2015 Matthias Kretz <kretz@kde.org>

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

static constexpr int UnrollOuterloop = 4;

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

    // The following function is a workaround for GCC 4.8, which fails to recognize
    // V::MemoryAlignment as a constant expression inside the alignas operator.
    static constexpr size_t dataAlignment() { return V::MemoryAlignment; }
    // The outer array stores N rows and does not require further padding. It must be
    // aligned correctly for Vc::Aligned loads and stores, though.
    alignas(dataAlignment()) std::array<RowArray, N> data;

public:
    Matrix()
    {
        for (int i = 0; i < int(N); ++i) {
            for (int j = N; j < int(NPadded); ++j) {
                data[i][j] = 0;
            }
        }
    }
    // returns a reference to the i-th row
    RowArray &operator[](size_t i) { return data[i]; }
    // const overload of the above
    const RowArray &operator[](size_t i) const { return data[i]; }
};

// vectorized matrix multiplication
template <typename T, size_t N>
inline Matrix<T, N> operator*(const Matrix<T, N> &a, const Matrix<T, N> &b)
{
    constexpr int NN = N;
    using V = Vc::Vector<T>;
    // resulting matrix c
    Matrix<T, N> c;

    // The row index (for a and c) is unrolled using the UnrollOuterloop stride. Therefore
    // the last rows may need special treatment if N is not a multiple of UnrollOuterloop.
    // N0 is the number of rows that can safely be iterated with a stride of
    // UnrollOuterloop.
    constexpr int N0 = N / UnrollOuterloop * UnrollOuterloop;
    for (int i = 0; i < N0; i += UnrollOuterloop) {
        // The iteration over the column index of b and c uses a stride of V::size(). This
        // enables row-vector loads (from b) and stores (to c). The matrix storage is
        // padded accordingly, ensuring correct bounds and alignment.
        for (int j = 0; j < NN; j += int(V::size())) {
            // This temporary variables are used to accumulate the results of the products
            // producing the new values for the c matrix. This variable is necessary
            // because we need a V object for data-parallel accumulation. Storing to c
            // directly stores to scalar objects and thus would drop the ability for
            // data-parallel (SIMD) addition.
            V c_ij[UnrollOuterloop];
            for (int n = 0; n < UnrollOuterloop; ++n) {
                c_ij[n] = a[i + n][0] * V(&b[0][j], Vc::Aligned);
            }
            for (int k = 1; k < NN - 1; ++k) {
                for (int n = 0; n < UnrollOuterloop; ++n) {
                    c_ij[n] += a[i + n][k] * V(&b[k][j], Vc::Aligned);
                }
            }
            for (int n = 0; n < UnrollOuterloop; ++n) {
                c_ij[n] += a[i + n][NN - 1] * V(&b[NN - 1][j], Vc::Aligned);
                c_ij[n].store(&c[i + n][j], Vc::Aligned);
            }
        }
    }
    // This final loop treats the remaining NN - N0 rows.
    for (int j = 0; j < NN; j += int(V::size())) {
        V c_ij[UnrollOuterloop];
        for (int n = N0; n < NN; ++n) {
            c_ij[n - N0] = a[n][0] * V(&b[0][j], Vc::Aligned);
        }
        for (int k = 1; k < NN - 1; ++k) {
            for (int n = N0; n < NN; ++n) {
                c_ij[n - N0] += a[n][k] * V(&b[k][j], Vc::Aligned);
            }
        }
        for (int n = N0; n < NN; ++n) {
            c_ij[n - N0] += a[n][NN - 1] * V(&b[N - 1][j], Vc::Aligned);
            c_ij[n - N0].store(&c[n][j], Vc::Aligned);
        }
    }
    return c;
}

// scalar matrix multiplication
template <typename T, size_t N>
Matrix<T, N> scalar_mul(const Matrix<T, N> &a, const Matrix<T, N> &b)
{
    constexpr int NN = N;
    Matrix<T, N> c;
    for (int i = 0; i < NN; ++i) {
        for (int j = 0; j < NN; ++j) {
            c[i][j] = a[i][0] * b[0][j];
            for (int k = 1; k < NN; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

// scalar matrix multiplication
template <typename T, size_t N>
Matrix<T, N> scalar_mul_blocked(const Matrix<T, N> &a, const Matrix<T, N> &b)
{
    constexpr int NN = N;
    Matrix<T, N> c;
    constexpr int N0 = N / UnrollOuterloop * UnrollOuterloop;
    for (int i = 0; i < N0; i += UnrollOuterloop) {
        for (int j = 0; j < NN; ++j) {
            for (int n = 0; n < UnrollOuterloop; ++n) {
                c[i + n][j] = a[i + n][0] * b[0][j];
            }
            for (int k = 1; k < NN; ++k) {
                for (int n = 0; n < UnrollOuterloop; ++n) {
                    c[i + n][j] += a[i + n][k] * b[k][j];
                }
            }
        }
    }
    for (int j = 0; j < NN; ++j) {
        for (int n = N0; n < NN; ++n) {
            c[n][j] = a[n][0] * b[0][j];
        }
        for (int k = 1; k < NN; ++k) {
            for (int n = N0; n < NN; ++n) {
                c[n][j] += a[n][k] * b[k][j];
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

#ifdef Vc_MSVC
#pragma optimize("", off)
template <typename T> void unused(T &&x) { x = x; }
#pragma optimize("", on)
#else
template <typename T> void unused(T &&x) { asm("" ::"m"(x)); }
#endif

template <size_t N, typename F> Vc_ALWAYS_INLINE void benchmark(F &&f)
{
    TimeStampCounter tsc;
    auto cycles = tsc.cycles();
    cycles = 0x7fffffff;
    for (int i = 0; i < 100; ++i) {
        tsc.start();
        for (int j = 0; j < 10; ++j) {
            auto C = f();
            unused(C);
        }
        tsc.stop();
        cycles = std::min(cycles, tsc.cycles());
    }
    //std::cout << cycles << " Cycles for " << N *N *(N + N - 1) << " FLOP => ";
    std::cout << std::setw(19) << std::setprecision(3)
              << double(N * N * (N + N - 1) * 10) / cycles;
    //<< " FLOP/cycle (" << variant << ")\n";
}

template <size_t N> void run()
{
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
    std::cout << std::setw(2) << N;
#if defined Vc_MSVC
    auto &&fakeModify = [](Matrix<float, N> &a, Matrix<float, N> &b) {
        unused(a);
        unused(b);
    };
#else
    auto &&fakeModify = [](Matrix<float, N> &a, Matrix<float, N> &b) {
#ifdef Vc_ICC
        asm("" ::"r"(&a), "r"(&b));
#else
        asm("" : "+m"(a), "+m"(b));
#endif
    };
#endif
    benchmark<N>([&] {
        fakeModify(A, B);
        return scalar_mul(A, B);
    });
    benchmark<N>([&] {
        fakeModify(A, B);
        return scalar_mul_blocked(A, B);
    });
    benchmark<N>([&] {
        fakeModify(A, B);
        return A * B;
    });
    benchmark<N>([&] {
        fakeModify(A, B);
        return AV * BV;
    });
    std::cout << std::endl;
}

int Vc_CDECL main()
{
    std::cout << " N             scalar   scalar & blocked          Vector<T>           valarray\n";
    run< 4>();
    run< 5>();
    run< 6>();
    run< 7>();
    run< 8>();
    run< 9>();
    run<10>();
    run<11>();
    run<12>();
    run<13>();
    run<14>();
    run<15>();
    run<16>();
    run<17>();
    run<18>();
    run<19>();
    run<20>();
    run<21>();
    run<22>();
    run<23>();
    run<24>();
    run<25>();
    run<26>();
    run<27>();
    run<28>();
    run<29>();
    run<30>();
    run<31>();
    run<32>();
    run<33>();
    run<34>();
    run<35>();
    return 0;
}
