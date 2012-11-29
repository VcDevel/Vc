/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#ifndef VC_MATRIX_H
#define VC_MATRIX_H

#include <cstdlib>

template<typename T, size_t Rows, size_t Cols = 1> struct Matrix;
template<typename T, size_t Rows, size_t Cols> struct MatrixScalar;
template<typename T, size_t Rows, size_t Cols, typename Impl> struct MatrixOperand;
template<typename T, size_t Rows, size_t Cols, typename Impl> struct MatrixOperandTransposed;
template<typename T, size_t Rows, size_t Cols, size_t FirstRow, size_t FirstCol, typename Impl> struct MatrixOperandSlice;
template<typename T, typename L, typename R, size_t K> struct EvaluateMatrixProduct
{
    static inline T evaluate(const L &left, const R &right, size_t r, size_t c, T sum) {
        return EvaluateMatrixProduct<T, L, R, K - 1>::evaluate(left, right, r, c, sum + left(r, K - 1) * right(K - 1, c));
    }
};
template<typename T, typename L, typename R> struct EvaluateMatrixProduct<T, L, R, 0>
{
    static inline T evaluate(const L &, const R &, size_t, size_t, T sum) { return sum; }
};

template<typename T, size_t Rows, size_t Cols, size_t K, typename Left, typename Right> struct MatrixProduct
    : MatrixOperand<T, Rows, Cols, MatrixProduct<T, Rows, Cols, K, Left, Right> >
{
    const Left &left;
    const Right &right;
    MatrixProduct(const Left &l, const Right &r) : left(l), right(r) {}

    inline const T operator()(size_t r, size_t c = 0) const {
        return EvaluateMatrixProduct<T, Left, Right, K - 1>::evaluate(left, right, r, c, left(r, K - 1) * right(K - 1, c));
    }
};
template<typename T, size_t Rows, size_t Cols, typename Left> struct MatrixProduct<T, Rows, Cols, 1, Left, T>
    : MatrixOperand<T, Rows, Cols, MatrixProduct<T, Rows, Cols, 1, Left, T> >
{
    const Left &left;
    const T right;
    MatrixProduct(const Left &l, const T r) : left(l), right(r) {}

    inline const T operator()(size_t r, size_t c = 0) const {
        return left(r, c) * right;
    }
};
template<typename T, size_t Rows, size_t Cols, typename Left, typename Right> struct MatrixSubtraction
    : MatrixOperand<T, Rows, Cols, MatrixSubtraction<T, Rows, Cols, Left, Right> >
{
    const Left &left;
    const Right &right;
    MatrixSubtraction(const Left &l, const Right &r) : left(l), right(r) {}

    inline const T operator()(size_t r, size_t c = 0) const {
        return left(r, c) - right(r, c);
    }
};
template<typename T, size_t Rows, size_t Cols, typename Left> struct MatrixSubtraction<T, Rows, Cols, Left, T>
    : MatrixOperand<T, Rows, Cols, MatrixSubtraction<T, Rows, Cols, Left, T> >
{
    const Left &left;
    const T right;
    MatrixSubtraction(const Left &l, const T r) : left(l), right(r) {}

    inline const T operator()(size_t r, size_t c = 0) const {
        return left(r, c) - right;
    }
};
template<typename T, size_t Rows, size_t Cols, typename Impl> struct MatrixOperandCastHelper {};
template<typename T, typename Impl> struct MatrixOperandCastHelper<T, 1, 1, Impl>
{
    // allow 1x1 matrices to cast to T
    inline operator const T() const { return static_cast<const Impl *>(this)->operator()(0, 0); }
};
template<typename T, size_t Rows, size_t Cols, typename Impl> struct MatrixOperand : public MatrixOperandCastHelper<T, Rows, Cols, Impl>
{
    T &operator()(size_t r, size_t c = 0) { return static_cast<Impl *>(this)->operator()(r, c); }
    const T operator()(size_t r, size_t c = 0) const { return static_cast<const Impl *>(this)->operator()(r, c); }
    template<size_t RhsCols, typename RhsImpl>
    inline MatrixProduct<T, Rows, RhsCols, Cols, Impl, RhsImpl> operator*(const MatrixOperand<T, Cols, RhsCols, RhsImpl> &rhs) const
    {
        return MatrixProduct<T, Rows, RhsCols, Cols, Impl, RhsImpl>(*static_cast<const Impl *>(this), static_cast<const RhsImpl &>(rhs));
    }
    inline MatrixProduct<T, Rows, Cols, 1, Impl, T> operator*(T scalar) const
    {
        return MatrixProduct<T, Rows, Cols, 1, Impl, T>(*static_cast<const Impl *>(this), scalar);
    }

    template<typename RhsImpl>
    inline MatrixSubtraction<T, Rows, Cols, Impl, RhsImpl> operator-(const MatrixOperand<T, Rows, Cols, RhsImpl> &rhs) const
    {
        return MatrixSubtraction<T, Rows, Cols, Impl, RhsImpl>(*static_cast<const Impl *>(this), static_cast<const RhsImpl &>(rhs));
    }
    inline MatrixSubtraction<T, Rows, Cols, Impl, T> operator-(T scalar) const
    {
        return MatrixSubtraction<T, Rows, Cols, Impl, T>(*static_cast<const Impl *>(this), scalar);
    }

    inline MatrixOperandTransposed<T, Cols, Rows, Impl> transposed() const {
        return MatrixOperandTransposed<T, Cols, Rows, Impl>(*this);
    }

    template<size_t FirstRow, size_t NewRows, size_t FirstCol, size_t NewCols>
    inline MatrixOperandSlice<T, NewRows, NewCols, FirstRow, FirstCol, MatrixOperand<T, Rows, Cols, Impl> > slice() const {
        return MatrixOperandSlice<T, NewRows, NewCols, FirstRow, FirstCol, MatrixOperand<T, Rows, Cols, Impl> >(*this);
    }

    template<size_t FirstRow, size_t NewRows>
    inline MatrixOperandSlice<T, NewRows, Cols, FirstRow, 0, MatrixOperand<T, Rows, Cols, Impl> > slice() const {
        return MatrixOperandSlice<T, NewRows, Cols, FirstRow, 0, MatrixOperand<T, Rows, Cols, Impl> >(*this);
    }
};

template<typename T, size_t Rows, size_t Cols> struct MatrixScalar : public MatrixOperand<T, Rows, Cols, MatrixScalar<T, Rows, Cols> >
// wrap one scalar like a matrix where all entries are equal
{
    T value;
    MatrixScalar(T v) : value(v) {}
    const T operator()(size_t, size_t) const { return value; }
};
template<typename T, size_t Rows, size_t Cols, typename Impl> struct MatrixOperandTransposed
    : public MatrixOperand<T, Rows, Cols, MatrixOperandTransposed<T, Rows, Cols, Impl> >
{
    const MatrixOperand<T, Cols, Rows, Impl> &m_operand;
    MatrixOperandTransposed(const MatrixOperand<T, Cols, Rows, Impl> &operand) : m_operand(operand) {}
    T &operator()(size_t r, size_t c = 0) { return m_operand(c, r); }
    const T operator()(size_t r, size_t c = 0) const { return m_operand(c, r); }
};

template<typename T, size_t Rows, size_t Cols, size_t FirstRow, size_t FirstCol, typename Impl> struct MatrixOperandSlice
    : public MatrixOperand<T, Rows, Cols, MatrixOperandSlice<T, Rows, Cols, FirstRow, FirstCol, Impl> >
{
    const Impl &m_operand;
    MatrixOperandSlice(const Impl &operand) : m_operand(operand) {}
    T &operator()(size_t r, size_t c = 0) { return m_operand(r + FirstRow, c + FirstCol); }
    const T operator()(size_t r, size_t c = 0) const { return m_operand(r + FirstRow, c + FirstCol); }
};

template<typename Lhs, typename Rhs, size_t R, size_t LastRow, size_t C, size_t LastCol> struct AssignMatrix
{
    static inline void evaluate(Lhs &lhs, const Rhs &rhs) {
        lhs(R, C) = rhs(R, C);
        AssignMatrix<Lhs, Rhs, R, LastRow, C + 1, LastCol>::evaluate(lhs, rhs);
    }
};
template<typename Lhs, typename Rhs, size_t R, size_t LastRow, size_t LastCol> struct AssignMatrix<Lhs, Rhs, R, LastRow, LastCol, LastCol>
{
    static inline void evaluate(Lhs &lhs, const Rhs &rhs) {
        lhs(R, LastCol) = rhs(R, LastCol);
        // next row
        AssignMatrix<Lhs, Rhs, R + 1, LastRow, 0, LastCol>::evaluate(lhs, rhs);
    }
};
template<typename Lhs, typename Rhs, size_t LastRow, size_t LastCol> struct AssignMatrix<Lhs, Rhs, LastRow, LastRow, LastCol, LastCol>
{
    static inline void evaluate(Lhs &lhs, const Rhs &rhs) {
        lhs(LastRow, LastCol) = rhs(LastRow, LastCol);
        // done
    }
};
template<typename T, size_t Rows, size_t Cols> struct Matrix : public MatrixOperand<T, Rows, Cols, Matrix<T, Rows, Cols> >
{
    T &operator()(size_t r, size_t c = 0) { return m_data[r][c]; }
    const T operator()(size_t r, size_t c = 0) const { return m_data[r][c]; }
    Matrix() {}
    template<typename Impl> Matrix(const MatrixOperand<T, Rows, Cols, Impl> &p) { operator=(p); }
    Matrix(T scalar) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                m_data[r][c] = scalar;
            }
        }
    }
    inline Matrix &operator=(T val) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                operator()(r, c) = val;
            }
        }
        return *this;
    }
    template<typename Impl>
    inline Matrix &operator=(const MatrixOperand<T, Rows, Cols, Impl> &p) {
        AssignMatrix<Matrix<T, Rows, Cols>, MatrixOperand<T, Rows, Cols, Impl>, 0, Rows - 1, 0, Cols - 1>::evaluate(*this, p);
        return *this;
    }
private:
    T m_data[Rows][Cols];
};

#endif // VC_MATRIX_H
