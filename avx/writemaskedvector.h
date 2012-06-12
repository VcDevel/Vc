/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

*/

#ifndef VC_AVX_WRITEMASKEDVECTOR_H
#define VC_AVX_WRITEMASKEDVECTOR_H

namespace Vc
{
namespace AVX
{

template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename DetermineEntryType<T>::Type EntryType;
    enum Constants { Size = sizeof(VectorType) / sizeof(EntryType) };
    typedef typename Vc::AVX::Mask<Size, sizeof(VectorType)> Mask;
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)
        //prefix
        Vector<T> ALWAYS_INLINE_L &operator++() ALWAYS_INLINE_R;
        Vector<T> ALWAYS_INLINE_L &operator--() ALWAYS_INLINE_R;
        //postfix
        Vector<T> ALWAYS_INLINE_L operator++(int) ALWAYS_INLINE_R;
        Vector<T> ALWAYS_INLINE_L operator--(int) ALWAYS_INLINE_R;

        Vector<T> ALWAYS_INLINE_L &operator+=(const Vector<T> &x) ALWAYS_INLINE_R;
        Vector<T> ALWAYS_INLINE_L &operator-=(const Vector<T> &x) ALWAYS_INLINE_R;
        Vector<T> ALWAYS_INLINE_L &operator*=(const Vector<T> &x) ALWAYS_INLINE_R;
        Vector<T> ALWAYS_INLINE_L &operator/=(const Vector<T> &x) ALWAYS_INLINE_R;
        Vector<T> ALWAYS_INLINE &operator+=(EntryType x) { return operator+=(Vector<T>(x)); }
        Vector<T> ALWAYS_INLINE &operator-=(EntryType x) { return operator-=(Vector<T>(x)); }
        Vector<T> ALWAYS_INLINE &operator*=(EntryType x) { return operator*=(Vector<T>(x)); }
        Vector<T> ALWAYS_INLINE &operator/=(EntryType x) { return operator/=(Vector<T>(x)); }

        Vector<T> ALWAYS_INLINE_L &operator=(const Vector<T> &x) ALWAYS_INLINE_R;
        Vector<T> ALWAYS_INLINE &operator=(EntryType x) { return operator=(Vector<T>(x)); }

        template<typename F> inline void INTRINSIC call(const F &f) const {
            return vec->call(f, mask);
        }
        template<typename F> inline void INTRINSIC call(F &f) const {
            return vec->call(f, mask);
        }
        template<typename F> inline Vector<T> INTRINSIC apply(const F &f) const {
            return vec->apply(f, mask);
        }
        template<typename F> inline Vector<T> INTRINSIC apply(F &f) const {
            return vec->apply(f, mask);
        }
    private:
        inline WriteMaskedVector(Vector<T> *v, const Mask &k) : vec(v), mask(k) {}
        Vector<T> *const vec;
        Mask mask;
};

} // namespace AVX
} // namespace Vc
#include "writemaskedvector.tcc"
#endif // VC_AVX_WRITEMASKEDVECTOR_H
