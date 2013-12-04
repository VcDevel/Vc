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

#ifndef VC_MIC_WRITEMASKEDVECTOR_H
#define VC_MIC_WRITEMASKEDVECTOR_H

#include <utility>

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<typename T> class WriteMaskedVector
{
    friend class Vector<T>;
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename DetermineEntryType<T>::Type EntryType;
    typedef typename DetermineVectorEntryType<T>::Type VectorEntryType;
    typedef MIC::Mask<T> Mask;
public:
    //prefix
    Vc_ALWAYS_INLINE Vector<T> &operator++() {
        vec->d = _add<VectorEntryType>(vec->d.v(), mask, vec->d.v(), _set1(VectorEntryType(1)));
        return *vec;
    }
    Vc_ALWAYS_INLINE Vector<T> &operator--() {
        vec->d = _sub<VectorEntryType>(vec->d.v(), mask, vec->d.v(), _set1(VectorEntryType(1)));
        return *vec;
    }
    //postfix
    Vc_ALWAYS_INLINE Vector<T> operator++(int) {
        Vector<T> ret(*vec);
        vec->d = _add<VectorEntryType>(vec->d.v(), mask, vec->d.v(), _set1(VectorEntryType(1)));
        return ret;
    }
    Vc_ALWAYS_INLINE Vector<T> operator--(int) {
        Vector<T> ret(*vec);
        vec->d = _sub<VectorEntryType>(vec->d.v(), mask, vec->d.v(), _set1(VectorEntryType(1)));
        return ret;
    }

    Vc_ALWAYS_INLINE Vector<T> &operator+=(Vector<T> x) {
        vec->d = _add<VectorEntryType>(vec->d.v(), mask, vec->d.v(), x.d.v());
        return *vec;
    }
    Vc_ALWAYS_INLINE Vector<T> &operator-=(Vector<T> x) {
        vec->d = _sub<VectorEntryType>(vec->d.v(), mask, vec->d.v(), x.d.v());
        return *vec;
    }
    Vc_ALWAYS_INLINE Vector<T> &operator*=(Vector<T> x) {
        vec->d = _mul<VectorEntryType>(vec->d.v(), mask, vec->d.v(), x.d.v());
        return *vec;
    }
    Vc_ALWAYS_INLINE Vector<T> &operator/=(Vector<T> x) {
        vec->d = _div<VectorEntryType>(vec->d.v(), mask, vec->d.v(), x.d.v());
        return *vec;
    }

    Vc_ALWAYS_INLINE Vector<T> &operator+=(EntryType x) { return operator+=(Vector<T>(x)); }
    Vc_ALWAYS_INLINE Vector<T> &operator-=(EntryType x) { return operator-=(Vector<T>(x)); }
    Vc_ALWAYS_INLINE Vector<T> &operator*=(EntryType x) { return operator*=(Vector<T>(x)); }
    Vc_ALWAYS_INLINE Vector<T> &operator/=(EntryType x) { return operator/=(Vector<T>(x)); }

    Vc_ALWAYS_INLINE Vector<T> &operator=(Vector<T> x) {
        vec->assign(x, mask);
        return *vec;
    }
    Vc_ALWAYS_INLINE Vector<T> &operator=(EntryType x) { return operator=(Vector<T>(x)); }

#ifdef VC_NO_MOVE_CTOR
    template<typename F> Vc_INTRINSIC void call(const F &f) const {
        return vec->call(f, mask);
    }
    template<typename F> Vc_INTRINSIC Vector<T> apply(const F &f) const {
        return vec->apply(f, mask);
    }
    template<typename F> Vc_INTRINSIC void call(F &f) const {
        return vec->call(f, mask);
    }
    template<typename F> Vc_INTRINSIC Vector<T> apply(F &f) const {
        return vec->apply(f, mask);
    }
#else
    template<typename F> Vc_INTRINSIC void call(F &&f) const {
        return vec->call(std::forward<F>(f), mask);
    }
    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f) const {
        return vec->apply(std::forward<F>(f), mask);
    }
#endif
private:
    constexpr WriteMaskedVector(Vector<T> *v, Mask k) : vec(v), mask(k.data()) {}
    Vector<T> *vec;
    typename MaskTypeHelper<EntryType>::Type mask;
};

Vc_NAMESPACE_END

#endif // VC_MIC_WRITEMASKEDVECTOR_H
