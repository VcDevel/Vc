/*  This file is part of the Vc library. {{{
Copyright © 2012-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_MIC_WRITEMASKEDVECTOR_H_
#define VC_MIC_WRITEMASKEDVECTOR_H_

#include <utility>

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{

template<typename T> class WriteMaskedVector
{
    friend class Vc::Vector<T, VectorAbi::Mic>;
    typedef typename VectorTypeHelper<T>::Type VectorType;
    using EntryType = T;
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

    template<typename F> Vc_INTRINSIC void call(F &&f) const {
        return vec->call(std::forward<F>(f), mask);
    }
    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f) const {
        return vec->apply(std::forward<F>(f), mask);
    }
private:
    constexpr WriteMaskedVector(Vector<T> *v, Mask k) : vec(v), mask(k.data()) {}
    Vector<T> *vec;
    typename MaskTypeHelper<EntryType>::Type mask;
};

}  // namespace MIC
}  // namespace Vc

#endif // VC_MIC_WRITEMASKEDVECTOR_H_
