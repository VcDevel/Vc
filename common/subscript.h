/*  This file is part of the Vc library. {{{
Copyright © 2013 Matthias Kretz <kretz@kde.org>
All rights reserved.

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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_COMMON_SUBSCRIPT_H
#define VC_COMMON_SUBSCRIPT_H

#include <type_traits>
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

template <typename Base> class AdaptSubscriptOperator : public Base
{
public:
    // perfect forward all Base constructors
    template <typename... Args>
    AdaptSubscriptOperator(Args &&... arguments)
        : Base(std::forward<Args>(arguments)...)
    {
    }

    // explicitly enable Base::operator[] because the following would hide it
    using Base::operator[];

    // forward to non-member subscript_operator function
    template <
        typename I,
        typename = typename std::enable_if<
            !std::is_arithmetic<typename std::decay<I>::type>::value>::type  // arithmetic types
                                                                             // should always use
                                                                             // Base::operator[] and
                                                                             // never match this one
        >
    auto operator[](I &&__arg) -> decltype(subscript_operator(*this, std::forward<I>(__arg)))
    {
        return subscript_operator(*this, std::forward<I>(__arg));
    }

    // const overload of the above
    template <typename I,
              typename = typename std::enable_if<
                  !std::is_arithmetic<typename std::decay<I>::type>::value>::type>
    auto operator[](I &&__arg) const -> decltype(subscript_operator(*this, std::forward<I>(__arg)))
    {
        return subscript_operator(*this, std::forward<I>(__arg));
    }
};

template <typename T, typename IndexVector> class SubscriptOperation
{
    IndexVector m_indexes;
    T *m_address;
    using ScalarType = typename std::decay<T>::type;
    using VectorType = Vector<ScalarType>;

public:
    SubscriptOperation(T *address, IndexVector indexes) : m_indexes(indexes), m_address(address)
    {
    }

    operator VectorType() const { return VectorType(m_address, m_indexes); }
};

template <typename Container, typename IndexVector>
SubscriptOperation<typename std::remove_reference<decltype(std::declval<Container>()[0])>::type,
                   typename std::decay<IndexVector>::type>
    subscript_operator(Container &&vec, IndexVector &&indexes)
{
    return {&vec[0], std::forward<IndexVector>(indexes)};
}

}  // namespace Common

namespace Scalar
{
    using Common::subscript_operator;
}  // namespace
namespace SSE
{
    using Common::subscript_operator;
}  // namespace
namespace AVX
{
    using Common::subscript_operator;
}  // namespace
namespace AVX2
{
    using Common::subscript_operator;
}  // namespace
namespace MIC
{
    using Common::subscript_operator;
}  // namespace

}  // namespace

#include "undomacros.h"

#endif // VC_COMMON_SUBSCRIPT_H
