/*  This file is part of the Vc library. {{{
Copyright © 2013-2014 Matthias Kretz <kretz@kde.org>
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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_COMMON_MASKENTRY_H
#define VC_COMMON_MASKENTRY_H

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

template<typename M> class MaskEntry
{
    M &mask;
    size_t offset;
public:
    constexpr MaskEntry(M &m, size_t o) : mask(m), offset(o) {}
    constexpr MaskEntry(const MaskEntry &) = default;
    constexpr MaskEntry(MaskEntry &&) = default;

    template <typename B, typename std::enable_if<std::is_same<B, bool>::value, int>::type = 0>
    Vc_ALWAYS_INLINE Vc_PURE operator B() const
    {
        const M &m = mask;
        return m[offset];
    }
    Vc_ALWAYS_INLINE MaskEntry &operator=(bool x) {
        mask.setEntry(offset, x);
        return *this;
    }
};

namespace
{
    template<size_t Bytes> struct MaskBoolStorage;
    template<> struct MaskBoolStorage<1> { typedef int8_t  type; };
    template<> struct MaskBoolStorage<2> { typedef int16_t type; };
    template<> struct MaskBoolStorage<4> { typedef int32_t type; };
    template<> struct MaskBoolStorage<8> { typedef int64_t type; };
} // anonymous namespace

template<size_t Bytes> class MaskBool
{
    typedef typename MaskBoolStorage<Bytes>::type storage_type Vc_MAY_ALIAS;
    storage_type data;
public:
    constexpr MaskBool(bool x) : data(x ? -1 : 0) {}
    Vc_ALWAYS_INLINE MaskBool &operator=(bool x) { data = x ? -1 : 0; return *this; }

    Vc_ALWAYS_INLINE MaskBool(const MaskBool &) = default;
    Vc_ALWAYS_INLINE MaskBool &operator=(const MaskBool &) = default;

    template <typename B, typename std::enable_if<std::is_same<B, bool>::value, int>::type = 0>
    constexpr operator B() const
    {
        return (data & 1) != 0;
    }
} Vc_MAY_ALIAS;

template <typename A,
          typename B,
          typename std::enable_if<
              std::is_convertible<A, bool>::value &&std::is_convertible<B, bool>::value,
              int>::type = 0>
constexpr bool operator==(A &&a, B &&b)
{
    return static_cast<bool>(a) == static_cast<bool>(b);
}
template <typename A,
          typename B,
          typename std::enable_if<
              std::is_convertible<A, bool>::value &&std::is_convertible<B, bool>::value,
              int>::type = 0>
constexpr bool operator!=(A &&a, B &&b)
{
    return static_cast<bool>(a) != static_cast<bool>(b);
}

static_assert(true == MaskBool<4>(true), "true == MaskBool<4>(true)");
static_assert(true != MaskBool<4>(false), "true != MaskBool<4>(false)");

}  // namespace Common
}  // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_MASKENTRY_H
