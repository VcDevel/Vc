/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_MASKBOOL_H_
#define VC_DATAPAR_MASKBOOL_H_

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace
{
    template<size_t Bytes> struct MaskBoolStorage;
    // the following for typedefs must use std::intN_t and NOT! Vc::intN_t. The latter
    // segfaults ICC 15.0.3.
    template<> struct MaskBoolStorage<1> { typedef std::int8_t  type; };
    template<> struct MaskBoolStorage<2> { typedef std::int16_t type; };
    template<> struct MaskBoolStorage<4> { typedef std::int32_t type; };
    template<> struct MaskBoolStorage<8> { typedef std::int64_t type; };
} // anonymous namespace

template<size_t Bytes> class MaskBool
{
    typedef typename MaskBoolStorage<Bytes>::type storage_type Vc_MAY_ALIAS;
    storage_type data;
public:
    constexpr MaskBool(bool x) noexcept : data(x ? -1 : 0) {}
    Vc_ALWAYS_INLINE MaskBool &operator=(bool x) noexcept { data = x ? -1 : 0; return *this; }
    template <typename T, typename = enable_if<(!std::is_same<T, bool>::value &&
                                                std::is_fundamental<T>::value)>>
    Vc_ALWAYS_INLINE MaskBool &operator=(T x) noexcept
    {
        data = reinterpret_cast<const storage_type &>(x);
        return *this;
    }

    Vc_ALWAYS_INLINE MaskBool(const MaskBool &) noexcept = default;
    Vc_ALWAYS_INLINE MaskBool &operator=(const MaskBool &) noexcept = default;

#ifdef Vc_ICC
    template <typename T, typename = enable_if<(std::is_same<T, bool>::value ||
                                                (std::is_fundamental<T>::value &&
                                                 sizeof(storage_type) == sizeof(T)))>>
    constexpr operator T() const noexcept
    {
        return std::is_same<T, bool>::value ? T((data & 1) != 0)
                                            : reinterpret_cast<const may_alias<T> &>(data);
    }
#else
    constexpr operator bool() const noexcept { return (data & 1) != 0; }
    constexpr operator storage_type() const noexcept { return data; }
    template <typename T, typename = enable_if<(std::is_fundamental<T>::value &&
                                                sizeof(storage_type) == sizeof(T))>>
    constexpr operator T() const noexcept
    {
        return reinterpret_cast<const may_alias<T> &>(data);
    }
#endif

#ifdef Vc_MSVC
};
#define friend template<size_t N>
#define MaskBool MaskBool<N>
#endif

    friend constexpr bool operator==(bool a, MaskBool &&b)
    {
        return a == static_cast<bool>(b);
    }
    friend constexpr bool operator==(bool a, const MaskBool &b)
    {
        return a == static_cast<bool>(b);
    }
    friend constexpr bool operator==(MaskBool &&b, bool a)
    {
        return static_cast<bool>(a) == static_cast<bool>(b);
    }
    friend constexpr bool operator==(const MaskBool &b, bool a)
    {
        return static_cast<bool>(a) == static_cast<bool>(b);
    }

    friend constexpr bool operator!=(bool a, MaskBool &&b)
    {
        return a != static_cast<bool>(b);
    }
    friend constexpr bool operator!=(bool a, const MaskBool &b)
    {
        return a != static_cast<bool>(b);
    }
    friend constexpr bool operator!=(MaskBool &&b, bool a)
    {
        return a != static_cast<bool>(b);
    }
    friend constexpr bool operator!=(const MaskBool &b, bool a)
    {
        return a != static_cast<bool>(b);
    }
#ifdef Vc_MSVC
#undef friend
#undef MaskBool
#else
} Vc_MAY_ALIAS;
#endif

static_assert(true == MaskBool<4>(true), "true == MaskBool<4>(true)");
static_assert(MaskBool<4>(true) == true, "true == MaskBool<4>(true)");
static_assert(true != MaskBool<4>(false), "true != MaskBool<4>(false)");
static_assert(MaskBool<4>(false) != true, "true != MaskBool<4>(false)");

namespace tests
{
namespace
{
// Assert that MaskBools operator== doesn't grab unrelated types. The test should compare "1 == 2"
// because A() implicitly converts to int(1). If the MaskBool operator were used, the executed
// compare is bool(1) == bool(2) and thus different and detectable.
struct A { constexpr operator int() const { return 1; } };
static_assert(!(A() == 2), "MaskBools operator== was incorrectly found and used");
}  // unnamed namespace
}  // namespace tests
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_MASKBOOL_H_
