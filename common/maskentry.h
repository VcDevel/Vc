/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_MASKENTRY_H
#define VC_COMMON_MASKENTRY_H

#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

template<typename M> class MaskEntry
{
    M &mask;
    size_t offset;
public:
    constexpr MaskEntry(M &m, size_t o) : mask(m), offset(o) {}
    constexpr MaskEntry(const MaskEntry &) = default;
#ifndef VC_NO_MOVE_CTOR
    constexpr MaskEntry(MaskEntry &&) = default;
#endif

    Vc_ALWAYS_INLINE Vc_PURE operator bool() const { const M &m = mask; return m[offset]; }
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
    Vc_ALWAYS_INLINE MaskBool(bool x) : data(x ? -1 : 0) {}
    Vc_ALWAYS_INLINE MaskBool &operator=(bool x) { data = x ? -1 : 0; return *this; }

    Vc_ALWAYS_INLINE MaskBool(const MaskBool &) = default;
#ifndef VC_NO_MOVE_CTOR
    Vc_ALWAYS_INLINE MaskBool(MaskBool &&) = default;
#endif
    Vc_ALWAYS_INLINE MaskBool &operator=(const MaskBool &) = default;

    Vc_ALWAYS_INLINE operator bool() const { return (data & 1) != 0; }
} Vc_MAY_ALIAS;

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_MASKENTRY_H
