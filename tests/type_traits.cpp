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

#include "unittest.h"
#include <Vc/type_traits>

using Vc::float_v;
using Vc::double_v;
using Vc::int_v;
using Vc::uint_v;
using Vc::short_v;
using Vc::ushort_v;

TEST_TYPES(V, isIntegral, (ALL_VECTORS))
{
    using T = typename V::EntryType;
    COMPARE(Vc::is_integral<V>::value, std::is_integral<T>::value);
}

TEST_TYPES(V, isFloatingPoint, (ALL_VECTORS))
{
    using T = typename V::EntryType;
    COMPARE(Vc::is_floating_point<V>::value, std::is_floating_point<T>::value);
}

TEST_TYPES(V, isSigned, (ALL_VECTORS))
{
    using T = typename V::EntryType;
    COMPARE(Vc::is_signed<V>::value, std::is_signed<T>::value);
    COMPARE(Vc::is_unsigned<V>::value, std::is_unsigned<T>::value);
}

TEST_TYPES(V, hasSubscript, (ALL_VECTORS))
{
    VERIFY(Vc::Traits::has_subscript_operator<V>::value);
}

TEST_TYPES(V, hasMultiply, (ALL_VECTORS))
{
    VERIFY(Vc::Traits::has_multiply_operator<V>::value);
}

TEST_TYPES(V, hasEquality, (ALL_VECTORS))
{
    VERIFY(Vc::Traits::has_equality_operator<V>::value);
}

TEST_TYPES(V, isSimdMask, (ALL_VECTORS))
{
    VERIFY(!Vc::is_simd_mask<V>::value);
    VERIFY( Vc::is_simd_mask<typename V::Mask>::value);
}

TEST_TYPES(V, isSimdVector, (ALL_VECTORS))
{
    VERIFY( Vc::is_simd_vector<V>::value);
    VERIFY(!Vc::is_simd_vector<typename V::Mask>::value);
}
