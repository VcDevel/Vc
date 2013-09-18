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

template<typename V> void isSigned();
template<> void isSigned< float_v>() { VERIFY( Vc::is_signed< float_v>::value); VERIFY(!Vc::is_unsigned< float_v>::value); }
template<> void isSigned<double_v>() { VERIFY( Vc::is_signed<double_v>::value); VERIFY(!Vc::is_unsigned<double_v>::value); }
template<> void isSigned<   int_v>() { VERIFY( Vc::is_signed<   int_v>::value); VERIFY(!Vc::is_unsigned<   int_v>::value); }
template<> void isSigned<  uint_v>() { VERIFY(!Vc::is_signed<  uint_v>::value); VERIFY( Vc::is_unsigned<  uint_v>::value); }
template<> void isSigned< short_v>() { VERIFY( Vc::is_signed< short_v>::value); VERIFY(!Vc::is_unsigned< short_v>::value); }
template<> void isSigned<ushort_v>() { VERIFY(!Vc::is_signed<ushort_v>::value); VERIFY( Vc::is_unsigned<ushort_v>::value); }

template<typename V> void isIntegral();
template<> void isIntegral< float_v>() { VERIFY(!Vc::is_integral< float_v>::value); }
template<> void isIntegral<double_v>() { VERIFY(!Vc::is_integral<double_v>::value); }
template<> void isIntegral<   int_v>() { VERIFY( Vc::is_integral<   int_v>::value); }
template<> void isIntegral<  uint_v>() { VERIFY( Vc::is_integral<  uint_v>::value); }
template<> void isIntegral< short_v>() { VERIFY( Vc::is_integral< short_v>::value); }
template<> void isIntegral<ushort_v>() { VERIFY( Vc::is_integral<ushort_v>::value); }

template<typename V> void isFloatingPoint();
template<> void isFloatingPoint< float_v>() { VERIFY( Vc::is_floating_point< float_v>::value); }
template<> void isFloatingPoint<double_v>() { VERIFY( Vc::is_floating_point<double_v>::value); }
template<> void isFloatingPoint<   int_v>() { VERIFY(!Vc::is_floating_point<   int_v>::value); }
template<> void isFloatingPoint<  uint_v>() { VERIFY(!Vc::is_floating_point<  uint_v>::value); }
template<> void isFloatingPoint< short_v>() { VERIFY(!Vc::is_floating_point< short_v>::value); }
template<> void isFloatingPoint<ushort_v>() { VERIFY(!Vc::is_floating_point<ushort_v>::value); }

void testmain()
{
    testAllTypes(isIntegral);
    testAllTypes(isFloatingPoint);
    testAllTypes(isSigned);
}
