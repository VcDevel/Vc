/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SCALAR_LIMITS_H
#define VC_SCALAR_LIMITS_H

#include <limits>
#include "types.h"

namespace std
{
    template<> inline Vc::Scalar::int_v    numeric_limits<Vc::Scalar::int_v   >::max() throw() { return Vc::Scalar::int_v   (std::numeric_limits<int>::max()); }
    template<> inline Vc::Scalar::int_v    numeric_limits<Vc::Scalar::int_v   >::min() throw() { return Vc::Scalar::int_v   (std::numeric_limits<int>::min()); }
    template<> inline Vc::Scalar::uint_v   numeric_limits<Vc::Scalar::uint_v  >::max() throw() { return Vc::Scalar::uint_v  (std::numeric_limits<unsigned int>::max()); }
    template<> inline Vc::Scalar::uint_v   numeric_limits<Vc::Scalar::uint_v  >::min() throw() { return Vc::Scalar::uint_v  (std::numeric_limits<unsigned int>::min()); }
    template<> inline Vc::Scalar::short_v  numeric_limits<Vc::Scalar::short_v >::max() throw() { return Vc::Scalar::short_v (std::numeric_limits<short>::max()); }
    template<> inline Vc::Scalar::short_v  numeric_limits<Vc::Scalar::short_v >::min() throw() { return Vc::Scalar::short_v (std::numeric_limits<short>::min()); }
    template<> inline Vc::Scalar::ushort_v numeric_limits<Vc::Scalar::ushort_v>::max() throw() { return Vc::Scalar::ushort_v(std::numeric_limits<unsigned short>::max()); }
    template<> inline Vc::Scalar::ushort_v numeric_limits<Vc::Scalar::ushort_v>::min() throw() { return Vc::Scalar::ushort_v(std::numeric_limits<unsigned short>::min()); }
    template<> inline Vc::Scalar::float_v  numeric_limits<Vc::Scalar::float_v >::max() throw() { return Vc::Scalar::float_v (std::numeric_limits<float>::max()); }
    template<> inline Vc::Scalar::float_v  numeric_limits<Vc::Scalar::float_v >::min() throw() { return Vc::Scalar::float_v (std::numeric_limits<float>::min()); }
    template<> inline Vc::Scalar::double_v numeric_limits<Vc::Scalar::double_v>::max() throw() { return Vc::Scalar::double_v(std::numeric_limits<double>::max()); }
    template<> inline Vc::Scalar::double_v numeric_limits<Vc::Scalar::double_v>::min() throw() { return Vc::Scalar::double_v(std::numeric_limits<double>::min()); }
} // namespace std
#endif // VC_SCALAR_LIMITS_H
