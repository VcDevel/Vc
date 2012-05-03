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

#ifndef VECTOR_H
#define VECTOR_H

#include "global.h"
#include "internal/namespace.h"

#if VC_IMPL_Scalar
# include "scalar/vector.h"
# include "scalar/helperimpl.h"
#elif VC_IMPL_AVX
# include "avx/vector.h"
# include "avx/helperimpl.h"
#elif VC_IMPL_SSE
# include "sse/vector.h"
# include "sse/helperimpl.h"
#endif

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc
{
  using VECTOR_NAMESPACE::VectorAlignment;
  using VECTOR_NAMESPACE::VectorAlignedBaseT;
  typedef VectorAlignedBaseT<> VectorAlignedBase;
  using namespace VectorSpecialInitializerZero;
  using namespace VectorSpecialInitializerOne;
  using namespace VectorSpecialInitializerIndexesFromZero;
  using VECTOR_NAMESPACE::min;
  using VECTOR_NAMESPACE::max;
  using VECTOR_NAMESPACE::sqrt;
  using VECTOR_NAMESPACE::rsqrt;
  using VECTOR_NAMESPACE::abs;
  using VECTOR_NAMESPACE::sin;
  using VECTOR_NAMESPACE::asin;
  using VECTOR_NAMESPACE::cos;
  using VECTOR_NAMESPACE::sincos;
  using VECTOR_NAMESPACE::floor;
  using VECTOR_NAMESPACE::ceil;
  using VECTOR_NAMESPACE::exp;
  using VECTOR_NAMESPACE::log;
  using VECTOR_NAMESPACE::log2;
  using VECTOR_NAMESPACE::log10;
  using VECTOR_NAMESPACE::reciprocal;
  using VECTOR_NAMESPACE::atan;
  using VECTOR_NAMESPACE::atan2;
  using VECTOR_NAMESPACE::frexp;
  using VECTOR_NAMESPACE::ldexp;
  using VECTOR_NAMESPACE::round;
  using VECTOR_NAMESPACE::isfinite;
  using VECTOR_NAMESPACE::isnan;
  using VECTOR_NAMESPACE::forceToRegisters;
  using VECTOR_NAMESPACE::Vector;

  namespace {
    VC_STATIC_ASSERT_NC(VECTOR_NAMESPACE::double_v::Size == VC_DOUBLE_V_SIZE, VC_DOUBLE_V_SIZE_MACRO_WRONG);
    VC_STATIC_ASSERT_NC(VECTOR_NAMESPACE::float_v::Size  == VC_FLOAT_V_SIZE , VC_FLOAT_V_SIZE_MACRO_WRONG );
    VC_STATIC_ASSERT_NC(VECTOR_NAMESPACE::sfloat_v::Size == VC_SFLOAT_V_SIZE, VC_SFLOAT_V_SIZE_MACRO_WRONG);
    VC_STATIC_ASSERT_NC(VECTOR_NAMESPACE::int_v::Size    == VC_INT_V_SIZE   , VC_INT_V_SIZE_MACRO_WRONG   );
    VC_STATIC_ASSERT_NC(VECTOR_NAMESPACE::uint_v::Size   == VC_UINT_V_SIZE  , VC_UINT_V_SIZE_MACRO_WRONG  );
    VC_STATIC_ASSERT_NC(VECTOR_NAMESPACE::short_v::Size  == VC_SHORT_V_SIZE , VC_SHORT_V_SIZE_MACRO_WRONG );
    VC_STATIC_ASSERT_NC(VECTOR_NAMESPACE::ushort_v::Size == VC_USHORT_V_SIZE, VC_USHORT_V_SIZE_MACRO_WRONG);
  }
} // namespace Vc

#ifndef VC_NO_STD_FUNCTIONS
namespace std
{
  using Vc::min;
  using Vc::max;

  using Vc::abs;
  using Vc::asin;
  using Vc::atan;
  using Vc::atan2;
  using Vc::ceil;
  using Vc::cos;
  using Vc::exp;
  using Vc::floor;
  using Vc::frexp;
  using Vc::ldexp;
  using Vc::log;
  using Vc::log10;
  using Vc::log2;
  using Vc::round;
  using Vc::sin;
  using Vc::sqrt;

  using Vc::isfinite;
  using Vc::isnan;
} // namespace std
#endif

#ifndef VC_CLEAN_NAMESPACE
#define foreach_bit(_it_, _mask_) Vc_foreach_bit(_it_, _mask_)
#endif

#undef VECTOR_NAMESPACE

#endif // VECTOR_H
