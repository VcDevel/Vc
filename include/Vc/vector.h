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

#ifdef VC_IMPL_Scalar
# include "scalar/vector.h"
# include "scalar/helperimpl.h"
#elif defined(VC_IMPL_MIC)
# include "mic/vector.h"
# include "mic/helperimpl.h"
#elif defined(VC_IMPL_AVX)
# include "avx/vector.h"
# include "avx/helperimpl.h"
#elif defined(VC_IMPL_SSE)
# include "sse/vector.h"
# include "sse/helperimpl.h"
#else
# error "No known Vc implementation was selected. This should not happen. The logic in Vc/global.h failed."
#endif

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

Vc_PUBLIC_NAMESPACE_BEGIN
  using Vc_IMPL_NAMESPACE::VectorAlignment;
  using Vc_IMPL_NAMESPACE::VectorAlignedBaseT;
  typedef VectorAlignedBaseT<> VectorAlignedBase;
  using namespace VectorSpecialInitializerZero;
  using namespace VectorSpecialInitializerOne;
  using namespace VectorSpecialInitializerIndexesFromZero;
  using Vc_IMPL_NAMESPACE::min;
  using Vc_IMPL_NAMESPACE::max;
  using Vc_IMPL_NAMESPACE::sqrt;
  using Vc_IMPL_NAMESPACE::rsqrt;
  using Vc_IMPL_NAMESPACE::abs;
  using Vc_IMPL_NAMESPACE::sin;
  using Vc_IMPL_NAMESPACE::asin;
  using Vc_IMPL_NAMESPACE::cos;
  using Vc_IMPL_NAMESPACE::sincos;
  using Vc_IMPL_NAMESPACE::trunc;
  using Vc_IMPL_NAMESPACE::floor;
  using Vc_IMPL_NAMESPACE::ceil;
  using Vc_IMPL_NAMESPACE::exp;
  using Vc_IMPL_NAMESPACE::log;
  using Vc_IMPL_NAMESPACE::log2;
  using Vc_IMPL_NAMESPACE::log10;
  using Vc_IMPL_NAMESPACE::reciprocal;
  using Vc_IMPL_NAMESPACE::atan;
  using Vc_IMPL_NAMESPACE::atan2;
  using Vc_IMPL_NAMESPACE::frexp;
  using Vc_IMPL_NAMESPACE::ldexp;
  using Vc_IMPL_NAMESPACE::round;
  using Vc_IMPL_NAMESPACE::isfinite;
  using Vc_IMPL_NAMESPACE::isinf;
  using Vc_IMPL_NAMESPACE::isnan;
  using Vc_IMPL_NAMESPACE::forceToRegisters;
  using Vc_IMPL_NAMESPACE::Vector;

  typedef Vc_IMPL_NAMESPACE::double_v double_v;
  typedef double_v::Mask double_m;
  typedef Vc_IMPL_NAMESPACE::float_v float_v;
  typedef float_v::Mask float_m;
  typedef Vc_IMPL_NAMESPACE::int_v int_v;
  typedef int_v::Mask int_m;
  typedef Vc_IMPL_NAMESPACE::uint_v uint_v;
  typedef uint_v::Mask uint_m;
  typedef Vc_IMPL_NAMESPACE::short_v short_v;
  typedef short_v::Mask short_m;
  typedef Vc_IMPL_NAMESPACE::ushort_v ushort_v;
  typedef ushort_v::Mask ushort_m;

  namespace {
    static_assert(double_v::Size == VC_DOUBLE_V_SIZE, "VC_DOUBLE_V_SIZE macro defined to an incorrect value");
    static_assert(float_v::Size  == VC_FLOAT_V_SIZE , "VC_FLOAT_V_SIZE macro defined to an incorrect value ");
    static_assert(int_v::Size    == VC_INT_V_SIZE   , "VC_INT_V_SIZE macro defined to an incorrect value   ");
    static_assert(uint_v::Size   == VC_UINT_V_SIZE  , "VC_UINT_V_SIZE macro defined to an incorrect value  ");
    static_assert(short_v::Size  == VC_SHORT_V_SIZE , "VC_SHORT_V_SIZE macro defined to an incorrect value ");
    static_assert(ushort_v::Size == VC_USHORT_V_SIZE, "VC_USHORT_V_SIZE macro defined to an incorrect value");
  }
Vc_NAMESPACE_END

#include "common/vectortuple.h"
#include "common/algorithms.h"
#include "common/where.h"
#include "common/iif.h"

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

#endif // VECTOR_H
