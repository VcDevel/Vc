/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VECTOR_H
#define VECTOR_H

#include "global.h"

// 1. forward declare all possible SIMD impl types
#include "common/types.h"
#include "scalar/types.h"
#include "sse/types.h"
#include "avx/types.h"
#include "mic/types.h"

// 2. forward declare simdarray types
#include "common/simdarrayfwd.h"

// 3. define all of Vc::Scalar - this one is always present, so it makes sense to put it first
#include "scalar/vector.h"
#include "common/simd_cast.h"
#include "scalar/simd_cast.h"

#ifdef VC_IMPL_AVX
# include "avx/intrinsics.h"
# undef VC_IMPL
# undef Vc_IMPL_NAMESPACE
# define VC_IMPL ::Vc::SSE42Impl
# define Vc_IMPL_NAMESPACE SSE
# include "sse/vector.h"
# undef VC_IMPL
# undef Vc_IMPL_NAMESPACE
# if defined(VC_IMPL_AVX2)
#  define VC_IMPL ::Vc::AVX2Impl
#  define Vc_IMPL_NAMESPACE AVX2
# elif defined(VC_IMPL_AVX)
#  define VC_IMPL ::Vc::AVXImpl
#  define Vc_IMPL_NAMESPACE AVX
# else
#  error "I lost track of the targeted implementation now. Something is messed up or there's a bug in Vc."
# endif
# include "avx/vector.h"
# include "sse/simd_cast.h"
# include "avx/simd_cast.h"
#elif defined(VC_IMPL_SSE)
# include "sse/vector.h"
# include "sse/simd_cast.h"
#endif

#if defined(VC_IMPL_MIC)
# include "mic/vector.h"
# include "mic/simd_cast.h"
#endif

namespace Vc_VERSIONED_NAMESPACE
{
  using Vc_IMPL_NAMESPACE::Vector;
  using Vc_IMPL_NAMESPACE::Mask;

  typedef Vc_IMPL_NAMESPACE::double_v double_v;
  typedef Vc_IMPL_NAMESPACE:: float_v  float_v;
  typedef Vc_IMPL_NAMESPACE::   int_v    int_v;
  typedef Vc_IMPL_NAMESPACE::  uint_v   uint_v;
  typedef Vc_IMPL_NAMESPACE:: short_v  short_v;
  typedef Vc_IMPL_NAMESPACE::ushort_v ushort_v;

  typedef Vc_IMPL_NAMESPACE::double_m double_m;
  typedef Vc_IMPL_NAMESPACE:: float_m  float_m;
  typedef Vc_IMPL_NAMESPACE::   int_m    int_m;
  typedef Vc_IMPL_NAMESPACE::  uint_m   uint_m;
  typedef Vc_IMPL_NAMESPACE:: short_m  short_m;
  typedef Vc_IMPL_NAMESPACE::ushort_m ushort_m;

  typedef Vector<std:: int_least64_t>  int_least64_v;
  typedef Vector<std::uint_least64_t> uint_least64_v;
  typedef Vector<std:: int_least32_t>  int_least32_v;
  typedef Vector<std::uint_least32_t> uint_least32_v;
  typedef Vector<std:: int_least16_t>  int_least16_v;
  typedef Vector<std::uint_least16_t> uint_least16_v;
  typedef Vector<std::  int_least8_t>   int_least8_v;
  typedef Vector<std:: uint_least8_t>  uint_least8_v;

  typedef Mask<std:: int_least64_t>  int_least64_m;
  typedef Mask<std::uint_least64_t> uint_least64_m;
  typedef Mask<std:: int_least32_t>  int_least32_m;
  typedef Mask<std::uint_least32_t> uint_least32_m;
  typedef Mask<std:: int_least16_t>  int_least16_m;
  typedef Mask<std::uint_least16_t> uint_least16_m;
  typedef Mask<std::  int_least8_t>   int_least8_m;
  typedef Mask<std:: uint_least8_t>  uint_least8_m;

  typedef Vector<std:: int_fast64_t>  int_fast64_v;
  typedef Vector<std::uint_fast64_t> uint_fast64_v;
  typedef Vector<std:: int_fast32_t>  int_fast32_v;
  typedef Vector<std::uint_fast32_t> uint_fast32_v;
  typedef Vector<std:: int_fast16_t>  int_fast16_v;
  typedef Vector<std::uint_fast16_t> uint_fast16_v;
  typedef Vector<std::  int_fast8_t>   int_fast8_v;
  typedef Vector<std:: uint_fast8_t>  uint_fast8_v;

  typedef Mask<std:: int_fast64_t>  int_fast64_m;
  typedef Mask<std::uint_fast64_t> uint_fast64_m;
  typedef Mask<std:: int_fast32_t>  int_fast32_m;
  typedef Mask<std::uint_fast32_t> uint_fast32_m;
  typedef Mask<std:: int_fast16_t>  int_fast16_m;
  typedef Mask<std::uint_fast16_t> uint_fast16_m;
  typedef Mask<std::  int_fast8_t>   int_fast8_m;
  typedef Mask<std:: uint_fast8_t>  uint_fast8_m;

#if defined INT64_MAX && defined UINT64_MAX
  typedef Vector<std:: int64_t>  int64_v;
  typedef Vector<std::uint64_t> uint64_v;
  typedef Mask<std:: int64_t>  int64_m;
  typedef Mask<std::uint64_t> uint64_m;
#endif
#if defined INT32_MAX && defined UINT32_MAX
  typedef Vector<std:: int32_t>  int32_v;
  typedef Vector<std::uint32_t> uint32_v;
  typedef Mask<std:: int32_t>  int32_m;
  typedef Mask<std::uint32_t> uint32_m;
#endif
#if defined INT16_MAX && defined UINT16_MAX
  typedef Vector<std:: int16_t>  int16_v;
  typedef Vector<std::uint16_t> uint16_v;
  typedef Mask<std:: int16_t>  int16_m;
  typedef Mask<std::uint16_t> uint16_m;
#endif
#if defined INT8_MAX && defined UINT8_MAX
  typedef Vector<std:: int8_t>  int8_v;
  typedef Vector<std::uint8_t> uint8_v;
  typedef Mask<std:: int8_t>  int8_m;
  typedef Mask<std::uint8_t> uint8_m;
#endif

  namespace {
    static_assert(double_v::Size == VC_DOUBLE_V_SIZE, "VC_DOUBLE_V_SIZE macro defined to an incorrect value");
    static_assert(float_v::Size  == VC_FLOAT_V_SIZE , "VC_FLOAT_V_SIZE macro defined to an incorrect value ");
    static_assert(int_v::Size    == VC_INT_V_SIZE   , "VC_INT_V_SIZE macro defined to an incorrect value   ");
    static_assert(uint_v::Size   == VC_UINT_V_SIZE  , "VC_UINT_V_SIZE macro defined to an incorrect value  ");
    static_assert(short_v::Size  == VC_SHORT_V_SIZE , "VC_SHORT_V_SIZE macro defined to an incorrect value ");
    static_assert(ushort_v::Size == VC_USHORT_V_SIZE, "VC_USHORT_V_SIZE macro defined to an incorrect value");
  }
}


// finally define the non-member operators
#include "common/operators.h"

#include "common/simdarray.h"
// XXX See bottom of common/simd_mask_array.h:
//#include "common/simd_cast_caller.tcc"

namespace Vc_VERSIONED_NAMESPACE {
  using Vc_IMPL_NAMESPACE::VectorAlignment;
} // namespace Vc_VERSIONED_NAMESPACE

#define VC_VECTOR_DECLARED__ 1

#include "scalar/helperimpl.h"
#include "scalar/math.h"
#include "scalar/simd_cast_caller.tcc"
#if defined(VC_IMPL_SSE)
# include "sse/helperimpl.h"
# include "sse/math.h"
# include "sse/simd_cast_caller.tcc"
#endif
#if defined(VC_IMPL_AVX)
# include "avx/helperimpl.h"
# include "avx/math.h"
# include "avx/simd_cast_caller.tcc"
#endif
#if defined(VC_IMPL_MIC)
# include "mic/helperimpl.h"
# include "mic/math.h"
# include "mic/simd_cast_caller.tcc"
#endif

#include "common/math.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc_VERSIONED_NAMESPACE
{
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
}

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
  using Vc::trunc;
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

#endif // VECTOR_H
