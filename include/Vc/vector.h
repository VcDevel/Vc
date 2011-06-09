/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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
  using namespace VECTOR_NAMESPACE::VectorSpecialInitializerZero;
  using namespace VECTOR_NAMESPACE::VectorSpecialInitializerOne;
  using namespace VECTOR_NAMESPACE::VectorSpecialInitializerIndexesFromZero;
  using VECTOR_NAMESPACE::min;
  using VECTOR_NAMESPACE::max;
  using VECTOR_NAMESPACE::sqrt;
  using VECTOR_NAMESPACE::rsqrt;
  using VECTOR_NAMESPACE::abs;
  using VECTOR_NAMESPACE::sin;
  using VECTOR_NAMESPACE::asin;
  using VECTOR_NAMESPACE::cos;
  using VECTOR_NAMESPACE::log;
  using VECTOR_NAMESPACE::log10;
  using VECTOR_NAMESPACE::reciprocal;
  using VECTOR_NAMESPACE::atan;
  using VECTOR_NAMESPACE::atan2;
  using VECTOR_NAMESPACE::round;
  using VECTOR_NAMESPACE::isfinite;
  using VECTOR_NAMESPACE::isnan;
  using VECTOR_NAMESPACE::forceToRegisters;
  using VECTOR_NAMESPACE::Vector;
} // namespace Vc

#ifndef VC_CLEAN_NAMESPACE
#define foreach_bit(_it_, _mask_) Vc_foreach_bit(_it_, _mask_)
#endif

#undef VECTOR_NAMESPACE

#endif // VECTOR_H
