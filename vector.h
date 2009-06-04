/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

*/

#ifndef VECTOR_H
#define VECTOR_H

#ifdef ENABLE_LARRABEE
#include "larrabee/vector.h"
#elif defined(USE_SSE)
#include "sse/vector.h"
#else
#include "simple/vector.h"
#endif

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc
{

#ifdef ENABLE_LARRABEE
  using ::Larrabee::Vector;
  using ::Larrabee::SwizzledVector;
  using ::Larrabee::Mask;
  using ::Larrabee::VectorAlignment;
  using namespace ::Larrabee::VectorSpecialInitializerZero;
  using namespace ::Larrabee::VectorSpecialInitializerOne;
  using namespace ::Larrabee::VectorSpecialInitializerRandom;
  using namespace ::Larrabee::VectorSpecialInitializerIndexesFromZero;
  using ::Larrabee::min;
  using ::Larrabee::max;
  using ::Larrabee::sqrt;
  using ::Larrabee::rsqrt;
  using ::Larrabee::abs;
  using ::Larrabee::sin;
  using ::Larrabee::cos;
  using ::Larrabee::log;
  using ::Larrabee::log10;
  using ::Larrabee::isfinite;
  using ::Larrabee::isnan;
  typedef Vector<signed int> short_v;
  typedef Vector<unsigned int> ushort_v;
  typedef Vector<float> sfloat_v;
#elif defined(USE_SSE)
  using ::SSE::Vector;
  using ::SSE::SwizzledVector;
  using ::SSE::Mask;
  using ::SSE::VectorAlignment;
  using namespace ::SSE::VectorSpecialInitializerZero;
  using namespace ::SSE::VectorSpecialInitializerOne;
  using namespace ::SSE::VectorSpecialInitializerRandom;
  using namespace ::SSE::VectorSpecialInitializerIndexesFromZero;
  using ::SSE::min;
  using ::SSE::max;
  using ::SSE::sqrt;
  using ::SSE::rsqrt;
  using ::SSE::abs;
  using ::SSE::sin;
  using ::SSE::cos;
  using ::SSE::log;
  using ::SSE::log10;
  using ::SSE::isfinite;
  using ::SSE::isnan;
  typedef Vector<signed short> short_v;
  typedef Vector<unsigned short> ushort_v;
  typedef Vector<SSE::float8> sfloat_v;
#else
  using ::Simple::Vector;
  using ::Simple::SwizzledVector;
  using ::Simple::Mask;
  using ::Simple::VectorAlignment;
  using namespace ::Simple::VectorSpecialInitializerZero;
  using namespace ::Simple::VectorSpecialInitializerOne;
  using namespace ::Simple::VectorSpecialInitializerRandom;
  using namespace ::Simple::VectorSpecialInitializerIndexesFromZero;
  using ::Simple::min;
  using ::Simple::max;
  using ::Simple::sqrt;
  using ::Simple::rsqrt;
  using ::Simple::abs;
  using ::Simple::sin;
  using ::Simple::cos;
  using ::Simple::log;
  using ::Simple::log10;
  using ::Simple::isfinite;
  using ::Simple::isnan;
  typedef Vector<signed short> short_v;
  typedef Vector<unsigned short> ushort_v;
  typedef Vector<float> sfloat_v;
#endif

  typedef Vector<double> double_v;
  typedef Vector<float>  float_v;
  typedef Vector<int>    int_v;
  typedef Vector<unsigned int> uint_v;

  template<typename T, unsigned int Dim>
  struct EuclideanVector
  {
    T v[Dim];

          T &operator[](int i)       { return v[i]; }
    const T &operator[](int i) const { return v[i]; }

    T &x() { return v[0]; }
    T &y() { return v[1]; }
    T &z() { return v[2]; }
    T &w() { return v[3]; }

    const T &x() const { return v[0]; }
    const T &y() const { return v[1]; }
    const T &z() const { return v[2]; }
    const T &w() const { return v[3]; }
  };

#define INIT_V2( a, b )    { { ( a ), ( b )        } }
#define INIT_V3( a, b, c ) { { ( a ), ( b ), ( c ) } }

  typedef EuclideanVector<double_v, 2> double_v2;
  typedef EuclideanVector<double_v, 3> double_v3;

  typedef EuclideanVector<float_v, 2> float_v2;
  typedef EuclideanVector<float_v, 3> float_v3;

  typedef EuclideanVector<int_v, 2> int_v2;
  typedef EuclideanVector<int_v, 3> int_v3;

  typedef EuclideanVector<uint_v, 2> uint_v2;
  typedef EuclideanVector<uint_v, 3> uint_v3;
} // namespace Vc

#endif // VECTOR_H
