/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

#ifndef VECTOR_H
#define VECTOR_H

#ifdef __LRB__
#include "larrabee/vector.h"
#elif USE_SSE
#include "sse/vector.h"
#else
#include "simple/vector.h"
#endif

namespace Vc
{

#ifdef __LRB__
  using ::Larrabee::Vector;
  using ::Larrabee::SwizzledVector;
  using ::Larrabee::Mask;
  using ::Larrabee::kFullMask;
  using ::Larrabee::VectorAlignment;
  using namespace ::Larrabee::VectorSpecialInitializerZero;
  using namespace ::Larrabee::VectorSpecialInitializerRandom;
  using ::Larrabee::min;
  using ::Larrabee::max;
  using ::Larrabee::sqrt;
  using ::Larrabee::abs;
  using ::Larrabee::sin;
  using ::Larrabee::cos;
  using ::Larrabee::maskNthElement;
#elif USE_SSE
  using ::SSE::Vector;
  using ::SSE::SwizzledVector;
  using ::SSE::Mask;
  using ::SSE::kFullMask;
  using ::SSE::VectorAlignment;
  using namespace ::SSE::VectorSpecialInitializerZero;
  using namespace ::SSE::VectorSpecialInitializerRandom;
  using ::SSE::min;
  using ::SSE::max;
  using ::SSE::sqrt;
  using ::SSE::abs;
  using ::SSE::sin;
  using ::SSE::cos;
  using ::SSE::maskNthElement;
#else
  using ::Simple::Vector;
  using ::Simple::SwizzledVector;
  using ::Simple::Mask;
  using ::Simple::kFullMask;
  using ::Simple::VectorAlignment;
  using namespace ::Simple::VectorSpecialInitializerZero;
  using namespace ::Simple::VectorSpecialInitializerRandom;
  using ::Simple::min;
  using ::Simple::max;
  using ::Simple::sqrt;
  using ::Simple::abs;
  using ::Simple::sin;
  using ::Simple::cos;
  using ::Simple::maskNthElement;
#endif

  typedef Vector<double> double_v;
  typedef Vector<float>  float_v;
  typedef Vector<int>    int_v;
  typedef Vector<unsigned int> uint_v;

  template<typename T, unsigned int Dim>
  struct EuclideanVector
  {
    T v[Dim];

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

#ifdef __LRB__
  typedef signed char PackedIndexes;
#else
  typedef signed int PackedIndexes;
#endif

#ifndef ALIGN
# ifdef __GNUC__
#  define ALIGN(n) __attribute__((aligned(n)))
# else
#  define ALIGN(n) __declspec(align(n))
# endif
#endif
  ALIGN( 16 ) static const PackedIndexes kIndexesFromZero[int_v::Size] = {
#ifdef __LRB__
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#elif USE_SSE
    0, 1, 2, 3
#else
    0
#endif
  };
#undef ALIGN

} // namespace Vc

#endif // VECTOR_H
