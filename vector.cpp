#ifndef ALIGN
# ifdef __GNUC__
#  define V_ALIGN(n) __attribute__((aligned(n)))
# else
#  define V_ALIGN(n) __declspec(align(n))
# endif
#endif

#ifdef __SSE2__
#include "sse/vector.h"
namespace SSE
{
  template<> V_ALIGN(16) const int VectorBase<int, Vector<int> >::IndexesFromZero[Vector<int>::Size] = { 0, 1, 2, 3 };
  template<> V_ALIGN(16) const int VectorBase<int, Mask>::IndexesFromZero[Mask::Size] = { 0, 1, 2, 3 };
  template<> V_ALIGN(16) const unsigned int VectorBase<unsigned int, Vector<unsigned int> >::IndexesFromZero[Vector<unsigned int>::Size] = { 0, 1, 2, 3 };
  template<> V_ALIGN(16) const short VectorBase<short, Vector<short> >::IndexesFromZero[Vector<short>::Size] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  template<> V_ALIGN(16) const unsigned short VectorBase<unsigned short, Vector<unsigned short> >::IndexesFromZero[Vector<unsigned short>::Size] = { 0, 1, 2, 3, 4, 5, 6, 7 };
} // namespace SSE
#elif defined(__LRB__)
#include "larrabee/vector.h"
namespace Larrabee
{
  template<> V_ALIGN(16) const signed char   VectorBase<int         , Vector<int>          >::IndexesFromZero[Vector<int         >::Size] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  template<> V_ALIGN(16) const unsigned char VectorBase<unsigned int, Vector<unsigned int> >::IndexesFromZero[Vector<unsigned int>::Size] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
} // namespace Larrabee
#endif

#include "simple/vector.h"
namespace Simple
{
  template<> const int VectorBase<int, Vector<int> >::IndexesFromZero[1] = { 0 };
  template<> const unsigned int VectorBase<unsigned int, Vector<unsigned int> >::IndexesFromZero[1] = { 0 };
  template<> const short VectorBase<short, Vector<short> >::IndexesFromZero[1] = { 0 };
  template<> const unsigned short VectorBase<unsigned short, Vector<unsigned short> >::IndexesFromZero[1] = { 0 };
} // namespace Simple

#undef V_ALIGN
