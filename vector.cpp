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
  template<> V_ALIGN(16) const int            VectorBase<int           , Vector<int>            >::_IndexesFromZero[           Vector<int>::Size] = { 0, 1, 2, 3 };
  template<> V_ALIGN(16) const unsigned int   VectorBase<unsigned int  , Vector<unsigned int>   >::_IndexesFromZero[  Vector<unsigned int>::Size] = { 0, 1, 2, 3 };
  template<> V_ALIGN(16) const short          VectorBase<short         , Vector<short>          >::_IndexesFromZero[         Vector<short>::Size] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  template<> V_ALIGN(16) const unsigned short VectorBase<unsigned short, Vector<unsigned short> >::_IndexesFromZero[Vector<unsigned short>::Size] = { 0, 1, 2, 3, 4, 5, 6, 7 };
} // namespace SSE
#endif

#ifdef ENABLE_LARRABEE
#include "larrabee/vector.h"
namespace Larrabee
{
    namespace Internal
    {
        V_ALIGN(16) const char _IndexesFromZero[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    } // namespace Internal
} // namespace Larrabee
#endif

#undef V_ALIGN
