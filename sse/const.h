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

#ifndef VC_SSE_CONST_H
#define VC_SSE_CONST_H

#include "const_data.h"
#include "vector.h"
#include "macros.h"

namespace Vc
{
namespace SSE
{
    template<typename T> class Vector;

    template<typename T> struct Const
    {
        typedef Vector<T> V;
        typedef typename V::Mask M;

        static inline ALWAYS_INLINE_L CONST_L V _1_2pi()       ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[0 * V::Size]); }
        static inline ALWAYS_INLINE_L CONST_L V _2pi()         ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[1 * V::Size]); }
        static inline ALWAYS_INLINE_L CONST_L V _pi_2()        ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[2 * V::Size]); }
        static inline ALWAYS_INLINE_L CONST_L V _pi()          ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[3 * V::Size]); }
        static inline ALWAYS_INLINE_L CONST_L V _1_3fac()      ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[4 * V::Size]); }
        static inline ALWAYS_INLINE_L CONST_L V _1_5fac()      ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[5 * V::Size]); }
        static inline ALWAYS_INLINE_L CONST_L V _1_7fac()      ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[6 * V::Size]); }
        static inline ALWAYS_INLINE_L CONST_L V _1_9fac()      ALWAYS_INLINE_R CONST_R { return V(&c_sin<T>::data[7 * V::Size]); }

        static inline ALWAYS_INLINE_L CONST_L M exponentMask() ALWAYS_INLINE_R CONST_R { return M(V(c_log<T>::d(1)).data()); }
        static inline ALWAYS_INLINE_L CONST_L V _1_2()         ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(18)); }
        static inline ALWAYS_INLINE_L CONST_L V _1_sqrt2()     ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(15)); }
        static inline ALWAYS_INLINE_L CONST_L V P(int i)       ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(2 + i)); }
        static inline ALWAYS_INLINE_L CONST_L V Q(int i)       ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(8 + i)); }
        static inline ALWAYS_INLINE_L CONST_L V min()          ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(14)); }
        static inline ALWAYS_INLINE_L CONST_L V ln2_small()    ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(17)); }
        static inline ALWAYS_INLINE_L CONST_L V ln2_large()    ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(16)); }
        static inline ALWAYS_INLINE_L CONST_L V neginf()       ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(13)); }
        static inline ALWAYS_INLINE_L CONST_L V log10_e()      ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(19)); }
        static inline ALWAYS_INLINE_L CONST_L V log2_e()       ALWAYS_INLINE_R CONST_R { return V(c_log<T>::d(20)); }
    };
#define VC_FLOAT8_CONST_IMPL(name) \
    template<> inline ALWAYS_INLINE CONST Vector<float8> Const<float8>::name() { \
        return M256::dup(Const<float>::name().data()); \
    }
    VC_FLOAT8_CONST_IMPL(_1_2pi)
    VC_FLOAT8_CONST_IMPL(_2pi)
    VC_FLOAT8_CONST_IMPL(_pi_2)
    VC_FLOAT8_CONST_IMPL(_pi)
    VC_FLOAT8_CONST_IMPL(_1_3fac)
    VC_FLOAT8_CONST_IMPL(_1_5fac)
    VC_FLOAT8_CONST_IMPL(_1_7fac)
    VC_FLOAT8_CONST_IMPL(_1_9fac)
    VC_FLOAT8_CONST_IMPL(_1_2)
    VC_FLOAT8_CONST_IMPL(_1_sqrt2)
    VC_FLOAT8_CONST_IMPL(min)
    VC_FLOAT8_CONST_IMPL(ln2_small)
    VC_FLOAT8_CONST_IMPL(ln2_large)
    VC_FLOAT8_CONST_IMPL(neginf)
    VC_FLOAT8_CONST_IMPL(log10_e)
    VC_FLOAT8_CONST_IMPL(log2_e)
    template<> inline ALWAYS_INLINE CONST Vector<float8> Const<float8>::P(int i) {
        return M256::dup(Const<float>::P(i).data());
    }
    template<> inline ALWAYS_INLINE CONST Vector<float8> Const<float8>::Q(int i) {
        return M256::dup(Const<float>::Q(i).data());
    }
    template<> inline ALWAYS_INLINE CONST Vector<float8>::Mask Const<float8>::exponentMask() {
        return M256::dup(Const<float>::exponentMask().data());
    }
#undef VC_FLOAT8_CONST_IMPL
} // namespace SSE
} // namespace Vc

#include "undomacros.h"

#endif // VC_SSE_CONST_H
