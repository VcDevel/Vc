/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

        static inline V CONST_L _1_2pi()  CONST_R { return V(&c_sin<T>::data[0 * V::Size]); }
        static inline V CONST_L _2pi()    CONST_R { return V(&c_sin<T>::data[1 * V::Size]); }
        static inline V CONST_L _pi_2()   CONST_R { return V(&c_sin<T>::data[2 * V::Size]); }
        static inline V CONST_L _pi()     CONST_R { return V(&c_sin<T>::data[3 * V::Size]); }
        static inline V CONST_L _1_3fac() CONST_R { return V(&c_sin<T>::data[4 * V::Size]); }
        static inline V CONST_L _1_5fac() CONST_R { return V(&c_sin<T>::data[5 * V::Size]); }
        static inline V CONST_L _1_7fac() CONST_R { return V(&c_sin<T>::data[6 * V::Size]); }
        static inline V CONST_L _1_9fac() CONST_R { return V(&c_sin<T>::data[7 * V::Size]); }

        static inline M CONST_L exponentMask() CONST_R { return M(V(c_log<T>::d(1)).data()); }
        static inline V CONST_L _1_2()         CONST_R { return V(c_log<T>::d(18)); }
        static inline V CONST_L _1_sqrt2()     CONST_R { return V(c_log<T>::d(15)); }
        static inline V CONST_L P(int i)       CONST_R { return V(c_log<T>::d(2 + i)); }
        static inline V CONST_L Q(int i)       CONST_R { return V(c_log<T>::d(8 + i)); }
        static inline V CONST_L min()          CONST_R { return V(c_log<T>::d(14)); }
        static inline V CONST_L ln2_small()    CONST_R { return V(c_log<T>::d(17)); }
        static inline V CONST_L ln2_large()    CONST_R { return V(c_log<T>::d(16)); }
        static inline V CONST_L neginf()       CONST_R { return V(c_log<T>::d(13)); }
        static inline V CONST_L log10_e()      CONST_R { return V(c_log<T>::d(19)); }
        static inline V CONST_L log2_e()       CONST_R { return V(c_log<T>::d(20)); }
    };
#define VC_FLOAT8_CONST_IMPL(name) \
    template<> inline Vector<float8> Const<float8>::name() { \
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
    template<> inline Vector<float8> Const<float8>::P(int i) {
        return M256::dup(Const<float>::P(i).data());
    }
    template<> inline Vector<float8> Const<float8>::Q(int i) {
        return M256::dup(Const<float>::Q(i).data());
    }
    template<> inline Vector<float8>::Mask Const<float8>::exponentMask() {
        return M256::dup(Const<float>::exponentMask().data());
    }
#undef VC_FLOAT8_CONST_IMPL
} // namespace SSE
} // namespace Vc

#include "undomacros.h"

#endif // VC_SSE_CONST_H
