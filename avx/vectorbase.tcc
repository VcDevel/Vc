/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_VECTORBASE_TCC
#define VC_AVX_VECTORBASE_TCC

#include "macros.h"

namespace Vc
{
namespace AVX
{

#define OP_IMPL(T, symbol) \
template<> inline Vector<T> &VectorBase<T>::operator symbol##=(const VectorBase<T> &x) \
{ \
    for_all_vector_entries(i, \
            d.m(i) symbol##= x.d.m(i); \
            ); \
    return *static_cast<Vector<T> *>(this); \
} \
template<> inline Vector<T>  VectorBase<T>::operator symbol(const VectorBase<T> &x) const \
{ \
    Vector<T> r; \
    for_all_vector_entries(i, \
            r.d.m(i) = d.m(i) symbol x.d.m(i); \
            ); \
    return r; \
}
OP_IMPL(int, <<)
OP_IMPL(int, >>)
OP_IMPL(unsigned int, <<)
OP_IMPL(unsigned int, >>)
OP_IMPL(short, <<)
OP_IMPL(short, >>)
OP_IMPL(unsigned short, <<)
OP_IMPL(unsigned short, >>)
#undef OP_IMPL

#define OP_IMPL(T, PREFIX, SUFFIX) \
template<> inline Vector<T> & INTRINSIC CONST VectorBase<T>::operator<<=(int x) \
{ \
    d.v() = CAT3(PREFIX, _slli_epi, SUFFIX)(d.v(), x); \
    return *static_cast<Vector<T> *>(this); \
} \
template<> inline Vector<T> & INTRINSIC CONST VectorBase<T>::operator>>=(int x) \
{ \
    d.v() = CAT3(PREFIX, _srli_epi, SUFFIX)(d.v(), x); \
    return *static_cast<Vector<T> *>(this); \
} \
template<> inline Vector<T> INTRINSIC CONST VectorBase<T>::operator<<(int x) const \
{ \
    return CAT3(PREFIX, _slli_epi, SUFFIX)(d.v(), x); \
} \
template<> inline Vector<T> INTRINSIC CONST VectorBase<T>::operator>>(int x) const \
{ \
    return CAT3(PREFIX, _srli_epi, SUFFIX)(d.v(), x); \
}
OP_IMPL(int, _mm256, 32)
OP_IMPL(unsigned int, _mm256, 32)
OP_IMPL(short, _mm, 16)
OP_IMPL(unsigned short, _mm, 16)
#undef OP_IMPL

} // namespace AVX
} // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_VECTORBASE_TCC
