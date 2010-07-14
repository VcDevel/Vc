/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

namespace Vc
{
namespace LRBni
{

template<> inline _M512D VectorHelper<double>::load(const double *x, AlignedFlag) {
    return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadq(x, _MM_FULLUPC64_NONE, _MM_BROADCAST_8X8));
}
template<> inline _M512D VectorHelper<double>::load(const double *x, UnalignedFlag) {
    return mm512_reinterpret_cast<VectorType>(_mm512_expandq( const_cast<double *>(x), _MM_FULLUPC64_NONE, _MM_HINT_NONE));
}

#define LOAD(T1, T2, conv) \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, AlignedFlag) { \
    return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadd(x, conv, _MM_BROADCAST_16X16, _MM_HINT_NONE)); } \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, UnalignedFlag) { \
    return mm512_reinterpret_cast<VectorType>(_mm512_expandd(const_cast<T2 *>(x), conv, _MM_HINT_NONE)); }

LOAD(float, float, _MM_FULLUPC_NONE)
LOAD(float, float16, _MM_FULLUPC_FLOAT16)
LOAD(float, unsigned char, _MM_FULLUPC_UINT8)
LOAD(float, signed char, _MM_FULLUPC_SINT8)
LOAD(float, char, _MM_FULLUPC_SINT8)
LOAD(float, unsigned short, _MM_FULLUPC_UINT16)
LOAD(float, signed short, _MM_FULLUPC_SINT16)

LOAD(int, int, _MM_FULLUPC_NONE)
LOAD(int, signed char, _MM_FULLUPC_SINT8I)
LOAD(int, signed short, _MM_FULLUPC_SINT16I)

LOAD(unsigned int, unsigned int, _MM_FULLUPC_NONE)
LOAD(unsigned int, unsigned char, _MM_FULLUPC_UINT8I)
LOAD(unsigned int, unsigned short, _MM_FULLUPC_UINT16I)

#undef LOAD

} // namespace LRBni
} // namespace Vc
