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

template<typename Parent, typename T> template<typename T2> inline void StoreMixin<Parent, T>::store(T2 *mem) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), Aligned);
}

template<typename Parent, typename T> template<typename T2> inline void StoreMixin<Parent, T>::store(T2 *mem, Mask mask) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), mask.data(), Aligned);
}

template<typename Parent, typename T> template<typename T2, typename A> inline void StoreMixin<Parent, T>::store(T2 *mem, A align) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), align);
}

template<typename Parent, typename T> template<typename T2, typename A> inline void StoreMixin<Parent, T>::store(T2 *mem, Mask mask, A align) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), mask.data(), align);
}

template<> inline Vector<double> INTRINSIC Vector<double>::operator-() const
{
    return lrb_cast<__m512d>(_mm512_xor_pi(lrb_cast<__m512i>(data.v()), _mm512_set_1to8_pq(0x8000000000000000ull)));
}
template<> inline Vector<float> INTRINSIC Vector<float>::operator-() const
{
    return lrb_cast<__m512>(_mm512_xor_pi(lrb_cast<__m512i>(data.v()), _mm512_set_1to16_pi(0x80000000u)));
}
template<> inline Vector<int> INTRINSIC Vector<int>::operator-() const
{
    return (~(*this)) + 1;
}
template<> inline Vector<int> INTRINSIC Vector<unsigned int>::operator-() const
{
    return Vector<int>(~(*this)) + 1;
}

template<typename T> inline typename Vector<T>::EntryType Vector<T>::min(Mask m) const
{
    return _mm512_mask_reduce_min_pi(m.data(), vdata());
}
template<> inline float Vector<float>::min(Mask m) const
{
    return _mm512_mask_reduce_min_ps(m.data(), vdata());
}
template<> inline double Vector<double>::min(Mask m) const
{
    return _mm512_mask_reduce_min_pd(m.data(), vdata());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::max(Mask m) const
{
    return _mm512_mask_reduce_max_pi(m.data(), vdata());
}
template<> inline float Vector<float>::max(Mask m) const
{
    return _mm512_mask_reduce_max_ps(m.data(), vdata());
}
template<> inline double Vector<double>::max(Mask m) const
{
    return _mm512_mask_reduce_max_pd(m.data(), vdata());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::product(Mask m) const
{
    return _mm512_mask_reduce_mul_pi(m.data(), vdata());
}
template<> inline float Vector<float>::product(Mask m) const
{
    return _mm512_mask_reduce_mul_ps(m.data(), vdata());
}
template<> inline double Vector<double>::product(Mask m) const
{
    return _mm512_mask_reduce_mul_pd(m.data(), vdata());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::sum(Mask m) const
{
    return _mm512_mask_reduce_add_pi(m.data(), vdata());
}
template<> inline float Vector<float>::sum(Mask m) const
{
    return _mm512_mask_reduce_add_ps(m.data(), vdata());
}
template<> inline double Vector<double>::sum(Mask m) const
{
    return _mm512_mask_reduce_add_pd(m.data(), vdata());
}

} // namespace LRBni
} // namespace Vc
