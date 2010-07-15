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
namespace SSE
{

template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum)
    : Base(VectorHelper<VectorType>::zero())
{
}

template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerOne::OEnum)
    : Base(VectorHelper<T>::one())
{
}

template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : Base(VectorHelper<VectorType>::load(Base::_IndexesFromZero(), Aligned))
{
}

template<typename T> inline Vector<T> Vector<T>::Zero()
{
    return VectorHelper<VectorType>::zero();
}

template<typename T> inline Vector<T> Vector<T>::IndexesFromZero()
{
    return VectorHelper<VectorType>::load(Base::_IndexesFromZero(), Aligned);
}

template<typename T> template<typename OtherT> inline Vector<T>::Vector(const Vector<OtherT> &x)
    : Base(StaticCastHelper<OtherT, T>::cast(x.data()))
{
}

template<typename T> inline Vector<T>::Vector(EntryType a)
    : Base(VectorHelper<T>::set(a))
{
}

template<typename T> inline Vector<T>::Vector(const EntryType *x)
    : Base(VectorHelper<VectorType>::load(x, Aligned))
{
}

template<typename T> template<typename A> inline Vector<T>::Vector(const EntryType *x, A align)
    : Base(VectorHelper<VectorType>::load(x, align))
{
}

template<typename T> inline Vector<T>::Vector(const Vector<typename CtorTypeHelper<T>::Type> *a)
    : Base(VectorHelper<T>::concat(a[0].data(), a[1].data()))
{
}

template<typename T> inline void Vector<T>::expand(Vector<typename ExpandTypeHelper<T>::Type> *x) const
{
    if (Size == 8u) {
        x[0].data() = VectorHelper<T>::expand0(data());
        x[1].data() = VectorHelper<T>::expand1(data());
    }
}

template<typename T> inline void Vector<T>::load(const EntryType *mem)
{
    data() = VectorHelper<VectorType>::load(mem, Aligned);
}

template<typename T> template<typename A> inline void Vector<T>::load(const EntryType *mem, A align)
{
    data() = VectorHelper<VectorType>::load(mem, align);
}

template<typename T> inline void Vector<T>::makeZero()
{
    data() = VectorHelper<VectorType>::zero();
}

template<typename T> inline void Vector<T>::makeZero(const Mask &k)
{
    data() = VectorHelper<VectorType>::andnot_(mm128_reinterpret_cast<VectorType>(k.data()), data());
}

template<typename T> inline void Vector<T>::store(EntryType *mem) const
{
    VectorHelper<VectorType>::store(mem, data(), Aligned);
}

template<typename T> inline void Vector<T>::store(EntryType *mem, const Mask &mask) const
{
    VectorHelper<VectorType>::store(mem, data(), mm128_reinterpret_cast<VectorType>(mask.data()));
}

template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, A align) const
{
    VectorHelper<VectorType>::store(mem, data(), align);
}

template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, const Mask &mask, A align) const
{
    store(mem, mask);
}

} // namespace SSE
} // namespace Vc
