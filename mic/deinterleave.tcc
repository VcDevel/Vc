/*  This file is part of the Vc library.

    Copyright (C) 2010-2013 Matthias Kretz <kretz@kde.org>

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

Vc_NAMESPACE_BEGIN(Internal)

template<typename V, typename M, typename A> inline void HelperImpl<MICImpl>::deinterleave(V &a, V &b, const M *m, A)
//template<> inline void deinterleave(float_v &a, float_v &b, const float *m, A align)
{
    typedef typename V::IndexType I;
    const I i = I::IndexesFromZero() << 1;
    a.data() = MicIntrinsics::gather(i.data(), m, MIC::UpDownConversion<typename V::VectorEntryType, M>());
    b.data() = MicIntrinsics::gather(i.data(), m + 1, MIC::UpDownConversion<typename V::VectorEntryType, M>());
}

Vc_NAMESPACE_END
