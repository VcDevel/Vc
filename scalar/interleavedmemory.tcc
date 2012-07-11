/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef VC_SCALAR_INTERLEAVEDMEMORY_TCC
#define VC_SCALAR_INTERLEAVEDMEMORY_TCC

namespace Vc
{
namespace Common
{

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1) const
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2) const
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3) const
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4) const
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5) const
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
    v5.data() = m_data[m_indexes.data() + 5];
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6) const
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
    v5.data() = m_data[m_indexes.data() + 5];
    v6.data() = m_data[m_indexes.data() + 6];
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6, float_v &v7) const
{
    v0.data() = m_data[m_indexes.data() + 0];
    v1.data() = m_data[m_indexes.data() + 1];
    v2.data() = m_data[m_indexes.data() + 2];
    v3.data() = m_data[m_indexes.data() + 3];
    v4.data() = m_data[m_indexes.data() + 4];
    v5.data() = m_data[m_indexes.data() + 5];
    v6.data() = m_data[m_indexes.data() + 6];
    v7.data() = m_data[m_indexes.data() + 7];
}

} // namespace Common
} // namespace Vc

#endif // VC_SCALAR_INTERLEAVEDMEMORY_TCC
