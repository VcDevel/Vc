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

#ifndef VC_LARRABEE_DEINTERLEAVE_H
#define VC_LARRABEE_DEINTERLEAVE_H

namespace Vc
{
namespace Internal
{

template<> struct HelperImpl<Vc::LRBniImpl>
{
    typedef LRBni::Vector<float> float_v;
    typedef LRBni::Vector<double> double_v;
    typedef LRBni::Vector<int> int_v;
    typedef LRBni::Vector<unsigned int> uint_v;

    template<typename V, typename M, typename A> static void deinterleave(V &a, V &b, const M *m, A);
//  template<typename A> static void deinterleave(float_v &, float_v &, const float *, A);
//  template<typename A> static void deinterleave(float_v &, float_v &, const short *, A);
//  template<typename A> static void deinterleave(float_v &, float_v &, const unsigned short *, A);
//
//  template<typename A> static void deinterleave(double_v &, double_v &, const double *, A);
//
//  template<typename A> static void deinterleave(int_v &, int_v &, const int *, A);
//  template<typename A> static void deinterleave(int_v &, int_v &, const short *, A);
//
//  template<typename A> static void deinterleave(uint_v &, uint_v &, const unsigned int *, A);
//  template<typename A> static void deinterleave(uint_v &, uint_v &, const unsigned short *, A);

    static inline void prefetchForOneRead(const void *addr) ALWAYS_INLINE;
    static inline void prefetchForModify(const void *addr) ALWAYS_INLINE;
    static inline void prefetchClose(const void *addr) ALWAYS_INLINE;
    static inline void prefetchMid(const void *addr) ALWAYS_INLINE;
    static inline void prefetchFar(const void *addr) ALWAYS_INLINE;
};

} // namespace Internal
} // namespace Vc

#include "deinterleave.tcc"
#include "prefetches.tcc"

#endif // VC_LARRABEE_DEINTERLEAVE_H
