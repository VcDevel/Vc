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

#ifndef VC_MIC_DEINTERLEAVE_H
#define VC_MIC_DEINTERLEAVE_H

#include "macros.h"

Vc_NAMESPACE_BEGIN(Internal)

template<> struct HelperImpl<Vc::MICImpl>
{
    typedef MIC::Vector<float> float_v;
    typedef MIC::Vector<double> double_v;
    typedef MIC::Vector<int> int_v;
    typedef MIC::Vector<unsigned int> uint_v;

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

    static Vc_ALWAYS_INLINE_L void prefetchForOneRead(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchForModify(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchClose(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchMid(const void *addr) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void prefetchFar(const void *addr) Vc_ALWAYS_INLINE_R;

    template<Vc::MallocAlignment A>
    static Vc_ALWAYS_INLINE_L void *malloc(size_t n) Vc_ALWAYS_INLINE_R;
    static Vc_ALWAYS_INLINE_L void free(void *p) Vc_ALWAYS_INLINE_R;
};

Vc_NAMESPACE_END

#include "deinterleave.tcc"
#include "prefetches.tcc"
#include "helperimpl.tcc"
#include "undomacros.h"

#endif // VC_MIC_DEINTERLEAVE_H
