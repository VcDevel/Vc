/*{{{
    Copyright © 2015 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.


This work is derived from a class in ALICE with the following copyright notice:
    **************************************************************************
    * This file is property of and copyright by the ALICE HLT Project        *
    * ALICE Experiment at CERN, All rights reserved.                         *
    *                                                                        *
    * Primary Authors: Sergey Gorbunov <sergey.gorbunov@cern.ch>             *
    *                  for The ALICE HLT Project.                            *
    *                                                                        *
    * Permission to use, copy, modify and distribute this software and its   *
    * documentation strictly for non-commercial purposes is hereby granted   *
    * without fee, provided that the above copyright notice appears in all   *
    * copies and that both the copyright notice and this permission notice   *
    * appear in the supporting documentation. The authors make no claims     *
    * about the suitability of this software for any purpose. It is          *
    * provided "as is" without express or implied warranty.                  *
    **************************************************************************
}}}*/

#include "spline.h"
#include <Vc/Vc>

#include <iostream>
#include <iomanip>

using namespace std;

Spline::Spline(float minA, float maxA, int nBinsA, float minB, float maxB,  //{{{1
               int nBinsB)
    : fNA(nBinsA < 4 ? 4 : nBinsA)
    , fNB(nBinsB < 4 ? 4 : nBinsB)
    , fN(fNA * fNB)
    , fMinA(minA)
    , fMinB(minB)
    , fStepA(((maxA <= minA ? minA + 1 : maxA) - minA) / (fNA - 1))
    , fStepB(((maxB <= minB ? minB + 1 : maxB) - minB) / (fNB - 1))
    , fScaleA(1.f / fStepA)
    , fScaleB(1.f / fStepB)
    , fXYZ(fN, DataPoint::Zero())
{
}

// spline 3-st order,  4 points, da = a - point 1 {{{1
template <typename T> static inline T GetSpline3(T v0, T v1, T v2, T v3, T x)
{
    const T dv = v2 - v1;
    const T z0 = 0.5f * (v2 - v0);
    const T z1 = 0.5f * (v3 - v1);
    return (x * x) * ((z1 - dv) * (x - 1) + (z0 - dv) * (x - 2)) + (z0 * x + v1);
    //return x * x * ((z1 - dv + z0 - dv) * (x - 1) - (z0 - dv)) + z0 * x + v1;
}

template <typename T> static inline T GetSpline3(const T *v, T x)
{
    return GetSpline3(v[0], v[1], v[2], v[3], x);
}

Point3 Spline::GetValue(Point2 ab) const  //{{{1
{
    float da1, db1;
    int iA, iB;
    std::tie(iA, iB, da1, db1) =
        evaluatePosition(ab, {{fMinA, fMinB}}, {{fScaleA, fScaleB}}, fNA, fNB);
    int ind = iA * fNB + iB;

    typedef Vc::SimdArray<float, 4> float4;
    const float4 da = da1;
    const float4 db = db1;

    float4 v[4];
    const float4 *m = &fXYZ[0];

    for (int i = 0; i < 4; i++) {
        v[i] = GetSpline3(m[ind + 0], m[ind + 1], m[ind + 2], m[ind + 3], db);
        ind += fNB;
    }
    float4 res = GetSpline3(v[0], v[1], v[2], v[3], da);
    return {{res[0], res[1], res[2]}};
}

Point3 Spline::GetValue16(Point2 ab) const  //{{{1
{
    float da1, db1;
    int iA, iB;
    std::tie(iA, iB, da1, db1) =
        evaluatePosition(ab, {{fMinA, fMinB}}, {{fScaleA, fScaleB}}, fNA, fNB);

    typedef Vc::SimdArray<float, 4> float4;
    typedef Vc::SimdArray<float, 16> float16;
    const float4 da = da1;
    const float16 db = db1;

    const float4 *m0 = &fXYZ[iA * fNB + iB];
    const float4 *m1 = m0 + fNB;
    const float4 *m2 = m1 + fNB;
    const float4 *m3 = m2 + fNB;
    const float16 v0123 =
        GetSpline3(Vc::simd_cast<float16>(m0[0], m1[0], m2[0], m3[0]),
                   Vc::simd_cast<float16>(m0[1], m1[1], m2[1], m3[1]),
                   Vc::simd_cast<float16>(m0[2], m1[2], m2[2], m3[2]),
                   Vc::simd_cast<float16>(m0[3], m1[3], m2[3], m3[3]), db);
    const float4 res =
        GetSpline3(Vc::simd_cast<float4, 0>(v0123), Vc::simd_cast<float4, 1>(v0123),
                   Vc::simd_cast<float4, 2>(v0123), Vc::simd_cast<float4, 3>(v0123), da);
    return {{res[0], res[1], res[2]}};
}

Point3 Spline::GetValueScalar(Point2 ab) const  //{{{1
{
    float da, db;
    int iA, iB;
    std::tie(iA, iB, da, db) =
        evaluatePosition(ab, {{fMinA, fMinB}}, {{fScaleA, fScaleB}}, fNA, fNB);
    int ind = iA * fNB + iB;

    float vx[4];
    float vy[4];
    float vz[4];
    for (int i = 0; i < 4; i++) {
        vx[i] = GetSpline3(fXYZ[ind][0], fXYZ[ind + 1][0], fXYZ[ind + 2][0],
                           fXYZ[ind + 3][0], db);
        vy[i] = GetSpline3(fXYZ[ind][1], fXYZ[ind + 1][1], fXYZ[ind + 2][1],
                           fXYZ[ind + 3][1], db);
        vz[i] = GetSpline3(fXYZ[ind][2], fXYZ[ind + 1][2], fXYZ[ind + 2][2],
                           fXYZ[ind + 3][2], db);
        ind += fNB;
    }
    return {{GetSpline3(vx, da), GetSpline3(vy, da), GetSpline3(vz, da)}};
}

Point3 Spline::GetValueAutovec(Point2 ab) const  //{{{1
{
    float da, db;
    int iA, iB;
    std::tie(iA, iB, da, db) =
        evaluatePosition(ab, {{fMinA, fMinB}}, {{fScaleA, fScaleB}}, fNA, fNB);
    int ind = iA * fNB + iB;

    float vx[4];
    float vy[4];
    float vz[4];
    const float *m = reinterpret_cast<const float *>(&fXYZ[0]);
    for (int i = 0; i < 4; i++) {
        int ind4 = ind * 4;
        vx[i] = GetSpline3(m[ind4 + 0], m[ind4 + 4], m[ind4 + 8], m[ind4 + 12], db);
        vy[i] = GetSpline3(m[ind4 + 1], m[ind4 + 5], m[ind4 + 9], m[ind4 + 13], db);
        vz[i] = GetSpline3(m[ind4 + 2], m[ind4 + 6], m[ind4 + 10], m[ind4 + 14], db);
        ind += fNB;
    }
    return {{GetSpline3(vx, da), GetSpline3(vy, da), GetSpline3(vz, da)}};
}

Point3V Spline::GetValue(const Point2V &ab) const  //{{{1
{
    index_v iA, iB;
    float_v da, db;
    std::tie(iA, iB, da, db) =
        evaluatePosition(ab, {{fMinA, fMinB}}, {{fScaleA, fScaleB}}, fNA, fNB);

    float_v vx[4];
    float_v vy[4];
    float_v vz[4];
    auto ind = iA * fNB + iB;
    const auto map = Vc::make_interleave_wrapper<float_v>(&fXYZ[0]);
    for (int i = 0; i < 4; i++) {
        float_v x[4], y[4], z[4];
        Vc::tie(x[0], y[0], z[0]) = map[ind];
        Vc::tie(x[1], y[1], z[1]) = map[ind + 1];
        Vc::tie(x[2], y[2], z[2]) = map[ind + 2];
        Vc::tie(x[3], y[3], z[3]) = map[ind + 3];
        vx[i] = GetSpline3<float_v>(x[0], x[1], x[2], x[3], db);
        vy[i] = GetSpline3<float_v>(y[0], y[1], y[2], y[3], db);
        vz[i] = GetSpline3<float_v>(z[0], z[1], z[2], z[3], db);
        ind += fNB;
    }
    Point3V XYZ;
    XYZ[0] = GetSpline3<float_v>(vx, da);
    XYZ[1] = GetSpline3<float_v>(vy, da);
    XYZ[2] = GetSpline3<float_v>(vz, da);
    return XYZ;
}

// Point3 Spline::GetValueAlice(Point2 ab) const {{{1
#pragma GCC diagnostic ignored "-Wold-style-cast"
__attribute__((optimize("no-tree-vectorize")))
Point3 Spline::GetValueAlice(Point2 ab) const
{
    float lA = (ab[0] - fMinA) * fScaleA - 1.f;
    int iA = (int)lA;
    if (lA < 0) iA = 0;
    else if (iA > fNA - 4) iA = fNA - 4;

    float lB = (ab[1] - fMinB) * fScaleB - 1.f;
    int iB = (int)lB;
    if (lB < 0) iB = 0;
    else if (iB > fNB - 4) iB = fNB - 4;

    Point3 XYZ;
    if (Vc::float_v::Size == 4) {
        Vc::float_v da = lA - iA;
        Vc::float_v db = lB - iB;

        Vc::float_v v[4];
        int ind = iA * fNB + iB;
        const Vc::float_v *m = reinterpret_cast<const Vc::float_v *>(&fXYZ[0]);

        for (int i = 0; i < 4; i++) {
            v[i] = GetSpline3(m[ind + 0], m[ind + 1], m[ind + 2], m[ind + 3], db);
            ind += fNB;
        }
        Vc::float_v res = GetSpline3(v[0], v[1], v[2], v[3], da);
        XYZ[0] = res[0];
        XYZ[1] = res[1];
        XYZ[2] = res[2];
    } else {
        float da = lA - iA;
        float db = lB - iB;

        float vx[4];
        float vy[4];
        float vz[4];
        int ind = iA * fNB + iB;
        const float *m = reinterpret_cast<const float *>(&fXYZ[0]);
        for (int i = 0; i < 4; i++) {
            int ind4 = ind * 4;
            vx[i] = GetSpline3(m[ind4 + 0], m[ind4 + 4], m[ind4 + 8], m[ind4 + 12], db);
            vy[i] = GetSpline3(m[ind4 + 1], m[ind4 + 5], m[ind4 + 9], m[ind4 + 13], db);
            vz[i] = GetSpline3(m[ind4 + 2], m[ind4 + 6], m[ind4 + 10], m[ind4 + 14], db);
            ind += fNB;
        }
        XYZ[0] = GetSpline3(vx, da);
        XYZ[1] = GetSpline3(vy, da);
        XYZ[2] = GetSpline3(vz, da);
    }
    return XYZ;
}
// vim: foldmethod=marker
