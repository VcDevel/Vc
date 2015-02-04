//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Sergey Gorbunov <sergey.gorbunov@cern.ch>             *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

#include "spline.h"
#include <Vc/Vc>
#include "../kdtree/simdize.h"

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
    , fXYZ(fN, DataPoint{0.f, 0.f, 0.f, 0.f})
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

std::array<float, 3> Spline::GetValue(std::array<float, 2> ab) const  //{{{1
{
    float lA = (ab[0] - fMinA) * fScaleA - 1.f;
    int iA = static_cast<int>(lA);
    if (lA < 0)
        iA = 0;
    else if (iA > fNA - 4)
        iA = fNA - 4;

    float lB = (ab[1] - fMinB) * fScaleB - 1.f;
    int iB = static_cast<int>(lB);
    if (lB < 0)
        iB = 0;
    else if (iB > fNB - 4)
        iB = fNB - 4;

    typedef simdize<float, 4> float_v;
    float_v da = lA - iA;
    float_v db = lB - iB;

    float_v v[4];
    int ind = iA * fNB + iB;
    const float_v *m = reinterpret_cast<const float_v *>(&fXYZ[0]);

    for (int i = 0; i < 4; i++) {
        v[i] = GetSpline3(m[ind + 0], m[ind + 1], m[ind + 2], m[ind + 3], db);
        ind += fNB;
    }
    float_v res = GetSpline3(v[0], v[1], v[2], v[3], da);
    std::array<float, 3> XYZ;
    XYZ[0] = res[0];
    XYZ[1] = res[1];
    XYZ[2] = res[2];
    return XYZ;
}

std::array<float, 3> Spline::GetValue16(std::array<float, 2> ab) const  //{{{1
{
    float lA = (ab[0] - fMinA) * fScaleA - 1.f;
    int iA = static_cast<int>(lA);
    if (lA < 0)
        iA = 0;
    else if (iA > fNA - 4)
        iA = fNA - 4;

    float lB = (ab[1] - fMinB) * fScaleB - 1.f;
    int iB = static_cast<int>(lB);
    if (lB < 0)
        iB = 0;
    else if (iB > fNB - 4)
        iB = fNB - 4;

    typedef Vc::simdarray<float, 4> float4;
    typedef Vc::simdarray<float, 16> float16;
    const float4 da = lA - iA;
    const float16 db = lB - iB;

    const float *m0 = &fXYZ[iA * fNB + iB].x;
    const float *m1 = m0 + fNB * 4;
    const float *m2 = m1 + fNB * 4;
    const float *m3 = m2 + fNB * 4;
    const float16 v0123 = GetSpline3(
        Vc::simd_cast<float16>(float4(m0, Vc::Aligned), float4(m1, Vc::Aligned),
                               float4(m2, Vc::Aligned), float4(m3, Vc::Aligned)),
        Vc::simd_cast<float16>(float4(m0 + 4, Vc::Aligned), float4(m1 + 4, Vc::Aligned),
                               float4(m2 + 4, Vc::Aligned), float4(m3 + 4, Vc::Aligned)),
        Vc::simd_cast<float16>(float4(m0 + 8, Vc::Aligned), float4(m1 + 8, Vc::Aligned),
                               float4(m2 + 8, Vc::Aligned), float4(m3 + 8, Vc::Aligned)),
        Vc::simd_cast<float16>(float4(m0 + 12, Vc::Aligned), float4(m1 + 12, Vc::Aligned),
                               float4(m2 + 12, Vc::Aligned),
                               float4(m3 + 12, Vc::Aligned)),
        db);
    const float4 res =
        GetSpline3(Vc::simd_cast<float4, 0>(v0123), Vc::simd_cast<float4, 1>(v0123),
                   Vc::simd_cast<float4, 2>(v0123), Vc::simd_cast<float4, 3>(v0123), da);
    std::array<float, 3> XYZ;
    XYZ[0] = res[0];
    XYZ[1] = res[1];
    XYZ[2] = res[2];
    return XYZ;
}

std::array<float, 3> Spline::GetValueScalar(std::array<float, 2> ab) const  //{{{1
{
    float lA = (ab[0] - fMinA) * fScaleA - 1.f;
    int iA = static_cast<int>(lA);
    if (lA < 0)
        iA = 0;
    else if (iA > fNA - 4)
        iA = fNA - 4;

    float lB = (ab[1] - fMinB) * fScaleB - 1.f;
    int iB = static_cast<int>(lB);
    if (lB < 0)
        iB = 0;
    else if (iB > fNB - 4)
        iB = fNB - 4;

    float da = lA - iA;
    float db = lB - iB;

    float vx[4];
    float vy[4];
    float vz[4];
    int ind = iA * fNB + iB;
    for (int i = 0; i < 4; i++) {
        vx[i] = GetSpline3(fXYZ[ind].x, fXYZ[ind + 1].x, fXYZ[ind + 2].x,
                           fXYZ[ind + 3].x, db);
        vy[i] = GetSpline3(fXYZ[ind].y, fXYZ[ind + 1].y, fXYZ[ind + 2].y,
                           fXYZ[ind + 3].y, db);
        vz[i] = GetSpline3(fXYZ[ind].z, fXYZ[ind + 1].z, fXYZ[ind + 2].z,
                           fXYZ[ind + 3].z, db);
        ind += fNB;
    }
    std::array<float, 3> XYZ;
    XYZ[0] = GetSpline3(vx, da);
    XYZ[1] = GetSpline3(vy, da);
    XYZ[2] = GetSpline3(vz, da);
    return XYZ;
}

Spline::Point3V Spline::GetValue(const Point2V &ab) const  //{{{1
{
    using Vc::float_v;

    const float_v lA = (ab[0] - fMinA) * fScaleA - 1.f;
    float_v iA = trunc(lA);
    iA.setZero(lA < 0);
    where(iA > fNA - 4) | iA = fNA - 4;

    const float_v lB = (ab[1] - fMinB) * fScaleB - 1.f;
    float_v iB = trunc(lB);
    iB.setZero(lB < 0);
    where(iB > fNB - 4) | iB = fNB - 4;

    const float_v da = lA - iA;
    const float_v db = lB - iB;

    float_v vx[4];
    float_v vy[4];
    float_v vz[4];
    auto ind = static_cast<float_v::IndexType>(iA * fNB + iB);
    const auto map = Vc::make_interleave_wrapper<float_v>(&fXYZ[0]);
    //std::cerr << typeid(map).name() << std::endl; exit(1);
    for (int i = 0; i < 4; i++) {
        float_v x[4], y[4], z[4];
        Vc::tie(x[0], y[0], z[0]) = map[ind + 0];
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

// vim: foldmethod=marker
