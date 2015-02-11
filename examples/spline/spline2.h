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

#ifndef SPLINE2_H_
#define SPLINE2_H_

#include <utility>
#include <array>
#include <tuple>
#include <Vc/Vc>
#include "../kdtree/simdize.h"
#include <Vc/vector>
#include "spline.h"

//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

class Spline2
{
public:
    typedef std::array<float, 2> Point2;
    typedef std::array<float, 3> Point3;

    typedef simdize<Point2> Point2V;
    typedef simdize<Point3> Point3V;

    Spline2(float minA, float maxA, int nBinsA, float minB, float maxB, int nBinsB);

    /**  Filling of points */
    template <typename F> void Fill(F &&func);
    /**  Filling of points */
    void Fill(int ind, float x, float y, float z);
    /**  Filling of points */
    void Fill(int ind, const float XYZ[]);

    /**  Get A,B by the point index */
    std::pair<float, float> GetAB(int ind) const;

    /** Calculate interpolated value at the given point(s) */
    Point3 GetValue(Point2) const;
    Point3V GetValue(Point2V ab) const;

    /**  Get size of the grid */
    int GetMapSize() const;

    /**  Get N of point on the grid */
    int GetNPoints() const;

private:
    /** copy constructor prohibited */
    Spline2(const Spline2 &);
    /** assignment operator prohibited */
    Spline2 &operator=(const Spline2 &);

    /** spline 2-nd order, 3 points, da = a - point 1 */
    static float GetSpline2(float *v, float da);

    const int fNA;        // N points A axis
    const int fNB;        // N points A axis
    const int fN;         // N points total
    const float fMinA;    // min A axis
    const float fMinB;    // min B axis
    const float fStepA;   // step between points A axis
    const float fStepB;   // step between points B axis
    const float fScaleA;  // scale A axis
    const float fScaleB;  // scale B axis
    Vc::vector<float, Vc::Allocator<float>> fXYZ;  // array of points, {X,Y,Z,0} values
};

inline void Spline2::Fill(int ind, float x, float y, float z)
{
    ind = ind / fNB + fNA * (ind % fNB);
    fXYZ[ind] = x;
    fXYZ[ind + fN] = y;
    fXYZ[ind + 2 * fN] = z;
}

inline void Spline2::Fill(int ind, const float XYZ[])
{
    Fill(ind, XYZ[0], XYZ[1], XYZ[2]);
}

template <typename F> inline void Spline2::Fill(F &&func)
{
    for (int i = 0; i < GetNPoints(); i++) {
        float a, b;
        std::tie(a, b) = GetAB(i);
        std::array<float, 3> xyz = func(a, b);
        Fill(i, xyz[0], xyz[1], xyz[2]);
    }
}

inline std::pair<float, float> Spline2::GetAB(int ind) const
{
    return std::make_pair(fMinA + (ind / fNA) * fStepA, fMinB + (ind % fNB) * fStepB);
}

inline int Spline2::GetMapSize() const { return 4 * sizeof(float) * fN; }

inline int Spline2::GetNPoints() const { return fN; }

inline float Spline2::GetSpline2(float *v, float x)
{
    return 0.5f * x * ((v[0] + v[2] - v[1] - v[1]) * x + v[2] - v[0]) + v[1];
}

// spline 3-st order,  4 points, da = a - point 1 {{{1
template <typename T> static __attribute__((always_inline)) inline T GetSpline3(T v0, T v1, T v2, T v3, T x)
{
    const T dv = v2 - v1;
    const T z0 = 0.5f * (v2 - v0);
    const T z1 = 0.5f * (v3 - v1);
    return (x * x) * ((z1 - dv) * (x - 1) + (z0 - dv) * (x - 2)) + (z0 * x + v1);
}

template <typename T> static __attribute__((always_inline)) inline T GetSpline3(const T v[], T x)
{
    return GetSpline3(v[0], v[1], v[2], v[3], x);
}

inline Point3 Spline2::GetValue(Point2 ab) const  //{{{1
{
    float da1, db1;
    int iA, iB;
    std::tie(iA, iB, da1, db1) =
        evaluatePosition(ab, {fMinA, fMinB}, {fScaleA, fScaleB}, fNA, fNB);

    typedef Vc::simdarray<float, 4> float4;
    typedef Vc::simdarray<float, 12> float12;
    const float4 da = da1;
    const float12 db = db1;

    const float *m0 = &fXYZ[iA + iB * fNA];
    const float *m1 = m0 + fNA;
    const float *m2 = m1 + fNA;
    const float *m3 = m2 + fNA;

    const float12 xyz = GetSpline3(
        Vc::simd_cast<float12>(float4(m0), float4(m0 + fN), float4(m0 + 2 * fN)),
        Vc::simd_cast<float12>(float4(m1), float4(m1 + fN), float4(m1 + 2 * fN)),
        Vc::simd_cast<float12>(float4(m2), float4(m2 + fN), float4(m2 + 2 * fN)),
        Vc::simd_cast<float12>(float4(m3), float4(m3 + fN), float4(m3 + 2 * fN)), db);

    float4 v[4];
    Vc::tie(v[0], v[1], v[2], v[3]) =
        Vc::transpose(Vc::simd_cast<float4, 0>(xyz), Vc::simd_cast<float4, 1>(xyz),
                      Vc::simd_cast<float4, 2>(xyz), float4::Zero());

    float4 res = GetSpline3(v[0], v[1], v[2], v[3], da);
    return {res[0], res[1], res[2]};
}

inline Spline2::Point3V Spline2::GetValue(Point2V ab) const  //{{{1
{
    index_v iA, iB;
    float_v da, db;
    std::tie(iA, iB, da, db) =
        evaluatePosition(ab, {fMinA, fMinB}, {fScaleA, fScaleB}, fNA, fNB);

    auto ind = iA + iB * fNA;
    Point3V xyz;

    {
        float_v x[4][4];
        Vc::tie(x[0][0], x[1][0], x[2][0], x[3][0]) = fXYZ[ind];
        Vc::tie(x[0][1], x[1][1], x[2][1], x[3][1]) = fXYZ[ind + fNA];
        Vc::tie(x[0][2], x[1][2], x[2][2], x[3][2]) = fXYZ[ind + 2 * fNA];
        Vc::tie(x[0][3], x[1][3], x[2][3], x[3][3]) = fXYZ[ind + 3 * fNA];
        xyz[0] = GetSpline3(GetSpline3(x[0], db), GetSpline3(x[1], db),
                            GetSpline3(x[2], db), GetSpline3(x[3], db), da);
    }
    ind += fN;
    {
        float_v y[4][4];
        Vc::tie(y[0][0], y[1][0], y[2][0], y[3][0]) = fXYZ[ind];
        Vc::tie(y[0][1], y[1][1], y[2][1], y[3][1]) = fXYZ[ind + fNA];
        Vc::tie(y[0][2], y[1][2], y[2][2], y[3][2]) = fXYZ[ind + 2 * fNA];
        Vc::tie(y[0][3], y[1][3], y[2][3], y[3][3]) = fXYZ[ind + 3 * fNA];
        xyz[1] = GetSpline3(GetSpline3(y[0], db), GetSpline3(y[1], db),
                            GetSpline3(y[2], db), GetSpline3(y[3], db), da);
    }
    ind += fN;
    {
        float_v z[4][4];
        Vc::tie(z[0][0], z[1][0], z[2][0], z[3][0]) = fXYZ[ind];
        Vc::tie(z[0][1], z[1][1], z[2][1], z[3][1]) = fXYZ[ind + fNA];
        Vc::tie(z[0][2], z[1][2], z[2][2], z[3][2]) = fXYZ[ind + 2 * fNA];
        Vc::tie(z[0][3], z[1][3], z[2][3], z[3][3]) = fXYZ[ind + 3 * fNA];
        xyz[2] = GetSpline3(GetSpline3(z[0], db), GetSpline3(z[1], db),
                            GetSpline3(z[2], db), GetSpline3(z[3], db), da);
    }
    return xyz;
}

inline Spline2::Spline2(float minA, float maxA, int nBinsA, float minB,  //{{{1
                        float maxB, int nBinsB)
    : fNA(nBinsA < 4 ? 4 : nBinsA)
    , fNB(nBinsB < 4 ? 4 : nBinsB)
    , fN(fNA * fNB)
    , fMinA(minA)
    , fMinB(minB)
    , fStepA(((maxA <= minA ? minA + 1 : maxA) - minA) / (fNA - 1))
    , fStepB(((maxB <= minB ? minB + 1 : maxB) - minB) / (fNB - 1))
    , fScaleA(1.f / fStepA)
    , fScaleB(1.f / fStepB)
    , fXYZ(3 * fN, 0.f)
{
}
//}}}1

#endif  // SPLINE2_H_

// vim: foldmethod=marker
