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

#ifndef SPLINE3_H_
#define SPLINE3_H_

#include <utility>
#include <array>
#include <tuple>
#include <Vc/Vc>
#include <Vc/vector>
#include "spline2.h"

//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

class Spline3
{
public:
    Spline3(float minA, float maxA, int nBinsA, float minB, float maxB, int nBinsB);

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
    Point3V GetValue(const Point2V &ab) const;

    /**  Get size of the grid */
    int GetMapSize() const;

    /**  Get N of point on the grid */
    int GetNPoints() const;

private:
    /** copy constructor prohibited */
    Spline3(const Spline3 &);
    /** assignment operator prohibited */
    Spline3 &operator=(const Spline3 &);

    const int fNA;        // N points A axis
    const int fNB;        // N points A axis
    const int fN;         // N points total
    const float fMinA;    // min A axis
    const float fMinB;    // min B axis
    const float fStepA;   // step between points A axis
    const float fStepB;   // step between points B axis
    const float fScaleA;  // scale A axis
    const float fScaleB;  // scale B axis
    Vc::vector<Point3> fXYZ;
};

inline void Spline3::Fill(int ind, float x, float y, float z)
{
    ind = ind / fNB + fNA * (ind % fNB);
    fXYZ[ind][0] = x;
    fXYZ[ind][1] = y;
    fXYZ[ind][2] = z;
}

inline void Spline3::Fill(int ind, const float XYZ[])
{
    Fill(ind, XYZ[0], XYZ[1], XYZ[2]);
}

template <typename F> inline void Spline3::Fill(F &&func)
{
    for (int i = 0; i < GetNPoints(); i++) {
        float a, b;
        std::tie(a, b) = GetAB(i);
        std::array<float, 3> xyz = func(a, b);
        Fill(i, xyz[0], xyz[1], xyz[2]);
    }
}

inline std::pair<float, float> Spline3::GetAB(int ind) const
{
    return std::make_pair(fMinA + (ind / fNA) * fStepA, fMinB + (ind % fNB) * fStepB);
}

inline int Spline3::GetMapSize() const { return 4 * sizeof(float) * fN; }

inline int Spline3::GetNPoints() const { return fN; }

inline Point3 Spline3::GetValue(Point2 ab) const  //{{{1
{
    float da1, db1;
    int iA, iB;
    std::tie(iA, iB, da1, db1) =
        evaluatePosition(ab, {fMinA, fMinB}, {fScaleA, fScaleB}, fNA, fNB);

    typedef Vc::simdarray<float, 4> float4;
    typedef Vc::simdarray<float, 12> float12;
    const float4 da = da1;
    const float12 db = db1;

    const float *m0 = &fXYZ[iA + iB * fNA][0];
    const float *m1 = m0 + fNA * 3;
    const float *m2 = m1 + fNA * 3;
    const float *m3 = m2 + fNA * 3;

    const float12 xyz = GetSpline3(float12(m0),  // x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3
                                   float12(m1), float12(m2), float12(m3), db);

    const float4 t0 = Vc::simd_cast<float4, 0>(xyz);  // x0 y0 z0 x1
    const float4 t1 = Vc::simd_cast<float4, 1>(xyz);  // y1 z1 x2 y2
    const float4 t2 = Vc::simd_cast<float4, 2>(xyz);  // z2 x3 y3 z3

    const float4 res =
        GetSpline3(t0, t0.shifted(3, t1), t1.shifted(2, t2), t2.shifted(1), da);
    return {res[0], res[1], res[2]};
}

Point3V Spline3::GetValue(const Point2V &ab) const  //{{{1
{
    index_v iA, iB;
    float_v da, db;
    std::tie(iA, iB, da, db) =
        evaluatePosition(ab, {fMinA, fMinB}, {fScaleA, fScaleB}, fNA, fNB);

    float_v vx[4];
    float_v vy[4];
    float_v vz[4];
    auto ind = iA + iB * fNA;
    for (int i = 0; i < 4; i++) {
        float_v x[4], y[4], z[4];
        Vc::tie(x[0], y[0], z[0]) = fXYZ[ind][0];
        Vc::tie(x[1], y[1], z[1]) = fXYZ[ind + fNA][0];
        Vc::tie(x[2], y[2], z[2]) = fXYZ[ind + 2 * fNA][0];
        Vc::tie(x[3], y[3], z[3]) = fXYZ[ind + 3 * fNA][0];
        vx[i] = GetSpline3<float_v>(x[0], x[1], x[2], x[3], db);
        vy[i] = GetSpline3<float_v>(y[0], y[1], y[2], y[3], db);
        vz[i] = GetSpline3<float_v>(z[0], z[1], z[2], z[3], db);
        ind += 1;
    }
    Point3V XYZ;
    XYZ[0] = GetSpline3<float_v>(vx, da);
    XYZ[1] = GetSpline3<float_v>(vy, da);
    XYZ[2] = GetSpline3<float_v>(vz, da);
    return XYZ;
}

inline Spline3::Spline3(float minA, float maxA, int nBinsA, float minB,  //{{{1
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
    , fXYZ(fN)
{
}
//}}}1

#endif  // SPLINE3_H_

// vim: foldmethod=marker
