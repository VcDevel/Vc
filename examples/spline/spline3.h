#ifndef SPLINE3_H_
#define SPLINE3_H_

#include <utility>
#include <array>
#include <tuple>
#include <Vc/Vc>
#include "../kdtree/simdize.h"
#include <Vc/vector>
#include "spline2.h"

//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

class Spline3
{
public:
    typedef std::array<float, 2> Point2;
    typedef std::array<float, 3> Point3;

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

inline std::array<float, 3> Spline3::GetValue(std::array<float, 2> ab) const  //{{{1
{
    float lA = (ab[0] - fMinA) * fScaleA - 1.f;
    const int iA = std::max(0, std::min(fNA - 4, static_cast<int>(lA)));

    float lB = (ab[1] - fMinB) * fScaleB - 1.f;
    const int iB = std::max(0, std::min(fNB - 4, static_cast<int>(lB)));

    typedef Vc::simdarray<float, 4> float4;
    typedef Vc::simdarray<float, 12> float12;
    const float4 da = lA - iA;
    const float12 db = lB - iB;

    const float *m0 = &fXYZ[iA + iB * fNA][0];
    const float *m1 = m0 + fNA * 3;
    const float *m2 = m1 + fNA * 3;
    const float *m3 = m2 + fNA * 3;

    const float12 xyz = GetSpline3(float12(m0),  // x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3
                                   float12(m1), float12(m2), float12(m3), db);

    const float4 t0 = Vc::simd_cast<float4, 0>(xyz);  // x0 y0 z0 x1
    const float4 t1 = Vc::simd_cast<float4, 1>(xyz);  // y1 z1 x2 y2
    const float4 t2 = Vc::simd_cast<float4, 2>(xyz);  // z2 x3 y3 z3

    float4 res = GetSpline3(t0, t0.shifted(3, t1), t1.shifted(2, t2), t2.shifted(1), da);
    std::array<float, 3> XYZ;
    XYZ[0] = res[0];
    XYZ[1] = res[1];
    XYZ[2] = res[2];
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
