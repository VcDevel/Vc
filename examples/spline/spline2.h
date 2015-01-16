#ifndef SPLINE2_H_
#define SPLINE2_H_

#include <utility>
#include <array>
#include <tuple>
#include <Vc/Vc>
#include "../kdtree/simdize.h"
#include <Vc/vector>

//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

class AliTPCParam;
class AliTPCTransform;

template <int NBinsA, int NBinsB> class Spline2
{
public:
    typedef std::array<float, 2> Point2;
    typedef std::array<float, 3> Point3;

    typedef simdize<Point2> Point2V;
    typedef simdize<Point3> Point3V;

    Spline2(float minA, float maxA, float minB, float maxB);

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
    Spline2(const Spline2 &);
    /** assignment operator prohibited */
    Spline2 &operator=(const Spline2 &);

    /** spline 2-nd order, 3 points, da = a - point 1 */
    static float GetSpline2(float *v, float da);

    static constexpr int fNA = NBinsA < 4 ? 4 : NBinsA;
    static constexpr int fNB = NBinsB < 4 ? 4 : NBinsB;
    static constexpr int fN = fNA * fNB;                 // N points total
    const float fMinA;    // min A axis
    const float fMinB;    // min B axis
    const float fStepA;   // step between points A axis
    const float fStepB;   // step between points B axis
    const float fScaleA;  // scale A axis
    const float fScaleB;  // scale B axis
    Vc::vector<float, Vc::Allocator<float>> fXYZ;  // array of points, {X,Y,Z,0} values
};

template <int NBinsA, int NBinsB>
inline void Spline2<NBinsA, NBinsB>::Fill(int ind, float x, float y, float z)
{
    ind = ind / fNB + fNA * (ind % fNB);
    fXYZ[ind] = x;
    fXYZ[ind + fN] = y;
    fXYZ[ind + 2 * fN] = z;
}

template <int NBinsA, int NBinsB>
inline void Spline2<NBinsA, NBinsB>::Fill(int ind, const float XYZ[])
{
    Fill(ind, XYZ[0], XYZ[1], XYZ[2]);
}

template <int NBinsA, int NBinsB>
template <typename F> inline void Spline2<NBinsA, NBinsB>::Fill(F &&func)
{
    for (int i = 0; i < GetNPoints(); i++) {
        float a, b;
        std::tie(a, b) = GetAB(i);
        std::array<float, 3> xyz = func(a, b);
        Fill(i, xyz[0], xyz[1], xyz[2]);
    }
}

template <int NBinsA, int NBinsB>
inline std::pair<float, float> Spline2<NBinsA, NBinsB>::GetAB(int ind) const
{
    return std::make_pair(fMinA + (ind / fNA) * fStepA, fMinB + (ind % fNB) * fStepB);
}

template <int NBinsA, int NBinsB>
inline int Spline2<NBinsA, NBinsB>::GetMapSize() const { return 4 * sizeof(float) * fN; }

template <int NBinsA, int NBinsB>
inline int Spline2<NBinsA, NBinsB>::GetNPoints() const { return fN; }

template <int NBinsA, int NBinsB>
inline float Spline2<NBinsA, NBinsB>::GetSpline2(float *v, float x)
{
    return 0.5f * x * ((v[0] + v[2] - v[1] - v[1]) * x + v[2] - v[0]) + v[1];
}

// spline 3-st order,  4 points, da = a - point 1 {{{1
template <typename T> static inline T GetSpline3(T v0, T v1, T v2, T v3, T x)
{
    const T dv = v2 - v1;
    const T z0 = 0.5f * (v2 - v0);
    const T z1 = 0.5f * (v3 - v1);
    return (x * x) * ((z1 - dv) * (x - 1) + (z0 - dv) * (x - 2)) + (z0 * x + v1);
}

template <int NBinsA, int NBinsB>
inline std::array<float, 3> Spline2<NBinsA, NBinsB>::GetValue(std::array<float, 2> ab) const  //{{{1
{
    float lA = (ab[0] - fMinA) * fScaleA - 1.f;
    const int iA = std::max(0, std::min(fNA - 4, static_cast<int>(lA)));

    float lB = (ab[1] - fMinB) * fScaleB - 1.f;
    const int iB = std::max(0, std::min(fNB - 4, static_cast<int>(lB)));

    typedef Vc::simdarray<float, 4> float4;
    typedef Vc::simdarray<float, 12> float12;
    const float4 da = lA - iA;
    const float12 db = lB - iB;

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
    std::array<float, 3> XYZ;
    XYZ[0] = res[0];
    XYZ[1] = res[1];
    XYZ[2] = res[2];
    return XYZ;
}

template <int NBinsA, int NBinsB>
Spline2<NBinsA, NBinsB>::Spline2(float minA, float maxA, float minB, float maxB)  //{{{1
    : fMinA(minA),
      fMinB(minB),
      fStepA(((maxA <= minA ? minA + 1 : maxA) - minA) / (fNA - 1)),
      fStepB(((maxB <= minB ? minB + 1 : maxB) - minB) / (fNB - 1)),
      fScaleA(1.f / fStepA),
      fScaleB(1.f / fStepB),
      fXYZ(3 * fN, 0.f)
{
}
//}}}1

#endif  // SPLINE2_H_

// vim: foldmethod=marker
