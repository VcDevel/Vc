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

#ifndef SPLINE_H_
#define SPLINE_H_

#include <utility>
#include <array>
#include <tuple>
#include "../kdtree/simdize.h"
#include <Vc/vector>

//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

typedef std::array<float, 2> Point2;
typedef std::array<float, 3> Point3;

typedef simdize<Point2> Point2V;
typedef simdize<Point3> Point3V;

class Spline
{
public:
    Spline(float minA, float maxA, int nBinsA, float minB, float maxB, int nBinsB);

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
    Point3 GetValue16(Point2 ab) const;
    Point3 GetValueScalar(Point2) const;
    Point3V GetValue(const Point2V &) const;

    /**  Get size of the grid */
    int GetMapSize() const;

    /**  Get N of point on the grid */
    int GetNPoints() const;

private:
    /** copy constructor prohibited */
    Spline(const Spline &);
    /** assignment operator prohibited */
    Spline &operator=(const Spline &);

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
    typedef Vc::simdarray<float, 4> DataPoint;
    Vc::vector<DataPoint> fXYZ;  // array of points, {X,Y,Z,0} values
};

inline void Spline::Fill(int ind, float x, float y, float z)
{
    fXYZ[ind][0] = x;
    fXYZ[ind][1] = y;
    fXYZ[ind][2] = z;
}

inline void Spline::Fill(int ind, const float XYZ[])
{
    Fill(ind, XYZ[0], XYZ[1], XYZ[2]);
}

template <typename F> inline void Spline::Fill(F &&func)
{
    for (int i = 0; i < GetNPoints(); i++) {
        float a, b;
        std::tie(a, b) = GetAB(i);
        std::array<float, 3> xyz = func(a, b);
        Fill(i, xyz[0], xyz[1], xyz[2]);
    }
}

inline std::pair<float, float> Spline::GetAB(int ind) const
{
    return std::make_pair(fMinA + (ind / fNA) * fStepA, fMinB + (ind % fNB) * fStepB);
}

inline int Spline::GetMapSize() const { return 4 * sizeof(float) * fN; }

inline int Spline::GetNPoints() const { return fN; }

inline float Spline::GetSpline2(float *v, float x)
{
    return 0.5f * x * ((v[0] + v[2] - v[1] - v[1]) * x + v[2] - v[0]) + v[1];
}

inline std::tuple<int, int, float, float> evaluatePosition(Point2 ab, Point2 min,
                                                           Point2 scale, int na, int nb)
{
    const float lA = (ab[0] - min[0]) * scale[0] - 1.f;
    const int iA = std::min(na - 4.f, std::max(lA, 0.f));

    const float lB = (ab[1] - min[1]) * scale[1] - 1.f;
    const int iB = std::min(nb - 4.f, std::max(lB, 0.f));

    const float da = lA - iA;
    const float db = lB - iB;

    return std::make_tuple(iA, iB, da, db);
}

using Vc::float_v;
typedef float_v::IndexType index_v;
inline std::tuple<index_v, index_v, float_v, float_v> evaluatePosition(Point2V ab,
                                                                       Point2 min,
                                                                       Point2 scale,
                                                                       int na, int nb)
{
    const float_v lA = (ab[0] - min[0]) * scale[0] - 1.f;
    const auto iA = static_cast<index_v>(std::min(na - 4.f, std::max(lA, 0.f)));

    const float_v lB = (ab[1] - min[1]) * scale[1] - 1.f;
    const auto iB = static_cast<index_v>(std::min(nb - 4.f, std::max(lB, 0.f)));

    const float_v da = lA - Vc::simd_cast<float_v>(iA);
    const float_v db = lB - Vc::simd_cast<float_v>(iB);

    return std::make_tuple(iA, iB, da, db);
}

#endif  // SPLINE_H_
