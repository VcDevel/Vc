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

class AliTPCParam;
class AliTPCTransform;

class Spline
{
public:
    typedef std::array<float, 2> Point2;
    typedef std::array<float, 3> Point3;

    typedef simdize<Point2> Point2V;
    typedef simdize<Point3> Point3V;

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
    struct DataPoint
    {
        float x, y, z, padding__;
    };
    Vc::vector<DataPoint, Vc::Allocator<DataPoint>> fXYZ;  // array of points, {X,Y,Z,0} values
};

inline void Spline::Fill(int ind, float x, float y, float z)
{
    fXYZ[ind].x = x;
    fXYZ[ind].y = y;
    fXYZ[ind].z = z;
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

#endif  // SPLINE_H_
