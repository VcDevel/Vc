#ifndef SPLINE_H_
#define SPLINE_H_

//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#include "Rtypes.h"

class AliTPCParam;
class AliTPCTransform;

class AliHLTTPCSpline2D3D
{
public:
    AliHLTTPCSpline2D3D();
    AliHLTTPCSpline2D3D(float minA, float maxA, int nBinsA, float minB,
                        float maxB, int nBinsB);
    ~AliHLTTPCSpline2D3D();
    void Init(float minA, float maxA, int nBinsA, float minB, float maxB,
              int nBinsB);

    /**  Filling of points */
    void Fill(void (*func)(float a, float b, float xyz[]));
    /**  Filling of points */
    void Fill(int ind, float x, float y, float z);
    /**  Filling of points */
    void Fill(int ind, float XYZ[]);
    /**  Filling of points */
    void Fill(int ind, double XYZ[]);

    /**  Get A,B by the point index */
    void GetAB(int ind, float &A, float &B) const;

    /** Consolidate the map*/
    void Consolidate();

    /**  Get Interpolated value at A,B */
    void GetValue(float A, float B, float XYZ[]) const;

    /**  Get Interpolated value at A,B */
    void GetValue(float A, float B, double XYZ[]) const;

    /**  Get size of the grid */
    int GetMapSize() const;

    /**  Get N of point on the grid */
    int GetNPoints() const;

private:
    /** copy constructor prohibited */
    AliHLTTPCSpline2D3D(const AliHLTTPCSpline2D3D &);
    /** assignment operator prohibited */
    AliHLTTPCSpline2D3D &operator=(const AliHLTTPCSpline2D3D &);

    /** spline 3-st order,  4 points, da = a - point 1 */
    static float GetSpline3(float v0, float v1, float v2, float v3, float da);
    static float GetSpline3(float *v, float da);

    /** spline 2-nd order, 3 points, da = a - point 1 */
    static float GetSpline2(float *v, float da);

    int fNA;        // N points A axis
    int fNB;        // N points A axis
    int fN;         // N points total
    float fMinA;    // min A axis
    float fMinB;    // min B axis
    float fStepA;   // step between points A axis
    float fStepB;   // step between points B axis
    float fScaleA;  // scale A axis
    float fScaleB;  // scale B axis
    float *fXYZ;    // array of points, {X,Y,Z,0} values
};

inline AliHLTTPCSpline2D3D::AliHLTTPCSpline2D3D()
    : fNA(0)
    , fNB(0)
    , fN(0)
    , fMinA(0)
    , fMinB(0)
    , fStepA(0)
    , fStepB(0)
    , fScaleA(0)
    , fScaleB(0)
    , fXYZ(0)
{
}

inline AliHLTTPCSpline2D3D::AliHLTTPCSpline2D3D(float minA, float maxA, int nBinsA,
                                                float minB, float maxB, int nBinsB)
    : fNA(0)
    , fNB(0)
    , fN(0)
    , fMinA(0)
    , fMinB(0)
    , fStepA(0)
    , fStepB(0)
    , fScaleA(0)
    , fScaleB(0)
    , fXYZ(0)
{
    Init(minA, maxA, nBinsA, minB, maxB, nBinsB);
}

inline void AliHLTTPCSpline2D3D::Fill(int ind, float x, float y, float z)
{
    int ind4 = ind * 4;
    fXYZ[ind4] = x;
    fXYZ[ind4 + 1] = y;
    fXYZ[ind4 + 2] = z;
}

inline void AliHLTTPCSpline2D3D::Fill(int ind, float XYZ[])
{
    Fill(ind, XYZ[0], XYZ[1], XYZ[2]);
}

inline void AliHLTTPCSpline2D3D::Fill(int ind, double XYZ[])
{
    Fill(ind, XYZ[0], XYZ[1], XYZ[2]);
}

inline void AliHLTTPCSpline2D3D::Fill(void (*func)(float a, float b, float xyz[]))
{
    for (int i = 0; i < GetNPoints(); i++) {
        float a, b, xyz[3];
        GetAB(i, a, b);
        (*func)(a, b, xyz);
        Fill(i, xyz);
    }
}

inline void AliHLTTPCSpline2D3D::GetAB(int ind, float &A, float &B) const
{
    A = fMinA + (ind / fNB) * fStepA;
    B = fMinB + (ind % fNB) * fStepB;
}

inline int AliHLTTPCSpline2D3D::GetMapSize() const { return 4 * sizeof(float) * fN; }

inline int AliHLTTPCSpline2D3D::GetNPoints() const { return fN; }

inline float AliHLTTPCSpline2D3D::GetSpline3(float v0, float v1, float v2,
                                               float v3, float x)
{
    float dv = v2 - v1;
    float z0 = 0.5f * (v2 - v0);
    float z1 = 0.5f * (v3 - v1);
    return x * x * ((z1 - dv + z0 - dv) * (x - 1) - (z0 - dv)) + z0 * x + v1;
}

inline float AliHLTTPCSpline2D3D::GetSpline3(float *v, float x)
{
    return GetSpline3(v[0], v[1], v[2], v[3], x);
}

inline float AliHLTTPCSpline2D3D::GetSpline2(float *v, float x)
{
    return 0.5 * x * ((v[0] + v[2] - v[1] - v[1]) * x + v[2] - v[0]) + v[1];
}

inline void AliHLTTPCSpline2D3D::GetValue(float A, float B, double XYZ[]) const
{
    float fxyz[3];
    GetValue(A, B, fxyz);
    for (int i = 0; i < 3; i++)
        XYZ[i] = fxyz[i];
}

#endif  // SPLINE_H_
