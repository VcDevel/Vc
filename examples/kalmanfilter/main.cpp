#include "matrix.h"
#include "runtimemean.h"
#include "../tsc.h"

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <Vc/Vc>

#define MUTE

using std::cout;
using std::endl;

constexpr int NFits = 1;
constexpr int MaxNTracks = 20000;
constexpr int Ntimes = 1;
constexpr int MaxNStations = 10;

using Vc::float_v;
using Vc::float_m;

/// read one float from \p strm and broadcast it to \p a
inline std::istream &operator>>(std::istream &strm, float_v &a)
{
    float tmp;
    strm >> tmp;
    a = tmp;
    return strm;
}

inline float_v rcp(const float_v &a) { return 1.f / a; }

struct FieldVector : public Vc::VectorAlignedBase
{
    float_v X, Y, Z;
    void Combine(FieldVector &H, const float_v &w)
    {
        X += w * (H.X - X);
        Y += w * (H.Y - Y);
        Z += w * (H.Z - Z);
    }
};

struct FieldSlice : public Vc::VectorAlignedBase
{
    float_v X[21], Y[21], Z[21];  // polinom coeff.

    FieldSlice()
    {
        for (int i = 0; i < 21; i++) {
            X[i] = Y[i] = Z[i] = 0;
        }
    }

    void GetField(const float_v &x, const float_v &y, float_v &Hx, float_v &Hy, float_v &Hz)
    {
        float_v x2 = x * x;
        float_v y2 = y * y;
        float_v xy = x * y;
        float_v x3 = x2 * x;
        float_v y3 = y2 * y;
        float_v xy2 = x * y2;
        float_v x2y = x2 * y;

        float_v x4 = x3 * x;
        float_v y4 = y3 * y;
        float_v xy3 = x * y3;
        float_v x2y2 = x2 * y2;
        float_v x3y = x3 * y;

        float_v x5 = x4 * x;
        float_v y5 = y4 * y;
        float_v xy4 = x * y4;
        float_v x2y3 = x2 * y3;
        float_v x3y2 = x3 * y2;
        float_v x4y = x4 * y;

        Hx = X[0] + X[1] * x + X[2] * y + X[3] * x2 + X[4] * xy + X[5] * y2 + X[6] * x3 + X[7] * x2y + X[8] * xy2 + X[9] * y3
            + X[10] * x4 + X[11] * x3y + X[12] * x2y2 + X[13] * xy3 + X[14] * y4
            + X[15] * x5 + X[16] * x4y + X[17] * x3y2 + X[18] * x2y3 + X[19] * xy4 + X[20] * y5;

        Hy = Y[0] + Y[1] * x + Y[2] * y + Y[3] * x2 + Y[4] * xy + Y[5] * y2 + Y[6] * x3 + Y[7] * x2y + Y[8] * xy2 + Y[9] * y3
            + Y[10] * x4 + Y[11] * x3y + Y[12] * x2y2 + Y[13] * xy3 + Y[14] * y4
            + Y[15] * x5 + Y[16] * x4y + Y[17] * x3y2 + Y[18] * x2y3 + Y[19] * xy4 + Y[20] * y5;

        Hz = Z[0] + Z[1] * x + Z[2] * y + Z[3] * x2 + Z[4] * xy + Z[5] * y2 + Z[6] * x3 + Z[7] * x2y + Z[8] * xy2 + Z[9] * y3
            + Z[10] * x4 + Z[11] * x3y + Z[12] * x2y2 + Z[13] * xy3 + Z[14] * y4
            + Z[15] * x5 + Z[16] * x4y + Z[17] * x3y2 + Z[18] * x2y3 + Z[19] * xy4 + Z[20] * y5;
    }

    void GetField(const float_v &x, const float_v &y, FieldVector &H)
    {
        GetField(x, y, H.X, H.Y, H.Z);
    }
};

struct FieldRegion : public Vc::VectorAlignedBase
{
    float_v x0, x1, x2;  // Hx(Z) = x0 + x1 * (Z - z) + x2 * (Z - z)^2
    float_v y0, y1, y2;  // Hy(Z) = y0 + y1 * (Z - z) + y2 * (Z - z)^2
    float_v z0, z1, z2;  // Hz(Z) = z0 + z1 * (Z - z) + z2 * (Z - z)^2
    float_v z;

    friend std::ostream &operator<<(std::ostream &os, const FieldRegion &a)
    {
        return os << a.x0 << '\n'
                  << a.x1 << '\n'
                  << a.x2 << '\n'
                  << a.y0 << '\n'
                  << a.y1 << '\n'
                  << a.y2 << '\n'
                  << a.z0 << '\n'
                  << a.z1 << '\n'
                  << a.z2 << '\n'
                  << a.z;
    }

    FieldRegion() { x0 = x1 = x2 = y0 = y1 = y2 = z0 = z1 = z2 = z = 0.; }

    void Get(const float_v z_, float_v *B) const
    {
        float_v dz = (z_ - z);
        float_v dz2 = dz * dz;
        B[0] = x0 + x1 * dz + x2 * dz2;
        B[1] = y0 + y1 * dz + y2 * dz2;
        B[2] = z0 + z1 * dz + z2 * dz2;
    }

    void Set(const FieldVector &H0, const float_v &H0z,
             const FieldVector &H1, const float_v &H1z,
             const FieldVector &H2, const float_v &H2z)
    {
        z = H0z;
        float_v dz1 = H1z - H0z, dz2 = H2z - H0z;
        float_v det = rcp(dz1 * dz2 * (dz2 - dz1));
        float_v w21 = - dz2 * det;
        float_v w22 = dz1 * det;
        float_v w11 = - dz2 * w21;
        float_v w12 = - dz1 * w22;

        float_v dH1 = H1.X - H0.X;
        float_v dH2 = H2.X - H0.X;
        x0 = H0.X;
        x1 = dH1 * w11 + dH2 * w12 ;
        x2 = dH1 * w21 + dH2 * w22 ;

        dH1 = H1.Y - H0.Y;
        dH2 = H2.Y - H0.Y;
        y0 = H0.Y;
        y1 = dH1 * w11 + dH2 * w12 ;
        y2 = dH1 * w21 + dH2 * w22  ;

        dH1 = H1.Z - H0.Z;
        dH2 = H2.Z - H0.Z;
        z0 = H0.Z;
        z1 = dH1 * w11 + dH2 * w12 ;
        z2 = dH1 * w21 + dH2 * w22 ;
    }

    void Shift(const float_v &Z0)
    {
        float_v dz = Z0 - z;
        float_v x2dz = x2 * dz;
        float_v y2dz = y2 * dz;
        float_v z2dz = z2 * dz;
        z = Z0;
        x0 += (x1 + x2dz) * dz;
        x1 += x2dz + x2dz;
        y0 += (y1 + y2dz) * dz;
        y1 += y2dz + y2dz;
        z0 += (z1 + z2dz) * dz;
        z1 += z2dz + z2dz;
    }

};

struct HitInfo : public Vc::VectorAlignedBase, public MatrixOperand<float_v, 1, 2, HitInfo>
{  // strip info
    float_v cos_phi, sin_phi, sigma2, sigma216;

    const float_v operator()(size_t, size_t c) const
    {
        switch (c) {
        case 0:  return cos_phi;
        default: return sin_phi;
        }
    }
};

struct HitXYInfo : public Vc::VectorAlignedBase
{
    float_v C00, C10, C11;
};

struct Station : public Vc::VectorAlignedBase
{
    float_v z, thick, zhit, RL, RadThick, logRadThick,
            SyF, SyL;  //  field intergals with respect to First(last) station

    HitInfo UInfo, VInfo;  // front and back
    HitXYInfo XYInfo;

    FieldSlice Map;

    using ArrayPtr = std::unique_ptr<Station[]>;
};

struct Hit : public Vc::VectorAlignedBase
{
    float_v::EntryType x, y;
    float_v::EntryType tmp1;
    int ista;
};

struct MCPoint : public Vc::VectorAlignedBase
{
    float_v::EntryType x, y, z;
    float_v::EntryType px, py, pz;
    int ista;
};

struct MCTrack : public Vc::VectorAlignedBase
{
    float_v::EntryType MC_x, MC_y, MC_z, MC_px, MC_py, MC_pz, MC_q;
    MCPoint vPoints[MaxNStations * 2];
    int NMCPoints;
};

struct Track : public Vc::VectorAlignedBase
{
    float_v::EntryType T[6];   // x, y, tx, ty, qp, z
    float_v::EntryType C[15];  // cov matr.
    float_v::EntryType Chi2;
    Hit vHits[MaxNStations];
    int NHits;
    int NDF;

    float_v::EntryType & x() { return T[0]; }
    float_v::EntryType & y() { return T[1]; }
    float_v::EntryType &tx() { return T[2]; }
    float_v::EntryType &ty() { return T[3]; }
    float_v::EntryType &qp() { return T[4]; }
    float_v::EntryType & z() { return T[5]; }
};

struct HitV : public Vc::VectorAlignedBase
{
    float_v x, y, w;
    FieldVector H;
};

struct CovV : public Vc::VectorAlignedBase, public MatrixOperand<float_v, 5, 5, CovV>
{
    float_v C00,
            C10, C11,
            C20, C21, C22,
            C30, C31, C32, C33,
            C40, C41, C42, C43, C44;

    inline const float_v operator()(size_t r, size_t c) const
    {
        switch (r) {
        case 0:
            switch (c) {
            case 4: return C40;
            case 3: return C30;
            case 2: return C20;
            default: return operator[](c);
            }
        case 1:
            switch (c) {
            case 4: return C41;
            case 3: return C31;
            case 2: return C21;
            default: return operator[](1 + c);
            }
        case 2:
            switch (c) {
            case 4: return C42;
            case 3: return C32;
            default: return operator[](3 + c);
            }
        case 3:
            switch (c) {
            case 4: return C43;
            default: return operator[](6 + c);
            }
        }
        return operator[](10 + c);
    }

    template <typename RhsImpl>
    inline CovV &operator-=(const MatrixOperand<float_v, 5, 5, RhsImpl> &rhs)
    {
        C00 -= rhs(0, 0);
        C10 -= rhs(1, 0);
        C11 -= rhs(1, 1);
        C20 -= rhs(2, 0);
        C21 -= rhs(2, 1);
        C22 -= rhs(2, 2);
        C30 -= rhs(3, 0);
        C31 -= rhs(3, 1);
        C32 -= rhs(3, 2);
        C33 -= rhs(3, 3);
        C40 -= rhs(4, 0);
        C41 -= rhs(4, 1);
        C42 -= rhs(4, 2);
        C43 -= rhs(4, 3);
        C44 -= rhs(4, 4);
        return *this;
    }
    const float_v &operator[](int i) const
    {
        const float_v *p = &C00;
        return p[i];
    }

    friend std::ostream &operator<<(std::ostream &os, const CovV &a)
    {
        return os << a.C00 << '\n'
                  << a.C10 << '\n'
                  << a.C11 << '\n'
                  << a.C20 << '\n'
                  << a.C21 << '\n'
                  << a.C22 << '\n'
                  << a.C30 << '\n'
                  << a.C31 << '\n'
                  << a.C32 << '\n'
                  << a.C33 << '\n'
                  << a.C40 << '\n'
                  << a.C41 << '\n'
                  << a.C42 << '\n'
                  << a.C43 << '\n'
                  << a.C44;
    }

    CovV()
      : C00(Vc::Zero),
        C10(Vc::Zero), C11(Vc::Zero),
        C20(Vc::Zero), C21(Vc::Zero), C22(Vc::Zero),
        C30(Vc::Zero), C31(Vc::Zero), C32(Vc::Zero), C33(Vc::Zero),
        C40(Vc::Zero), C41(Vc::Zero), C42(Vc::Zero), C43(Vc::Zero), C44(Vc::Zero)
    {}
};

typedef CovV CovVConventional;

struct TrackV : public Vc::VectorAlignedBase, public MatrixOperand<float_v, 6, 1, TrackV>
{
    HitV vHits[MaxNStations];

    float_v T[6];  // x, y, tx, ty, qp, z
    CovV C;        // cov matr.

    float_v Chi2;
    float_v NDF;

    FieldRegion f;  // field at first hit (needed for extrapolation to MC and check of results)

    float_v &x() { return T[0]; }
    float_v &y() { return T[1]; }
    float_v &tx() { return T[2]; }
    float_v &ty() { return T[3]; }
    float_v &qp() { return T[4]; }
    float_v &z() { return T[5]; }

    TrackV() : Chi2(Vc::Zero), NDF(Vc::Zero)
    {
        T[0] = float_v::Zero();
        T[1] = float_v::Zero();
        T[2] = float_v::Zero();
        T[3] = float_v::Zero();
        T[4] = float_v::Zero();
        T[5] = float_v::Zero();
    }

    template <typename RhsImpl>
    inline TrackV &operator-=(const MatrixOperand<float_v, 5, 1, RhsImpl> &rhs)
    {
        T[0] -= rhs(0);
        T[1] -= rhs(1);
        T[2] -= rhs(2);
        T[3] -= rhs(3);
        T[4] -= rhs(4);
        return *this;
    }

    template <typename RhsImpl>
    inline TrackV &operator-=(const MatrixOperand<float_v, 6, 1, RhsImpl> &rhs)
    {
        T[0] -= rhs(0);
        T[1] -= rhs(1);
        T[2] -= rhs(2);
        T[3] -= rhs(3);
        T[4] -= rhs(4);
        T[5] -= rhs(5);
        return *this;
    }

    inline const float_v operator()(size_t r, size_t = 0) const { return T[r]; }
};

//constants
#define cnst static const float_v

static const float_v INF = .01f;
static const float_v INF2 = .0001f;
static const float_v c_light = 0.000299792458f;
static const float_v c_light_i = 1.f / c_light;
static const float_v PipeRadThick = 0.0009f;

class Jacobian_t{ // jacobian elements // j[0][0] - j[3][2] are j02 - j34
    public:
        float_v &operator()(int i, int j) { assert(i >= 0 && j >= 2); return fj[i][j - 2]; };

    private:
        //     1 0 ? ? ?
        //     0 1 ? ? ?
        // j = 0 0 ? ? ?
        //     0 0 ? ? ?
        //     0 0 0 0 1
        float_v fj[4][3];
};

class FitFunctional
{  // base class for all approaches
public:
    void Fit(TrackV &t, const Station::ArrayPtr &vStations, int NStations) const;

    /// extrapolates track parameters
    void ExtrapolateALight(float_v T[],
                           CovV &C,
                           const float_v &z_out,
                           float_v &qp0,
                           FieldRegion &F,
                           float_v w = float_v::Zero()) const;

protected:
    void Filter(TrackV &track,
                const HitInfo &info,
                const float_v u,
                const float_v w = float_v::One()) const;
    void FilterFirst(TrackV &track, HitV &hit, Station &st) const;

    void ExtrapolateWithMaterial(TrackV &track,
                                 const float_v &z_out,
                                 float_v &qp0,
                                 FieldRegion &F,
                                 Station &st,
                                 bool isPipe = false,
                                 float_v w = float_v::Zero()) const;

    void AddMaterial(CovV &C, float_v Q22, float_v Q32, float_v Q33) const;
    /// initial aproximation
    void GuessVec(TrackV &t, const Station::ArrayPtr &vStations, int NStations, bool dir = false) const;

    void AddMaterial(TrackV &track, Station &st, const float_v qp0, bool isPipe = false) const;

    /// extrapolates track parameters and returns jacobian for extrapolation of CovMatrix
    void ExtrapolateJ(float_v *T,     // input track parameters (x,y,tx,ty,Q / p) and cov.matrix
                      float_v z_out,  // extrapolate to this z position
                      float_v qp0,    // use Q / p linearisation at this value
                      const FieldRegion &F,
                      Jacobian_t &j,
                      float_v w = float_v::Zero()) const;

    /// calculate covMatrix for Multiple Scattering
    void GetMSMatrix(const float_v &tx,
                     const float_v &ty,
                     const float_v &radThick,
                     const float_v &logRadThick,
                     float_v qp0,
                     float_v &Q22,
                     float_v &Q32,
                     float_v &Q33) const;

private:
    /// extrapolates track parameters and returns jacobian for extrapolation of CovMatrix
    void ExtrapolateJAnalytic(float_v T[],    // input track parameters (x,y,tx,ty,Q / p)
                              float_v z_out,  // extrapolate to this z position
                              float_v qp0,    // use Q / p linearisation at this value
                              const FieldRegion &F,
                              Jacobian_t &j,
                              float_v w = float_v::Zero()) const;
};

inline void FitFunctional::GuessVec(TrackV &t,
                                    const Station::ArrayPtr &vStations,
                                    int nStations,
                                    bool dir) const
{
    float_v * T = t.T;

    int NHits = nStations;

    float_v A0, A1 = float_v::Zero(), A2 = float_v::Zero(), A3 = float_v::Zero(),
                A4 = float_v::Zero(), A5 = float_v::Zero(), a0, a1 = float_v::Zero(),
                a2 = float_v::Zero(), b0, b1 = float_v::Zero(), b2 = float_v::Zero();
    float_v z0, x, y, z, S, w, wz, wS;

    int i = NHits - 1;
    if (dir) i = 0;
    z0 = vStations[i].zhit;
    HitV * hlst = &(t.vHits[i]);
    w = hlst->w;
    A0 = w;
    a0 = w * hlst->x;
    b0 = w * hlst->y;
    HitV * h = t.vHits;
    const Station *stationIt = &vStations[0];
    int st_add = 1;
    if (dir)
    {
        st_add = - 1;
        h += NHits - 1;
        stationIt += NHits - 1;
    }

    for (; h!= hlst; h += st_add, stationIt += st_add) {
        x = h->x;
        y = h->y;
        w = h->w;
        z = stationIt->zhit - z0;
        if (!dir) S = stationIt->SyL;
        else S = stationIt->SyF;
        wz = w * z;
        wS = w * S;
        A0 += w;
        A1 += wz;  A2 += wz * z;
        A3 += wS;  A4 += wS * z; A5 += wS * S;
        a0 += w * x; a1 += wz * x; a2 += wS * x;
        b0 += w * y; b1 += wz * y; b2 += wS * y;
    }

    float_v A3A3 = A3 * A3;
    float_v A3A4 = A3 * A4;
    float_v A1A5 = A1 * A5;
    float_v A2A5 = A2 * A5;
    float_v A4A4 = A4 * A4;

    float_v det = rcp(- A2 * A3A3 + A1 * (A3A4 + A3A4 - A1A5) + A0 * (A2A5 - A4A4));
    float_v Ai0 = (- A4A4 + A2A5);
    float_v Ai1 = (A3A4 - A1A5);
    float_v Ai2 = (- A3A3 + A0 * A5);
    float_v Ai3 = (- A2 * A3 + A1 * A4);
    float_v Ai4 = (A1 * A3 - A0 * A4);
    float_v Ai5 = (- A1 * A1 + A0 * A2);

    float_v L, L1;
    T[0] = (Ai0 * a0 + Ai1 * a1 + Ai3 * a2) * det;
    T[2] = (Ai1 * a0 + Ai2 * a1 + Ai4 * a2) * det;
    float_v txtx1 = float_v::One() +  T[2] * T[2];
    L    = (Ai3 * a0 + Ai4 * a1 + Ai5 * a2) * det * rcp(txtx1);
    L1 = L * T[2];
    A1 = A1 + A3 * L1;
    A2 = A2 + (A4 + A4 + A5 * L1) * L1;
    b1 += b2 * L1;
    det = rcp(- A1 * A1 + A0 * A2);

    T[1] = (A2 * b0 - A1 * b1) * det;
    T[3] = (- A1 * b0 + A0 * b1) * det;
    T[4] = - L * c_light_i / sqrt(txtx1 + T[3] * T[3]);
    //T[4] = - L * c_light_i * Vc::rsqrt(txtx1 + T[3] * T[3]);
    T[5] = z0;
}

inline void FitFunctional::ExtrapolateJAnalytic(  // extrapolates track parameters and returns
                                                  // jacobian for extrapolation of CovMatrix
    float_v T[],                                  // input track parameters (x,y,tx,ty,Q / p)
    float_v z_out,                                // extrapolate to this z position
    float_v qp0,                                  // use Q / p linearisation at this value
    const FieldRegion &F,
    Jacobian_t &j,
    float_v) const
{
    // cout << "Extrapolation..." << endl;
    //
    //  Part of the analytic extrapolation formula with error (c_light * B * dz)^4 / 4!
    //

    cnst c1 = 1.f;
    cnst c2 = 2.f;
    cnst c3 = 3.f;
    cnst c4 = 4.f;
    cnst c6 = 6.f;
    cnst c9 = 9.f;
    cnst c15 = 15.f;
    cnst c18 = 18.f;
    cnst c45 = 45.f;
    cnst c2i = .5f;
    cnst c3i = .3333333432674407958984375f;
    cnst c6i = .16666667163372039794921875f;
    cnst c12i = .083333335816860198974609375f;

    const float_v qp = T[4];
    const float_v dz = (z_out - T[5]);
    const float_v dz2 = dz * dz;
    const float_v dz3 = dz2 * dz;

    // construct coefficients

    const float_v x   = T[2];
    const float_v y   = T[3];
    const float_v xx  = x * x;
    const float_v xy = x * y;
    const float_v yy = y * y;
    const float_v y2 = y * c2;
    const float_v x2 = x * c2;
    const float_v x4 = x * c4;
    const float_v xx31 = xx * c3 + c1;
    const float_v xx159 = xx * c15 + c9;

    const float_v Ay = - xx - c1;
    const float_v Ayy = x * (xx * c3 + c3);
    const float_v Ayz = - c2 * xy;
    const float_v Ayyy = - (c15 * xx * xx + c18 * xx + c3);

    const float_v Ayy_x = c3 * xx31;
    const float_v Ayyy_x = - x4 * xx159;

    const float_v Bx = yy + c1;
    const float_v Byy = y * xx31;
    const float_v Byz = c2 * xx + c1;
    const float_v Byyy = - xy * xx159;

    const float_v Byy_x = c6 * xy;
    const float_v Byyy_x = - y * (c45 * xx + c9);
    const float_v Byyy_y = - x * xx159;

    // end of coefficients calculation

    const float_v t2   = c1 + xx + yy;
    const float_v t    = sqrt(t2);
    const float_v h    = qp0 * c_light;
    const float_v ht   = h * t;

    // get field integrals
    const float_v ddz = T[5] - F.z;
    const float_v Fx0 = F.x0 + F.x1 * ddz + F.x2 * ddz * ddz;
    const float_v Fx1 = (F.x1 + c2 * F.x2 * ddz) * dz;
    const float_v Fx2 = F.x2 * dz2;
    const float_v Fy0 = F.y0 + F.y1 * ddz + F.y2 * ddz * ddz;
    const float_v Fy1 = (F.y1 + c2 * F.y2 * ddz) * dz;
    const float_v Fy2 = F.y2 * dz2;
    const float_v Fz0 = F.z0 + F.z1 * ddz + F.z2 * ddz * ddz;
    const float_v Fz1 = (F.z1 + c2 * F.z2 * ddz) * dz;
    const float_v Fz2 = F.z2 * dz2;

    //

    const float_v sx = (Fx0 + Fx1 * c2i + Fx2 * c3i );
    const float_v sy = (Fy0 + Fy1 * c2i + Fy2 * c3i);
    const float_v sz = (Fz0 + Fz1 * c2i + Fz2 * c3i);

    const float_v Sx = (Fx0 * c2i + Fx1 * c6i + Fx2 * c12i);
    const float_v Sy = (Fy0 * c2i + Fy1 * c6i + Fy2 * c12i);
    const float_v Sz = (Fz0 * c2i + Fz1 * c6i + Fz2 * c12i);

    float_v syz;
    {
        cnst c00 = .5f;
        cnst c01 = .16666667163372039794921875f;
        cnst c02 = .083333335816860198974609375f;
        cnst c10 = .3333333432674407958984375f;
        cnst c11 = .125f;
        cnst c12 = .066666670143604278564453125f;
        cnst c20 = .25f;
        cnst c21 = .100000001490116119384765625f;
        cnst c22 = .0555555559694766998291015625f;
        syz = Fy0 * (c00 * Fz0 + c01 * Fz1 + c02 * Fz2)
            + Fy1 * (c10 * Fz0 + c11 * Fz1 + c12 * Fz2)
            + Fy2 * (c20 * Fz0 + c21 * Fz1 + c22 * Fz2);
    }

    float_v Syz;
    {
        cnst c00 = 21.f * 20.f / 2520.f;
        cnst c01 = 21.f *  5.f / 2520.f;
        cnst c02 = 21.f *  2.f / 2520.f;
        cnst c10 =  7.f * 30.f / 2520.f;
        cnst c11 =  7.f *  9.f / 2520.f;
        cnst c12 =  7.f *  4.f / 2520.f;
        cnst c20 =  2.f * 63.f / 2520.f;
        cnst c21 =  2.f * 21.f / 2520.f;
        cnst c22 =  2.f * 10.f / 2520.f;
        Syz = Fy0 * (c00 * Fz0 + c01 * Fz1 + c02 * Fz2)
            + Fy1 * (c10 * Fz0 + c11 * Fz1 + c12 * Fz2)
            + Fy2 * (c20 * Fz0 + c21 * Fz1 + c22 * Fz2);
    }

    const float_v syy  = sy * sy * c2i;
    const float_v syyy = syy * sy * c3i;

    float_v Syy;
    {
        cnst c00 = 420.f / 2520.f;
        cnst c01 =  21.f * 15.f / 2520.f;
        cnst c02 =  21.f * 8.f / 2520.f;
        cnst c03 =  63.f / 2520.f;
        cnst c04 =  70.f / 2520.f;
        cnst c05 =  20.f / 2520.f;
        Syy =  Fy0 * (c00 * Fy0 + c01 * Fy1 + c02 * Fy2) + Fy1 * (c03 * Fy1 + c04 * Fy2) + c05 * Fy2 * Fy2 ;
    }

    float_v Syyy;
    {
        cnst c000 =       7560.f / 181440.f;
        cnst c001 = 9.f * 1008.f / 181440.f;
        cnst c002 = 5.f * 1008.f / 181440.f;
        cnst c011 = 21.f * 180.f / 181440.f;
        cnst c012 = 24.f * 180.f / 181440.f;
        cnst c022 =  7.f * 180.f / 181440.f;
        cnst c111 =        540.f / 181440.f;
        cnst c112 =        945.f / 181440.f;
        cnst c122 =        560.f / 181440.f;
        cnst c222 =        112.f / 181440.f;
        const float_v Fy22 = Fy2 * Fy2;
        Syyy = Fy0 * (Fy0 * (c000 * Fy0 + c001 * Fy1 + c002 * Fy2) + Fy1 * (c011 * Fy1 + c012 * Fy2) + c022 * Fy22)
            +    Fy1 * (Fy1 * (c111 * Fy1 + c112 * Fy2) + c122 * Fy22) + c222 * Fy22 * Fy2                  ;
    }


    const float_v sA1   = sx * xy   + sy * Ay   + sz * y ;
    const float_v sA1_x = sx * y - sy * x2 ;
    const float_v sA1_y = sx * x + sz ;

    const float_v sB1   = sx * Bx   - sy * xy   - sz * x ;
    const float_v sB1_x = - sy * y - sz ;
    const float_v sB1_y = sx * y2 - sy * x ;

    const float_v SA1   = Sx * xy   + Sy * Ay   + Sz * y ;
    const float_v SA1_x = Sx * y - Sy * x2 ;
    const float_v SA1_y = Sx * x + Sz;

    const float_v SB1   = Sx * Bx   - Sy * xy   - Sz * x ;
    const float_v SB1_x = - Sy * y - Sz;
    const float_v SB1_y = Sx * y2 - Sy * x;


    const float_v sA2   = syy * Ayy   + syz * Ayz ;
    const float_v sA2_x = syy * Ayy_x - syz * y2 ;
    const float_v sA2_y = - syz * x2 ;
    const float_v sB2   = syy * Byy   + syz * Byz  ;
    const float_v sB2_x = syy * Byy_x + syz * x4 ;
    const float_v sB2_y = syy * xx31 ;

    const float_v SA2   = Syy * Ayy   + Syz * Ayz ;
    const float_v SA2_x = Syy * Ayy_x - Syz * y2 ;
    const float_v SA2_y = - Syz * x2 ;
    const float_v SB2   = Syy * Byy   + Syz * Byz ;
    const float_v SB2_x = Syy * Byy_x + Syz * x4 ;
    const float_v SB2_y = Syy * xx31 ;

    const float_v sA3   = syyy * Ayyy  ;
    const float_v sA3_x = syyy * Ayyy_x;
    const float_v sB3   = syyy * Byyy  ;
    const float_v sB3_x = syyy * Byyy_x;
    const float_v sB3_y = syyy * Byyy_y;


    const float_v SA3   = Syyy * Ayyy  ;
    const float_v SA3_x = Syyy * Ayyy_x;
    const float_v SB3   = Syyy * Byyy  ;
    const float_v SB3_x = Syyy * Byyy_x;
    const float_v SB3_y = Syyy * Byyy_y;

    const float_v ht1 = ht * dz;
    const float_v ht2 = ht * ht * dz2;
    const float_v ht3 = ht * ht * ht * dz3;
    const float_v ht1sA1 = ht1 * sA1;
    const float_v ht1sB1 = ht1 * sB1;
    const float_v ht1SA1 = ht1 * SA1;
    const float_v ht1SB1 = ht1 * SB1;
    const float_v ht2sA2 = ht2 * sA2;
    const float_v ht2SA2 = ht2 * SA2;
    const float_v ht2sB2 = ht2 * sB2;
    const float_v ht2SB2 = ht2 * SB2;
    const float_v ht3sA3 = ht3 * sA3;
    const float_v ht3sB3 = ht3 * sB3;
    const float_v ht3SA3 = ht3 * SA3;
    const float_v ht3SB3 = ht3 * SB3;

    T[0]  += ((x + ht1SA1 + ht2SA2 + ht3SA3) * dz);
    T[1]  += ((y + ht1SB1 + ht2SB2 + ht3SB3) * dz);
    T[2]  += (ht1sA1 + ht2sA2 + ht3sA3);
    T[3]  += (ht1sB1 + ht2sB2 + ht3sB3);
    T[5]  += (dz);

    const float_v ctdz  = c_light * t * dz;
    const float_v ctdz2 = c_light * t * dz2;

    const float_v dqp = qp - qp0;
    const float_v t2i = c1 * rcp(t2); // / t2;
    const float_v xt2i = x * t2i;
    const float_v yt2i = y * t2i;
    const float_v tmp0 = ht1SA1 + c2 * ht2SA2 + c3 * ht3SA3;
    const float_v tmp1 = ht1SB1 + c2 * ht2SB2 + c3 * ht3SB3;
    const float_v tmp2 = ht1sA1 + c2 * ht2sA2 + c3 * ht3sA3;
    const float_v tmp3 = ht1sB1 + c2 * ht2sB2 + c3 * ht3sB3;

    //     1 0 ? ? ?
    //     0 1 ? ? ?
    // j = 0 0 ? ? ?
    //     0 0 ? ? ?
    //     0 0 0 0 1

    j(0,2) = dz * (c1 + xt2i * tmp0 + ht1 * SA1_x + ht2 * SA2_x + ht3 * SA3_x);
    j(1,2) = dz * (xt2i * tmp1 + ht1 * SB1_x + ht2 * SB2_x + ht3 * SB3_x);
    j(2,2) =     c1 + xt2i * tmp2 + ht1 * sA1_x + ht2 * sA2_x + ht3 * sA3_x ;
    j(3,2) =          xt2i * tmp3 + ht1 * sB1_x + ht2 * sB2_x + ht3 * sB3_x ;

    j(0,3) = dz * (yt2i * tmp0 + ht1 * SA1_y + ht2 * SA2_y);
    j(1,3) = dz * (c1 + yt2i * tmp1 + ht1 * SB1_y + ht2 * SB2_y + ht3 * SB3_y);
    j(2,3) =          yt2i * tmp2 + ht1 * sA1_y + ht2 * sA2_y  ;
    j(3,3) =     c1 + yt2i * tmp3 + ht1 * sB1_y + ht2 * sB2_y + ht3 * sB3_y ;

    j(0,4) = ctdz2 * (SA1 + c2 * ht1 * SA2 + c3 * ht2 * SA3);
    j(1,4) = ctdz2 * (SB1 + c2 * ht1 * SB2 + c3 * ht2 * SB3);
    j(2,4) = ctdz * (sA1 + c2 * ht1 * sA2 + c3 * ht2 * sA3);
    j(3,4) = ctdz * (sB1 + c2 * ht1 * sB2 + c3 * ht2 * sB3);

    // extrapolate inverse momentum

    T[0] += (j(0,4) * dqp);
    T[1] += (j(1,4) * dqp);
    T[2] += (j(2,4) * dqp);
    T[3] += (j(3,4) * dqp);
}

inline void FitFunctional::ExtrapolateJ(  // extrapolates track parameters and returns jacobian for
                                          // extrapolation of CovMatrix
    float_v *T,                           // input track parameters (x,y,tx,ty,Q / p) and cov.matrix
    float_v z_out,                        // extrapolate to this z position
    float_v qp0,                          // use Q / p linearisation at this value
    const FieldRegion &F,
    Jacobian_t &j,
    float_v w) const
{
    ExtrapolateJAnalytic(T, z_out, qp0, F, j, w);
}

/// calculate covMatrix for Multiple Scattering
inline void FitFunctional::GetMSMatrix(const float_v &tx,
                                       const float_v &ty,
                                       const float_v &radThick,
                                       const float_v &logRadThick,
                                       float_v qp0,
                                       float_v &Q22,
                                       float_v &Q32,
                                       float_v &Q33) const
{
    cnst mass2 = 0.1396f * 0.1396f;

    float_v txtx = tx * tx;
    float_v tyty = ty * ty;
    float_v txtx1 = txtx + float_v::One();
    float_v h = txtx + tyty;
    float_v t = sqrt(txtx1 + tyty);
    float_v h2 = h * h;
    float_v qp0t = qp0 * t;

    cnst c1 = 0.0136f;
    cnst c2 = c1 * 0.038f;
    cnst c3 = c2 * 0.5f;
    cnst c4 = - c3 / 2.0f;
    cnst c5 = c3 / 3.0f;
    cnst c6 = - c3 / 4.0f;

    float_v s0 = (c1 + c2 * logRadThick + c3 * h + h2 * (c4 + c5 * h + c6 * h2)) * qp0t;

    float_v a = (float_v::One() + mass2 * qp0 * qp0t) * radThick * s0 * s0;


    Q22 = txtx1 * a;
    Q32 = tx * ty * a;
    Q33 = (float_v::One() + tyty) * a;
}

inline void FitFunctional::AddMaterial(TrackV &track, Station &st, const float_v qp0, bool isPipe)
    const
{
    float_v Q22, Q32, Q33;
    if (isPipe) {
        GetMSMatrix(track.T[2],
                    track.T[3],
                    st.RadThick + PipeRadThick,
                    log(st.RadThick + PipeRadThick),
                    qp0,
                    Q22,
                    Q32,
                    Q33);
    } else {
        GetMSMatrix(track.T[2], track.T[3], st.RadThick, st.logRadThick, qp0, Q22, Q32, Q33);
    }

    AddMaterial(track.C, Q22, Q32, Q33);
}

inline void FitFunctional::Fit(TrackV &t,
                               const std::unique_ptr<Station[]> &vStations,
                               int nStations) const
{
    // upstream

    GuessVec(t, vStations, nStations);

    // downstream

    FieldRegion f;
    float_v z0, z1, z2, dz;
    FieldVector H0, H1, H2;

    float_v qp0 = t.T[4];
    int i = nStations - 1;
    HitV *h = &t.vHits[i];

    FilterFirst(t, *h, vStations[i]);
    AddMaterial(t, vStations[i], qp0);

    z1 = vStations[i].z;
    vStations[i].Map.GetField(t.T[0], t.T[1], H1);
    H1.Combine(h->H, h->w);

    z2 = vStations[i - 2].z;
    dz = z2 - z1;
    vStations[i - 2].Map.GetField(t.T[0] + t.T[2] * dz, t.T[1] + t.T[3] * dz, H2);
    h = &t.vHits[i - 2];
    H2.Combine(h->H, h->w);

    const int iFirstStation = 0;
    for (--i; i >= iFirstStation; i--) {
        h = &t.vHits[i];
        Station &st = vStations[i];
        z0 = st.z;
        dz = (z1 - z0);
        st.Map.GetField(t.T[0] - t.T[2] * dz, t.T[1] - t.T[3] * dz, H0);
        H0.Combine(h->H, h->w);
        f.Set(H0, z0, H1, z1, H2, z2);
        if (i == iFirstStation) {
            t.f = f;
        }

        // ExtrapolateALight(t.T, t.C, st.zhit, qp0, f);
        // AddMaterial(t, st, qp0);
        if (i == 1) {                                               // 2nd MVD
            ExtrapolateWithMaterial(t, st.zhit, qp0, f, st, true);  // add pipe
        } else {
            ExtrapolateWithMaterial(t, st.zhit, qp0, f, st);
        }

        float_v u = h->x * st.UInfo.cos_phi + h->y * st.UInfo.sin_phi;
        float_v v = h->x * st.VInfo.cos_phi + h->y * st.VInfo.sin_phi;
        Filter(t, st.UInfo, u, h->w);
        Filter(t, st.VInfo, v, h->w);
        H2 = H1;
        z2 = z1;
        H1 = H0;
        z1 = z0;
    }
}

// inline // --> causes a runtime overhead and problems for the MS compiler (error C2603)
void FitFunctional::ExtrapolateALight(float_v T[],  // input track parameters (x,y,tx,ty,Q / p)
                                      CovV &C,      // input covariance matrix
                                      const float_v &z_out,  // extrapolate to this z position
                                      float_v &qp0,  // use Q / p linearisation at this value
                                      FieldRegion &F,
                                      float_v w) const
{
    //
    //  Part of the analytic extrapolation formula with error (c_light * B * dz)^4 / 4!
    //
    Jacobian_t j;
    ExtrapolateJ(T, z_out, qp0, F, j, w);

    //          covariance matrix transport

    const float_v c42 = C.C42, c43 = C.C43;

    const float_v cj00 = C.C00 + C.C20 * j(0,2) + C.C30 * j(0,3) + C.C40 * j(0,4);
    // const float_v cj10 = C.C10 + C.C21 * j(0,2) + C.C31 * j(0,3) + C.C41 * j(0,4);
    const float_v cj20 = C.C20 + C.C22 * j(0,2) + C.C32 * j(0,3) + c42 * j(0,4);
    const float_v cj30 = C.C30 + C.C32 * j(0,2) + C.C33 * j(0,3) + c43 * j(0,4);

    const float_v cj01 = C.C10 + C.C20 * j(1,2) + C.C30 * j(1,3) + C.C40 * j(1,4);
    const float_v cj11 = C.C11 + C.C21 * j(1,2) + C.C31 * j(1,3) + C.C41 * j(1,4);
    const float_v cj21 = C.C21 + C.C22 * j(1,2) + C.C32 * j(1,3) + c42 * j(1,4);
    const float_v cj31 = C.C31 + C.C32 * j(1,2) + C.C33 * j(1,3) + c43 * j(1,4);

    // const float_v cj02 = C.C20 * j(2,2) + C.C30 * j(2,3) + C.C40 * j(2,4);
    // const float_v cj12 = C.C21 * j(2,2) + C.C31 * j(2,3) + C.C41 * j(2,4);
    const float_v cj22 = C.C22 * j(2,2) + C.C32 * j(2,3) + c42 * j(2,4);
    const float_v cj32 = C.C32 * j(2,2) + C.C33 * j(2,3) + c43 * j(2,4);

    // const float_v cj03 = C.C20 * j(3,2) + C.C30 * j(3,3) + C.C40 * j(3,4);
    // const float_v cj13 = C.C21 * j(3,2) + C.C31 * j(3,3) + C.C41 * j(3,4);
    const float_v cj23 = C.C22 * j(3,2) + C.C32 * j(3,3) + c42 * j(3,4);
    const float_v cj33 = C.C32 * j(3,2) + C.C33 * j(3,3) + c43 * j(3,4);

    C.C40 += c42 * j(0,2) + c43 * j(0,3) + C.C44 * j(0,4); // cj40
    C.C41 += c42 * j(1,2) + c43 * j(1,3) + C.C44 * j(1,4); // cj41
    C.C42 = c42 * j(2,2) + c43 * j(2,3) + C.C44 * j(2,4);
    C.C43 = c42 * j(3,2) + c43 * j(3,3) + C.C44 * j(3,4);

    C.C00 = (cj00 + j(0,2) * cj20 + j(0,3) * cj30 + j(0,4) * C.C40);
    C.C10 = (cj01 + j(0,2) * cj21 + j(0,3) * cj31 + j(0,4) * C.C41);
    C.C11 = (cj11 + j(1,2) * cj21 + j(1,3) * cj31 + j(1,4) * C.C41);

    C.C20 = (j(2,2) * cj20 + j(2,3) * cj30 + j(2,4) * C.C40);
    C.C30 = (j(3,2) * cj20 + j(3,3) * cj30 + j(3,4) * C.C40);
    C.C21 = (j(2,2) * cj21 + j(2,3) * cj31 + j(2,4) * C.C41);
    C.C31 = (j(3,2) * cj21 + j(3,3) * cj31 + j(3,4) * C.C41);
    C.C22 = (j(2,2) * cj22 + j(2,3) * cj32 + j(2,4) * C.C42);
    C.C32 = (j(3,2) * cj22 + j(3,3) * cj32 + j(3,4) * C.C42);
    C.C33 = (j(3,2) * cj23 + j(3,3) * cj33 + j(3,4) * C.C43);
}

/**
 * \param measurementModel Information about the coordinate system of the measurement
 * \param u Is a measurement that we want to add - Strip coordinate (may be x or y)
 * \param w Weight. At this point either 1 or 0, simply masking invalid entries in the SIMD-vector.
 */
inline void FitFunctional::Filter(TrackV &track,
                                  const HitInfo &measurementModel,
                                  const float_v m,
                                  const float_v weight) const
{
    static RuntimeMean timer;
    timer.start();

    const float_v sigma2 = measurementModel.sigma2;
    const float_v sigma216 = measurementModel.sigma216;
    const Matrix<float_v, 5> F = track.C.slice<0, 5, 0, 2>() * measurementModel.transposed(); // CHᵀ
    const float_v hch = measurementModel * F.slice<0, 2>();                                  // HCHᵀ
    const float_v residual = measurementModel * track.slice<0, 2>() - m; // ζ = Hr - m

    const float_v denominator = Vc::iif (hch < sigma216, hch + sigma2, hch);
    const float_v zetawi = residual / denominator;             //           (float_v' + HCHᵀ)⁻¹ ζ
    track -= F * zetawi;                                       // r  -= CHᵀ (float_v' + HCHᵀ)⁻¹ ζ
    track.C -= F * (weight / (sigma2 + hch)) * F.transposed(); // C  -= CHᵀ (float_v  + HCHᵀ)⁻¹ HC
    track.Chi2(hch < sigma216) += residual * zetawi;           // χ² +=  ζ  (float_v' + HCHᵀ)⁻¹ ζ
    track.NDF += weight;

    timer.stop();
}

inline void FitFunctional::FilterFirst(TrackV &track, HitV &hit, Station &st) const
{

    CovV &C = track.C;
    float_v w1 = float_v::One() - hit.w;
    float_v c00 = hit.w * st.XYInfo.C00 + w1 * INF;
    float_v c10 = hit.w * st.XYInfo.C10 + w1 * INF;
    float_v c11 = hit.w * st.XYInfo.C11 + w1 * INF;

    // // initialize covariance matrix
    C.C00 = c00;
    C.C10 = c10;       C.C11 = c11;
    C.C20 = float_v::Zero();      C.C21 = float_v::Zero();      C.C22 = INF2; // needed for stability of smoother. improve FilterTracks and CHECKME
    C.C30 = float_v::Zero();      C.C31 = float_v::Zero();      C.C32 = float_v::Zero(); C.C33 = INF2;
    C.C40 = float_v::Zero();      C.C41 = float_v::Zero();      C.C42 = float_v::Zero(); C.C43 = float_v::Zero(); C.C44 = INF;

    track.T[0] = hit.w * hit.x + w1 * track.T[0];
    track.T[1] = hit.w * hit.y + w1 * track.T[1];
    track.NDF = - 3.0;
    track.Chi2 = float_v::Zero();
}

inline void FitFunctional::AddMaterial(CovV &C, float_v Q22, float_v Q32, float_v Q33) const
{
    C.C22 += Q22;
    C.C32 += Q32; C.C33 += Q33;
}

inline void FitFunctional::ExtrapolateWithMaterial(TrackV &track,
                                                   const float_v &z_out,
                                                   float_v &qp0,
                                                   FieldRegion &F,
                                                   Station &st,
                                                   bool isPipe,
                                                   float_v w) const
{
    ExtrapolateALight(track.T, track.C, z_out, qp0, F, w);
    FitFunctional::AddMaterial(track, st, qp0, isPipe);  // FIXME
}

class KalmanFilter : public Vc::VectorAlignedBase
{
    FieldRegion field0;
    FitFunctional fitter;
    Track vTracks[MaxNTracks];
    MCTrack vMCTracks[MaxNTracks];
    std::unique_ptr<Station[]> vStations;
    int NStations;
    int NTracks;
    int NTracksV;

public:
    KalmanFilter() : vStations{new Station[MaxNStations]}, NStations(0), NTracks(0), NTracksV(0) {}

    void readInput()
    {
        std::fstream FileGeo, FileTracks, FileMCTracks;

        FileGeo.open("geo.dat", std::ios::in);
        FileTracks.open("tracks.dat", std::ios::in);
        FileMCTracks.open("mctracksin.dat", std::ios::in);
        {
            FieldVector H[3];
            float_v Hz[3];
            for (int i = 0; i < 3; i++) {
                float_v::EntryType Bx, By, Bz, z;
                FileGeo >> z >> Bx >> By >> Bz;
                Hz[i] = z;
                H[i].X = Bx;
                H[i].Y = By;
                H[i].Z = Bz;
#ifndef MUTE
                cout << "Input Magnetic field:" << z << " " << Bx << " " << By << " " << Bz << endl;
#endif
            }
            field0.Set(H[0], Hz[0], H[1], Hz[1], H[2], Hz[2]);
        }
        FileGeo >> NStations;
#ifndef MUTE
        cout << "Input " << NStations << " Stations:" << endl;
#endif

        for (int i = 0; i < NStations; i++) {
            int ist;
            FileGeo >> ist;
            if (ist != i) {
                break;
            }
            Station &st = vStations[i];

            FileGeo >> st.z >> st.thick >> st.RL >> st.UInfo.sigma2 >> st.VInfo.sigma2;
            st.UInfo.sigma2 *= st.UInfo.sigma2;
            st.VInfo.sigma2 *= st.VInfo.sigma2;
            st.UInfo.sigma216 = st.UInfo.sigma2 * 16.f;
            st.VInfo.sigma216 = st.VInfo.sigma2 * 16.f;

            if (i < 2) {  // mvd // TODO From Geo File!!!
                st.UInfo.cos_phi = float_v::One();
                st.UInfo.sin_phi = float_v::Zero();
                st.VInfo.cos_phi = float_v::Zero();
                st.VInfo.sin_phi = float_v::One();
            } else {
                st.UInfo.cos_phi = float_v::One();           // 0 degree
                st.UInfo.sin_phi = float_v::Zero();
                st.VInfo.cos_phi = 0.9659258244f; // 15 degree
                st.VInfo.sin_phi = 0.2588190521f;
            }

            float_v idet =
                st.UInfo.cos_phi * st.VInfo.sin_phi - st.UInfo.sin_phi * st.VInfo.cos_phi;
            idet = float_v::One() / (idet * idet);
            st.XYInfo.C00 = (st.VInfo.sin_phi * st.VInfo.sin_phi * st.UInfo.sigma2 +
                    st.UInfo.sin_phi * st.UInfo.sin_phi * st.VInfo.sigma2) * idet;
            st.XYInfo.C10 = - (st.VInfo.sin_phi * st.VInfo.cos_phi * st.UInfo.sigma2 +
                    st.UInfo.sin_phi * st.UInfo.cos_phi * st.VInfo.sigma2) * idet;
            st.XYInfo.C11 = (st.VInfo.cos_phi * st.VInfo.cos_phi * st.UInfo.sigma2 +
                    st.UInfo.cos_phi * st.UInfo.cos_phi * st.VInfo.sigma2) * idet;

#ifndef MUTE
            cout << "    " << st.z[0] << " " << st.thick[0] << " " << st.RL[0] << ", ";
#endif
            st.zhit = st.z;
            st.RadThick = st.thick / st.RL;
            st.logRadThick = log(st.RadThick);

            int N = 0;
            FileGeo >> N;
#ifndef MUTE
            cout << N << " field coeff." << endl;
#endif
            for (int j = 0; j < N; j++) FileGeo >> st.Map.X[j];
            for (int j = 0; j < N; j++) FileGeo >> st.Map.Y[j];
            for (int j = 0; j < N; j++) FileGeo >> st.Map.Z[j];
        }
        {
            // field intergals with respect to Last station
            float_v z0 = vStations[NStations - 1].z;
            float_v sy = 0.f, Sy = 0.f;
            for (int i = NStations - 1; i >= 0; i--) {
                Station &st = vStations[i];
                float_v dz = st.z - z0;
                float_v Hy = vStations[i].Map.Y[0];
                Sy += dz * sy + dz * dz * Hy * 0.5f;
                sy += dz * Hy;
                st.SyL = Sy;
                z0 = st.z;
            }
            // field intergals with respect to First station
            z0 = vStations[0].z;
            sy = 0.f, Sy = 0.f;
            for (int i = 0; i < NStations; i++) {
                Station &st = vStations[i];
                float_v dz = st.z - z0;
                float_v Hy = vStations[i].Map.Y[0];
                Sy += dz * sy + dz * dz * Hy * 0.5f;
                sy += dz * Hy;
                st.SyF = Sy;
                z0 = st.z;
            }
        }

        FileGeo.close();

        NTracks = 0;
        int TrackIndex[MaxNTracks];
        while (!FileTracks.eof()) {
            int itr;
            FileTracks >> itr;
            // if (itr!= NTracks) break;
            if (NTracks >= MaxNTracks)
                break;

            Track &t = vTracks[NTracks];
            MCTrack &mc = vMCTracks[NTracks];
            FileTracks >> mc.MC_x >> mc.MC_y >> mc.MC_z >> mc.MC_px >> mc.MC_py >> mc.MC_pz >>
                mc.MC_q >> t.NHits;
            for (int i = 0; i < t.NHits; i++) {
                int ist;
                FileTracks >> ist;
                t.vHits[i].ista = ist;
                FileTracks >> t.vHits[i].x >> t.vHits[i].y;
            }
            TrackIndex[NTracks] = itr;
            if (t.NHits == NStations)
                NTracks++;
        }
        int NMCTracks = 0;
        int iPoint = 0;
        while (!FileMCTracks.eof()) {
            int itr;
            FileMCTracks >> itr;
            // if (itr!= NTracks) break;
            if (NMCTracks >= MaxNTracks)
                break;
            MCTrack &mc = vMCTracks[NMCTracks];
            float_v::EntryType temp;
            int NMCPoints;
            FileMCTracks >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> NMCPoints;
            mc.NMCPoints = NMCPoints;
            for (int i = 0; i < NMCPoints; i++) {
                int ist;
                FileMCTracks >> ist;
                mc.vPoints[i].ista = ist;
                FileMCTracks >> mc.vPoints[i].x >> mc.vPoints[i].y >> mc.vPoints[i].z >>
                    mc.vPoints[i].px >> mc.vPoints[i].py >> mc.vPoints[i].pz;
            }

            iPoint = 0;  // compare paraments at the first station
            // iPoint = NMCPoints - 1;
            mc.MC_x = mc.vPoints[iPoint].x;
            mc.MC_y = mc.vPoints[iPoint].y;
            mc.MC_z = mc.vPoints[iPoint].z;
            mc.MC_px = mc.vPoints[iPoint].px;
            mc.MC_py = mc.vPoints[iPoint].py;
            mc.MC_pz = mc.vPoints[iPoint].pz;

            if (itr == TrackIndex[NMCTracks])
                NMCTracks++;
        }
        // cout << NTracks << " " << NMCTracks << " reco and Mc tracks have been read" << endl;
        FileTracks.close();
        FileMCTracks.close();

        NTracksV = NTracks / float_v::Size;
        NTracks = NTracksV * float_v::Size;
    }

#define _STRINGIFY(_x) #_x
#define STRINGIFY(_x) _STRINGIFY(_x)

    void writeOutput()
    {
        std::fstream Out, Diff;
        Out.open(STRINGIFY(VC_IMPL) "_fit.dat", std::ios::out);
        Out << "Fitter" << endl;
        Out << MaxNTracks << endl;

        for (int it = 0, itt = 0; itt < NTracks; itt++) {
            Track &t = vTracks[itt];
            MCTrack &mc = vMCTracks[itt];

            bool ok = 1;
            for (int i = 0; i < 6; i++) {
                ok = ok && finite(t.T[i]);
            }
            for (int i = 0; i < 15; i++) {
                ok = ok && finite(t.C[i]);
            }
            if (!ok) {
                cout << " infinite " << endl;
            }

            const int iPoint = 0;
            Out << it << '\n'
                << std::setw(15) << mc.vPoints[iPoint].x
                << std::setw(15) << mc.vPoints[iPoint].y
                << std::setw(15) << mc.vPoints[iPoint].z
                << std::setw(15) << mc.vPoints[iPoint].px
                << std::setw(15) << mc.vPoints[iPoint].py
                << std::setw(15) << mc.vPoints[iPoint].pz
                << '\n'
                << std::setw(15) << t.x()
                << std::setw(15) << t.y()
                << std::setw(15) << t.z()
                << std::setw(15) << t.tx()
                << std::setw(15) << t.ty()
                << std::setw(15) << t.qp()
                << '\n';

            for (int i = 0; i < 15; i++) {
                Out << std::setw(13) << t.C[i];
            }
            Out << endl;

            it++;
        }
        Out.close();
    }

    void fitTracks()
    {
        std::unique_ptr<TrackV[]> TracksV{new TrackV[MaxNTracks / float_v::Size + 1]};
        float_v *Z0 =
            new float_v[MaxNTracks / float_v::Size + 1];  // mc - z, used for result comparison
        float_v *Z0s[MaxNStations];
        for (int is = 0; is < NStations; ++is) {
            Z0s[is] = new float_v[MaxNTracks / float_v::Size + 1];
        }

        float_v::Memory Z0mem;
        float_v::Memory Z0smem[MaxNStations];
#ifndef MUTE
        cout << "Prepare data..." << endl;
#endif
        TimeStampCounter timer1;
        timer1.start();

        for (int iV = 0; iV < NTracksV; iV++) {  // loop on set of 4 tracks
#ifndef MUTE
            if (iV * float_v::Size % 100 == 0) {
                cout << iV *float_v::Size << endl;
            }
#endif
            TrackV &t = TracksV[iV];
            for (int ist = 0; ist < NStations; ist++) {
                HitV &h = t.vHits[ist];

                h.x = 0.;
                h.y = 0.;
                h.w = 0.;
                h.H.X = 0.;
                h.H.Y = 0.;
                h.H.Z = 0.;
            }

            for (std::size_t it = 0; it < float_v::Size; it++) {
                Track &ts = vTracks[iV * float_v::Size + it];

                Z0mem[it] = vMCTracks[iV * float_v::Size + it].MC_z;
                for (int is = 0; is < NStations; ++is) {
                    Z0smem[is][it] = vMCTracks[iV * float_v::Size + it].vPoints[is].z;
                }

                for (int ista = 0, ih = 0; ista < NStations; ista++) {
                    Hit &hs = ts.vHits[ih];
                    if (hs.ista != ista) {
                        continue;
                    }
                    ih++;

                    t.vHits[ista].x[it] = hs.x;
                    t.vHits[ista].y[it] = hs.y;
                    t.vHits[ista].w[it] = 1.;
                }

            }

            float_v Z0temp(Z0mem);
            Z0[iV] = Z0temp;

            for (int is = 0; is < NStations; ++is) {
                float_v Z0stemp(Z0smem[is]);
                Z0s[is][iV] = Z0stemp;
            }

            if (0) {  // output for check
                cout << "track " << iV << "  ";
                for (int ista = 0; ista < NStations; ista++) {
                    cout << t.vHits[ista].x << " ";
                }
                cout << endl;
            }

            for (int ist = 0; ist < NStations; ist++) {
                HitV &h = t.vHits[ist];
                vStations[ist].Map.GetField(h.x, h.y, h.H);
            }
        }
        timer1.stop();
#ifndef MUTE
        cout << "Start fit..." << endl;
#endif
        TimeStampCounter timer;
        timer.start();
        for (int times = 0; times < Ntimes; times++) {
            int ifit;
            int iV;
            {
                for (iV = 0; iV < NTracksV; iV++) {  // loop on set of 4 tracks
                    for (ifit = 0; ifit < NFits; ifit++) {
                        fitter.Fit(TracksV[iV], vStations, NStations);
                    }
                }
            }
        }
        timer.stop();

        for (int iV = 0; iV < NTracksV; iV++) {  // loop on set of 4 tracks
            TrackV &t = TracksV[iV];
            fitter.ExtrapolateALight(t.T, t.C, Z0[iV], TracksV[iV].T[4], t.f);
        }

        cout << "             preparation: " << std::setw(8) << timer1.cycles() / NTracks / NFits << '\n'
             << "cycles per track and fit: " << std::setw(8) << timer.cycles() / (NTracks * NFits) / Ntimes << '\n'
             << "     cycles for all fits: " << std::setw(8) << timer.cycles() << endl;

        for (int iV = 0; iV < NTracksV; iV++) {  // loop on set of 4 tracks
            TrackV &t = TracksV[iV];

            for (std::size_t it = 0; it < float_v::Size; it++) {
                Track &ts = vTracks[iV * float_v::Size + it];

                for (int i = 0; i < 6; i++) {
                    ts.T[i] = t.T[i][it];
                }
                for (int i = 0; i < 15; i++) {
                    ts.C[i] = t.C[i][it];
                }
            }
        }

        delete[] Z0;
        for (int is = 0; is < NStations; ++is) {
            delete[] Z0s[is];
        }
    }
};

int main()
{
    KalmanFilter *filter = new KalmanFilter;
    filter->readInput();
    filter->fitTracks();
    filter->writeOutput();
    delete filter;
    return 0;
}
