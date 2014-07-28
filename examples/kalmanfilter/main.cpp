#include "matrix.h"
#include "runtimemean.h"
#include <Vc/Vc>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "../tsc.h"
#include <assert.h>

#define MUTE

using std::istream;
using std::ostream;
using std::fstream;
using std::cout;
using std::endl;

const int NFits = 1;
const int MaxNTracks = 20000;
const int Ntimes = 1;
const int MaxNStations = 10;

typedef Vc::float_v V;
using Vc::float_v;
using Vc::float_m;

inline istream & operator>>(istream &strm, V &a) {
    float tmp;
    strm >> tmp;
    a = tmp;
    return strm;
}

inline V rcp(const V &a) {
    return 1.f / a;
}

struct FieldVector : public Vc::VectorAlignedBase {
    V X, Y, Z;
    void Combine(FieldVector &H, const V &w) {
        X += w * (H.X - X);
        Y += w * (H.Y - Y);
        Z += w * (H.Z - Z);
    }
};

struct FieldSlice : public Vc::VectorAlignedBase {
    V X[21], Y[21], Z[21]; // polinom coeff.

    FieldSlice() { for (int i = 0; i < 21; i++) X[i] = Y[i] = Z[i] = 0; }

    void GetField(const V &x, const V &y, V &Hx, V &Hy, V &Hz) {

        V x2 = x * x;
        V y2 = y * y;
        V xy = x * y;
        V x3 = x2 * x;
        V y3 = y2 * y;
        V xy2 = x * y2;
        V x2y = x2 * y;

        V x4 = x3 * x;
        V y4 = y3 * y;
        V xy3 = x * y3;
        V x2y2 = x2 * y2;
        V x3y = x3 * y;

        V x5 = x4 * x;
        V y5 = y4 * y;
        V xy4 = x * y4;
        V x2y3 = x2 * y3;
        V x3y2 = x3 * y2;
        V x4y = x4 * y;

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

    void GetField(const V &x, const V &y, FieldVector &H) {
        GetField(x, y, H.X, H.Y, H.Z);
    }
};

struct FieldRegion : public Vc::VectorAlignedBase {
    V x0, x1, x2 ; // Hx(Z) = x0 + x1 * (Z - z) + x2 * (Z - z)^2
    V y0, y1, y2 ; // Hy(Z) = y0 + y1 * (Z - z) + y2 * (Z - z)^2
    V z0, z1, z2 ; // Hz(Z) = z0 + z1 * (Z - z) + z2 * (Z - z)^2
    V z;

    friend ostream& operator<<(ostream &os, const FieldRegion &a) {
        os << a.x0 << endl
            << a.x1 << endl
            << a.x2 << endl
            << a.y0 << endl
            << a.y1 << endl
            << a.y2 << endl
            << a.z0 << endl
            << a.z1 << endl
            << a.z2 << endl
            << a.z;
        return os;
    }

    FieldRegion() {
        x0 = x1 = x2 = y0 = y1 = y2 = z0 = z1 = z2 = z = 0.;
    }

    void Get(const V z_, V * B) const{
        V dz = (z_ - z);
        V dz2 = dz * dz;
        B[0] = x0 + x1 * dz + x2 * dz2;
        B[1] = y0 + y1 * dz + y2 * dz2;
        B[2] = z0 + z1 * dz + z2 * dz2;
    }

    void Set(const FieldVector &H0, const V &H0z,
            const FieldVector &H1, const V &H1z,
            const FieldVector &H2, const V &H2z) {
        z = H0z;
        V dz1 = H1z - H0z, dz2 = H2z - H0z;
        V det = rcp(dz1 * dz2 * (dz2 - dz1));
        V w21 = - dz2 * det;
        V w22 = dz1 * det;
        V w11 = - dz2 * w21;
        V w12 = - dz1 * w22;

        V dH1 = H1.X - H0.X;
        V dH2 = H2.X - H0.X;
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

    void Shift(const V &Z0) {
        V dz = Z0 - z;
        V x2dz = x2 * dz;
        V y2dz = y2 * dz;
        V z2dz = z2 * dz;
        z = Z0;
        x0 += (x1 + x2dz) * dz;
        x1 += x2dz + x2dz;
        y0 += (y1 + y2dz) * dz;
        y1 += y2dz + y2dz;
        z0 += (z1 + z2dz) * dz;
        z1 += z2dz + z2dz;
    }

};

struct HitInfo : public Vc::VectorAlignedBase, public MatrixOperand<V, 1, 2, HitInfo> { // strip info
    V cos_phi, sin_phi, sigma2, sigma216;

    const V operator()(size_t, size_t c) const {
        switch (c) {
        case 0:  return cos_phi;
        default: return sin_phi;
        }
    }
};

struct HitXYInfo : public Vc::VectorAlignedBase {
    V C00, C10, C11;
};

struct Station : public Vc::VectorAlignedBase {
    V z, thick, zhit, RL,  RadThick, logRadThick,
      SyF, SyL; //  field intergals with respect to First(last) station

    HitInfo UInfo, VInfo; // front and back
    HitXYInfo XYInfo;

    FieldSlice Map;
};

struct Hit : public Vc::VectorAlignedBase {
    V::EntryType x, y;
    V::EntryType tmp1;
    int ista;
};

struct MCPoint : public Vc::VectorAlignedBase {
    V::EntryType x, y, z;
    V::EntryType px, py, pz;
    int ista;
};

struct MCTrack : public Vc::VectorAlignedBase {
    V::EntryType MC_x, MC_y, MC_z, MC_px, MC_py, MC_pz, MC_q;
    MCPoint vPoints[MaxNStations * 2];
    int NMCPoints;
};

struct Track : public Vc::VectorAlignedBase {
    V::EntryType T[6]; // x, y, tx, ty, qp, z
    V::EntryType C[15]; // cov matr.
    V::EntryType Chi2;
    Hit vHits[MaxNStations];
    int NHits;
    int NDF;

    V::EntryType & x() { return T[0]; }
    V::EntryType & y() { return T[1]; }
    V::EntryType &tx() { return T[2]; }
    V::EntryType &ty() { return T[3]; }
    V::EntryType &qp() { return T[4]; }
    V::EntryType & z() { return T[5]; }
};

struct HitV : public Vc::VectorAlignedBase {
    V x, y, w;
    FieldVector H;
};

struct CovV : public Vc::VectorAlignedBase, public MatrixOperand<V, 5, 5, CovV> {
    V C00,
      C10, C11,
      C20, C21, C22,
      C30, C31, C32, C33,
      C40, C41, C42, C43, C44;

    inline const V operator()(size_t r, size_t c) const {
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

    template<typename RhsImpl> inline CovV &operator-=(const MatrixOperand<V, 5, 5, RhsImpl> &rhs) {
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
    const V &operator[](int i) const {
        const V *p = &C00;
        return p[i];
    }

    friend ostream& operator<<(ostream &os, const CovV &a) {
        os << a.C00 << endl
            << a.C10 << endl
            << a.C11 << endl
            << a.C20 << endl
            << a.C21 << endl
            << a.C22 << endl
            << a.C30 << endl
            << a.C31 << endl
            << a.C32 << endl
            << a.C33 << endl
            << a.C40 << endl
            << a.C41 << endl
            << a.C42 << endl
            << a.C43 << endl
            << a.C44;
        return os;
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

struct TrackV : public Vc::VectorAlignedBase, public MatrixOperand<V, 6, 1, TrackV> {
    HitV vHits[MaxNStations];

    V T[6]; // x, y, tx, ty, qp, z
    CovV   C;    // cov matr.

    V Chi2;
    V NDF;

    FieldRegion f; // field at first hit (needed for extrapolation to MC and check of results)

    V &x() { return T[0]; }
    V &y() { return T[1]; }
    V &tx() { return T[2]; }
    V &ty() { return T[3]; }
    V &qp() { return T[4]; }
    V &z() { return T[5]; }

    TrackV()
        : Chi2(Vc::Zero),
        NDF(Vc::Zero)
    {
        T[0] = V::Zero();
        T[1] = V::Zero();
        T[2] = V::Zero();
        T[3] = V::Zero();
        T[4] = V::Zero();
        T[5] = V::Zero();
    }

    template<typename RhsImpl> inline TrackV &operator-=(const MatrixOperand<V, 5, 1, RhsImpl> &rhs)
    {
        T[0] -= rhs(0);
        T[1] -= rhs(1);
        T[2] -= rhs(2);
        T[3] -= rhs(3);
        T[4] -= rhs(4);
        return *this;
    }

    template<typename RhsImpl> inline TrackV &operator-=(const MatrixOperand<V, 6, 1, RhsImpl> &rhs)
    {
        T[0] -= rhs(0);
        T[1] -= rhs(1);
        T[2] -= rhs(2);
        T[3] -= rhs(3);
        T[4] -= rhs(4);
        T[5] -= rhs(5);
        return *this;
    }
    inline const V operator()(size_t r, size_t = 0) const {
        return T[r];
    }
};

//constants
#define cnst static const V

cnst INF = .01f;
cnst INF2 = .0001f;
cnst c_light = 0.000299792458f;
cnst c_light_i = 1.f / c_light;

cnst PipeRadThick = 0.0009f;

class Jacobian_t{ // jacobian elements // j[0][0] - j[3][2] are j02 - j34
    public:
        V &operator()(int i, int j) { assert(i >= 0 && j >= 2); return fj[i][j - 2]; };

    private:
        //     1 0 ? ? ?
        //     0 1 ? ? ?
        // j = 0 0 ? ? ?
        //     0 0 ? ? ?
        //     0 0 0 0 1
        V fj[4][3];
};

class FitFunctional { // base class for all approaches
    public:
        void Fit(TrackV &t, Station vStations[], int NStations) const;

        /// extrapolates track parameters
        void ExtrapolateALight(V T[], CovV &C,  const V &z_out,  V& qp0, FieldRegion &F, V w = V::Zero()) const;

    protected:
        void Filter(TrackV &track, const HitInfo &info, const V u, const V w = V::One()) const;
        void FilterFirst(TrackV &track, HitV &hit, Station &st) const;

        void ExtrapolateWithMaterial(TrackV &track, const V &z_out,  V& qp0, FieldRegion &F, Station &st, bool isPipe = false, V w = V::Zero()) const;

        void AddMaterial(CovV &C, V Q22, V Q32, V Q33) const;
        /// initial aproximation
        void GuessVec(TrackV &t, Station * vStations, int NStations, bool dir = false) const;

        void AddMaterial(TrackV &track, Station &st, const V qp0, bool isPipe = false) const;

        /// extrapolates track parameters and returns jacobian for extrapolation of CovMatrix
        void ExtrapolateJ (
                V * T, // input track parameters (x,y,tx,ty,Q / p) and cov.matrix
                V      z_out  , // extrapolate to this z position
                V       qp0    , // use Q / p linearisation at this value
                const FieldRegion &F,
                Jacobian_t &j,
                V w = V::Zero()) const;

        /// calculate covMatrix for Multiple Scattering
        void GetMSMatrix(const V &tx, const V &ty, const V &radThick, const V &logRadThick, V qp0, V &Q22, V &Q32, V &Q33) const;
    private:

        /// extrapolates track parameters and returns jacobian for extrapolation of CovMatrix
        void ExtrapolateJAnalytic
            (
             V T [], // input track parameters (x,y,tx,ty,Q / p)
             V        z_out  , // extrapolate to this z position
             V       qp0    , // use Q / p linearisation at this value
             const FieldRegion &F,
             Jacobian_t &j,
             V w = V::Zero()) const;

};

inline void FitFunctional::GuessVec(TrackV &t, Station * vStations, int nStations, bool dir) const
{
    V * T = t.T;

    int NHits = nStations;

    V A0, A1 = V::Zero(), A2 = V::Zero(), A3 = V::Zero(), A4 = V::Zero(), A5 = V::Zero(), a0, a1 = V::Zero(), a2 = V::Zero(),
      b0, b1 = V::Zero(), b2 = V::Zero();
    V z0, x, y, z, S, w, wz, wS;

    int i = NHits - 1;
    if (dir) i = 0;
    z0 = vStations[i].zhit;
    HitV * hlst = &(t.vHits[i]);
    w = hlst->w;
    A0 = w;
    a0 = w * hlst->x;
    b0 = w * hlst->y;
    HitV * h = t.vHits;
    Station * st = vStations;
    int st_add = 1;
    if (dir)
    {
        st_add = - 1;
        h += NHits - 1;
        st += NHits - 1;
    }

    for (; h!= hlst; h += st_add, st += st_add) {
        x = h->x;
        y = h->y;
        w = h->w;
        z = st->zhit - z0;
        if (!dir) S = st->SyL;
        else S = st->SyF;
        wz = w * z;
        wS = w * S;
        A0 += w;
        A1 += wz;  A2 += wz * z;
        A3 += wS;  A4 += wS * z; A5 += wS * S;
        a0 += w * x; a1 += wz * x; a2 += wS * x;
        b0 += w * y; b1 += wz * y; b2 += wS * y;
    }

    V A3A3 = A3 * A3;
    V A3A4 = A3 * A4;
    V A1A5 = A1 * A5;
    V A2A5 = A2 * A5;
    V A4A4 = A4 * A4;

    V det = rcp(- A2 * A3A3 + A1 * (A3A4 + A3A4 - A1A5) + A0 * (A2A5 - A4A4));
    V Ai0 = (- A4A4 + A2A5);
    V Ai1 = (A3A4 - A1A5);
    V Ai2 = (- A3A3 + A0 * A5);
    V Ai3 = (- A2 * A3 + A1 * A4);
    V Ai4 = (A1 * A3 - A0 * A4);
    V Ai5 = (- A1 * A1 + A0 * A2);

    V L, L1;
    T[0] = (Ai0 * a0 + Ai1 * a1 + Ai3 * a2) * det;
    T[2] = (Ai1 * a0 + Ai2 * a1 + Ai4 * a2) * det;
    V txtx1 = V::One() +  T[2] * T[2];
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

inline void FitFunctional::ExtrapolateJAnalytic // extrapolates track parameters and returns jacobian for extrapolation of CovMatrix
(
 V T [], // input track parameters (x,y,tx,ty,Q / p)
 V        z_out  , // extrapolate to this z position
 V       qp0    , // use Q / p linearisation at this value
 const FieldRegion &F,
 Jacobian_t &j,
 V
 ) const
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

    const V qp = T[4];
    const V dz = (z_out - T[5]);
    const V dz2 = dz * dz;
    const V dz3 = dz2 * dz;

    // construct coefficients

    const V x   = T[2];
    const V y   = T[3];
    const V xx  = x * x;
    const V xy = x * y;
    const V yy = y * y;
    const V y2 = y * c2;
    const V x2 = x * c2;
    const V x4 = x * c4;
    const V xx31 = xx * c3 + c1;
    const V xx159 = xx * c15 + c9;

    const V Ay = - xx - c1;
    const V Ayy = x * (xx * c3 + c3);
    const V Ayz = - c2 * xy;
    const V Ayyy = - (c15 * xx * xx + c18 * xx + c3);

    const V Ayy_x = c3 * xx31;
    const V Ayyy_x = - x4 * xx159;

    const V Bx = yy + c1;
    const V Byy = y * xx31;
    const V Byz = c2 * xx + c1;
    const V Byyy = - xy * xx159;

    const V Byy_x = c6 * xy;
    const V Byyy_x = - y * (c45 * xx + c9);
    const V Byyy_y = - x * xx159;

    // end of coefficients calculation

    const V t2   = c1 + xx + yy;
    const V t    = sqrt(t2);
    const V h    = qp0 * c_light;
    const V ht   = h * t;

    // get field integrals
    const V ddz = T[5] - F.z;
    const V Fx0 = F.x0 + F.x1 * ddz + F.x2 * ddz * ddz;
    const V Fx1 = (F.x1 + c2 * F.x2 * ddz) * dz;
    const V Fx2 = F.x2 * dz2;
    const V Fy0 = F.y0 + F.y1 * ddz + F.y2 * ddz * ddz;
    const V Fy1 = (F.y1 + c2 * F.y2 * ddz) * dz;
    const V Fy2 = F.y2 * dz2;
    const V Fz0 = F.z0 + F.z1 * ddz + F.z2 * ddz * ddz;
    const V Fz1 = (F.z1 + c2 * F.z2 * ddz) * dz;
    const V Fz2 = F.z2 * dz2;

    //

    const V sx = (Fx0 + Fx1 * c2i + Fx2 * c3i );
    const V sy = (Fy0 + Fy1 * c2i + Fy2 * c3i);
    const V sz = (Fz0 + Fz1 * c2i + Fz2 * c3i);

    const V Sx = (Fx0 * c2i + Fx1 * c6i + Fx2 * c12i);
    const V Sy = (Fy0 * c2i + Fy1 * c6i + Fy2 * c12i);
    const V Sz = (Fz0 * c2i + Fz1 * c6i + Fz2 * c12i);

    V syz;
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

    V Syz;
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
            +   Fy1 * (c10 * Fz0 + c11 * Fz1 + c12 * Fz2)
            +   Fy2 * (c20 * Fz0 + c21 * Fz1 + c22 * Fz2) ;
    }

    const V syy  = sy * sy * c2i;
    const V syyy = syy * sy * c3i;

    V Syy ;
    {
        cnst c00 = 420.f / 2520.f;
        cnst c01 =  21.f * 15.f / 2520.f;
        cnst c02 =  21.f * 8.f / 2520.f;
        cnst c03 =  63.f / 2520.f;
        cnst c04 =  70.f / 2520.f;
        cnst c05 =  20.f / 2520.f;
        Syy =  Fy0 * (c00 * Fy0 + c01 * Fy1 + c02 * Fy2) + Fy1 * (c03 * Fy1 + c04 * Fy2) + c05 * Fy2 * Fy2 ;
    }

    V Syyy;
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
        const V Fy22 = Fy2 * Fy2;
        Syyy = Fy0 * (Fy0 * (c000 * Fy0 + c001 * Fy1 + c002 * Fy2) + Fy1 * (c011 * Fy1 + c012 * Fy2) + c022 * Fy22)
            +    Fy1 * (Fy1 * (c111 * Fy1 + c112 * Fy2) + c122 * Fy22) + c222 * Fy22 * Fy2                  ;
    }


    const V sA1   = sx * xy   + sy * Ay   + sz * y ;
    const V sA1_x = sx * y - sy * x2 ;
    const V sA1_y = sx * x + sz ;

    const V sB1   = sx * Bx   - sy * xy   - sz * x ;
    const V sB1_x = - sy * y - sz ;
    const V sB1_y = sx * y2 - sy * x ;

    const V SA1   = Sx * xy   + Sy * Ay   + Sz * y ;
    const V SA1_x = Sx * y - Sy * x2 ;
    const V SA1_y = Sx * x + Sz;

    const V SB1   = Sx * Bx   - Sy * xy   - Sz * x ;
    const V SB1_x = - Sy * y - Sz;
    const V SB1_y = Sx * y2 - Sy * x;


    const V sA2   = syy * Ayy   + syz * Ayz ;
    const V sA2_x = syy * Ayy_x - syz * y2 ;
    const V sA2_y = - syz * x2 ;
    const V sB2   = syy * Byy   + syz * Byz  ;
    const V sB2_x = syy * Byy_x + syz * x4 ;
    const V sB2_y = syy * xx31 ;

    const V SA2   = Syy * Ayy   + Syz * Ayz ;
    const V SA2_x = Syy * Ayy_x - Syz * y2 ;
    const V SA2_y = - Syz * x2 ;
    const V SB2   = Syy * Byy   + Syz * Byz ;
    const V SB2_x = Syy * Byy_x + Syz * x4 ;
    const V SB2_y = Syy * xx31 ;

    const V sA3   = syyy * Ayyy  ;
    const V sA3_x = syyy * Ayyy_x;
    const V sB3   = syyy * Byyy  ;
    const V sB3_x = syyy * Byyy_x;
    const V sB3_y = syyy * Byyy_y;


    const V SA3   = Syyy * Ayyy  ;
    const V SA3_x = Syyy * Ayyy_x;
    const V SB3   = Syyy * Byyy  ;
    const V SB3_x = Syyy * Byyy_x;
    const V SB3_y = Syyy * Byyy_y;

    const V ht1 = ht * dz;
    const V ht2 = ht * ht * dz2;
    const V ht3 = ht * ht * ht * dz3;
    const V ht1sA1 = ht1 * sA1;
    const V ht1sB1 = ht1 * sB1;
    const V ht1SA1 = ht1 * SA1;
    const V ht1SB1 = ht1 * SB1;
    const V ht2sA2 = ht2 * sA2;
    const V ht2SA2 = ht2 * SA2;
    const V ht2sB2 = ht2 * sB2;
    const V ht2SB2 = ht2 * SB2;
    const V ht3sA3 = ht3 * sA3;
    const V ht3sB3 = ht3 * sB3;
    const V ht3SA3 = ht3 * SA3;
    const V ht3SB3 = ht3 * SB3;

    T[0]  += ((x + ht1SA1 + ht2SA2 + ht3SA3) * dz);
    T[1]  += ((y + ht1SB1 + ht2SB2 + ht3SB3) * dz);
    T[2]  += (ht1sA1 + ht2sA2 + ht3sA3);
    T[3]  += (ht1sB1 + ht2sB2 + ht3sB3);
    T[5]  += (dz);

    const V ctdz  = c_light * t * dz;
    const V ctdz2 = c_light * t * dz2;

    const V dqp = qp - qp0;
    const V t2i = c1 * rcp(t2); // / t2;
    const V xt2i = x * t2i;
    const V yt2i = y * t2i;
    const V tmp0 = ht1SA1 + c2 * ht2SA2 + c3 * ht3SA3;
    const V tmp1 = ht1SB1 + c2 * ht2SB2 + c3 * ht3SB3;
    const V tmp2 = ht1sA1 + c2 * ht2sA2 + c3 * ht3sA3;
    const V tmp3 = ht1sB1 + c2 * ht2sB2 + c3 * ht3sB3;

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

inline void FitFunctional::ExtrapolateJ // extrapolates track parameters and returns jacobian for extrapolation of CovMatrix
(
 V * T, // input track parameters (x,y,tx,ty,Q / p) and cov.matrix
 V      z_out  , // extrapolate to this z position
 V       qp0    , // use Q / p linearisation at this value
 const FieldRegion &F,
 Jacobian_t &j,
 V w
 ) const
{
    ExtrapolateJAnalytic(T, z_out, qp0, F, j, w);
}

/// calculate covMatrix for Multiple Scattering
inline void FitFunctional::GetMSMatrix(const V &tx, const V &ty, const V &radThick, const V &logRadThick, V qp0, V &Q22, V &Q32, V &Q33) const
{
    cnst mass2 = 0.1396f * 0.1396f;

    V txtx = tx * tx;
    V tyty = ty * ty;
    V txtx1 = txtx + V::One();
    V h = txtx + tyty;
    V t = sqrt(txtx1 + tyty);
    V h2 = h * h;
    V qp0t = qp0 * t;

    cnst c1 = 0.0136f;
    cnst c2 = c1 * 0.038f;
    cnst c3 = c2 * 0.5f;
    cnst c4 = - c3 / 2.0f;
    cnst c5 = c3 / 3.0f;
    cnst c6 = - c3 / 4.0f;

    V s0 = (c1 + c2 * logRadThick + c3 * h + h2 * (c4 + c5 * h + c6 * h2)) * qp0t;

    V a = (V::One() + mass2 * qp0 * qp0t) * radThick * s0 * s0;


    Q22 = txtx1 * a;
    Q32 = tx * ty * a;
    Q33 = (V::One() + tyty) * a;
}

inline void FitFunctional::AddMaterial(TrackV &track, Station &st, const V qp0, bool isPipe) const
{
    V Q22, Q32, Q33;
    if (isPipe)
        GetMSMatrix(track.T[2], track.T[3], st.RadThick + PipeRadThick, log(st.RadThick + PipeRadThick), qp0, Q22, Q32, Q33);
    else
        GetMSMatrix(track.T[2], track.T[3], st.RadThick, st.logRadThick, qp0, Q22, Q32, Q33);

    AddMaterial(track.C, Q22, Q32, Q33);
}

inline void FitFunctional::Fit(TrackV &t, Station vStations[], int nStations) const
{
    // upstream

    GuessVec(t, vStations, nStations);

    // downstream

    FieldRegion f;
    V z0,z1,z2, dz;
    FieldVector H0, H1, H2;

    V qp0 = t.T[4];
    int i = nStations - 1;
    HitV * h = &t.vHits[i];

    FilterFirst(t, * h, vStations[i]);
    AddMaterial(t, vStations[ i ], qp0);

    z1 = vStations[ i ].z;
    vStations[i].Map.GetField(t.T[0],t.T[1], H1);
    H1.Combine(h->H, h->w);

    z2 = vStations[ i - 2 ].z;
    dz = z2 - z1;
    vStations[ i - 2 ].Map.GetField(t.T[0] + t.T[2] * dz,t.T[1] + t.T[3] * dz,H2);
    h = &t.vHits[i - 2];
    H2.Combine(h->H, h->w);

    const int iFirstStation = 0;
    for (--i; i >= iFirstStation; i--) {
        h = &t.vHits[i];
        Station &st = vStations[i];
        z0 = st.z;
        dz = (z1 - z0);
        st.Map.GetField(t.T[0] - t.T[2] * dz,t.T[1] - t.T[3] * dz,H0);
        H0.Combine(h->H, h->w);
        f.Set(H0, z0, H1, z1, H2, z2);
        if (i == iFirstStation)
            t.f = f;

        // ExtrapolateALight(t.T, t.C, st.zhit, qp0, f);
        // AddMaterial(t, st, qp0);
        if (i == 1) { // 2nd MVD
            ExtrapolateWithMaterial(t, st.zhit, qp0, f, st, true); // add pipe
        }
        else
            ExtrapolateWithMaterial(t, st.zhit, qp0, f, st);

        V u = h->x * st.UInfo.cos_phi + h->y * st.UInfo.sin_phi;
        V v = h->x * st.VInfo.cos_phi + h->y * st.VInfo.sin_phi;
        Filter(t, st.UInfo, u, h->w);
        Filter(t, st.VInfo, v, h->w);
        H2 = H1;
        z2 = z1;
        H1 = H0;
        z1 = z0;
    }
}

// inline // --> causes a runtime overhead and problems for the MS compiler (error C2603)
void FitFunctional::ExtrapolateALight
(
 V T [], // input track parameters (x,y,tx,ty,Q / p)
 CovV &C,     // input covariance matrix
 const V &z_out  , // extrapolate to this z position
 V       &qp0    , // use Q / p linearisation at this value
 FieldRegion &F,
 V w
 ) const
{
    //
    //  Part of the analytic extrapolation formula with error (c_light * B * dz)^4 / 4!
    //
    Jacobian_t j;
    ExtrapolateJ(T, z_out, qp0, F, j, w);

    //          covariance matrix transport

    const V c42 = C.C42, c43 = C.C43;

    const V cj00 = C.C00 + C.C20 * j(0,2) + C.C30 * j(0,3) + C.C40 * j(0,4);
    // const V cj10 = C.C10 + C.C21 * j(0,2) + C.C31 * j(0,3) + C.C41 * j(0,4);
    const V cj20 = C.C20 + C.C22 * j(0,2) + C.C32 * j(0,3) + c42 * j(0,4);
    const V cj30 = C.C30 + C.C32 * j(0,2) + C.C33 * j(0,3) + c43 * j(0,4);

    const V cj01 = C.C10 + C.C20 * j(1,2) + C.C30 * j(1,3) + C.C40 * j(1,4);
    const V cj11 = C.C11 + C.C21 * j(1,2) + C.C31 * j(1,3) + C.C41 * j(1,4);
    const V cj21 = C.C21 + C.C22 * j(1,2) + C.C32 * j(1,3) + c42 * j(1,4);
    const V cj31 = C.C31 + C.C32 * j(1,2) + C.C33 * j(1,3) + c43 * j(1,4);

    // const V cj02 = C.C20 * j(2,2) + C.C30 * j(2,3) + C.C40 * j(2,4);
    // const V cj12 = C.C21 * j(2,2) + C.C31 * j(2,3) + C.C41 * j(2,4);
    const V cj22 = C.C22 * j(2,2) + C.C32 * j(2,3) + c42 * j(2,4);
    const V cj32 = C.C32 * j(2,2) + C.C33 * j(2,3) + c43 * j(2,4);

    // const V cj03 = C.C20 * j(3,2) + C.C30 * j(3,3) + C.C40 * j(3,4);
    // const V cj13 = C.C21 * j(3,2) + C.C31 * j(3,3) + C.C41 * j(3,4);
    const V cj23 = C.C22 * j(3,2) + C.C32 * j(3,3) + c42 * j(3,4);
    const V cj33 = C.C32 * j(3,2) + C.C33 * j(3,3) + c43 * j(3,4);

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
inline void FitFunctional::Filter(TrackV &track, const HitInfo &measurementModel, const float_v m, const float_v weight) const
{
    static RuntimeMean timer;
    timer.start();

    const float_v sigma2 = measurementModel.sigma2;
    const float_v sigma216 = measurementModel.sigma216;
    const Matrix<float_v, 5> F = track.C.slice<0, 5, 0, 2>() * measurementModel.transposed(); // CHᵀ
    const float_v hch = measurementModel * F.slice<0, 2>();                                  // HCHᵀ
    const float_v residual = measurementModel * track.slice<0, 2>() - m; // ζ = Hr - m

    const float_v denominator = Vc::iif (hch < sigma216, hch + sigma2, hch);
    const float_v zetawi = residual / denominator;             //           (V' + HCHᵀ)⁻¹ ζ
    track -= F * zetawi;                                       // r  -= CHᵀ (V' + HCHᵀ)⁻¹ ζ
    track.C -= F * (weight / (sigma2 + hch)) * F.transposed(); // C  -= CHᵀ (V  + HCHᵀ)⁻¹ HC
    track.Chi2(hch < sigma216) += residual * zetawi;           // χ² +=  ζ  (V' + HCHᵀ)⁻¹ ζ
    track.NDF += weight;

    timer.stop();
}

inline void FitFunctional::FilterFirst(TrackV &track, HitV &hit, Station &st) const
{

    CovV &C = track.C;
    V w1 = V::One() - hit.w;
    V c00 = hit.w * st.XYInfo.C00 + w1 * INF;
    V c10 = hit.w * st.XYInfo.C10 + w1 * INF;
    V c11 = hit.w * st.XYInfo.C11 + w1 * INF;

    // // initialize covariance matrix
    C.C00 = c00;
    C.C10 = c10;       C.C11 = c11;
    C.C20 = V::Zero();      C.C21 = V::Zero();      C.C22 = INF2; // needed for stability of smoother. improve FilterTracks and CHECKME
    C.C30 = V::Zero();      C.C31 = V::Zero();      C.C32 = V::Zero(); C.C33 = INF2;
    C.C40 = V::Zero();      C.C41 = V::Zero();      C.C42 = V::Zero(); C.C43 = V::Zero(); C.C44 = INF;

    track.T[0] = hit.w * hit.x + w1 * track.T[0];
    track.T[1] = hit.w * hit.y + w1 * track.T[1];
    track.NDF = - 3.0;
    track.Chi2 = V::Zero();
}

inline void FitFunctional::AddMaterial(CovV &C, V Q22, V Q32, V Q33) const
{
    C.C22 += Q22;
    C.C32 += Q32; C.C33 += Q33;
}

inline void FitFunctional::ExtrapolateWithMaterial(TrackV &track,  const V &z_out,  V& qp0, FieldRegion &F, Station &st, bool isPipe, V w) const
{
    ExtrapolateALight(track.T, track.C, z_out, qp0, F, w);
    FitFunctional::AddMaterial(track, st, qp0, isPipe); // FIXME
}

class KalmanFilter : public Vc::VectorAlignedBase
{
    FieldRegion field0;
    FitFunctional fitter;
    Track vTracks[MaxNTracks];
    MCTrack vMCTracks[MaxNTracks];
    Station * vStations;
    int NStations;
    int NTracks;
    int NTracksV;

public:
    KalmanFilter()
        : vStations(new Station[MaxNStations]),
          NStations(0),
          NTracks(0),
          NTracksV(0)
    {
    }

    ~KalmanFilter()
    {
        delete[] vStations;
    }

    void readInput() {

        fstream FileGeo, FileTracks, FileMCTracks;

        FileGeo.open("geo.dat", std::ios::in);
        FileTracks.open("tracks.dat", std::ios::in);
        FileMCTracks.open("mctracksin.dat", std::ios::in);
        {
            FieldVector H[3];
            V Hz[3];
            for (int i = 0; i < 3; i++) {
                V::EntryType Bx, By, Bz, z;
                FileGeo >> z >> Bx >> By >> Bz;
                Hz[i] = z; H[i].X = Bx;   H[i].Y = By; H[i].Z = Bz;
#ifndef MUTE
                cout << "Input Magnetic field:" << z << " " << Bx << " " << By << " " << Bz << endl;
#endif
            }
            field0.Set(H[0],Hz[0], H[1],Hz[1], H[2],Hz[2]);
        }
        FileGeo >> NStations;
#ifndef MUTE
        cout << "Input " << NStations << " Stations:" << endl;
#endif

        for (int i = 0; i < NStations; i++) {
            int ist;
            FileGeo >> ist;
            if (ist!= i) break;
            Station &st = vStations[i];

            FileGeo >> st.z >> st.thick >> st.RL >> st.UInfo.sigma2 >> st.VInfo.sigma2;
            st.UInfo.sigma2 *= st.UInfo.sigma2;
            st.VInfo.sigma2 *= st.VInfo.sigma2;
            st.UInfo.sigma216 = st.UInfo.sigma2 * 16.f;
            st.VInfo.sigma216 = st.VInfo.sigma2 * 16.f;

            if (i < 2) { // mvd // TODO From Geo File!!!
                st.UInfo.cos_phi = V::One();
                st.UInfo.sin_phi = V::Zero();
                st.VInfo.cos_phi = V::Zero();
                st.VInfo.sin_phi = V::One();
            }
            else{
                st.UInfo.cos_phi = V::One();           // 0 degree
                st.UInfo.sin_phi = V::Zero();
                st.VInfo.cos_phi = 0.9659258244f; // 15 degree
                st.VInfo.sin_phi = 0.2588190521f;
            }

            V idet = st.UInfo.cos_phi * st.VInfo.sin_phi - st.UInfo.sin_phi * st.VInfo.cos_phi;
            idet = V::One() / (idet * idet);
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
            V z0  = vStations[NStations - 1].z;
            V sy = 0.f, Sy = 0.f;
            for (int i = NStations - 1; i >= 0; i--) {
                Station &st = vStations[i];
                V dz = st.z - z0;
                V Hy = vStations[i].Map.Y[0];
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
                V dz = st.z - z0;
                V Hy = vStations[i].Map.Y[0];
                Sy += dz * sy + dz * dz * Hy * 0.5f;
                sy += dz * Hy;
                st.SyF = Sy;
                z0 = st.z;
            }
        }

        FileGeo.close();

        NTracks = 0;
        int TrackIndex[MaxNTracks];
        while(!FileTracks.eof()) {

            int itr;
            FileTracks >> itr;
            // if (itr!= NTracks) break;
            if (NTracks >= MaxNTracks) break;

            Track &t = vTracks[NTracks];
            MCTrack &mc = vMCTracks[NTracks];
            FileTracks >> mc.MC_x   >> mc.MC_y  >> mc.MC_z
                >> mc.MC_px >> mc.MC_py >> mc.MC_pz >> mc.MC_q
                >> t.NHits;
            for (int i = 0; i < t.NHits; i++) {
                int ist;
                FileTracks >> ist;
                t.vHits[i].ista = ist;
                FileTracks >> t.vHits[i].x >> t.vHits[i].y;
            }
            TrackIndex[NTracks] = itr;
            if (t.NHits == NStations) NTracks++;
        }
        int NMCTracks = 0;
        int iPoint = 0;
        while(!FileMCTracks.eof()) {

            int itr;
            FileMCTracks >> itr;
            // if (itr!= NTracks) break;
            if (NMCTracks >= MaxNTracks) break;
            MCTrack &mc = vMCTracks[NMCTracks];
            V::EntryType temp;
            int NMCPoints;
            FileMCTracks >> temp   >> temp  >> temp
                >> temp >> temp >> temp >> temp
                >> NMCPoints;
            mc.NMCPoints = NMCPoints;
            for (int i = 0; i < NMCPoints; i++) {
                int ist;
                FileMCTracks >> ist;
                mc.vPoints[i].ista = ist;
                FileMCTracks >> mc.vPoints[i].x >> mc.vPoints[i].y >> mc.vPoints[i].z >> mc.vPoints[i].px >> mc.vPoints[i].py >> mc.vPoints[i].pz;

            }

            iPoint = 0; // compare paraments at the first station
            // iPoint = NMCPoints - 1;
            mc.MC_x = mc.vPoints[iPoint].x;
            mc.MC_y = mc.vPoints[iPoint].y;
            mc.MC_z = mc.vPoints[iPoint].z;
            mc.MC_px = mc.vPoints[iPoint].px;
            mc.MC_py = mc.vPoints[iPoint].py;
            mc.MC_pz = mc.vPoints[iPoint].pz;

            if (itr == TrackIndex[NMCTracks]) NMCTracks++;
        }
        //cout << NTracks << " " << NMCTracks << " reco and Mc tracks have been read" << endl;
        FileTracks.close();
        FileMCTracks.close();

        NTracksV = NTracks / V::Size;
        NTracks =  NTracksV * V::Size;
    }

#define _STRINGIFY(_x) #_x
#define STRINGIFY(_x) _STRINGIFY(_x)

    void writeOutput() {

        fstream Out, Diff;

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
            for (int i = 0; i < 15; i++) ok = ok && finite(t.C[i]);

            if (!ok) { cout << " infinite " << endl; }

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
        TrackV * TracksV = new TrackV[MaxNTracks / V::Size + 1];
        V * Z0      = new V[MaxNTracks / V::Size + 1]; // mc - z, used for result comparison
        V * Z0s[MaxNStations];
        for (int is = 0; is < NStations; ++is)
            Z0s[is] = new V[MaxNTracks / V::Size + 1];

        V::Memory Z0mem;
        V::Memory Z0smem[MaxNStations];
#ifndef MUTE
        cout << "Prepare data..." << endl;
#endif
        TimeStampCounter timer1;
        timer1.start();

        for (int iV = 0; iV < NTracksV; iV++) { // loop on set of 4 tracks
#ifndef MUTE
            if (iV * V::Size%100 == 0) cout << iV * V::Size << endl;
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

            for (int it = 0; it < V::Size; it++) {
                Track &ts = vTracks[iV * V::Size + it];

                Z0mem[it] = vMCTracks[iV * V::Size + it].MC_z;
                for (int is = 0; is < NStations; ++is)
                    Z0smem[is][it] = vMCTracks[iV * V::Size + it].vPoints[is].z;

                for (int ista = 0, ih = 0; ista < NStations; ista++) {
                    Hit &hs = ts.vHits[ih];
                    if (hs.ista != ista) continue;
                    ih++;

                    t.vHits[ista].x[it] = hs.x;
                    t.vHits[ista].y[it] = hs.y;
                    t.vHits[ista].w[it] = 1.;
                }

            }

            V Z0temp(Z0mem);
            Z0[iV] = Z0temp;

            for (int is = 0; is < NStations; ++is) {
                V Z0stemp(Z0smem[is]);
                Z0s[is][iV] = Z0stemp;
            }

            if (0) {    // output for check
                cout << "track " << iV << "  ";
                for (int ista = 0; ista < NStations; ista++)
                    cout << t.vHits[ista].x << " ";
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
                for (iV = 0; iV < NTracksV; iV++) { // loop on set of 4 tracks
                    for (ifit = 0; ifit < NFits; ifit++) {
                        fitter.Fit(TracksV[iV], vStations, NStations);
                    }
                }
            }
        }
        timer.stop();

        for (int iV = 0; iV < NTracksV; iV++) { // loop on set of 4 tracks
            TrackV &t = TracksV[iV];
            fitter.ExtrapolateALight(t.T, t.C, Z0[iV], TracksV[iV].T[4], t.f);
        }

        cout << "             preparation: " << std::setw(8) << timer1.cycles() / NTracks / NFits << '\n'
             << "cycles per track and fit: " << std::setw(8) << timer.cycles() / (NTracks * NFits) / Ntimes << '\n'
             << "     cycles for all fits: " << std::setw(8) << timer.cycles() << endl;

        for (int iV = 0; iV < NTracksV; iV++) { // loop on set of 4 tracks
            TrackV &t = TracksV[iV];

            for (int it = 0; it < V::Size; it++) {
                Track &ts = vTracks[iV * V::Size + it];

                for (int i = 0; i < 6; i++)
                    ts.T[i] = t.T[i][it];
                for (int i = 0; i < 15; i++)
                    ts.C[i] = t.C[i][it];
            }
        }

        delete [] Z0;
        for (int is = 0; is < NStations; ++is)
            delete [] Z0s[is];
        delete [] TracksV;
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
