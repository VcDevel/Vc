#include <Vc/Vc>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "../../tsc.h"

#define MUTE

using Vc::float_v;
using Vc::float_m;

typedef float_v Vec;
typedef int    Int_t;
enum Constants {
    vecN = float_v::Size,
    MaxNTracks = 20000,//20000
    NFits = 1,//10;
    Ntimes = 100 //int(3000.*coeff);//100;
};


std::istream & operator>>(std::istream &strm, Vec &a ){
    Vec::EntryType tmp;
    strm>>tmp;
    a = tmp;
    return strm;
}

struct FieldVector : public Vc::VectorAlignedBase
{
    Vec X, Y, Z;
    void Combine( FieldVector &H, const Vec &w ){
        X+= w*(H.X-X);
        Y+= w*(H.Y-Y);
        Z+= w*(H.Z-Z);
    }
};

struct FieldSlice : public Vc::VectorAlignedBase
{
    Vec X[10], Y[10], Z[10]; // polinom coeff.
    //Vec z;
    FieldSlice(){ for( int i=0; i<10; i++ ) X[i]=Y[i]=Z[i]=0; }

    void GetField( const Vec &x, const Vec &y, Vec &Hx, Vec &Hy, Vec &Hz ){

        Vec x2 = x*x;
        Vec y2 = y*y;
        Vec x3 = x2*x;
        Vec y3 = y2*y;
        Vec xy = x*y;
        Vec xy2 = x*y2;
        Vec x2y = x2*y;
        //Vec x4 = x3*x;
        //Vec xy3 = x*y3;
        //Vec x2y2 = x*xy2;
        //Vec x3y = x*x2y;
        //Vec y4 = y*y3;

        Hx = X[0] +X[1]*x +X[2]*y +X[3]*x2 +X[4]*xy +X[5]*y2 +X[6]*x3 +X[7]*x2y +X[8]*xy2 +X[9]*y3;
        //+ X[10]*x4 + X[11]*x3y +X[12]*x2y2 +X[13]*xy3 + X[14]*y4;
        Hy = Y[0] +Y[1]*x +Y[2]*y +Y[3]*x2 +Y[4]*xy +Y[5]*y2 +Y[6]*x3 +Y[7]*x2y +Y[8]*xy2 +Y[9]*y3;
        //+ Y[10]*x4 + Y[11]*x3y +Y[12]*x2y2 +Y[13]*xy3 + Y[14]*y4;
        Hz = Z[0] +Z[1]*x +Z[2]*y +Z[3]*x2 +Z[4]*xy +Z[5]*y2 +Z[6]*x3 +Z[7]*x2y +Z[8]*xy2 +Z[9]*y3;
        //+ Z[10]*x4 + Z[11]*x3y +Z[12]*x2y2 +Z[13]*xy3 + Z[14]*y4;
    }

    void GetField( const Vec &x, const Vec &y, FieldVector &H ){
        GetField( x, y, H.X, H.Y, H.Z );
    }
};


struct FieldRegion : public Vc::VectorAlignedBase
{
    Vec x0, x1, x2 ; // Hx(Z) = x0 + x1*(Z-z) + x2*(Z-z)^2
    Vec y0, y1, y2 ; // Hy(Z) = y0 + y1*(Z-z) + y2*(Z-z)^2
    Vec z0, z1, z2 ; // Hz(Z) = z0 + z1*(Z-z) + z2*(Z-z)^2
    Vec z;

    FieldRegion()
        : x0(Vc::Zero), x1(Vc::Zero), x2(Vc::Zero),
        y0(Vc::Zero), y1(Vc::Zero), y2(Vc::Zero),
        z0(Vc::Zero), z1(Vc::Zero), z2(Vc::Zero),
        z(Vc::Zero)
    {
    }

    void Set(const FieldVector &H0, const Vec &H0z,
             const FieldVector &H1, const Vec &H1z,
             const FieldVector &H2, const Vec &H2z)
    {
        z = H0z;
        Vec dz1 = H1z-H0z, dz2 = H2z-H0z;
        Vec det = Vc::reciprocal(dz1*dz2*(dz2-dz1));
        Vec w21 = -dz2*det;
        Vec w22 = dz1*det;
        Vec w11 = -dz2*w21;
        Vec w12 = -dz1*w22;

        Vec dH1 = H1.X - H0.X;
        Vec dH2 = H2.X - H0.X;
        x0 = H0.X;
        x1 = dH1*w11 + dH2*w12 ;
        x2 = dH1*w21 + dH2*w22 ;

        dH1 = H1.Y - H0.Y;
        dH2 = H2.Y - H0.Y;
        y0 = H0.Y;
        y1 = dH1*w11 + dH2*w12 ;
        y2 = dH1*w21 + dH2*w22  ;

        dH1 = H1.Z - H0.Z;
        dH2 = H2.Z - H0.Z;
        z0 = H0.Z;
        z1 = dH1*w11 + dH2*w12 ;
        z2 = dH1*w21 + dH2*w22 ;
    }

    void Shift( const Vec &Z0){
        Vec dz = Z0-z;
        Vec x2dz = x2*dz;
        Vec y2dz = y2*dz;
        Vec z2dz = z2*dz;
        z = Z0;
        x0+= (x1 + x2dz)*dz;
        x1+= x2dz+x2dz;
        y0+= (y1 + y2dz)*dz;
        y1+= y2dz+y2dz;
        z0+= (z1 + z2dz)*dz;
        z1+= z2dz+z2dz;
    }
};


struct Station : public Vc::VectorAlignedBase
{
    Vec z, thick, zhit, RL,  RadThick, logRadThick,
        Sigma, Sigma2, Sy;
    FieldSlice Map;
    Station() {
        Sigma = 20.E-4f;
        Sigma2 = Sigma*Sigma;
    }
};


struct Hit : public Vc::VectorAlignedBase
{
    Vec::EntryType x, y;
    Int_t ista;
    Vec::EntryType tmp1;
};

struct MCTrack : public Vc::VectorAlignedBase
{
    Vec::EntryType MC_x, MC_y, MC_z, MC_px, MC_py, MC_pz, MC_q;
};

struct Track : public Vc::VectorAlignedBase
{
    Int_t NHits;
    Hit vHits[12];
    Vec::EntryType T[6]; // x, y, tx, ty, qp, z
    Vec::EntryType C[15]; // cov matr.
    Vec::EntryType Chi2;
    Int_t NDF;
};

struct HitV : public Vc::VectorAlignedBase
{
    Vec x, y, w;
    FieldVector H;
};


struct CovV : public Vc::VectorAlignedBase
{
#ifdef SQRT_FILTER
    Vec C00, C01, C02, C03, C04,
        C10, C11, C12, C13, C14,
        C20, C21, C22, C23, C24,
        C30, C31, C32, C33, C34,
        C40, C41, C42, C43, C44;
#else
    Vec C00,
        C10, C11,
        C20, C21, C22,
        C30, C31, C32, C33,
        C40, C41, C42, C43, C44;
#endif
};

struct TrackV : public Vc::VectorAlignedBaseT<Vec> {
    HitV vHits[12];
    Vec T[6]; // x, y, tx, ty, qp, z
    CovV   C;    // cov matr.
    Vec Chi2;
    Vec NDF;
};

static const Vec::EntryType INF = .01, ZERO = 0.0, ONE = 1.;
static const Vec::EntryType c_light = 0.000299792458, c_light_i = 1./c_light;

//inline // --> causes a runtime overhead and problems for the MS compiler (error C2603)
void ExtrapolateALight
(
 Vec T [], // input track parameters (x,y,tx,ty,Q/p)
 CovV &C,     // input covariance matrix
 const Vec &z_out  , // extrapolate to this z position
 Vec       &qp0    , // use Q/p linearisation at this value
 FieldRegion &F
 )
{
    //
    //  Part of the analytic extrapolation formula with error (c_light*B*dz)^4/4!
    //
    // TimeStampCounter unitimer2[3];
    // unitimer2[0].Start();
    static const Vec::EntryType
        c1 = 1., c2 = 2., c3 = 3., c4 = 4., c6 = 6., c9 = 9., c15 = 15., c18 = 18., c45 = 45.,
    c2i = 1./2., c3i = 1./3., c6i = 1./6., c12i = 1./12.;

    Vec qp = T[4];
    // unitimer2[0].Stop();

    Vec dz = (z_out - T[5]);
    // unitimer2[1].Start();
    Vec dz2 = dz*dz;
    // unitimer2[1].Stop();
    Vec dz3 = dz2*dz;

    Vec T0 = T[0];
    Vec T1 = T[1];
    Vec T2 = T[2];
    Vec T3 = T[3];
    //Vec T4 = T[4];
    Vec T5 = T[5];

    // construct coefficients

    Vec x   = T[2]; // tx !!
    Vec y   = T[3]; // ty !!
    Vec xx  = x*x;
    Vec xy = x*y;
    Vec yy = y*y;
    Vec y2 = y*c2;
    Vec x2 = x*c2;
    Vec x4 = x*c4;
    Vec xx31 = xx*c3+c1;
    Vec xx159 = xx*c15+c9;

    Vec Ay = -xx-c1;
    Vec Ayy = x*(xx*c3+c3);
    Vec Ayz = -c2*xy;
    Vec Ayyy = -(c15*xx*xx+c18*xx+c3);

    Vec Ayy_x = c3*xx31;
    Vec Ayyy_x = -x4*xx159;

    Vec Bx = yy+c1;
    Vec Byy = y*xx31;
    Vec Byz = c2*xx+c1;
    Vec Byyy = -xy*xx159;

    Vec Byy_x = c6*xy;
    Vec Byyy_x = -y*(c45*xx+c9);
    Vec Byyy_y = -x*xx159;

    // end of coefficients calculation

    Vec t2   = c1 + xx + yy;
    Vec t    = sqrt( t2 );
    Vec h    = qp0*c_light;
    Vec ht   = h*t;

    // get field integrals
    Vec ddz = T5-F.z;
    Vec Fx0 = F.x0 + F.x1*ddz + F.x2*ddz*ddz;
    Vec Fx1 = (F.x1 + c2*F.x2*ddz)*dz;
    Vec Fx2 = F.x2*dz2;
    Vec Fy0 = F.y0 + F.y1*ddz + F.y2*ddz*ddz;
    Vec Fy1 = (F.y1 + c2*F.y2*ddz)*dz;
    Vec Fy2 = F.y2*dz2;
    Vec Fz0 = F.z0 + F.z1*ddz + F.z2*ddz*ddz;
    Vec Fz1 = (F.z1 + c2*F.z2*ddz)*dz;
    Vec Fz2 = F.z2*dz2;

    //
    // cout << "1: Cycles = " << unitimer2[0].Cycles() << "\t";
    // cout << "2: Cycles = " << unitimer2[1].Cycles() << "\t";
    // cout << "2Fil: Cycles = " << unitimer2[2].Cycles() << " ";
    Vec sx = ( Fx0 + Fx1*c2i + Fx2*c3i );
    Vec sy = ( Fy0 + Fy1*c2i + Fy2*c3i );
    Vec sz = ( Fz0 + Fz1*c2i + Fz2*c3i );

    Vec Sx = ( Fx0*c2i + Fx1*c6i + Fx2*c12i );
    Vec Sy = ( Fy0*c2i + Fy1*c6i + Fy2*c12i );
    Vec Sz = ( Fz0*c2i + Fz1*c6i + Fz2*c12i );

    Vec syz;
    {
        static const Vec::EntryType
            d = 1./360.,
              c00 = 30.*6.*d, c01 = 30.*2.*d,   c02 = 30.*d,
              c10 = 3.*40.*d, c11 = 3.*15.*d,   c12 = 3.*8.*d,
              c20 = 2.*45.*d, c21 = 2.*2.*9.*d, c22 = 2.*2.*5.*d;
        syz = Fy0*( c00*Fz0 + c01*Fz1 + c02*Fz2)
            +   Fy1*( c10*Fz0 + c11*Fz1 + c12*Fz2)
            +   Fy2*( c20*Fz0 + c21*Fz1 + c22*Fz2) ;
    }

    Vec Syz;
    {
        static const Vec::EntryType
            d = 1./2520.,
              c00 = 21.*20.*d, c01 = 21.*5.*d, c02 = 21.*2.*d,
              c10 =  7.*30.*d, c11 =  7.*9.*d, c12 =  7.*4.*d,
              c20 =  2.*63.*d, c21 = 2.*21.*d, c22 = 2.*10.*d;
        Syz = Fy0*( c00*Fz0 + c01*Fz1 + c02*Fz2 )
            +   Fy1*( c10*Fz0 + c11*Fz1 + c12*Fz2 )
            +   Fy2*( c20*Fz0 + c21*Fz1 + c22*Fz2 ) ;
    }

    Vec syy  = sy*sy*c2i;
    Vec syyy = syy*sy*c3i;

    Vec Syy ;
    {
        static const Vec::EntryType
            d= 1./2520., c00= 420.*d, c01= 21.*15.*d, c02= 21.*8.*d,
            c03= 63.*d, c04= 70.*d, c05= 20.*d;
        Syy =  Fy0*(c00*Fy0+c01*Fy1+c02*Fy2) + Fy1*(c03*Fy1+c04*Fy2) + c05*Fy2*Fy2 ;
    }

    Vec Syyy;
    {
        static const Vec::EntryType
            d = 1./181440.,
              c000 =   7560*d, c001 = 9*1008*d, c002 = 5*1008*d,
              c011 = 21*180*d, c012 = 24*180*d, c022 =  7*180*d,
              c111 =    540*d, c112 =    945*d, c122 =    560*d, c222 = 112*d;
        Vec Fy22 = Fy2*Fy2;
        Syyy = Fy0*( Fy0*(c000*Fy0+c001*Fy1+c002*Fy2)+ Fy1*(c011*Fy1+c012*Fy2)+c022*Fy22 )
            +    Fy1*( Fy1*(c111*Fy1+c112*Fy2)+c122*Fy22) + c222*Fy22*Fy2                  ;
    }


    Vec sA1   = sx*xy   + sy*Ay   + sz*y ;
    Vec sA1_x = sx*y - sy*x2 ;
    Vec sA1_y = sx*x + sz ;

    Vec sB1   = sx*Bx   - sy*xy   - sz*x ;
    Vec sB1_x = -sy*y - sz ;
    Vec sB1_y = sx*y2 - sy*x ;

    Vec SA1   = Sx*xy   + Sy*Ay   + Sz*y ;
    Vec SA1_x = Sx*y - Sy*x2 ;
    Vec SA1_y = Sx*x + Sz;

    Vec SB1   = Sx*Bx   - Sy*xy   - Sz*x ;
    Vec SB1_x = -Sy*y - Sz;
    Vec SB1_y = Sx*y2 - Sy*x;


    Vec sA2   = syy*Ayy   + syz*Ayz ;
    Vec sA2_x = syy*Ayy_x - syz*y2 ;
    Vec sA2_y = -syz*x2 ;
    Vec sB2   = syy*Byy   + syz*Byz  ;
    Vec sB2_x = syy*Byy_x + syz*x4 ;
    Vec sB2_y = syy*xx31 ;

    Vec SA2   = Syy*Ayy   + Syz*Ayz ;
    Vec SA2_x = Syy*Ayy_x - Syz*y2 ;
    Vec SA2_y = -Syz*x2 ;
    Vec SB2   = Syy*Byy   + Syz*Byz ;
    Vec SB2_x = Syy*Byy_x + Syz*x4 ;
    Vec SB2_y = Syy*xx31 ;

    Vec sA3   = syyy*Ayyy  ;
    Vec sA3_x = syyy*Ayyy_x;
    Vec sB3   = syyy*Byyy  ;
    Vec sB3_x = syyy*Byyy_x;
    Vec sB3_y = syyy*Byyy_y;


    Vec SA3   = Syyy*Ayyy  ;
    Vec SA3_x = Syyy*Ayyy_x;
    Vec SB3   = Syyy*Byyy  ;
    Vec SB3_x = Syyy*Byyy_x;
    Vec SB3_y = Syyy*Byyy_y;

    Vec ht1 = ht*dz;
    Vec ht2 = ht*ht*dz2;
    Vec ht3 = ht*ht*ht*dz3;
    Vec ht1sA1 = ht1*sA1;
    Vec ht1sB1 = ht1*sB1;
    Vec ht1SA1 = ht1*SA1;
    Vec ht1SB1 = ht1*SB1;
    Vec ht2sA2 = ht2*sA2;
    Vec ht2SA2 = ht2*SA2;
    Vec ht2sB2 = ht2*sB2;
    Vec ht2SB2 = ht2*SB2;
    Vec ht3sA3 = ht3*sA3;
    Vec ht3sB3 = ht3*sB3;
    Vec ht3SA3 = ht3*SA3;
    Vec ht3SB3 = ht3*SB3;

    T[0] = T0 + (x + ht1SA1 + ht2SA2 + ht3SA3)*dz ;
    T[1] = T1 + (y + ht1SB1 + ht2SB2 + ht3SB3)*dz ;
    T[2] = T2 + ht1sA1 + ht2sA2 + ht3sA3;
    T[3] = T3 + ht1sB1 + ht2sB2 + ht3sB3;
    T[5]+= dz;

    Vec ctdz  = c_light*t*dz;
    Vec ctdz2 = c_light*t*dz2;

    Vec dqp = qp - qp0;
    Vec t2i = c1*Vc::reciprocal(t2);// /t2;
    Vec xt2i = x*t2i;
    Vec yt2i = y*t2i;
    Vec tmp0 = ht1SA1 + c2*ht2SA2 + c3*ht3SA3;
    Vec tmp1 = ht1SB1 + c2*ht2SB2 + c3*ht3SB3;
    Vec tmp2 = ht1sA1 + c2*ht2sA2 + c3*ht3sA3;
    Vec tmp3 = ht1sB1 + c2*ht2sB2 + c3*ht3sB3;

    Vec j02 = dz*(c1 + xt2i*tmp0 + ht1*SA1_x + ht2*SA2_x + ht3*SA3_x);
    Vec j12 = dz*(     xt2i*tmp1 + ht1*SB1_x + ht2*SB2_x + ht3*SB3_x);
    Vec j22 =     c1 + xt2i*tmp2 + ht1*sA1_x + ht2*sA2_x + ht3*sA3_x ;
    Vec j32 =          xt2i*tmp3 + ht1*sB1_x + ht2*sB2_x + ht3*sB3_x ;

    Vec j03 = dz*(     yt2i*tmp0 + ht1*SA1_y + ht2*SA2_y );
    Vec j13 = dz*(c1 + yt2i*tmp1 + ht1*SB1_y + ht2*SB2_y + ht3*SB3_y );
    Vec j23 =          yt2i*tmp2 + ht1*sA1_y + ht2*sA2_y  ;
    Vec j33 =     c1 + yt2i*tmp3 + ht1*sB1_y + ht2*sB2_y + ht3*sB3_y ;

    Vec j04 = ctdz2*( SA1 + c2*ht1*SA2 + c3*ht2*SA3 );
    Vec j14 = ctdz2*( SB1 + c2*ht1*SB2 + c3*ht2*SB3 );
    Vec j24 = ctdz *( sA1 + c2*ht1*sA2 + c3*ht2*sA3 );
    Vec j34 = ctdz *( sB1 + c2*ht1*sB2 + c3*ht2*sB3 );

    // extrapolate inverse momentum

    T[0]+=j04*dqp;
    T[1]+=j14*dqp;
    T[2]+=j24*dqp;
    T[3]+=j34*dqp;

    //          covariance matrix transport

    Vec c42 = C.C42, c43 = C.C43;

    Vec cj00 = C.C00 + C.C20*j02 + C.C30*j03 + C.C40*j04;
    //Vec cj10 = C.C10 + C.C21*j02 + C.C31*j03 + C.C41*j04;
    Vec cj20 = C.C20 + C.C22*j02 + C.C32*j03 + c42*j04;
    Vec cj30 = C.C30 + C.C32*j02 + C.C33*j03 + c43*j04;

    Vec cj01 = C.C10 + C.C20*j12 + C.C30*j13 + C.C40*j14;
    Vec cj11 = C.C11 + C.C21*j12 + C.C31*j13 + C.C41*j14;
    Vec cj21 = C.C21 + C.C22*j12 + C.C32*j13 + c42*j14;
    Vec cj31 = C.C31 + C.C32*j12 + C.C33*j13 + c43*j14;

    //Vec cj02 = C.C20*j22 + C.C30*j23 + C.C40*j24;
    //Vec cj12 = C.C21*j22 + C.C31*j23 + C.C41*j24;
    Vec cj22 = C.C22*j22 + C.C32*j23 + c42*j24;
    Vec cj32 = C.C32*j22 + C.C33*j23 + c43*j24;

    //Vec cj03 = C.C20*j32 + C.C30*j33 + C.C40*j34;
    //Vec cj13 = C.C21*j32 + C.C31*j33 + C.C41*j34;
    Vec cj23 = C.C22*j32 + C.C32*j33 + c42*j34;
    Vec cj33 = C.C32*j32 + C.C33*j33 + c43*j34;

    C.C40+= c42*j02 + c43*j03 + C.C44*j04; // cj40
    C.C41+= c42*j12 + c43*j13 + C.C44*j14; // cj41
    C.C42 = c42*j22 + c43*j23 + C.C44*j24; // cj42
    C.C43 = c42*j32 + c43*j33 + C.C44*j34; // cj43

    C.C00 = cj00 + j02*cj20 + j03*cj30 + j04*C.C40;
    C.C10 = cj01 + j02*cj21 + j03*cj31 + j04*C.C41;
    C.C11 = cj11 + j12*cj21 + j13*cj31 + j14*C.C41;

    C.C20 = j22*cj20 + j23*cj30 + j24*C.C40 ;
    C.C30 = j32*cj20 + j33*cj30 + j34*C.C40 ;
    C.C21 = j22*cj21 + j23*cj31 + j24*C.C41 ;
    C.C31 = j32*cj21 + j33*cj31 + j34*C.C41 ;
    C.C22 = j22*cj22 + j23*cj32 + j24*C.C42 ;
    C.C32 = j32*cj22 + j33*cj32 + j34*C.C42 ;
    C.C33 = j32*cj23 + j33*cj33 + j34*C.C43 ;

}

struct HitInfo{
    Vec cos_phi, sin_phi, sigma2, sigma216;
};

/**
 * \param info Information about the coordinate system of the measurement
 * \param u Is a measurement that we want to add - Strip coordinate (may be x or y)
 * \param w Mask which entries of u to use (just 1 or 0)
 */
inline void Filter( TrackV &track, HitInfo &info, Vec &u, Vec &w )
{
    // convert input
    Vec *T = track.T;  // track paramenters: x, y, tx, ty, qp, z
    CovV &C = track.C; // covariance matrix

    Vec weightMatrix, zeta, zetawi, HCH; // model of the measurement * cov-matrix * transposed model of the measurement
    Vec F0, F1, F2, F3, F4;
    Vec gain1, gain2, gain3, gain4;

    // residual: difference to new measurement
    residual = info.cos_phi*T[0] + info.sin_phi*T[1] - u; // cos(phi) * x + sin(phi) * y - u
    // F = CH'

    F0 = info.cos_phi*C.C00 + info.sin_phi*C.C10;
    F1 = info.cos_phi*C.C10 + info.sin_phi*C.C11;
    F2 = info.cos_phi*C.C20 + info.sin_phi*C.C21;
    F3 = info.cos_phi*C.C30 + info.sin_phi*C.C31;
    F4 = info.cos_phi*C.C40 + info.sin_phi*C.C41;

    HCH = ( F0*info.cos_phi +F1*info.sin_phi );
    float_m initialised = HCH < info.sigma216; // fix roundoff errors: if HCH is too small

    weightMatrix = w * Vc::reciprocal(info.sigma2 +HCH); // matrix S (1x1)
    Vec tmp = Vec::Zero();
    tmp(initialised) = info.sigma2;
    residual_S = w * residual * Vc::reciprocal(tmp + HCH); // residual * S
    tmp.setZero();
    tmp(initialised) = residual * residual_S;
    track.Chi2 += tmp; // residual * S * residual

    track.NDF  += w;

    // K0 = F0*weightMatrix
    gain1 = F1*weightMatrix;
    gain2 = F2*weightMatrix;
    gain3 = F3*weightMatrix;
    gain4 = F4*weightMatrix;

    T[0]-= F0*residual_S;
    T[1]-= F1*residual_S;
    T[2]-= F2*residual_S;
    T[3]-= F3*residual_S;
    T[4]-= F4*residual_S;

    C.C00-= F0*F0*weightMatrix;
    C.C10-= gain1*F0;
    C.C11-= gain1*F1;
    C.C20-= K2*F0;
    C.C21-= K2*F1;
    C.C22-= K2*F2;
    C.C30-= K3*F0;
    C.C31-= K3*F1;
    C.C32-= K3*F2;
    C.C33-= K3*F3;
    C.C40-= K4*F0;
    C.C41-= K4*F1;
    C.C42-= K4*F2;
    C.C43-= K4*F3;
    C.C44-= K4*F4;
}

inline void FilterFirst( TrackV &track, HitV &hit, Station &st )
{

    CovV &C = track.C;
    Vec w1 = ONE-hit.w;
    Vec sigma2 = hit.w*st.Sigma2 + w1*INF;
    // initialize covariance matrix
    C.C00= sigma2;
    C.C10= ZERO;      C.C11= sigma2;
    C.C20= ZERO;      C.C21= ZERO;      C.C22= INF;
    C.C30= ZERO;      C.C31= ZERO;      C.C32= ZERO; C.C33= INF;
    C.C40= ZERO;      C.C41= ZERO;      C.C42= ZERO; C.C43= ZERO; C.C44= INF;

    track.T[0] = hit.w*hit.x + w1*track.T[0];
    track.T[1] = hit.w*hit.y + w1*track.T[1];
    track.NDF = -3.0;
    track.Chi2 = ZERO;
}

inline void AddMaterial( TrackV &track, Station &st, Vec &qp0 )
{
    static const Vec::EntryType mass2 = 0.1396*0.1396;

    Vec tx = track.T[2];
    Vec ty = track.T[3];
    Vec txtx = tx*tx;
    Vec tyty = ty*ty;
    Vec txtx1 = txtx + ONE;
    Vec h = txtx + tyty;
    Vec t = sqrt(txtx1 + tyty);
    Vec h2 = h*h;
    Vec qp0t = qp0*t;

    static const Vec::EntryType c1=0.0136, c2=c1*0.038, c3=c2*0.5, c4=-c3/2.0, c5=c3/3.0, c6=-c3/4.0;

    Vec s0 = (c1+c2*st.logRadThick + c3*h + h2*(c4 + c5*h +c6*h2) )*qp0t;

    Vec a = (ONE+mass2*qp0*qp0t)*st.RadThick*s0*s0;

    CovV &C = track.C;

    C.C22 += txtx1*a;
    C.C32 += tx*ty*a; C.C33 += (ONE+tyty)*a;
}

inline void GuessVec( TrackV &t, Station *vStations, int NStations )
{
    Vec *T = t.T;

    Int_t NHits = NStations;

    Vec A0, A1=ZERO, A2=ZERO, A3=ZERO, A4=ZERO, A5=ZERO, a0, a1=ZERO, a2=ZERO,
        b0, b1=ZERO, b2=ZERO;
    Vec z0, x, y, z, S, w, wz, wS;

    Int_t i=NHits-1;
    z0 = vStations[i].zhit;
    HitV *hlst = &(t.vHits[i]);
    w = hlst->w;
    A0 = w;
    a0 = w*hlst->x;
    b0 = w*hlst->y;
    HitV *h = t.vHits;
    Station *st = vStations;
    for( ; h!=hlst; h++, st++ ){
        x = h->x;
        y = h->y;
        w = h->w;
        z = st->zhit - z0;
        S = st->Sy;
        wz = w*z;
        wS = w*S;
        A0+=w;
        A1+=wz;  A2+=wz*z;
        A3+=wS;  A4+=wS*z; A5+=wS*S;
        a0+=w*x; a1+=wz*x; a2+=wS*x;
        b0+=w*y; b1+=wz*y; b2+=wS*y;
    }

    Vec A3A3 = A3*A3;
    Vec A3A4 = A3*A4;
    Vec A1A5 = A1*A5;
    Vec A2A5 = A2*A5;
    Vec A4A4 = A4*A4;

    Vec det = Vc::reciprocal(-A2*A3A3 + A1*( A3A4+A3A4 - A1A5) + A0*(A2A5-A4A4));
    Vec Ai0 = ( -A4A4 + A2A5 );
    Vec Ai1 = (  A3A4 - A1A5 );
    Vec Ai2 = ( -A3A3 + A0*A5 );
    Vec Ai3 = ( -A2*A3 + A1*A4 );
    Vec Ai4 = (  A1*A3 - A0*A4 );
    Vec Ai5 = ( -A1*A1 + A0*A2 );

    Vec L, L1;
    T[0] = (Ai0*a0 + Ai1*a1 + Ai3*a2)*det;
    T[2] = (Ai1*a0 + Ai2*a1 + Ai4*a2)*det;
    Vec txtx1 = T[2]*T[2]+1.f;
    L    = (Ai3*a0 + Ai4*a1 + Ai5*a2)*det *Vc::reciprocal(txtx1);
    L1 = L*T[2];
    A1 = A1 + A3*L1;
    A2 = A2 + ( A4 + A4 + A5*L1 )*L1;
    b1+= b2 * L1;
    det = Vc::reciprocal(-A1*A1+A0*A2);

    T[1] = (  A2*b0 - A1*b1 )*det;
    T[3] = ( -A1*b0 + A0*b1 )*det;
    T[4] = -L*c_light_i*rsqrt(txtx1 +T[3]*T[3]);
    T[5] = z0;
}

inline void Fit( TrackV &t, Station vStations[], int NStations )
{
    HitInfo Xinfo, Yinfo;
    static const Vec::EntryType c16 = 16.;
    Xinfo.cos_phi = ONE;
    Xinfo.sin_phi = ZERO;
    Xinfo.sigma2  = vStations[0].Sigma2;
    Xinfo.sigma216 = Xinfo.sigma2*c16;
    Yinfo.cos_phi = ZERO;
    Yinfo.sin_phi = ONE;
    Yinfo.sigma2  = Xinfo.sigma2;
    Yinfo.sigma216 = Xinfo.sigma216;
    // upstream

    GuessVec( t, vStations,NStations );

    // downstream

    FieldRegion f;
    Vec z0,z1,z2, dz;
    FieldVector H0, H1, H2;

    Vec qp0 = t.T[4];
    Int_t i= NStations-1;
    HitV *h = &t.vHits[i];

    FilterFirst( t, *h, vStations[i] );
    AddMaterial( t, vStations[ i ], qp0 );

    z1 = vStations[ i ].z;
    vStations[i].Map.GetField(t.T[0],t.T[1], H1);
    H1.Combine( h->H, h->w );

    z2 = vStations[ i-2 ].z;
    dz = z2-z1;
    vStations[ i-2 ].Map.GetField(t.T[0]+t.T[2]*dz,t.T[1]+t.T[3]*dz,H2);
    h = &t.vHits[i-2];
    H2.Combine( h->H, h->w );

    for( --i; i>=0; i-- ){
        h = &t.vHits[i];
        Station &st = vStations[i];
        z0 = st.z;
        dz = (z1-z0);
        st.Map.GetField(t.T[0]-t.T[2]*dz,t.T[1]-t.T[3]*dz,H0);
        H0.Combine( h->H, h->w );
        f.Set( H0, z0, H1, z1, H2, z2);

        ExtrapolateALight( t.T, t.C, st.zhit, qp0, f );
        AddMaterial( t, st, qp0 );
        Filter( t, Xinfo, h->x, h->w );
        Filter( t, Yinfo, h->y, h->w );
        H2 = H1;
        z2 = z1;
        H1 = H0;
        z1 = z0;
    }
}

int tasks = 80;  /* #threads <= #tasks */

class KalmanFilter
{
    Track vTracks[MaxNTracks];
    MCTrack vMCTracks[MaxNTracks];
    Station* vStations;
    int NStations;
    int NTracks;
    int NTracksV;

    FieldRegion field0;

    void ReadInput()
    {
        std::fstream FileGeo, FileTracks;

        FileGeo.open("geo.dat", std::ios::in );
        FileTracks.open("tracks.dat", std::ios::in );
        {
            FieldVector H[3];
            Vec Hz[3];
            for( int i=0; i<3; i++) {
                float Bx, By, Bz, z;
                FileGeo >> z >> Bx >> By >> Bz;
                Hz[i] = z;
                H[i].X = Bx;
                H[i].Y = By;
                H[i].Z = Bz;
            }
            field0.Set(H[0],Hz[0], H[1],Hz[1], H[2],Hz[2] );
        }
        FileGeo >> NStations;
        for( int i=0; i<NStations; i++ ){
            int ist;
            FileGeo >> ist;
            if( ist!=i ) break;
            Station &st = vStations[i];
            FileGeo >> st.z >> st.thick >> st.RL;
            st.zhit = st.z - st.thick * 0.5f;
            st.RadThick = st.thick/st.RL;
            st.logRadThick = log(st.RadThick);
            int N=0;
            FileGeo >> N;
            for( int j=0; j<N; j++ ) FileGeo >> st.Map.X[j];
            for( int j=0; j<N; j++ ) FileGeo >> st.Map.Y[j];
            for( int j=0; j<N; j++ ) FileGeo >> st.Map.Z[j];
        }
        {
            Vec z0  = vStations[NStations-1].z;
            Vec sy(Vc::Zero);
            Vec Sy(Vc::Zero);
            for( int i=NStations-1; i>=0; i-- ){
                Station &st = vStations[i];
                Vec dz = st.z-z0;
                Vec Hy = vStations[i].Map.Y[0];
                Sy += dz*sy + dz*dz*Hy/2.f;
                sy += dz*Hy;
                st.Sy = Sy;
                z0 = st.z;
            }
        }

        FileGeo.close();

        NTracks = 0;
        while( !FileTracks.eof() ){

            int itr;
            FileTracks>>itr;
            if( NTracks>=MaxNTracks ) break;

            Track &t = vTracks[NTracks];
            MCTrack &mc = vMCTracks[NTracks];
            FileTracks >> mc.MC_x   >> mc.MC_y  >> mc.MC_z
                >> mc.MC_px >> mc.MC_py >> mc.MC_pz >> mc.MC_q
                >> t.NHits;
            for( int i=0; i<t.NHits; i++ ){
                int ist;
                FileTracks >> ist;
                t.vHits[i].ista = ist;
                FileTracks >> t.vHits[i].x >> t.vHits[i].y;
            }
            if( t.NHits==NStations )   NTracks++;
        }
        FileTracks.close();

        NTracksV = NTracks/vecN;
        NTracks =  NTracksV*vecN;
    }

    void WriteOutput(){

        std::fstream Out, Diff;

        Out.open("fit.dat", std::ios::out );
        Diff.open("fitdiff.dat", std::ios::out );

        for( int it=0, itt=0; itt<NTracks; itt++ ){
            Track &t = vTracks[itt];
            MCTrack &mc = vMCTracks[itt];

            // convert matrix
            double C[15];
            {
                Vec::EntryType *tC = &t.C[0];
                for( int i=0, n=0; i<5; i++)
                    for( int j=0; j<=i; j++, n++ ){
                        C[n]=0;
                        C[n] = tC[n];
                    }
            }

            bool ok = 1;
            for( int i=0; i<6; i++ ){
                ok = ok && finite(t.T[i]);
            }
            for( int i=0; i<15; i++ ) ok = ok && finite(C[i]);

            if(!ok){ std::cout<<" infinite\n"; continue; }

            Out << it << "\n   "
                << " " << mc.MC_x  << " " << mc.MC_y  << " " << mc.MC_z
                << " " << mc.MC_px << " " << mc.MC_py << " " << mc.MC_pz
                << " " << mc.MC_q;
            Out<<"\n   ";
            for( int i=0; i<6; i++ ) Out<< " " <<t.T[i];
            Out << "\n   ";
            for( int i=0; i<15; i++ ) Out<< " " <<C[i];
            Out << '\n';

            float tmc[6] = {
                mc.MC_x,
                mc.MC_y,
                mc.MC_px / mc.MC_pz,
                mc.MC_py / mc.MC_pz,
                mc.MC_q / std::sqrt(mc.MC_px*mc.MC_px+mc.MC_py*mc.MC_py+mc.MC_pz*mc.MC_pz), mc.MC_z};
            Diff << it << "\n   ";
            for( int i=0; i<6; i++ ) Diff<< " " <<t.T[i]-tmc[i];
            Diff << "\n   ";
            for( int i=0; i<15; i++ ) Diff<< " " <<C[i];
            Diff << '\n';
            it++;
        }
        Out.close();
        Diff.close();
    }

    void FitTracksV(){

        double TimeTable[Ntimes];

        TrackV *TracksV = new TrackV[MaxNTracks / vecN + 1];
        Vec *Z0      = new Vec[MaxNTracks/vecN+1];

        Vec::Memory Z0mem;
#ifndef MUTE
        cout<<"Prepare data..."<<endl;
#endif
        TimeStampCounter timer1;

        for( int iV=0; iV<NTracksV; iV++ ){ // loop on set of 4 tracks
#ifndef MUTE
            if( iV*vecN%100==0 ) cout<<iV*vecN<<endl;
#endif
            TrackV &t = TracksV[iV];
            for( int ist=0; ist<NStations; ist++ ){
                HitV &h = t.vHits[ist];

                /* some obsure debug here?
                if((((unsigned long)(&h.x)) & 0x3f) != 0) {
                    std::cout << iV << ", " << ist << ": " << std::hex << (unsigned long)&h.x << std::endl;
                    std::cout << std::hex << (unsigned long)&h.y << std::endl;
                    std::cout << std::hex << (unsigned long)&h << std::endl;
                }
                */

                h.x = 0.;
                h.y = 0.;
                h.w = 0.;
                h.H.X = 0.;
                h.H.Y = 0.;
                h.H.Z = 0.;
            }

            Vc::Memory<Vec> hxmem(NStations * vecN), hymem(NStations * vecN), hwmem(NStations * vecN);
            for( int it=0; it<vecN; it++ ){
                Track &ts = vTracks[iV*vecN+it];

                Z0mem[it] = vMCTracks[iV*vecN+it].MC_z;

                for( int ista=0, ih=0; ista<NStations; ista++ ){
                    Hit &hs = ts.vHits[ih];
                    if (hs.ista != ista) continue;
                    ih++;

                    hxmem[ista * vecN + it] = hs.x;
                    hymem[ista * vecN + it] = hs.y;
                    hwmem[ista * vecN + it] = 1.;
                }

            }
            for( int ista=0; ista<NStations; ista++ ){
                Vec hxtemp( hxmem[ista] );
                Vec hytemp( hymem[ista] );
                Vec hwtemp( hwmem[ista] );
                t.vHits[ista].x = hxtemp;
                t.vHits[ista].y = hytemp;
                t.vHits[ista].w = hwtemp;
            }


            Vec Z0temp(Z0mem);
            Z0[iV] = Z0temp;


            if (0){    // output for check
                std::cout << "track " << iV << "  ";
                for( int ista=0; ista<NStations; ista++ )
                    std::cout << t.vHits[ista].x << " ";
                std::cout << '\n';
            }


            for( int ist=0; ist<NStations; ist++ ){
                HitV &h = t.vHits[ist];
                vStations[ist].Map.GetField(h.x, h.y, h.H);
            }
        }
        timer1.Stop();
#ifndef MUTE
        cout<<"Start fit..."<<endl;
#endif
        TimeStampCounter timer;
        TimeStampCounter timer2;
        //   TimeStampCounter timer_test;
        timer.Start();
        for( int times=0; times<Ntimes; times++){
            timer2.Start();
            int ifit;
            int iV;
    //#pragma omp parallel num_threads(tasks)
            {
    //#pragma omp for
                for( iV=0; iV<NTracksV; iV++ ){ // loop on set of 4 tracks
                    // timer_test.Start();
                    for( ifit=0; ifit<NFits; ifit++){
                        Fit( TracksV[iV], vStations, NStations );
                    }
                    // timer_test.Stop();
                    // cout<<"test time = "<<timer_test.RealTime()*1.e6<<" [us]"<<endl;
                }
            }
            timer2.Stop();
            TimeTable[times]=timer2.Cycles();
        }
        timer.Stop();


        for( int iV=0; iV<NTracksV; iV++ ){ // loop on set of 4 tracks
            TrackV &t = TracksV[iV];
            ExtrapolateALight( t.T, t.C, Z0[iV], TracksV[iV].T[4], field0 );
        }

        double realtime=0;
        std::fstream TimeFile;
        TimeFile.open("time.dat", std::ios::out );
        for( int times=0; times<Ntimes; times++ ){
            TimeFile << TimeTable[times]*1.e6/(NTracks*NFits)<<std::endl;
            realtime += TimeTable[times]*1.e6/(NTracks*NFits);
        }
        TimeFile.close();
        realtime /= Ntimes;

#ifndef MUTE
        std::cout<<"Preparation time/track = "<<timer1.Cycles()/NTracks/NFits<<" [us]\n";
        std::cout<<"CPU  fit time/track = "<<timer.Cycles()/(NTracks*NFits)/Ntimes<<" [us]\n";
        std::cout<<"Real fit time/track = "<<realtime <<" [us]\n";
        std::cout<<"Total fit time = "<<timer.Cycles()<<" [sec]\n";
#else
        std::cout<<"Prep[us], CPU fit/tr[us], Real fit/tr[us], CPU[sec], Real[sec] = "<<timer1.Cycles()/NTracks/NFits<<"\t";
        std::cout<<timer.Cycles()/(NTracks*NFits)/Ntimes<<"\t";
        std::cout<<realtime <<"\t";
        std::cout<<timer.Cycles()<<std::endl;
#endif

        for( int iV=0; iV<NTracksV; iV++ ){ // loop on set of 4 tracks
            TrackV &t = TracksV[iV];
            for( int it=0; it<vecN; it++ ){
                Track &ts = vTracks[iV*vecN+it];
                Vec *C = &t.C.C00;
                Vec::EntryType *sC = &ts.C[0];
                for( int i=0; i<6; i++ ) ts.T[i] = t.T[i][it];
#ifdef SQRT_FILTER
                for( int i=0,n=0; i<5; i++ )
                    for( int j=0; j<=i; j++,n++){
                        sC[n]=0;
                        for(int k=0;k<5;k++) sC[n]+= C[i*5+k][it]*C[j*5+k][it];
                    }
#else
                for( int i=0; i<15; i++ ) sC[i] = C[i][it];
#endif // SQRT_FILTER
            }
        }

        delete [] Z0;
        delete [] TracksV;
    }
public:
    KalmanFilter()
        : vStations(new Station[8]),
        NStations(0), NTracks(0), NTracksV(0)
    {
        ReadInput();
        FitTracksV();
        WriteOutput();
    }

    ~KalmanFilter()
    {
        delete[] vStations;
    }
};

int main()
{
    KalmanFilter *f = new KalmanFilter;
    delete f;
    return 0;
}
