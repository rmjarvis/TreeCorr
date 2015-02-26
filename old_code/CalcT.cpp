
#include <cmath>
#include "dbg.h"
#include "Form.h"
#include "CalcT.h"

template <class T> inline T SQR(T x) { return x*x; }

double norm(double x) { return x*x; }
double norm(const std::complex<double>& x) { return std::norm(x); }


// In the below functions, the UForm can be either Schneider or Crittenden.
//
// Schneider's Map: U = 9/Pi (1-r^2) (1/3-r^2)
//                  Q = 6/Pi r^2 (1-r^2)
//
// Crittenden's Map: U = 1/2Pi (1-r^2/2) exp(-r^2/2)
//                   Q = 1/4Pi r^2 exp(-r^2/2)



void CalcT(
    UForm uform, double s, std::complex<double> t, 
    std::complex<double>* T0, std::complex<double>* T1,
    std::complex<double>* T2, std::complex<double>* T3,
    double k1, double k2, double k3)
// Calculate T0, T1, T2, T3 according to JBJ formula for k2=k3=1
// If k2 or k3 != 1, then return versions of these for SKL's 
// modification: <M^3>(t,k2 t,k3 t) and <M^2M*>(t, k2 t, k3 t)
{
    // I don't know the functional form for this/ function with Schneider's Map.
    if (uform == Schneider) 
        myerror("Sorry, the Schneider aperture mass cannot be used with m3");

    std::complex<double> q1 = (s+t)/3.;
    std::complex<double> q2 = q1-s;
    std::complex<double> q3 = q1-t;
    std::complex<double> q1q2 = q1*q2;
    std::complex<double> q1q3 = q1*q3;
    std::complex<double> q2q3 = q2*q3;
    std::complex<double> q1q2q3 = q1*q2q3;

    if (k2 ==1. && k3 == 1.) {
        double ksq = k1*k1; 
        double kto4 = ksq*ksq;
        double kto6 = ksq*kto4; 
        Assert(ksq != 0.);
        double expfactor = -exp(-(norm(q1)+norm(q2)+norm(q3))/2./ksq);
        *T0 = SQR(conj(q1q2q3))*expfactor/kto6/24.; 
        *T1 = ( SQR(q1*conj(q2q3))/kto6/24. - q1*conj(q1q2q3)/kto4/9. + 
                (SQR(conj(q1))+2.*conj(q2q3))/ksq/27. )*expfactor;
        *T2 = ( SQR(q2*conj(q1q3))/kto6/24. - q2*conj(q1q2q3)/kto4/9. +
                (SQR(conj(q2))+2.*conj(q1q3))/ksq/27. )*expfactor;
        *T3 = ( SQR(q3*conj(q1q2))/kto6/24. - q3*conj(q1q2q3)/kto4/9. +
                (SQR(conj(q3))+2.*conj(q1q2))/ksq/27. )*expfactor; 
    } else {
        std::complex<double> s1 = t-s; // = q2-q3
        std::complex<double> s2 = -t; // = q3-q1
        double s3 = s; // = q1-q2

        double k1sq = k1*k1;
        double k2sq = k2*k2;
        double k3sq = k3*k3;
        double Theta2 = sqrt((k1sq*k2sq + k1sq*k3sq + k2sq*k3sq)/3.);
        Assert(Theta2 != 0.);
        k1sq /= Theta2;
        k2sq /= Theta2;
        k3sq /= Theta2;
        double Theta4 = Theta2*Theta2;
        double Theta6 = Theta4*Theta2;

        double S = k1sq*k2sq*k3sq/Theta4;
        double Z1 = ( k1sq * norm(s1) + k2sq * norm(s2) + k3sq * norm(s3) )/6./Theta2;
        double Z2 = ( k1sq * norm(s2) + k2sq * norm(s3) + k3sq * norm(s1) )/6./Theta2;
        double Z3 = ( k1sq * norm(s3) + k2sq * norm(s1) + k3sq * norm(s2) )/6./Theta2;
        std::complex<double> q1f1 = (k2sq + k3sq)/2.*q1 + (k2sq-k3sq)/6.*s1;
        std::complex<double> q2f1 = (k2sq + k3sq)/2.*q2 + (k2sq-k3sq)/6.*s2;
        std::complex<double> q3f1 = (k2sq + k3sq)/2.*q3 + (k2sq-k3sq)/6.*s3;
        std::complex<double> q1f2 = (k3sq + k1sq)/2.*q1 + (k3sq-k1sq)/6.*s1;
        std::complex<double> q2f2 = (k3sq + k1sq)/2.*q2 + (k3sq-k1sq)/6.*s2;
        std::complex<double> q3f2 = (k3sq + k1sq)/2.*q3 + (k3sq-k1sq)/6.*s3;
        std::complex<double> q1f3 = (k1sq + k2sq)/2.*q1 + (k1sq-k2sq)/6.*s1;
        std::complex<double> q2f3 = (k1sq + k2sq)/2.*q2 + (k1sq-k2sq)/6.*s2;
        std::complex<double> q3f3 = (k1sq + k2sq)/2.*q3 + (k1sq-k2sq)/6.*s3;
        std::complex<double> q1g1 = (k2sq * k3sq)*q1 - k1sq*(k2sq-k3sq)/3.*s1;
        std::complex<double> q2g1 = (k2sq * k3sq)*q2 - k1sq*(k2sq-k3sq)/3.*s2;
        std::complex<double> q3g1 = (k2sq * k3sq)*q3 - k1sq*(k2sq-k3sq)/3.*s3;

        *T0 = -S*exp(-Z1)*SQR(conj(q1f1*q2f2*q3f3))/Theta6/24.;
        *T1 = -S*exp(-Z1)*(
            SQR(q1f1*conj(q2f2*q3f3))/Theta6/24. 
            - q1f1*conj(q1g1*q2f2*q3f3)/Theta4/9.
            + (SQR(conj(q1g1)) + 2.*k2sq*k3sq*conj(q2f2*q3f3))/Theta2/27.);
        *T2 = -S*exp(-Z2)*(
            SQR(q2f1*conj(q3f2*q1f3))/Theta6/24. 
            - q2f1*conj(q2g1*q3f2*q1f3)/Theta4/9.
            + (SQR(conj(q2g1)) + 2.*k2sq*k3sq*conj(q3f2*q1f3))/Theta2/27.);
        *T3 = -S*exp(-Z3)*(
            SQR(q3f1*conj(q1f2*q2f3))/Theta6/24. 
            - q3f1*conj(q3g1*q1f2*q2f3)/Theta4/9.
            + (SQR(conj(q3g1)) + 2.*k2sq*k3sq*conj(q1f2*q2f3))/Theta2/27.);
    }
}

double Tplus(UForm uform, double r) 
{
    if (uform == Schneider) {
        // T+ = 12/5Pi (2-15r^2) arccos(r/2)
        //      - 1/(100Pi) r sqrt(4-r^2) (9r^8 - 132 r^6 + 754r^4 - 2320r^2 - 120)
        // see Schneider et al 2001: astroph 0112441 for more about these four
        // functions: T+,T-,S+,S-
        Assert(r>=0.);
        if (r>=2.) return 0.;
        double rsq = r*r;
        double temp = ((((9.*rsq-132.)*rsq+754)*rsq-2320.)*rsq-120.);
        temp *= r*sqrt(4.-rsq)/100.;
        return (2.4*acos(r/2.)*(2.-15.*rsq) - temp)/M_PI;
    } else {
        // T+ = (r^4 - 16r^2 + 32)/128  exp(-r^2/4)
        double rsq = r*r;
        return (rsq*rsq - 16.*rsq + 32.)/128. *exp(-rsq/4.);
    }
}

double Tminus(UForm uform, double r)
{
    if (uform == Schneider) {
        // T- = 3/70Pi r^3 (4-r^2)^(7/2)
        Assert(r>=0.);
        if (r>=2.) return 0.;
        return 3.*pow(r,3.)*pow(4.-r*r,3.5)/70./M_PI;
    } else {
        // T- = r^4/128  exp(-r^2/4)
        double rsq = r*r;
        return rsq*rsq/128. *exp(-rsq/4.);
    }
}

double Tcross(UForm uform, double r)
{
    if (uform == Schneider) {
        // Tx = 18/Pi r^2 arccos(r/2)
        //      + 3/40Pi r^3 sqrt(4-r^2) (r^6 - 14r^4 + 74r^2 - 196)
        Assert(r>=0.);
        if (r>=2.) return 0.;
        double rsq = r*r;
        double temp = (((rsq-14.)*rsq+74.)*rsq)-196.;
        temp *= 0.075 * r*rsq * sqrt(4.-rsq);
        return (18*rsq*acos(r/2.) + temp)/M_PI;
    } else {
        // Tx = r^2/128 (12-r^2) exp(-r^2/4)
        double rsq = r*r;
        return rsq*(12.-rsq)/128. * exp(-rsq/4.);
    }
}

double Splus(double r)
    // S+ = 1/Pi * (4*arccos(r/2) - r sqrt(4-r^2) )
{
    Assert(r>=0.);
    if (r>=2.) return 0.;
    double rsq = r*r;
    return (4.*acos(r/2.) - r*sqrt(4.-rsq))/M_PI;
}

double Sminus(double r)
    // S- = (r<=2):  1/(Pi r^4) * ( r sqrt(4-r^2) (6-r^2) - 8(3-r^2) arcsin(r/2) )
    //      (r>=2):  4(r^2-3)/r^4
{
    Assert(r>=0.);
    double rsq = SQR(r);
    if (r<2) {
        double sm = r*sqrt(4.-rsq)*(6.-rsq) - 8.*(3.-rsq)*asin(r/2.);
        sm /= M_PI * SQR(rsq);
        return sm;
    } else {
        return 4.*(rsq-3.)/SQR(rsq);
    }
}

