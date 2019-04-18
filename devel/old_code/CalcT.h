#ifndef CalcT_H
#define CalcT_H


// In the below functions, the UForm can be either Schneider or Crittenden.
//
// Schneider's Map: U = 9/Pi (1-r^2) (1/3-r^2)
//                  Q = 6/Pi r^2 (1-r^2)
//
// Crittenden's Map: U = 1/2Pi (1-r^2/2) exp(-r^2/2)
//                   Q = 1/4Pi r^2 exp(-r^2/2)

enum UForm { Schneider, Crittenden };

void CalcT(
    UForm uform, double s, std::complex<double> t, 
    std::complex<double>* T0, std::complex<double>* T1,
    std::complex<double>* T2, std::complex<double>* T3,
    double R1, double R2=0., double R3=0.);

double Tplus(UForm uform, double x);
double Tminus(UForm uform, double x);
double Tcross(UForm uform, double x);
double Splus(double r);
double Sminus(double r);

#endif
