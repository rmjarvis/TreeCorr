
#include <iostream>
#include <fstream>
#include <complex>
#include <cstdlib>
#include <cassert>

int main()
{
    std::ofstream fout1("test1.dat");  // Flat Sky
    std::ofstream fout2("test2.dat");  // Spherical near equator
    std::ofstream fout3("test3.dat");  // Spherical around N pole.
    std::ofstream fout4("test4.dat");  // Test1 with random offsets
    std::ofstream fout4c("test4_centers.dat");  // Corresponding centers for test4.dat
    fout1.precision(12);
    fout2.precision(12);
    fout3.precision(12);
    fout4.precision(12);
    fout4c.precision(12);

    double ra0 = 0.; // RA, Dec of center for test2.
    double dec0 = 80.;

    const long ngal = 100000;
    const double rscale = 1.; // Scale radius in degrees.
    const double rmax = 5.; // In units of rscale

    // The functional form imprinted is purely radial:
    // gamma_t(r) = gamma0/r
    // kappa(r) = kappa0/r
    // n(r) ~ exp(-r^2/2)
    // where r is in units of rscale
    const double gamma0 = 1.e-3;
    const double kappa0 = 3.e-3;
    // Make sure gamma_t < 0.5
    const double rmin = gamma0/0.5;

    srand(1234); // To make it deterministic.
    // Normally for the Box-Muller Transformation, one would restrict rsq < 1
    // But we then want to clip the resulting distribution at rmax, so
    // rsq * fac^2 < rmax^2
    // rsq * (-2.*log(rsq)/rsq) < rmax^2
    // -2. * log(rsq) < rmax^2
    // rsq > exp(-rmax^2/2)
    const double rsqmin = exp(-rmax*rmax/2.);
    const double rsqmax = exp(-rmin*rmin/2.);
    //std::cout<<"rsq min/max = "<<rsqmin<<','<<rsqmax<<std::endl;

    double xdeg_min=0., xdeg_max=0., var_xdeg=0.;
    double ydeg_min=0., ydeg_max=0., var_ydeg=0.;

    for (long n=0; n<ngal; ++n) {
        double x,y,rsq;
        do {
            x = double(rand()) / RAND_MAX; // Random number from 0..1
            y = double(rand()) / RAND_MAX;
            //std::cout<<"x,y = "<<x<<','<<y<<std::endl;
            x = 2.*x-1.; // Now from -1..1
            y = 2.*y-1.;
            //std::cout<<"x,y => "<<x<<','<<y<<std::endl;
            rsq = x*x+y*y;
            //std::cout<<"rsq = "<<rsq<<std::endl;
        } while (rsq <= rsqmin || rsq >= rsqmax);
        // Use Box-Muller Transformation to convert to Gaussian distribution in x,y.
        double fac = sqrt(-2.*log(rsq)/rsq);
        x *= fac;
        y *= fac;

        double r = fac*sqrt(rsq);
        double theta = atan2(y,x);
        double g = gamma0 / r;
        double k = kappa0 / r;
        assert(g < 0.5);

        double xdeg = x*rscale;
        double ydeg = y*rscale;
        double rdeg = r*rscale;

        // Do some sanity checks:
        if (xdeg < xdeg_min) xdeg_min = xdeg;
        if (xdeg > xdeg_max) xdeg_max = xdeg;
        if (ydeg < ydeg_min) ydeg_min = ydeg;
        if (ydeg > ydeg_max) ydeg_max = ydeg;
        var_xdeg += xdeg*xdeg;
        var_ydeg += ydeg*ydeg;

        //
        // Flat sky:
        // 
        double g1 = -g * cos(2.*theta);
        double g2 = -g * sin(2.*theta);
        fout1 << xdeg <<"  "<< ydeg <<"  "<< g1 <<"  "<< g2 <<"  "<< k <<std::endl;


        // With offsets in position:
        double dx = 2.*double(rand()) / RAND_MAX - 1.; // Random number from -1..1
        double dy = 2.*double(rand()) / RAND_MAX - 1.;
        dx *= rscale;
        dy *= rscale;
        fout4 << xdeg + dx <<"  "<< ydeg + dy <<"  "<< g1 <<"  "<< g2 <<"  "<< k <<std::endl;
        fout4c << dx <<"  "<< dy <<"  1. "<<std::endl;

        // 
        // Spherical near equator:
        //

        // Use spherical triangle with A = point, B = (ra0,dec0), C = N. pole
        // a = Pi/2-dec0
        // c = 2*atan(r/2)
        // B = Pi/2 - theta

        // Solve the rest of the triangle with spherical trig:
        double c = 2.*atan( (rdeg*M_PI/180.) / 2.);
        double a = M_PI/2. - (dec0*M_PI/180.);
        double B = x > 0 ? M_PI/2. - theta : theta - M_PI/2.;
        if (B < 0) B += 2.*M_PI;
        if (B > 2.*M_PI) B -= 2.*M_PI;
        double cosb = cos(a)*cos(c) + sin(a)*sin(c)*cos(B);
        double b = std::abs(cosb) < 1. ? acos(cosb) : 0.;
        double cosA = (cos(a) - cos(b)*cos(c)) / (sin(b)*sin(c));
        double A = std::abs(cosA) < 1. ? acos(cosA) : 0.;
        double cosC = (cos(c) - cos(a)*cos(b)) / (sin(a)*sin(b));
        double C = std::abs(cosC) < 1. ? acos(cosC) : 0.;

        //std::cout<<"x,y = "<<x<<','<<y<<std::endl;
        //std::cout<<"a = "<<a<<std::endl;
        //std::cout<<"b = "<<b<<std::endl;
        //std::cout<<"c = "<<c<<std::endl;
        //std::cout<<"A = "<<A<<std::endl;
        //std::cout<<"B = "<<B<<std::endl;
        //std::cout<<"C = "<<C<<std::endl;

        // Compute ra,dec from these.
        // Note: increasing x is decreasing ra.  East is left on the sky!
        double ra = x>0 ? -C : C;
        double dec = M_PI/2. - b;
        ra *= 180. / M_PI;
        dec *= 180. / M_PI;
        ra += ra0;
        //std::cout<<"ra = "<<ra<<std::endl;
        //std::cout<<"dec = "<<dec<<std::endl;

        // Rotate shear relative to local west
        std::complex<double> gamma(g1,g2);
        double beta = M_PI - (A+B);
        if (x > 0) beta = -beta;
        //std::cout<<"gamma = "<<gamma<<std::endl;
        //std::cout<<"beta = "<<beta<<std::endl;
        std::complex<double> exp2ibeta(cos(2.*beta),sin(2.*beta));
        gamma *= exp2ibeta;
        //std::cout<<"gamma => "<<gamma<<std::endl;
        fout2 << ra <<"  "<< dec <<"  "<< real(gamma) <<"  "<<imag(gamma) <<"  "<<k <<std::endl;

        //
        // Spherical around N pole
        //

        dec = 90. - c * 180./M_PI;
        ra = theta * 12. / M_PI;
        fout3 << ra <<"  "<< dec <<"  "<< g <<"  "<<0. <<"  "<<k <<std::endl;
    }
    var_xdeg /= ngal;
    var_ydeg /= ngal;
    std::cout<<"Min/Max x = "<<xdeg_min<<"  "<<xdeg_max<<std::endl;;
    std::cout<<"Min/Max y = "<<ydeg_min<<"  "<<ydeg_max<<std::endl;;
    std::cout<<"sqrt(Var(x)) = "<<sqrt(var_xdeg)<<std::endl;
    std::cout<<"sqrt(Var(y)) = "<<sqrt(var_ydeg)<<std::endl;

    // Make random catalogs
    std::ofstream foutr1("rand1.dat");
    std::ofstream foutr2("rand2.dat");
    std::ofstream foutr3("rand3.dat");
    foutr1.precision(12);
    foutr2.precision(12);
    foutr3.precision(12);
    xdeg_min=xdeg_max=var_xdeg=0.;
    ydeg_min=ydeg_max=var_ydeg=0.;
    for (long n=0; n<10*ngal; ++n) {
        double x,y,rsq;
        do {
            x = double(rand()) / RAND_MAX; // Random number from 0..1
            y = double(rand()) / RAND_MAX;
            x = 2.*x-1.; // Now from -1..1
            y = 2.*y-1.;
            rsq = x*x+y*y;
        } while (rsq >= 1.);
        x *= rmax;
        y *= rmax;

        double r = rmax*sqrt(rsq);
        double theta = atan2(y,x);
        double xdeg = x*rscale;
        double ydeg = y*rscale;
        double rdeg = r*rscale;

        // Do some sanity checks:
        if (xdeg < xdeg_min) xdeg_min = xdeg;
        if (xdeg > xdeg_max) xdeg_max = xdeg;
        if (ydeg < ydeg_min) ydeg_min = ydeg;
        if (ydeg > ydeg_max) ydeg_max = ydeg;
        var_xdeg += xdeg*xdeg;
        var_ydeg += ydeg*ydeg;


        //
        // flat sky:
        // 
        foutr1 << xdeg <<"  "<< ydeg << std::endl;

        // 
        // Spherical near equator:
        //

        double c = 2.*atan( (rdeg*M_PI/180.) / 2.);
        double a = M_PI/2. - (dec0*M_PI/180.);
        double B = x > 0 ? M_PI/2. - theta : theta - M_PI/2.;
        if (B < 0) B += 2.*M_PI;
        if (B > 2.*M_PI) B -= 2.*M_PI;
        double cosb = cos(a)*cos(c) + sin(a)*sin(c)*cos(B);
        double b = std::abs(cosb) < 1. ? acos(cosb) : 0.;
        double cosC = (cos(c) - cos(a)*cos(b)) / (sin(a)*sin(b));
        double C = std::abs(cosC) < 1. ? acos(cosC) : 0.;

        double ra = x>0 ? -C : C;
        double dec = M_PI/2. - b;
        ra *= 180. / M_PI;
        dec *= 180. / M_PI;
        ra += ra0;

        foutr2 << ra <<"  "<< dec <<std::endl;

        //
        // Spherical around N pole
        //

        dec = 90. - c * 180./M_PI;
        ra = theta * 12. / M_PI;
        foutr3 << ra <<"  "<< dec <<std::endl;
    }
    var_xdeg /= ngal;
    var_ydeg /= ngal;
    std::cout<<"For randoms:\n";
    std::cout<<"Min/Max x = "<<xdeg_min<<"  "<<xdeg_max<<std::endl;;
    std::cout<<"Min/Max y = "<<ydeg_min<<"  "<<ydeg_max<<std::endl;;
    std::cout<<"sqrt(Var(x)) = "<<sqrt(var_xdeg)<<std::endl;
    std::cout<<"sqrt(Var(y)) = "<<sqrt(var_ydeg)<<std::endl;
    return 0;
}

