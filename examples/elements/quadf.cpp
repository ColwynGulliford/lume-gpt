#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <math.h>

using namespace std;

int get_fields(double X, double Y, double Z, double G0, double L, double b1, double dl, double& BX, double&  BY, double& BZ, double& G, double& Gp, double& Gpp);

int main(int argc, char ** argv) {

  for(unsigned int ii=0; ii<argc; ++ii ) 
    cout << ii << "   " << argv[ii] << endl;

  double b1 = 100;
  double dl= 0;

  double G0 = 0.01;
  double L = 0.25;

  char * filename;

  if(argc>=5) {
    filename = argv[1];
    G0 = strtod(argv[2], NULL);
    L = strtod(argv[3], NULL);
    b1 = strtod(argv[4], NULL);
    dl= strtod(argv[5], NULL);
  }

  cout << filename <<  " " << G0 << " " << L << " " << b1 << " " << dl << endl;

  double X=0.001;
  double Y=0.005;
  double Z;

  double BX, BY, BZ;
  double g, gp, gpp;

  double zmax = L/2.0 + dl + 10/b1;

  int npts = 500;
  double zstep = (L + 2*dl + 2*(10/b1))/(double(npts));

  ofstream outfile;
  outfile.open(filename);

  outfile << "Z     G     Gp     Gpp" << endl;

  for(unsigned int ii=0; ii<npts; ii++) {

    Z = -zmax + ii*zstep;

    get_fields(X, Y, Z, G0, L, b1, dl, BX, BY, BZ, g, gp, gpp); 

    outfile << Z << "   " << g << "   " << gp << "   " << gpp << endl;

    //cout << Z << " " << BX << " " << BY << " " << BZ << " " << g << " " << gp << " " << gpp << endl;
  }

  outfile.close();

  return (0);
}

int get_fields(double X, double Y, double Z, double G0, double L, double b1, double dl, double& BX, double&  BY, double& BZ, double& G, double& Gp, double& Gpp) {
   

  double NGAP = 10;
  double dz = fabs(Z) - (dl + L/2);  //distance from fringe location
  double f;

  if(b1==0) {// Fringe b1 parameter is zero

    if( dz > 0) return( 0 ) ;

    BX = G0 * Y ;
    BY = G0 * X ;

    return( 1 ) ;

  } else {

    if(dz*b1 > NGAP) return (0);
    else {

      f = exp(b1*dz);   // Fringe fraction of nominal gradient

      G   =  G0/(1+f);
      Gp  = -G0*b1*f/pow(1+f, 2);
      Gpp =  G0*b1*b1*f*(f-1)/pow(1+f, 3);

      if(Z<0) Gp=-Gp;

      BX = Y*(G-(3*X*X + Y*Y)*Gpp/12);
      BY = X*(G-(3*Y*Y + X*X)*Gpp/12);
      BZ = Gp*X*Y;

      return (1);

    }

  }

}