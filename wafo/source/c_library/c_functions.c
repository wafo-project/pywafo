#include "math.h"
/*
*  Install gfortran and run the following to build the module on windows:
 *   f2py c_library.pyf c_functions.c -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
 */

/*
 * findrfc.c -
 *
 *  Returns indices to RFC turningpoints of a vector
 *  of turningpoints
 *
 * 1998 by Per Andreas Brodtkorb.
 */

void findrfc(double *y1,double hmin, int *ind, int n,int *info) {
   double xminus,xplus,Tpl,Tmi,*y,Tstart;
   int i,j,ix=0,NC,iy;
   info[0] = 0;
   if (*(y1+0)> *(y1+1)){
   /* if first is a max , ignore the first max*/
      y=&(*(y1+1));
      NC=floor((n-1)/2);
      Tstart=1;
   }
   else {
      y=y1;
      NC=floor(n/2);
      Tstart=0;
   }

   if (NC<1){
      return; /* No RFC cycles*/
   }


   if (( *(y+0) > *(y+1)) && ( *(y+1) > *(y+2)) ){
      info[0] = -1;
      return; /*This is not a sequence of turningpoints, exit */
   }
   if ((*(y+0) < *(y+1)) && (*(y+1)< *(y+2))){
      info[0]=-1;
      return; /*This is not a sequence of turningpoints, exit */
   }


   for (i=0; i<NC; i++) {

      Tmi=Tstart+2*i;
      Tpl=Tstart+2*i+2;
      xminus=*(y+2*i);
      xplus=*(y+2*i+2);

      if(i!=0){
     j=i-1;
     while((j>=0) && (*(y+2*j+1)<=*(y+2*i+1))){
        if( (*(y+2*j)<xminus) ){
           xminus=*(y+2*j);
           Tmi=Tstart+2*j;
        } /*if */
        j--;
     } /*while j*/
      } /*if i */
      if ( xminus >= xplus){
     if ( (*(y+2*i+1)-xminus) >= hmin){
        *(ind+ix)=Tmi;
        ix++;
        *(ind+ix)=(Tstart+2*i+1);
        ix++;
     } /*if*/
     goto L180;
      }

      j=i+1;
      while((j<NC) ) {
     if (*(y+2*j+1) >= *(y+2*i+1)) goto L170;
     if( (*(y+2*j+2) <= xplus) ){
        xplus=*(y+2*j+2);
        Tpl=(Tstart+2*j+2);
     }/*if*/
        j++;
      } /*while*/


      if ( (*(y+2*i+1)-xminus) >= hmin) {
     *(ind+ix)=Tmi;
     ix++;
     *(ind+ix)=(Tstart+2*i+1);
     ix++;

      } /*if*/
      goto L180;
   L170:
      if (xplus <= xminus ) {
     if ( (*(y+2*i+1)-xminus) >= hmin){
        *(ind+ix)=Tmi;
        ix++;
        *(ind+ix)=(Tstart+2*i+1);
        ix++;
     } /*if*/
     /*goto L180;*/
      }
      else{
     if ( (*(y+2*i+1)-xplus) >= hmin) {
        *(ind+ix)=(Tstart+2*i+1);
        ix++;
        *(ind+ix)=Tpl;
        ix++;
     } /*if*/
      } /*elseif*/
   L180:
     iy=i;
   }  /* for i */
   info[0] = ix;
  return ;
}



/*
 * findcross.c -
 *
 *  Returns indices to level v crossings of argument vector
 *
 * 1998 by Per Andreas Brodtkorb. last modified 23.06-98
 */


void findcross(double *y, double v, int *ind, int n, int *info)
{ int i,start, ix=0,dcross=0;
    start=0;
    if  ( y[0]< v){
        dcross=-1; /* first is a up-crossing*/
    }
    else if  ( y[0]> v){
        dcross=1;  /* first is a down-crossing*/
    }
    else if  ( y[0]== v){
        /* Find out what type of crossing we have next time.. */
        for (i=1; i<n; i++) {
            start=i;
            if  ( y[i]< v){
                ind[ix] = i-1; /* first crossing is a down crossing*/
                ix++;
                dcross=-1; /* The next crossing is a up-crossing*/
                goto L120;
            }
            else if  ( y[i]> v){
                ind[ix] = i-1; /* first crossing is a up-crossing*/
                ix++;
                dcross=1;  /*The next crossing is a down-crossing*/
                goto L120;
            }
        }
    }
    L120:
    for (i=start; i<n-1; i++) {
        if (( (dcross==-1) && (y[i]<=v) && (y[i+1] > v)  )  || ((dcross==1 ) && (y[i]>=v) && (y[i+1] < v) ) )  {

            ind[ix] = i;
            ix++;
            dcross=-dcross;
        }
    }
    info[0] = ix;
    return;
}


/*
 * DISUFQ  Is an internal function to spec2nlsdat
 *
 *  CALL:  disufq(rvec,ivec,rA,iA, w,kw,h,g,nmin,nmax,m,n)
 *
 * rvec, ivec = real and imaginary parts of the resultant  (size m X n).
 * rA, iA     = real and imaginary parts of the amplitudes (size m X n).
 * w          = vector with angular frequencies (w>=0)
 * kw         = vector with wavenumbers (kw>=0)
 * h          = water depth             (h >=0)
 * g          = constant acceleration of gravity
 * nmin       = minimum index where rA(:,nmin) and iA(:,nmin) is
 *              greater than zero.
 * nmax       = maximum index where rA(:,nmax) and iA(:,nmax) is
 *              greater than zero.
 * m          = size(rA,1),size(iA,1)
 * n          = size(rA,2),size(iA,2), or size(rvec,2),size(ivec,2)
 *
 * DISUFQ returns the summation of difference frequency and sum
 * frequency effects in the vector vec = rvec +sqrt(-1)*ivec.
 * The 2'nd order contribution to the Stokes wave is then calculated by
 * a simple 1D Fourier transform, real(FFT(vec)).
 *
 *  Install gfortran and run the following to build the module:
 *   f2py diffsumfunq.pyf disufq1.c -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
 *
 * by Per Andreas Brodtkorb 15.08.2001
 * revised pab 14.03.2002, 01.05.2002 22.07.2002, oct 2008
 */

void disufq(double *rvec, double *ivec,
       double *rA,   double *iA,
       double *w,    double *kw,
       double h,     double g,
       int nmin, int nmax,
       int m,    int n)
{
  double Epij, Edij;
  double tmp1, tmp2, tmp3, tmp4, kfact;
  double w1, w2, kw1, kw2, Cg;
  double rrA, iiA, riA, irA;
  int i,jy,ix,iz1,iv1,ixi,jyi;
  //int iz2, iv2;
  //Initialize rvec and ivec to zero
  for (ix=0;ix<n*m;ix++) {
    rvec[ix] = 0.0;
    ivec[ix] = 0.0;
  }

  // kfact is set to 2 in order to exploit the symmetry.
  // If you set kfact to 1, you must uncomment all statements
  // including the expressions: rvec[iz2], rvec[iv2], ivec[iz2] and ivec[iv2].

  kfact = 2.0;
  if (h>10000){ /* deep water /Inifinite water depth */
    for (ix = nmin-1;ix<nmax;ix++) {
      ixi = ix*m;
      iz1 = 2*ixi;
      //iz2 = n*m-ixi;
      kw1  = kw[ix];
      Epij = kw1;
      for (i=0;i<m;i++,ixi++,iz1++) {
          rrA = rA[ixi]*rA[ixi]; ///
          iiA = iA[ixi]*iA[ixi]; ///
          riA = rA[ixi]*iA[ixi]; ///

          /// Sum frequency effects along the diagonal
          tmp1 = kfact*(rrA-iiA)*Epij;
          tmp2 = kfact*2.0*riA*Epij;
          rvec[iz1] += tmp1;
          ivec[iz1] += tmp2;

          //rvec[iz2] += tmp1;
          //ivec[iz2] -= tmp2;
          //iz2++;

          /// Difference frequency effects are zero along the diagonal
          /// and are thus not contributing to the mean.
      }
      for (jy = ix+1;jy<nmax;jy++){
          kw2  = kw[jy];
          Epij = 0.5*(kw2 + kw1);
          Edij = -0.5*(kw2 - kw1);
          //printf("Edij = %f Epij = %f \n", Edij,Epij);

          ixi = ix*m;
          jyi = jy*m;
          iz1 = ixi+jyi;
          iv1 = jyi-ixi;
          //iz2 = (n*m-iz1);
          //iv2 = (n*m-iv1);
          for (i = 0;i<m;i++,ixi++,jyi++,iz1++,iv1++) {

              rrA = rA[ixi]*rA[jyi]; ///rrA = rA[i][ix]*rA[i][jy];
              iiA = iA[ixi]*iA[jyi]; ///iiA = iA[i][ix]*iA[i][jy];
              riA = rA[ixi]*iA[jyi]; ///riA = rA[i][ix]*iA[i][jy];
              irA = iA[ixi]*rA[jyi]; ///irA = iA[i][ix]*rA[i][jy];

              /* Sum frequency effects */
              tmp1 = kfact*2.0*(rrA-iiA)*Epij;
              tmp2 = kfact*2.0*(riA+irA)*Epij;
              rvec[iz1] += tmp1;///rvec[i][ix+jy] += tmp1;
              ivec[iz1] += tmp2;///ivec[i][ix+jy] += tmp2;
              //rvec[iz2] += tmp1;///rvec[i][n*m-(ix+jy)] +=  tmp1;
              //ivec[iz2] -= tmp2;///ivec[i][n*m-(ix+jy)] -=  tmp2;
              // iz2++;

              /* Difference frequency effects */
              tmp1 = kfact*2.0*(rrA+iiA)*Edij;
              tmp2 = kfact*2.0*(riA-irA)*Edij;

              rvec[iv1] += tmp1;///rvec[i][jy-ix] += tmp1;
              ivec[iv1] += tmp2;///ivec[i][jy-ix] += tmp2;

              //rvec[iv2] += tmp1;///rvec[i][n*m-(jy-ix)] += tmp1;
              //ivec[iv2] -= tmp2;///ivec[i][n*m-(jy-ix)] -= tmp2;
              //iv2++;
          }
      }
    }
  }
  else{ /* Finite water depth */
    for (ix = nmin-1;ix<nmax;ix++) {
     kw1  = kw[ix];
     w1   = w[ix];
     tmp1 = tanh(kw1*h);
     /// Cg, wave group velocity
     Cg   = 0.5*g*(tmp1 + kw1*h*(1.0- tmp1*tmp1))/w1; /// OK
     tmp1 = 0.5*g*(kw1/w1)*(kw1/w1);
     tmp2 = 0.5*w1*w1/g;
     tmp3 = g*kw1/(w1*Cg);

     if (kw1*h<300.0){
       tmp4 = kw1/sinh(2.0*kw1*h);
     }
     else{ // To ensure sinh does not overflow.
       tmp4 = 0.0;
     }
     // Difference frequency effects finite water depth
     Edij = (tmp1-tmp2+tmp3)/(1.0-g*h/(Cg*Cg))-tmp4; /// OK

     // Sum frequency effects finite water depth
     Epij = (3.0*(tmp1-tmp2)/(1.0-tmp1/kw1*tanh(2.0*kw1*h))+3.0*tmp2-tmp1); /// OK
     //printf("Edij = %f Epij = %f \n", Edij,Epij);

     ixi = ix*m;
     iz1 = 2*ixi;
     //iz2 = n*m-ixi;
     for (i=0;i<m;i++,ixi++,iz1++) {

       rrA = rA[ixi]*rA[ixi]; ///
       iiA = iA[ixi]*iA[ixi]; ///
       riA = rA[ixi]*iA[ixi]; ///


       /// Sum frequency effects along the diagonal
       rvec[iz1] +=  kfact*(rrA-iiA)*Epij;
       ivec[iz1] +=  kfact*2.0*riA*Epij;
       //rvec[iz2] +=  kfact*(rrA-iiA)*Epij;
       //ivec[iz2] -=  kfact*2.0*riA*Epij;
       //iz2++;

       /// Difference frequency effects along the diagonal
       /// are only contributing to the mean
       rvec[i] +=  2.0*(rrA+iiA)*Edij;
     }
     for (jy = ix+1;jy<nmax;jy++) {
       // w1  = w[ix];
       // kw1 = kw[ix];
       w2   = w[jy];
       kw2  = kw[jy];
       tmp1 = g*(kw1/w1)*(kw2/w2);
       tmp2 = 0.5/g*(w1*w1+w2*w2+w1*w2);
       tmp3 = 0.5*g*(w1*kw2*kw2+w2*kw1*kw1)/(w1*w2*(w1+w2));
       tmp4 = (1-g*(kw1+kw2)/(w1+w2)/(w1+w2)*tanh((kw1+kw2)*h));
       Epij = (tmp1-tmp2+tmp3)/tmp4+tmp2-0.5*tmp1; /* OK */

       tmp2 = 0.5/g*(w1*w1+w2*w2-w1*w2); /*OK*/
       tmp3 = -0.5*g*(w1*kw2*kw2-w2*kw1*kw1)/(w1*w2*(w1-w2));
       tmp4 = (1.0-g*(kw1-kw2)/(w1-w2)/(w1-w2)*tanh((kw1-kw2)*h));
       Edij = (tmp1-tmp2+tmp3)/tmp4+tmp2-0.5*tmp1; /* OK */
       //printf("Edij = %f Epij = %f \n", Edij,Epij);

       ixi = ix*m;
       jyi = jy*m;
       iz1 = ixi+jyi;
       iv1 = jyi-ixi;
       //       iz2 = (n*m-iz1);
       //       iv2 = n*m-iv1;
       for (i=0;i<m;i++,ixi++,jyi++,iz1++,iv1++) {
     rrA = rA[ixi]*rA[jyi]; ///rrA = rA[i][ix]*rA[i][jy];
     iiA = iA[ixi]*iA[jyi]; ///iiA = iA[i][ix]*iA[i][jy];
     riA = rA[ixi]*iA[jyi]; ///riA = rA[i][ix]*iA[i][jy];
     irA = iA[ixi]*rA[jyi]; ///irA = iA[i][ix]*rA[i][jy];

     /* Sum frequency effects */
     tmp1 = kfact*2.0*(rrA-iiA)*Epij;
     tmp2 = kfact*2.0*(riA+irA)*Epij;
     rvec[iz1] += tmp1;///rvec[i][jy+ix] += tmp1;
     ivec[iz1] += tmp2;///ivec[i][jy+ix] += tmp2;
     //rvec[iz2] += tmp1;///rvec[i][n*m-(jy+ix)] += tmp1;
     //ivec[iz2] -= tmp2;///ivec[i][n*m-(jy+ix)] -= tmp2;
     //iz2++;

     /* Difference frequency effects */
     tmp1 = kfact*2.0*(rrA+iiA)*Edij;
     tmp2 = kfact*2.0*(riA-irA)*Edij;
     rvec[iv1] += tmp1;///rvec[i][jy-ix] += tmp1;
     ivec[iv1] += tmp2;///ivec[i][jy-ix] -= tmp2;

     //rvec[iv2] += tmp1;
     //ivec[iv2] -= tmp2;
     //iv2++;
       }
     }
   }
  }
  //return i;
}
/*
 * DISUFQ2  Is an internal function to spec2nlsdat
 *
 *  CALL:  disufq2(rsvec,isvec,rdvec,idvec,rA,iA, w,kw,h,g,nmin,nmax,m,n)
 *
 * rsvec, isvec = real and imaginary parts of the sum frequency
 *                effects  (size m X n).
 * rdvec, idvec = real and imaginary parts of the difference frequency
 *                effects  (size m X n).
 * rA, iA     = real and imaginary parts of the amplitudes (size m X n).
 * w          = vector with angular frequencies (w>=0)
 * kw         = vector with wavenumbers (kw>=0)
 * h          = water depth             (h >=0)
 * g          = constant acceleration of gravity
 * nmin       = minimum index where rA(:,nmin) and iA(:,nmin) is
 *              greater than zero.
 * nmax       = maximum index where rA(:,nmax) and iA(:,nmax) is
 *              greater than zero.
 * m          = size(rA,1),size(iA,1)
 * n          = size(rA,2),size(iA,2), or size(rvec,2),size(ivec,2)
 *
 * DISUFQ2 returns the summation of sum and difference frequency
 * frequency effects in the vectors svec = rsvec +sqrt(-1)*isvec and
 * dvec =  rdvec +sqrt(-1)*idvec.
 * The 2'nd order contribution to the Stokes wave is then calculated by
 * a simple 1D Fourier transform, real(FFT(svec+dvec)).
 *
 *
 * This is a MEX-file for MATLAB.
 * by Per Andreas Brodtkorb 15.08.2001
 * revised pab 14.03.2002, 01.05.2002
 */

void disufq2(double *rsvec, double *isvec,
        double *rdvec, double *idvec,
        double *rA,   double *iA,
        double *w,    double *kw,
        double h,     double g,
        int nmin, int nmax,
        int m,    int n)
{
  double Epij, Edij;
  double tmp1, tmp2, tmp3, tmp4, kfact;
  double w1, w2, kw1, kw2, Cg;
  double rrA, iiA, riA, irA;
  int i,jy,ix,iz1,iv1,ixi,jyi;
  //int iz2,iv2

  //Initialize rvec and ivec to zero
  for (ix=0;ix<n*m;ix++) {
    rsvec[ix] = 0.0;
    isvec[ix] = 0.0;
    rdvec[ix] = 0.0;
    idvec[ix] = 0.0;
  }

  // kfact is set to 2 in order to exploit the symmetry.
  // If you set kfact to 1, you must uncomment all statements
  // including the expressions: rvec[iz2], rvec[iv2], ivec[iz2] and ivec[iv2].

  kfact = 2.0;
  if (h>10000){ /* deep water /Inifinite water depth */
    for (ix = nmin-1;ix<nmax;ix++) {
      ixi = ix*m;
      iz1 = 2*ixi;
      //iz2 = n*m-ixi;
      kw1  = kw[ix];
      Epij = kw1;
      for (i=0;i<m;i++,ixi++,iz1++) {
    rrA = rA[ixi]*rA[ixi]; ///
    iiA = iA[ixi]*iA[ixi]; ///
    riA = rA[ixi]*iA[ixi]; ///

    /// Sum frequency effects along the diagonal
    tmp1 = kfact*(rrA-iiA)*Epij;
    tmp2 = kfact*2.0*riA*Epij;
    rsvec[iz1] += tmp1;
    isvec[iz1] += tmp2;

    //rsvec[iz2] += tmp1;
    //isvec[iz2] -= tmp2;
    //iz2++;

    /// Difference frequency effects are zero along the diagonal
    /// and are thus not contributing to the mean.
      }
      for (jy = ix+1;jy<nmax;jy++){
    kw2  = kw[jy];
    Epij = 0.5*(kw2 + kw1);
    Edij = -0.5*(kw2 - kw1);
    //printf("Edij = %f Epij = %f \n", Edij,Epij);

    ixi = ix*m;
    jyi = jy*m;
    iz1 = ixi+jyi;
    iv1 = jyi-ixi;
    //iz2 = (n*m-iz1);
    //iv2 = (n*m-iv1);
    for (i = 0;i<m;i++,ixi++,jyi++,iz1++,iv1++) {

      rrA = rA[ixi]*rA[jyi]; ///rrA = rA[i][ix]*rA[i][jy];
      iiA = iA[ixi]*iA[jyi]; ///iiA = iA[i][ix]*iA[i][jy];
      riA = rA[ixi]*iA[jyi]; ///riA = rA[i][ix]*iA[i][jy];
      irA = iA[ixi]*rA[jyi]; ///irA = iA[i][ix]*rA[i][jy];

     /* Sum frequency effects */
      tmp1 = kfact*2.0*(rrA-iiA)*Epij;
      tmp2 = kfact*2.0*(riA+irA)*Epij;
      rsvec[iz1] += tmp1;  ///rvec[i][ix+jy] += tmp1;
      isvec[iz1] += tmp2;  ///ivec[i][ix+jy] += tmp2;
      //rsvec[iz2] += tmp1;///rvec[i][n*m-(ix+jy)] += tmp1;
      //isvec[iz2] -= tmp2;///ivec[i][n*m-(ix+jy)] += tmp2;
      //iz2++;

      /* Difference frequency effects */
      tmp1 = kfact*2.0*(rrA+iiA)*Edij;
      tmp2 = kfact*2.0*(riA-irA)*Edij;

      rdvec[iv1] += tmp1;///rvec[i][jy-ix] += tmp1;
      idvec[iv1] += tmp2;///ivec[i][jy-ix] += tmp2;

      //rdvec[iv2] += tmp1;///rvec[i][n*m-(jy-ix)] += tmp1;
      //idvec[iv2] -= tmp2;///ivec[i][n*m-(jy-ix)] -= tmp2;
      //  iv2++;
    }
      }
   }
  }
  else{ /* Finite water depth */
    for (ix = nmin-1;ix<nmax;ix++) {
     kw1  = kw[ix];
     w1   = w[ix];
     tmp1 = tanh(kw1*h);
     /// Cg, wave group velocity
     Cg   = 0.5*g*(tmp1 + kw1*h*(1.0- tmp1*tmp1))/w1; /// OK
     tmp1 = 0.5*g*(kw1/w1)*(kw1/w1);
     tmp2 = 0.5*w1*w1/g;
     tmp3 = g*kw1/(w1*Cg);

     if (kw1*h<300.0){
       tmp4 = kw1/sinh(2.0*kw1*h);
     }
     else{ // To ensure sinh does not overflow.
       tmp4 = 0.0;
     }
     // Difference frequency effects finite water depth
     Edij = (tmp1-tmp2+tmp3)/(1.0-g*h/(Cg*Cg))-tmp4; /// OK

     // Sum frequency effects finite water depth
     Epij = (3.0*(tmp1-tmp2)/(1.0-tmp1/kw1*tanh(2.0*kw1*h))+3.0*tmp2-tmp1); /// OK
     //printf("Edij = %f Epij = %f \n", Edij,Epij);

     ixi = ix*m;
     iz1 = 2*ixi;
     //iz2 = n*m-ixi;
     for (i=0;i<m;i++,ixi++,iz1++) {

       rrA = rA[ixi]*rA[ixi]; ///
       iiA = iA[ixi]*iA[ixi]; ///
       riA = rA[ixi]*iA[ixi]; ///


       /// Sum frequency effects along the diagonal
       rsvec[iz1] +=  kfact*(rrA-iiA)*Epij;
       isvec[iz1] +=  kfact*2.0*riA*Epij;
       //rsvec[iz2] +=  kfact*(rrA-iiA)*Epij;
       //isvec[iz2] -=  kfact*2.0*riA*Epij;

       /// Difference frequency effects along the diagonal
       /// are only contributing to the mean
       //printf(" %f \n",2.0*(rrA+iiA)*Edij);
       rdvec[i] +=  2.0*(rrA+iiA)*Edij;
     }
     for (jy = ix+1;jy<nmax;jy++) {
       // w1  = w[ix];
       // kw1 = kw[ix];
       w2   = w[jy];
       kw2  = kw[jy];
       tmp1 = g*(kw1/w1)*(kw2/w2);
       tmp2 = 0.5/g*(w1*w1+w2*w2+w1*w2);
       tmp3 = 0.5*g*(w1*kw2*kw2+w2*kw1*kw1)/(w1*w2*(w1+w2));
       tmp4 = (1-g*(kw1+kw2)/(w1+w2)/(w1+w2)*tanh((kw1+kw2)*h));
       Epij = (tmp1-tmp2+tmp3)/tmp4+tmp2-0.5*tmp1; /* OK */

       tmp2 = 0.5/g*(w1*w1+w2*w2-w1*w2); /*OK*/
       tmp3 = -0.5*g*(w1*kw2*kw2-w2*kw1*kw1)/(w1*w2*(w1-w2));
       tmp4 = (1.0-g*(kw1-kw2)/(w1-w2)/(w1-w2)*tanh((kw1-kw2)*h));
       Edij = (tmp1-tmp2+tmp3)/tmp4+tmp2-0.5*tmp1; /* OK */
       //printf("Edij = %f Epij = %f \n", Edij,Epij);

       ixi = ix*m;
       jyi = jy*m;
       iz1 = ixi+jyi;
       iv1 = jyi-ixi;
       //       iz2 = (n*m-iz1);
       //       iv2 = (n*m-iv1);
       for (i=0;i<m;i++,ixi++,jyi++,iz1++,iv1++) {
     rrA = rA[ixi]*rA[jyi]; ///rrA = rA[i][ix]*rA[i][jy];
     iiA = iA[ixi]*iA[jyi]; ///iiA = iA[i][ix]*iA[i][jy];
     riA = rA[ixi]*iA[jyi]; ///riA = rA[i][ix]*iA[i][jy];
     irA = iA[ixi]*rA[jyi]; ///irA = iA[i][ix]*rA[i][jy];

     /* Sum frequency effects */
     tmp1 = kfact*2.0*(rrA-iiA)*Epij;
     tmp2 = kfact*2.0*(riA+irA)*Epij;
     rsvec[iz1] += tmp1;///rsvec[i][jy+ix] += tmp1;
     isvec[iz1] += tmp2;///isvec[i][jy+ix] += tmp2;
     //rsvec[iz2] += tmp1;///rsvec[i][n*m-(jy+ix)] += tmp1;
     //isvec[iz2] -= tmp2;///isvec[i][n*m-(jy-ix)] += tmp2;
     //iz2++;

     /* Difference frequency effects */
     tmp1 = kfact*2.0*(rrA+iiA)*Edij;
     tmp2 = kfact*2.0*(riA-irA)*Edij;
     rdvec[iv1] += tmp1;
     idvec[iv1] += tmp2;

     //rdvec[iv2] += tmp1;
     //idvec[iv2] -= tmp2;
     // iv2++;
       }
     }
   }
  }
 // return i;
}


/* ++++++++++ BEGIN RF3 [ampl ampl_mean nr_of_cycle] */
/* ++++++++++ Rain flow without time analysis */
//By Adam Nieslony
//Visit the MATLAB Central File Exchange for latest version
//http://www.mathworks.com/matlabcentral/fileexchange/3026
void findrfc3_astm(double *array_ext, double *array_out, int n, int *nout) {

    double *pr, *po, a[16384], ampl, mean;
    int tot_num, index, j, cNr1, cNr2;

    tot_num = n;

    // pointers to the first element of the arrays
    pr = &array_ext[0];
    po = &array_out[0];

    // The original rainflow counting by Nieslony, unchanged
    j = -1;
    cNr1 = 1;
    for (index=0; index<tot_num; index++) {
        a[++j]=*pr++;
        while ( (j >= 2) && (fabs(a[j-1]-a[j-2]) <= fabs(a[j]-a[j-1])) ) {
            ampl=fabs( (a[j-1]-a[j-2])/2 );
            switch(j) {
                case 0: { break; }
                case 1: { break; }
                case 2: {
                    mean=(a[0]+a[1])/2;
                    a[0]=a[1];
                    a[1]=a[2];
                    j=1;
                    if (ampl > 0) {
                        *po++=ampl;
                        *po++=mean;
                        *po++=0.50;
                    }
                    break;
                }
                default: {
                    mean=(a[j-1]+a[j-2])/2;
                    a[j-2]=a[j];
                    j=j-2;
                    if (ampl > 0) {
                        *po++=ampl;
                        *po++=mean;
                        *po++=1.00;
                        cNr1++;
                    }
                    break;
                }
            }
        }
    }
    cNr2 = 1;
    for (index=0; index<j; index++) {
        ampl=fabs(a[index]-a[index+1])/2;
        mean=(a[index]+a[index+1])/2;
        if (ampl > 0){
            *po++=ampl;
            *po++=mean;
            *po++=0.50;
            cNr2++;
        }
    }
    // array of ints nout is outputted
    nout[0] = cNr1;
    nout[1] = cNr2;
}
/* ++++++++++ END RF3 */


// ++ BEGIN RF5 [ampl ampl_mean nr_of_cycle cycle_begin_time cycle_period_time]
/* ++++++++++ Rain flow with time analysis */
//By Adam Nieslony
//Visit the MATLAB Central File Exchange for latest version
//http://www.mathworks.com/matlabcentral/fileexchange/3026
void
findrfc5_astm(double *array_ext, double *array_t, double *array_out, int n, int *nout) {
    double *pr, *pt, *po, a[16384], t[16384], ampl, mean, period, atime;
    int tot_num, index, j, cNr1, cNr2;


//    tot_num = mxGetM(array_ext) * mxGetN(array_ext);
    tot_num = n;

    // pointers to the first element of the arrays
    pr = &array_ext[0];
    pt = &array_t[0];
    po = &array_out[0];

//    array_out = mxCreateDoubleMatrix(5, tot_num-1, mxREAL);

    // The original rainflow counting by Nieslony, unchanged
    j = -1;
    cNr1 = 1;
    for (index=0; index<tot_num; index++) {
        a[++j]=*pr++;
        t[j]=*pt++;
        while ( (j >= 2) && (fabs(a[j-1]-a[j-2]) <= fabs(a[j]-a[j-1])) ) {
            ampl=fabs( (a[j-1]-a[j-2])/2 );
            switch(j)
{
                case 0: { break; }
                case 1: { break; }
                case 2: {
                    mean=(a[0]+a[1])/2;
                    period=(t[1]-t[0])*2;
                    atime=t[0];
                    a[0]=a[1];
                    a[1]=a[2];
                    t[0]=t[1];
                    t[1]=t[2];
                    j=1;
                    if (ampl > 0) {
                        *po++=ampl;
                        *po++=mean;
                        *po++=0.50;
                        *po++=atime;
                        *po++=period;
                    }
                    break;
                }
                default: {
                    mean=(a[j-1]+a[j-2])/2;
                    period=(t[j-1]-t[j-2])*2;
                    atime=t[j-2];
                    a[j-2]=a[j];
                    t[j-2]=t[j];
                    j=j-2;
                    if (ampl > 0) {
                        *po++=ampl;
                        *po++=mean;
                        *po++=1.00;
                        *po++=atime;
                        *po++=period;
                        cNr1++;
                    }
                    break;
                }
            }
        }
    }
    cNr2 = 1;
    for (index=0; index<j; index++) {
        ampl=fabs(a[index]-a[index+1])/2;
        mean=(a[index]+a[index+1])/2;
        period=(t[index+1]-t[index])*2;
        atime=t[index];
        if (ampl > 0){
            *po++=ampl;
            *po++=mean;
            *po++=0.50;
            *po++=atime;
            *po++=period;
            cNr2++;
        }
    }
//  /* free the memeory !!!*/
//    mxSetN(array_out, tot_num - cNr);
    nout[0] = cNr1;
    nout[1] = cNr2;
}
/* ++++++++++ END RF5 */
