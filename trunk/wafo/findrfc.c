#include "math.h"
/*
 * findrfc.c - 
 *
 *  Returns indices to RFC turningpoints of a vector
 *  of turningpoints
 *
 *  Install gfortran and run the following to build the module:
 *   f2py rfc.pyf findrfc.c -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
*
 * 1998 by Per Andreas Brodtkorb.
 */

void findrfc(double *y1,double hmin, double *ind, int n,int info) {
   double xminus,xplus,Tpl,Tmi,*y,Tstart;
   int i,j,ix=0,NC,iy;
   
   if (*(y1+0)> *(y1+1)){ /* if first is a max*/
      y=&(*(y1+1));  /* ignore the first max*/
      NC=floor((n-1)/2);
      Tstart=2;
   }
   else {
      y=y1;
      NC=floor(n/2);
      Tstart=1;
   }
    
   if (NC<1){
	  info = 0;
      return; /* No RFC cycles*/
   }
   

   if (( *(y+0) > *(y+1)) && ( *(y+1) > *(y+2)) ){
	  info = -1;
      return; /*This is not a sequence of turningpoints, exit */
   }
   if ((*(y+0) < *(y+1)) && (*(y+1)< *(y+2))){
      info=-1;
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
   info = ix;
  return ;
}



