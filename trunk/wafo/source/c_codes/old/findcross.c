

/*
 * findcross.c - 
 *
 *  Returns indices to level v crossings of argument vector
 *
 * 1998 by Per Andreas Brodtkorb. last modified 23.06-98
 */ 


void findcross(double *y, double v, double *ind, int n, int info)
{ int i,start, ix=0,dcross=0;

 if  ( *(y +0)< v){
    dcross=-1; /* first is a up-crossing*/ 
 }
 if  ( *(y +0)> v){
    dcross=1;  /* first is a down-crossing*/ 
 }
 start=0;
 if  ( *(y +0)== v){
    /* Find out what type of crossing we have next time.. */
    for (i=1; i<n; i++) {
       start=i;
       if  ( *(y +i)< v){
	  *(ind + ix) = i; /* first crossing is a down crossing*/ 
	  ix++; 
	  dcross=-1; /* The next crossing is a up-crossing*/ 
	  break;
       }
       if  ( *(y +i)> v){
	  *(ind + ix) = i; /* first crossing is a up-crossing*/ 
	  ix++; 
	  dcross=1;  /*The next crossing is a down-crossing*/ 
	  break;
       }
    }
 }
 
 for (i=start; i<n-1; i++) {
    if (( (dcross==-1) && (*(y +i)<=h) && (*(y+i+1) > h)  )  || ((dcross==1 ) && (*(y +i)>=h) && (*(y+i+1) < h) ) )  { 
      
       *(ind + ix) = i+1 ;
       ix++;
       dcross=-dcross;
    }  
 }
 info = ix
 return;
}


