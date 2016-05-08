      PROGRAM sp2tccpdf1 
C*********************************************************************** 
C     This program computes upper and lower bounds for the:            * 
C                                                                      * 
C     density of     T=  T_1+T_2 in a gaussian process i.e.            * 
C                                                                      * 
C      wavelengthes for crests <h1 and troughs >h2                     *     
C                                                                      * 
C     Sylvie  and Igor 7 dec. 1999                                    * 
C*********************************************************************** 
      use GLOBALDATA, only : Nt,Nj,Nd,Nc,Ntd,Ntdc,NI,Mb, 
     &   NIT,Nx,TWOPI,XSPLT,SCIS,NSIMmax,COV 
      use rind 
      IMPLICIT NONE 
      double precision, dimension(:,:),allocatable :: BIG 
      double precision, dimension(:,:),allocatable :: ansrup 
      double precision, dimension(:,:),allocatable :: ansrlo 
      double precision, dimension(:  ),allocatable :: ex,CY1,CY2 
      double precision, dimension(:,:),allocatable :: xc 
      double precision, dimension(:,:),allocatable ::fxind 
      double precision, dimension(:  ),allocatable :: h1,h2 
      double precision, dimension(:  ),allocatable :: hh1,hh2 
      double precision, dimension(:  ),allocatable :: R0,R1,R2 
      double precision             ::CC,U,XddInf,XdInf,XtInf   
      double precision, dimension(:,:),allocatable :: a_up,a_lo 
      integer         , dimension(:  ),allocatable :: seed 
      integer ,dimension(7) :: indI 
      integer :: Ntime,N0,tn,ts,speed,ph,seed1,seed_size,Nx1,Nx2 
      integer :: icy,icy2 
      double precision :: ds,dT ! lag spacing for covariances  
! DIGITAL: 
! f90 -g2 -C -automatic -o ~/WAT/V4/sp2tthpdf.exe rind48.f sp2tthpdf.f 
! SOLARIS: 
!f90 -g -O -w3 -Bdynamic -fixed -o ../sp2tthpdf.exe rind48.f sp2tthpdf.f 
 
      !print *,'enter sp2thpdf' 
      CALL INIT_LEVELS(U,Ntime,N0,NIT,speed,SCIS,seed1,Nx1,Nx2,dT) 
       
      !print *,'U,Ntime,NIT,speed,SCIS,seed1,Nx,dT' 
      !print *,U,Ntime,NIT,speed,SCIS,seed1,Nx,dT 
      !Nx1=1 
      !Nx2=1 
 
      Nx=Nx1*Nx2 
      !print *,'NN',Nx1,Nx2,Nx 
       
       
      !XSPLT=1.5d0 
      if (SCIS.GT.0) then 
        allocate(COV(1:Nx)) 
        call random_seed(SIZE=seed_size)  
        allocate(seed(seed_size))  
        call random_seed(GET=seed(1:seed_size))  ! get current seed 
        seed(1)=seed1                            ! change seed 
        call random_seed(PUT=seed(1:seed_size))  
        deallocate(seed) 
      endif 
      CALL INITDATA(speed)   
      !print *,ntime,speed,u,NIT 
      allocate(R0(1:Ntime+1)) 
      allocate(R1(1:Ntime+1)) 
      allocate(R2(1:Ntime+1)) 
       
      allocate(h1(1:Nx1)) 
      allocate(h2(1:Nx2)) 
      CALL INIT_AMPLITUDES(h1,Nx1,h2,Nx2) 
      CALL INIT_COVARIANCES(Ntime,R0,R1,R2) 
       
 
      allocate(hh1(1:Nx)) 
      allocate(hh2(1:Nx)) 
      !h transformation 
      do icy=1,Nx1 
         do icy2=1,Nx2 
         hh1((icy-1)*Nx2+icy2)=h1(icy); 
         hh2((icy-1)*Nx2+icy2)=h2(icy2);  
         enddo  
      enddo 
 
      Nj=0 
      indI(1)=0 
 
C     ***** The bound 'infinity' is set to 10*sigma ***** 
      XdInf=10.d0*SQRT(-R2(1)) 
      XtInf=10.d0*SQRT(R0(1)) 
      !h1(1)=XtInf 
      !h2(1)=XtInf 
      ! normalizing constant 
      CC=TWOPI*SQRT(-R0(1)/R2(1))*exp(u*u/(2.d0*R0(1)) ) 
      allocate(CY1(1:Nx))  
      allocate(CY2(1:Nx))  
      do icy=1,Nx 
        CY1(icy)=exp(-0.5*hh1(icy)*hh1(icy)/100)/(10*sqrt(twopi)) 
        CY2(icy)=exp(-0.5*hh2(icy)*hh2(icy)/100)/(10*sqrt(twopi)) 
      enddo 
      !print *,CY1 
      allocate(ansrup(1:Ntime,1:Nx)) 
      allocate(ansrlo(1:Ntime,1:Nx)) 
      ansrup=0.d0 
      ansrlo=0.d0 
      allocate(fxind(1:Nx,1:2))     
      !fxind=0.d0 this is not needed 
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
! Y={X(t2)..,X(ts),..X(tn-1)||X'(ts) X'(t1) X'(tn)||Y1 Y2 X(ts) X(t1) X(tn)} !! 
! = [Xt                          Xd                    Xc]                   !! 
!                                                                            !! 
! Nt=tn-2, Nd=3, Nc=2+3                                                      !! 
!                                                                            !! 
! Xt= contains Nt time points in the indicator function                      !! 
! Xd=    "     Nd    derivatives                                             !! 
! Xc=    "     Nc    variables to condition on                               !! 
! (Y1,Y2) dummy variables ind. of all other v. inputing h1,h2 into rindd     !! 
!                                                                            !! 
! There are 6 ( NI=7) regions with constant bariers:                         !! 
! (indI(1)=0);     for i\in (indI(1),indI(2)] u<Y(i)<h1                      !! 
! (indI(2)=ts-2);  for i\in (indI(2),indI(2)], inf<Y(i)<inf (no restr.)      !! 
! (indI(3)=ts-1);  for i\in (indI(3),indI(4)], h2 <Y(i)<u                    !! 
! (indI(4)=Nt)  ;  for i\in (indI(4),indI(5)], Y(i)<0 (deriv. X'(ts))        !! 
! (indI(5)=Nt+1);  for i\in (indI(5),indI(6)], Y(i)>0 (deriv. X'(t1))        !! 
! (indI(6)=Nt+2);  for i\in (indI(6),indI(7)], Y(i)>0 (deriv. X'(tn))        !! 
! (indI(7)=Nt+3);  NI=7.                                                     !! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
 
       
      NI=7; Nd=3 
      Nc=5; Mb=3 
      allocate(a_up(1:Mb,1:(NI-1))) 
      allocate(a_lo(1:Mb,1:(NI-1))) 
      a_up=0.d0 
      a_lo=0.d0 
      allocate(BIG(1:(Ntime+Nc+1),1:(Ntime+Nc+1)))  
      ALLOCATE(xc(1:Nc,1:Nx)) 
      allocate(ex(1:(Ntime+Nc+1))) 
      !print *,size(ex),Ntime 
      ex=0.d0 
      !print *,size(ex),ex 
      xc(1,1:Nx)=hh1(1:Nx) 
      xc(2,1:Nx)=hh2(1:Nx) 
      xc(3,1:Nx)=u 
      xc(4,1:Nx)=u 
      xc(5,1:Nx)=u 
         ! upp- down- upp-crossings at t1,ts,tn 
        
        a_lo(1,1)=u 
        a_up(1,2)=XtInf   ! X(ts) is redundant   
        a_lo(1,2)=-Xtinf 
        a_up(1,3)=u 
    
  
        a_lo(1,4)=-XdInf 
        a_up(1,5)= XdInf 
        a_up(1,6)= XdInf   
 
        a_up(2,1)=1.d0 
        a_lo(3,3)=1.d0 !signe a voir!!!!!! 
!        print *,a_up 
!        print *,a_lo 
      do tn=N0,Ntime,1 
!      do tn=Ntime,Ntime,1 
        Ntd=tn+1 
        Nt=Ntd-Nd 
        Ntdc=Ntd+Nc 
        indI(4)=Nt 
        indI(5)=Nt+1 
        indI(6)=Nt+2 
        indI(7)=Ntd 
        if (SCIS.gt.0) then 
           if (SCIS.EQ.2) then            
              Nj=max(Nt,0) 
           else 
              Nj=min(max(Nt-5, 0),0) 
           endif 
        endif 
        do ts=3,tn-2 
         !print *,'ts,tn' ,ts,tn,Ntdc 
         CALL COV_INPUT(Big(1:Ntdc,1:Ntdc),tn,ts,R0,R1,R2)!positive wave period 
         indI(2)=ts-2 
         indI(3)=ts-1 
          
 
         CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex(1:Ntdc), 
     &              xc,indI,a_lo,a_up) 
          
        ds=dt 
         do icy=1,Nx 
         ! ansr(tn,:)=ansr(tn,:)+fxind*CC*ds./(CY1.*CY2) 
         ansrup(tn,icy)=ansrup(tn,icy)+fxind(icy,1)*CC*ds 
     &                          /(CY1(icy)*CY2(icy)) 
         ansrlo(tn,icy)=ansrlo(tn,icy)+fxind(icy,2)*CC*ds 
     &                          /(CY1(icy)*CY2(icy)) 
         enddo 
        enddo ! ts 
        print *,'Ready: ',tn,' of ',Ntime 
 
      enddo !tn 
   
 300  open (unit=11, file='dens.out',  STATUS='unknown') 
    
      do ts=1,Ntime 
         do ph=1,Nx 
            !write(11,*)  ansrup(ts,ph),ansrlo(ts,ph) 
            write(11,111)  ansrup(ts,ph),ansrlo(ts,ph) 
         enddo 
      enddo 
 111  FORMAT(2x,F12.8,2x,F12.8)   
      close(11) 
 900  deallocate(big) 
      deallocate(fxind) 
      deallocate(ansrup) 
      deallocate(ansrlo) 
      deallocate(xc) 
      deallocate(ex) 
      deallocate(R0) 
      deallocate(R1) 
      deallocate(R2) 
      if (allocated(COV) ) then 
         deallocate(COV) 
      endif 
        deallocate(h1) 
        deallocate(h2) 
        deallocate(hh1) 
        deallocate(hh2) 
        deallocate(a_up) 
        deallocate(a_lo) 
      stop 
      !return 
 
      CONTAINS 
       
     
             
      SUBROUTINE INIT_LEVELS 
     & (U,Ntime,N0,NIT,speed,SCIS,seed1,Nx1,Nx2,dT) 
      IMPLICIT NONE 
      integer, intent(out):: Ntime,N0,NIT,speed,Nx1,Nx2,SCIS,seed1 
      double precision ,intent(out) :: U,dT 
      
 
      OPEN(UNIT=14,FILE='reflev.in',STATUS= 'UNKNOWN') 
      READ (14,*) U 
      READ (14,*) Ntime 
      READ (14,*) N0 
      READ (14,*) NIT 
      READ (14,*) speed    
      READ (14,*) SCIS    
      READ (14,*) seed1    
 
 
        READ (14,*) Nx1,Nx2 
        READ (14,*) dT 
        if (Ntime.lt.5) then 
           print *,'The number of wavelength points is too small, stop' 
           stop 
        end if 
       
      CLOSE(UNIT=14) 
        
      RETURN 
      END SUBROUTINE INIT_LEVELS 
  
C****************************************************** 
      SUBROUTINE INIT_AMPLITUDES(h1,Nx1,h2,Nx2)       
      IMPLICIT NONE 
      double precision, dimension(:), intent(out) :: h1,h2     
      integer, intent(in) :: Nx1,Nx2 
      integer :: ix 
       
  
      OPEN(UNIT=4,FILE='h.in',STATUS= 'UNKNOWN') 
 
C 
C Reading in amplitudes 
C 
      do ix=1,Nx1 
        READ (4,*) H1(ix) 
      enddo 
      do ix=1,Nx2 
        READ (4,*) H2(ix) 
      enddo 
      CLOSE(UNIT=4) 
       
      RETURN 
      END SUBROUTINE INIT_AMPLITUDES 
 
C************************************************** 
 
C*********************************************************************** 
C*********************************************************************** 
        
      SUBROUTINE INIT_COVARIANCES(Ntime,R0,R1,R2) 
      IMPLICIT NONE 
      double precision, dimension(:),intent(out) :: R0,R1,R2 
      integer,intent(in) :: Ntime 
      integer :: i 
      open (unit=1, file='Cd0.in',STATUS='unknown') 
      open (unit=2, file='Cd1.in',STATUS='unknown') 
      open (unit=3, file='Cd2.in',STATUS='unknown') 
 
      do i=1,Ntime 
         read(1,*) R0(i) 
         read(2,*) R1(i) 
         read(3,*) R2(i) 
      enddo 
      close(1) 
      close(2) 
      close(3) 
       
      return 
      END SUBROUTINE INIT_COVARIANCES 
       
C*********************************************************************** 
C*********************************************************************** 
 
C********************************************************************** 
 
      SUBROUTINE COV_INPUT(BIG,tn,ts, R0,R1,R2) 
      IMPLICIT NONE 
      double precision, dimension(:,:),intent(inout) :: BIG 
      double precision, dimension(:),intent(in) :: R0,R1,R2 
      integer ,intent(in) :: tn,ts 
      integer :: i,j,Ntd1,N !=Ntdc 
      double precision :: tmp  
! the order of the variables in the covariance matrix 
! are organized as follows:  
!  
! ||X(t2)..X(ts),..X(tn-1)||X'(ts) X'(t1) X'(tn)||Y1 Y2 X(ts) X(t1) X(tn)||  
! = [Xt                          Xd                    Xc] 
! where  
! 
! Xt= time points in the indicator function 
! Xd= derivatives 
! Xc=variables to condition on 
 
! Computations of all covariances follows simple rules: Cov(X(t),X(s))=r(t,s), 
! then  Cov(X'(t),X(s))=dr(t,s)/dt.  Now for stationary X(t) we have 
! a function r(tau) such that Cov(X(t),X(s))=r(s-t) (or r(t-s) will give the same result). 
! 
! Consequently  Cov(X'(t),X(s))    = -r'(s-t)    = -sign(s-t)*r'(|s-t|) 
!               Cov(X'(t),X'(s))   = -r''(s-t)   = -r''(|s-t|) 
!               Cov(X''(t),X'(s))  =  r'''(s-t)  =  sign(s-t)*r'''(|s-t|) 
!               Cov(X''(t),X(s))   =  r''(s-t)   =   r''(|s-t|) 
!               Cov(X''(t),X''(s)) =  r''''(s-t) = r''''(|s-t|) 
 
      Ntd1=tn+1 
      N=Ntd1+Nc 
      do i=1,tn-2 
      !cov(Xt) 
         do j=i,tn-2 
           BIG(i,j) = R0(j-i+1) ! cov(X(ti+1),X(tj+1)) 
         enddo 
      !cov(Xt,Xc) 
	 BIG(i      ,Ntd1+1) = 0.d0          !cov(X(ti+1),Y1)   
         BIG(i      ,Ntd1+2) = 0.d0          !cov(X(ti+1),Y2)   
         BIG(i      ,Ntd1+4) = R0(i+1)       !cov(X(ti+1),X(t1))   
         BIG(tn-1-i ,Ntd1+5) = R0(i+1)       !cov(X(t.. ),X(tn))   
 
      !Cov(Xt,Xd)=cov(X(ti+1),x(tj) 
         BIG(i,Ntd1-1)   =-R1(i+1)         !cov(X(ti+1),X'(t1))   
         BIG(tn-1-i,Ntd1)= R1(i+1)         !cov(X(ti+1),X'(tn))  
      enddo 
!cov(Xd) 
      BIG(Ntd1  ,Ntd1  ) = -R2(1) 
      BIG(Ntd1-1,Ntd1  ) = -R2(tn)        !cov(X'(t1),X'(tn)) 
      BIG(Ntd1-1,Ntd1-1) = -R2(1) 
      BIG(Ntd1-2,Ntd1-1) = -R2(ts)        !cov(X'(ts),X'(t1)) 
      BIG(Ntd1-2,Ntd1-2) = -R2(1) 
      BIG(Ntd1-2,Ntd1  ) = -R2(tn+1-ts)   !cov(X'(ts),X'(tn)) 
 
!cov(Xc) 
      BIG(Ntd1+1,Ntd1+1) = 100.d0        ! cov(Y1 Y1)  
      BIG(Ntd1+1,Ntd1+2) = 0.d0          ! cov(Y1 Y2)  
      BIG(Ntd1+1,Ntd1+3) = 0.d0          ! cov(Y1 X(ts))  
      BIG(Ntd1+1,Ntd1+4) = 0.d0          ! cov(Y1 X(t1))  
      BIG(Ntd1+1,Ntd1+5) = 0.d0          ! cov(Y1 X(tn)) 
      BIG(Ntd1+2,Ntd1+2) = 100.d0        ! cov(Y2 Y2)  
      BIG(Ntd1+2,Ntd1+3) = 0.d0          ! cov(Y2 X(ts))  
      BIG(Ntd1+2,Ntd1+4) = 0.d0          ! cov(Y2 X(t1))  
      BIG(Ntd1+2,Ntd1+5) = 0.d0          ! cov(Y2 X(tn)) 
 
      BIG(Ntd1+3,Ntd1+3) = R0(1)        ! cov(X(ts),X (ts) 
      BIG(Ntd1+3,Ntd1+4) = R0(ts)       ! cov(X(ts),X (t1)) 
      BIG(Ntd1+3,Ntd1+5) = R0(tn+1-ts)  ! cov(X(ts),X (tn)) 
      BIG(Ntd1+4,Ntd1+4) = R0(1)        ! cov(X(t1),X (t1))  
      BIG(Ntd1+4,Ntd1+5) = R0(tn)       ! cov(X(t1),X (tn)) 
      BIG(Ntd1+5,Ntd1+5) = R0(1)        ! cov(X(tn),X (tn)) 
   
 
!cov(Xd,Xc) 
      BIG(Ntd1  ,Ntd1+1) = 0.d0        !cov(X'(tn),Y1)  
      BIG(Ntd1  ,Ntd1+2) = 0.d0        !cov(X'(tn),Y2)  
      BIG(Ntd1-1  ,Ntd1+1) = 0.d0        !cov(X'(t1),Y1)  
      BIG(Ntd1-1  ,Ntd1+2) = 0.d0        !cov(X'(t1),Y2)  
      BIG(Ntd1-2  ,Ntd1+1) = 0.d0        !cov(X'(ts),Y1)  
      BIG(Ntd1-2  ,Ntd1+2) = 0.d0        !cov(X'(ts),Y2)  
   
      BIG(Ntd1  ,Ntd1+4) = R1(tn)        !cov(X'(tn),X(t1))      
      BIG(Ntd1  ,Ntd1+5) = 0.d0          !cov(X'(tn),X(tn)) 
      BIG(Ntd1-1,Ntd1+4) = 0.d0          !cov(X'(t1),X(t1)) 
      BIG(Ntd1-1,Ntd1+5) =-R1(tn)        !cov(X'(t1),X(tn)) 
      BIG(Ntd1  ,Ntd1+3) = R1(tn+1-ts)   !cov(X'(tn),X (ts))   
      BIG(Ntd1-1,Ntd1+3) =-R1(ts)        !cov(X'(t1),X (ts))    
      BIG(Ntd1-2,Ntd1+3) = 0.d0          !cov(X'(ts),X (ts) 
      BIG(Ntd1-2,Ntd1+4) = R1(ts)        !cov(X'(ts),X (t1)) 
      BIG(Ntd1-2,Ntd1+5) = -R1(tn+1-ts)  !cov(X'(ts),X (tn)) 
 
 
        do i=1,tn-2 
          j=abs(i+1-ts) 
!cov(Xt,Xc) 
          BIG(i,Ntd1+3)   = R0(j+1)      !cov(X(ti+1),X(ts))           
!Cov(Xt,Xd) 
          if ((i+1-ts).lt.0) then 
              BIG(i,Ntd1-2)  =  R1(j+1) 
            else                         !cov(X(ti+1),X'(ts))   
              BIG(i,Ntd1-2)  = -R1(j+1) 
          endif 
      enddo 
     
! make lower triangular part equal to upper  
      do j=1,N-1 
        do i=j+1,N 
           tmp =BIG(j,i) 
 
           BIG(i,j)=tmp 
        enddo 
      enddo 
 
C     write (*,10) ((BIG(j,i),i=N+1,N+6),j=N+1,N+6) 
C 10  format(6F8.4) 
      RETURN 
      END  SUBROUTINE COV_INPUT 
 
      SUBROUTINE COV_INPUT2(BIG,pt, R0,R1,R2) 
      IMPLICIT NONE 
      double precision, dimension(:,:), intent(out) :: BIG 
      double precision, dimension(:), intent(in) ::  R0,R1,R2 
      integer :: pt,i,j 
! the order of the variables in the covariance matrix 
! are organized as follows; 
! X(t2)...X(tn-1) X'(t1) X'(tn) X(t1) X(tn) = [Xt Xd Xc] 
!  
! where Xd is the derivatives  
! 
! Xt= time points in the indicator function 
! Xd= derivatives 
! Xc=variables to condition on 
 
!cov(Xc) 
      BIG(pt+2,pt+2) = R0(1) 
      BIG(pt+1,pt+1) = R0(1) 
      BIG(pt+1,pt+2) = R0(pt) 
!cov(Xd) 
      BIG(pt,pt)     = -R2(1) 
      BIG(pt-1,pt-1) = -R2(1) 
      BIG(pt-1,pt)   = -R2(pt) 
!cov(Xd,Xc) 
      BIG(pt,pt+2)   =  0.d0 
      BIG(pt,pt+1)   = R1(pt) 
      BIG(pt-1,pt+2) = -R1(pt) 
      BIG(pt-1,pt+1) =  0.d0 
  
      if (pt.GT.2) then 
!cov(Xt) 
         do i=1,pt-2 
           do j=i,pt-2 
              BIG(i,j) = R0(j-i+1) 
           enddo 
        enddo 
!cov(Xt,Xc) 
        do i=1,pt-2 
           BIG(i,pt+1)      = R0(i+1) 
           BIG(pt-1-i,pt+2) = R0(i+1) 
        enddo 
!Cov(Xt,Xd)=cov(X(ti+1),x(tj)) 
        do i=1,pt-2 
           BIG(i,pt-1)   = -R1(i+1) 
           BIG(pt-1-i,pt)=  R1(i+1) 
        enddo 
      endif 
 
 
      ! make lower triangular part equal to upper  
      do j=1,pt+1 
         do i=j+1,pt+2 
            BIG(i,j)=BIG(j,i) 
         enddo 
      enddo 
C      write (*,10) ((BIG(j,i),i=N+1,N+6),j=N+1,N+6) 
C 10   format(6F8.4) 
      RETURN 
      END  SUBROUTINE COV_INPUT2  
 
 
      END  PROGRAM  sp2tccpdf1 
        
 
 
 
 
 
 
 
 
