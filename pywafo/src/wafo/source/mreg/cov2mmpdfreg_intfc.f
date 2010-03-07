C     Version  1994-X-18 

C     This is a new version of WAMP program computing crest-trough wavelength 
C     and amplitude  density.
C     
C     revised pab 2007
C     -moved all common blocks into modules
C     -renamed from minmax to sp2mmpdfreg + fixed some bugs
C     revised pab July 2007
!     -renamed from sp2mmpdfreg to cov2mmpdfreg
! gfortran -W -Wall -pedantic-errors -fbounds-check -Werror -c dsvdc.f  mregmodule.f cov2mmpdfreg.f

      module cov2mmpdfmod
      IMPLICIT NONE
      PRIVATE
      PUBLIC cov2mmpdfreg, EPS_, EPSS_, EPS0_, C_, IAC_, ISQ_
      DOUBLE PRECISION :: EPS_ = 1.d-2   
      DOUBLE PRECISION :: EPSS_ = 5.d-5 
! used in GAUSSLE1 to implicitly  ! determ. # nodes  
      DOUBLE PRECISION :: EPS0_ = 5.d-5 
      DOUBLE PRECISION :: C_ = 4.5d0
      INTEGER :: IAC_=1
      INTEGER :: ISQ_=0

      contains

      subroutine cov2mmpdfreg(UVdens,t,COV,ULev,VLev,Tg,Xg,Nt,Nu,Nv,Ng,
     !   NIT)
      USE SIZEMOD
      USE EPSMOD
      USE CHECKMOD
      USE MREGMOD
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: Nt, Nu, Nv, Ng, NIT
      REAL*8, DIMENSION(Nt,5), intent(in):: COV
      REAL*8, DIMENSION(Nu,Nv), intent(out):: UVdens
      REAL*8, DIMENSION(Nu), intent(in):: ULev
      REAL*8, DIMENSION(Nv), intent(in):: VLev
      REAL*8, DIMENSION(Ng), intent(in):: Tg, Xg
      REAL*8, dimension(Nt), intent(in):: T
Cf2py integer, intent(hide), depend(t) :: Nt = len(t)
Cf2py integer, intent(hide), depend(Ulev) :: Nu = len(Ulev)
Cf2py integer, intent(hide), depend(Vlev) :: Nv = len(Vlev)
Cf2py integer, intent(hide), depend(Tg) :: Ng = len(Tg)
Cf2py integer, optional :: NIT = 2
Cf2py real*8, intent(out), depend(Nu,Nv) :: UVdens
Cf2py depend(Ng)  Xg
Cf2py depend(Nt,5)  COV 




      real*8 Q0,SQ0,Q1,SQ1, U,V,VV, XL0, XL2, XL4
      REAL*8 VDERI, CDER,SDER, DER, CONST, F, HHHH,FM, VALUE
C     INTEGER, PARAMETER :: MMAX = 5, NMAX = 101, RDIM = 10201
      REAL*8, DIMENSION(NMAX) :: HHT,VT,UT,Vdd,Udd
      REAL*8, DIMENSION(RDIM) :: R,R1,R2,R3
      REAL*8:: AA(MMAX-2,MMAX-2),AI((MMAX+1)*NMAX)
      REAL*8, DIMENSION(MMAX+1) :: BB, DAI
C      DIMENSION UVdens(NMAX,NMAX),HHT(NMAX)
C      DIMENSION T(NMAX),Ulev(NMAX),Vlev(NMAX)
C      DIMENSION VT(NMAX),UT(NMAX),Vdd(NMAX),Udd(NMAX)
C      DIMENSION COV(5*NMAX),R(RDIM),R1(RDIM),R2(RDIM),R3(RDIM)


C
C   The program computes the joint density of maximum the following minimum
C   and the distance between Max and min for a zero-mean stationary 
C   Gaussian process with covariance function defined explicitely with 4 
C   derivatives. The process should be normalized so that the first and 
C   the second spectral moments are equal to 1. The values of Max are taken
C   as the nodes at Hermite-Quadrature and then integrated out so that
C   the output is a joint density of wavelength T and amplitude H=Max-min.
C   The Max values are defined by subroutine Gauss_M with the accuracy
C   input  epsu. The principle is that the integral of the marginal density 
C   of f_Max is computed with sufficient accuracy.
C
      REAL*8, DIMENSION(NMAX) :: B0,DB0,DDB0,B1,DB1,DDB1,DB2,DDB2
      REAL*8, DIMENSION(NMAX) :: Q,SQ,VDER,DBI,BI
C      DIMENSION B0(NMAX),DB0(NMAX),DDB0(NMAX)
C      DIMENSION B1(NMAX),DB1(NMAX),DDB1(NMAX)
C      DIMENSION DB2(NMAX),DDB2(NMAX)
C      DIMENSION Q(NMAX),SQ(NMAX),VDER(NMAX),DBI(NMAX),BI(NMAX)
      INTEGER :: J,I,I1,I2,I3,IU, IV,N, NNIT, INF
      INTEGER ::  fffff
C     REAL*8 EPS0
C     INTEGER III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101 , III0
C      COMMON/CHECK1/III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101 
C      COMMON/CHECKQ/III0 
C      COMMON /EPS/  EPS,EPSS,CEPSS

C
C  Initiation of all constants and integration nodes 'INITINTEG'
C
      CALL INITINTEG()

!     OPEN(UNIT=8,FILE='min.out')
!      OPEN(UNIT=9,FILE='Max.out')
!      OPEN(UNIT=10,FILE='Maxmin.out')
!      OPEN(UNIT=11,FILE='Maxmin.log')
c
c   OBS. we are using the variables R,R1,R2 R3 as a temporary storage 
C   for transformation  g  of the process.

c
      CALL INITLEVELS(T,HHT,Nt,NG,NU,Nv)
C      CALL INITLEVELS(Ulev,NU,Vlev,NV,T,HHT,Nt,R1,R2,NG)
      IF( Tg(1) .gt. Tg(ng))  then
       print *,'Error Tg must be strictly increasing'
       return
      end if
      if(abs(Tg(ng)-Tg(1))*abs(Xg(ng)-Xg(1)).lt.0.01d0) then
       print *,'The transformation  g is singular, stop'
       stop
      end if
      DO IV=1,Nv
         V=Vlev(IV)
         CALL TRANSF(NG,V,Xg,Tg,VALUE,DER) 
         VT(IV)=VALUE
         Vdd(IV)=DER
14    continue
      enddo
      DO IU=1,Nu
         U = Ulev(IU)
         CALL TRANSF(NG,U,Xg,Tg,VALUE,DER) 
         UT(IU)  = VALUE
         Udd(IU) = DER
         do IV=1,Nv
             UVdens(IU,IV)=0.0d0
16           CONTINUE
         enddo
      enddo
      
      CALL COVG(XL0,XL2,XL4,COV,T,Nt)
      
      
      Q0=XL4
      IF (Q0.le.1.0D0+EPS) then
      Print *,'Covariance structure is singular, stop.'
      stop
      end if
      SQ0 = SQRT(Q0)
      Q1  = XL0-XL2*XL2/XL4
      IF (Q1.le.EPS) then
      Print *,'Covariance structure is singular, stop.'
      stop
      end if
      SQ1 = SQRT(Q1)
      DO I=1,Nt
        B0(I)  =-COV(I,3)
        DB0(I) =-COV(I,4)
        DDB0(I)=-COV(I,5)
      
        B1(I)  =COV(I,1)+COV(I,3)*(XL2/XL4)
        DB1(I) =COV(I,2)+COV(I,4)*(XL2/XL4)
        DDB1(I)=COV(I,3)+XL2*(COV(I,5)/XL4)
C
C       Q(I) contains Var(X(T(i))|X'(0),X''(0),X(0))
C    VDER(I) contains Var(X''(T(i))|X'(0),X''(0),X(0))
C
        Q(I)=XL0 - COV(I,2)*(COV(I,2)/XL2) - B0(I)*(B0(I)/Q0)
     1     -B1(I)*(B1(I)/Q1)
        VDER(I)=XL4 - (COV(I,4)*COV(I,4))/XL2 - (DDB0(I)*DDB0(I))/Q0
     1     - (DDB1(I)*DDB1(I))/Q1 
      

C
C       DDB2(I) contains Cov(X''(T(i)),X(T(i))|X'(0),X''(0),X(0))
C
        DDB2(I)=-XL2 - (COV(I,2)*COV(I,4))/XL2 - DDB0(I)*(B0(I)/Q0)
     1     -DDB1(I)*(B1(I)/Q1)
        IF(Q(I).LE.eps) then
          SQ(i)  =0.0d0
          DDB2(i)=0.0d0
        else
          SQ(I)=SQRT(Q(I))
C 
C       VDER(I) contains Var(X''(T(i))|X'(0),X''(0),X(0),X(T(i))
C
      
          VDER(I)=VDER(I) - (DDB2(I)*DDB2(I))/Q(I)
        end if
      
10    CONTINUE
      enddo
      DO I=1,Nt
      DO J=1,Nt
C      
C   R1 contains Cov(X(T(I)),X'(T(J))|X'(0),X''(0),X(0)) 
C
      R1(J+(I-1)*N)=R1(J+(I-1)*N) -  COV(I,2)*(COV(J,3)/XL2) 
     1 -  (B0(I)*DB0(J)/Q0) -  (B1(I)*DB1(J)/Q1) 
      
C      
C   R2 contains Cov(X'(T(I)),X'(T(J))|X'(0),X''(0),X(0))
C
      R2(J+(I-1)*N) = -R2(J+(I-1)*N) - COV(I,3)*(COV(J,3)/XL2) 
     1   - DB0(I)*DB0(J)/Q0  - DB1(I)*(DB1(J)/Q1) 
C      
C   R3 contains Cov(X''(T(I)),X'(T(J))|X'(0),X''(0),X(0))
C
      R3(J+(I-1)*N) = R3(J+(I-1)*N) - COV(I,4)*(COV(J,3)/XL2) 
     1   - DB0(J)*(DDB0(I)/Q0)  - DDB1(I)*(DB1(J)/Q1) 
15    CONTINUE
      enddo
      enddo

C  The initiations are finished and we are beginning with 3 loops
C  on T=T(I), U=Ulevels(IU), V=Ulevels(IV), U>V.
      
      DO I=1,Nt

           NNIT=NIT
           IF (Q(I).LE.EPS) GO TO 20

         DO I1=1,I
            DB2(I1)=R1(I1+(I-1)*N) 

C     Cov(X'(T(I1)),X(T(i))|X'(0),X''(0),X(0)) 
C     DDB2(I) contains Cov(X''(T(i)),X(T(i))|X'(0),X''(0),X(0))

 30      CONTINUE
         enddo
      
         DO I3=1,I
            DBI(I3) = R3(I3+(I-1)*N) - (DDB2(I)*DB2(I3)/Q(I)) 
            BI(I3)  = R2(I3+(I-1)*N) - (DB2(I)*DB2(I3)/Q(I))
 50      CONTINUE
         enddo
         DO I3=1,I-1
            AI(I3)=0.0d0
            AI(I3+I-1)=DB0(I3)/SQ0 
            AI(I3+2*(I-1))=DB1(I3)/SQ1
            AI(I3+3*(I-1))=DB2(I3)/SQ(I)
 51      CONTINUE
         enddo
         VDERI=VDER(I)
         DAI(1)=0.0d0
         DAI(2)=DDB0(I)/SQ0
         DAI(3)=DDB1(I)/SQ1 
         DAI(4)=DDB2(I)/SQ(I) 
         AA(1,1)=DB0(I)/SQ0 
         AA(1,2)=DB1(I)/SQ1 
         AA(1,3)=DB2(I)/SQ(I) 
         AA(2,1)=XL2/SQ0
         AA(2,2)=SQ1
         AA(2,3)=0.0d0
         AA(3,1)=B0(I)/SQ0 
         AA(3,2)=B1(I)/SQ1 
         AA(3,3)=SQ(I)
         IF (BI(I).LE.EPS) NNIT=0
         IF (NNIT.GT.1) THEN
            IF(I.LT.1) GO TO 41
            DO I1=1,I-1
               DO I2=1,I-1

C   R contains Cov(X'(T(I1)),X'(T(I2))|X'(0),X''(0),X(0),X(I)) 

      R(I2+(I1-1)*(I-1))=R2(I2+(I1-1)*N)-(DB2(I1)*DB2(I2)/Q(I)) 

 40         CONTINUE
            enddo
            enddo
 41         CONTINUE
         END IF

C  Here the covariance of the problem would be innitiated

            INF=0
            Print *,'   Laps to go:',N-I+1
         DO IV=1,Nv
            V=VT(IV)
!            IF (ABS(V).GT.5.0D0) GO TO 80                  
            IF (Vdd(IV).LT.EPS0) GO TO 80
            DO IU=1,Nu
                  U=UT(IU)
                  IF (U.LE.V) go to 60
!                  IF (ABS(U).GT.5.0D0) GO TO 60                  
                  IF (Udd(IU).LT.EPS0) GO TO 60
                  BB(1)=0.0d0
                  BB(2)=U
                  BB(3)=V
!     if (IV.EQ.2.AND.IU.EQ.1) THEN   
!     fffff = 10
!     endif
                 
      CALL MREG(F,R,BI,DBI,AA,BB,AI,DAI,VDERI,3,I-1,NNIT,INF)
             INF=1
             UVdens(IU,IV) = UVdens(IU,IV) + Udd(IU)*Vdd(IV)*HHT(I)*F

!     if (F.GT.0.01.AND.U.GT.2.AND.V.LT.-2) THEN  
!     if (N-I+1 .eq. 38.and.IV.EQ.26.AND.IU.EQ.16) THEN
!     if (IV.EQ.32.AND.IU.EQ.8.and.I.eq.11) THEN      
!          PRINT * ,' R:', R(1:I)
!         PRINT * ,' BI:', BI(1:I)
!         PRINT * ,' DBI:', DBI(1:I)
!         PRINT * ,' DB2:', DB2(1:I)
!         PRINT * ,' DB0(1):', DB0(1)
!         PRINT * ,' DB1(1):', DB1(1)
!          PRINT * ,' DAI:', DAI
!         PRINT * ,' BB:', BB
!         PRINT * ,' VDERI:', VDERI
!          PRINT * ,' F    :', F
!         PRINT * ,' UVDENS :',  UVdens(IU,IV)
!         fffff = 10
!     endif

 60         CONTINUE
            enddo
 80       continue
          enddo
 20   CONTINUE
      enddo

!       hhhh=0.0d0
!       do 90 Iu=1,Nu
!       do 90 Iv=1,Nv
!       WRITE(10,300) Ulev(iu),Vlev(iv),UVdens(iu,iv)
!       hhhh=hhhh+UVdens(iu,iv)
! 90    continue
!      if (nu.gt.1.and.nv.gt.1) then
!      write(11,*) 'SumSum f_uv *du*dv='
!     1,(Ulev(2)-Ulev(1))*(Vlev(2)-Vlev(1))*hhhh
!      end if
      
C      sder=sqrt(XL4-XL2*XL2/XL0)
C      cder=-XL2/sqrt(XL0)
C      const=1/sqrt(XL0*XL4)
C      DO 95 IU=1,NU     
C        U=UT(IU)
C        FM=Udd(IU)*const*exp(-0.5*U*U/XL0)*PMEAN(-cder*U,sder)
C        WRITE(9,300) Ulev(IU),FM
C 95   continue      
C      DO 105 IV=1,NV     
C        V=VT(IV)
C        VV=cder*V
C        Fm=Vdd(IV)*const*exp(-0.5*V*V/XL0)*PMEAN(VV,sder)
C        WRITE(8,300) Vlev(IV),Fm
C 105   continue 
      if (III0.eq.0) III0=1

      PRINT *, 'Rate of calls RINDT0:',float(iii01)/float(III0)
      PRINT *, 'Rate of calls RINDT1:',float(iii11)/float(III0)
      PRINT *, 'Rate of calls RINDT2:',float(iii21)/float(III0)
      PRINT *, 'Rate of calls RINDT3:',float(iii31)/float(III0)
      PRINT *, 'Rate of calls RINDT4:',float(iii41)/float(III0)
      PRINT *, 'Rate of calls RINDT5:',float(iii51)/float(III0)
      PRINT *, 'Rate of calls RINDT6:',float(iii61)/float(III0)
      PRINT *, 'Rate of calls RINDT7:',float(iii71)/float(III0)
      PRINT *, 'Rate of calls RINDT8:',float(iii81)/float(III0)
      PRINT *, 'Rate of calls RINDT9:',float(iii91)/float(III0)
      PRINT *, 'Rate of calls RINDT10:',float(iii101)/float(III0)
      PRINT *, 'Number of calls of RINDT*',III0


      return
      END subroutine cov2mmpdfreg

      SUBROUTINE INITLEVELS(T,HT,N,NG,NU,Nv)
      USE TBRMOD
      USE SIZEMOD
      IMPLICIT NONE
C     INTEGER, PARAMETER:: NMAX = 101, RDIM = 10201
C      DIMENSION ULEVELS(1),Vlevels(1),T(1),HT(1),TG(1),XG(1),HH(101)
      REAL*8, DIMENSION(:), intent(in) :: T
      REAL*8, DIMENSION(:), intent(out) :: HT
      INTEGER, intent(in) :: NG
      REAL*8 :: UMIN,UMAX,VMIN,VMAX, HU,HV
      integer :: N, I, NU, NV
C     REAL*8, DIMENSION(NMAX) :: HH
C      COMMON/TBR/HH
      
      IF (NG.GT.501) THEN
      PRINT *,'Vector defining transformation of data > 501, stop'
      STOP
      END IF

      
      IF(N.ge.NMAX) then
      print *,'The number of wavelength points >',NMAX-1, ' stop'
      stop
      end if
      IF(N.lt.2) then
      print *,'The number of wavelength points < 2, stop'
      stop
      end if

      HT(1)=0.5d0*(T(2)-T(1))
      HT(N)=0.5d0*(T(N)-T(N-1))
      HH(1)=-100.0d0
      HH(N)=-100.0d0
      DO I=2,N-1
         HT(I)=0.5d0*(T(I+1)-T(I-1))
         HH(I)=-100.0d0
10    CONTINUE
      enddo


      IF(NU.gt.NMAX) then
      print *,'The number of maxima >',NMAX,' stop'
      stop
      end if
      IF(NV.gt.NMAX) then
      print *,'The number of minima >',NMAX,' stop'
      stop
      end if

      IF(NU.LT.1) Then
      print *,'The number of maxima < 1, stop'
      stop
      end if
      IF(NV.LT.1) Then
      print *,'The number of minima < 1, stop'
      stop
      end if

      RETURN
      END SUBROUTINE INITLEVELS


      SUBROUTINE TRANSF(N,T,A,TIMEV,VALUE,DER) 
C
C N number of data points
C TIMEV vector of time points
C A a vector of values of a function G(TIME)
C T independent time point
C VALUE is a value of a function at T, i.e. VALUE=G(T).
c DER=G'(t)
C
      USE SIZEMOD
      IMPLICIT NONE
      REAL*8, intent(inout):: VALUE, DER,T
C      INTEGER, PARAMETER :: RDIM = 10201
      REAL*8, DIMENSION(:), intent(in) :: A,TIMEV
      integer, intent(in) :: N
      REAL*8:: T1
      integer :: I
      
      IF (T.LT.TIMEV(1))  then
      der=(A(2)-A(1))/(TIMEV(2)-TIMEV(1))
      T1=T-TIMEV(1)
      VALUE=A(1)+T1*DER
      return
      end if
      IF (T.GT.TIMEV(N)) then
      der = (A(N)-A(N-1))/(TIMEV(N)-TIMEV(N-1))
      T1  = T-TIMEV(N)
      VALUE=A(N)+T1*DER
      return
      end if
      DO 5 I=2,N
      IF (T.LT.TIMEV(I)) GO TO 10
5     CONTINUE
10    I=I-1
      T1=T-TIMEV(I)
      DER=(A(I+1)-A(I))/(TIMEV(i+1)-TIMEV(I))
      VALUE=A(I)+T1*DER
      RETURN
      END SUBROUTINE TRANSF

      REAL*8 FUNCTION SPLE(N,T,A,TIMEV)
C
C N number of data points
C TIME vector of time points
C A a vector of values of a function G(TIME)
C T independent time point
C SPLE is a value of a function at T, i.e. SPLE=G(T).
C
      USE SIZEMOD
      IMPLICIT NONE
      INTEGER, INTENT(IN):: N

      REAL*8, INTENT(IN) :: T
      REAL*8, DIMENSION(:), INTENT(IN) :: A,TIMEV
      REAL*8 :: T1
      INTEGER :: I
      SPLE=-9.9d0
      IF (T.LT.TIMEV(1) .OR. T.GT.TIMEV(N)) RETURN
      DO 5 I=2,N
      IF (T.LT.TIMEV(I)) GO TO 10
5     CONTINUE
10    I=I-1
      T1=T-TIMEV(I)
      SPLE=A(I)+T1*(A(I+1)-A(I))/(TIMEV(i+1)-TIMEV(I))
      RETURN
      END FUNCTION SPLE



      SUBROUTINE COVG(XL0,XL2,XL4,COV,T,N)
C
C  COVG  evaluates: 
C
C  XL0,XL2,XL4 - spectral moments.
C
C  Covariance function and its four derivatives for a vector T of length N. 
C  It is saved in a vector COV; COV(1,...,N)=r(T), COV(N+1,...,2N)=r'(T), etc.
C  The vector COV should be of the length 5*N.
C
C  Covariance matrices COV1=r'(T-T), COV2=r''(T-T) and COV3=r'''(T-T) 
C  Dimension of  COV1, COV2  should be   N*N.
C
!      USE SIZEMOD
!     IMPLICIT NONE
C     INTEGER, PARAMETER:: NMAX = 101, RDIM = 10201
      REAL*8, PARAMETER:: ZERO = 0.0d0
      REAL*8, intent(inout) :: XL0,XL2,XL4
      REAL*8, DIMENSION(N,5), intent(in) :: COV
      REAL*8, DIMENSION(N), intent(in) :: T
      INTEGER, intent(in) :: N
      
      
C
C   COV(Y(T),Y(0)) = COV(:,1)
C
      XL0 = COV(1,1) 
!     XL0 = SPLE(NT,ZERO,COV(:,1),T)
C      
C    DERIVATIVE  COV(Y(T),Y(0)) = COV(:,2)
C
C    2-DERIVATIVE  COV(Y(T),Y(0)) = COV(:,3)
      XL2 = -COV(1,3) 
!     XL2 = -SPLE(NT,ZERO,COV(:,3),T)
C    3-DERIVATIVE  COV(Y(T),Y(0)) = COV(:,4)

C    4-DERIVATIVE  COV(Y(T),Y(0)) = COV(:,5)
      
      XL4 = COV(1,5) 
!     XL4 = SPLE(NT,ZERO,COV(:,5),T)

      RETURN
      END SUBROUTINE COVG

      SUBROUTINE INITINTEG()
      USE RINTMOD
      USE EPSMOD
      USE INFCMOD
      USE MREGMOD
!     IMPLICIT NONE      
C      COMMON /RINT/   C,FC
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /INFC/   ISQ,INF,INFO
     
      IAC = IAC_
      ISQ = ISQ_
      EPS = EPS_
      EPSS = EPSS_
      EPS0 = EPS0_
      C = C_
      
      FC = FI(C)-FI(-C)
!      CEPSS = 1.0d0-EPSS

      RETURN
      END SUBROUTINE INITINTEG

      END module cov2mmpdfmod