C     Version  1994-X-18 

C     This is a new version of WAMP program computing crest-trough wavelength 
C     and amplitude  density.
C     
C     revised pab 2007
C     -moved all common blocks into modules
C     -renamed from minmax to sp2mmpdfreg + fixed some bugs
C     revised pab July 2007
!     -renamed from sp2mmpdfreg to cov2mmpdfreg

      PROGRAM cov2mmpdfreg
      USE SIZEMOD
      USE EPSMOD
      USE CHECKMOD
      USE MREGMOD
      IMPLICIT NONE
      real*8 Q0,SQ0,Q1,SQ1, AA, BB, DAI, AI , U,V,VV, XL0, XL2, XL4
      REAL*8 VDERI, CDER,SDER, DER, CONST, F, HHHH,FM, VALUE
C	INTEGER, PARAMETER :: MMAX = 5, NMAX = 101, RDIM = 10201
      REAL*8, DIMENSION(NMAX) :: HHT,T,Ulev,Vlev,VT,UT,Vdd,Udd
      REAL*8, DIMENSION(RDIM) :: R,R1,R2,R3
      REAL*8, DIMENSION(5*NMAX) :: COV
      REAL*8, DIMENSION(NMAX,NMAX) :: UVdens
C      DIMENSION UVdens(NMAX,NMAX),HHT(NMAX)
C      DIMENSION T(NMAX),Ulev(NMAX),Vlev(NMAX)
C      DIMENSION VT(NMAX),UT(NMAX),Vdd(NMAX),Udd(NMAX)
C      DIMENSION COV(5*NMAX),R(RDIM),R1(RDIM),R2(RDIM),R3(RDIM)
      DIMENSION AA(MMAX-2,MMAX-2),BB(MMAX+1),DAI(MMAX),AI((MMAX+1)*NMAX)

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
      INTEGER :: J,I,I1,I2,I3,IU, IV, NU,NV,NG,N,NIT, NNIT, INF
	  INTEGER ::  fffff
C	REAL*8 EPS0
C	INTEGER III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101 , III0
C      COMMON/CHECK1/III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101 
C      COMMON/CHECKQ/III0 
C      COMMON /EPS/  EPS,EPSS,CEPSS

C
C  Initiation of all constants and integration nodes 'INITINTEG'
C
      CALL INITINTEG(NIT)
c
c   OBS. we are using the variables R,R1,R2 R3 as a temporary storage 
C   for transformation  g  of the process.



c
      CALL INITLEVELS(Ulev,NU,Vlev,NV,T,HHT,N,R1,R2,NG)
      IF( R1(1) .gt. R1(ng))  then
      do 13 I=1,ng
      R3(I)=R1(I)
      R(I) =R2(I)
13    continue
      do 17 i=1,ng
      R1(i) = R3(ng-i+1)
      R2(i) = R(ng-i+1)
17    continue
      end if
      if(abs(R1(ng)-R1(1))*abs(R2(ng)-R2(1)).lt.0.01d0) then
      print *,'The transformation  g is singular, stop'
      stop
      end if
      DO 14 IV=1,Nv
         V=Vlev(IV)
         CALL TRANSF(NG,V,R2,R1,VALUE,DER) 
         VT(IV)=VALUE
         Vdd(IV)=DER
14    continue
      DO 16 IU=1,Nu
         U = Ulev(IU)
         CALL TRANSF(NG,U,R2,R1,VALUE,DER) 
         UT(IU)  = VALUE
         Udd(IU) = DER
         do 16 IV=1,Nv
             UVdens(IU,IV)=0.0d0
16    CONTINUE
      
      
      CALL COVG(XL0,XL2,XL4,COV,R1,R2,R3,T,N)
      
      
      Q0=XL4
      IF (Q0.le.1.0D0+EPS) then
      Print *,'Covariance structure is singular, stop.'
      stop
      end if
      SQ0 = SQRT(Q0)
      Q1  = XL0-XL2*XL2/XL4
      IF (Q1.le.eps) then
      Print *,'Covariance structure is singular, stop.'
      stop
      end if
      SQ1 = SQRT(Q1)
      DO 10 I=1,N
        B0(I)  =-COV(I+2*N)
        DB0(I) =-COV(I+3*N)
        DDB0(I)=-COV(I+4*N)
      
        B1(I)  =COV(I)+COV(I+2*N)*(XL2/XL4)
        DB1(I) =COV(I+N)+COV(I+3*N)*(XL2/XL4)
        DDB1(I)=COV(I+2*N)+XL2*(COV(I+4*N)/XL4)
C
C       Q(I) contains Var(X(T(i))|X'(0),X''(0),X(0))
C    VDER(I) contains Var(X''(T(i))|X'(0),X''(0),X(0))
C
        Q(I)=XL0 - COV(I+N)*(COV(I+N)/XL2) - B0(I)*(B0(I)/Q0)
     1     -B1(I)*(B1(I)/Q1)
        VDER(I)=XL4 - (COV(I+3*N)*COV(I+3*N))/XL2 - (DDB0(I)*DDB0(I))/Q0
     1     - (DDB1(I)*DDB1(I))/Q1 
      

C
C       DDB2(I) contains Cov(X''(T(i)),X(T(i))|X'(0),X''(0),X(0))
C
        DDB2(I)=-XL2 - (COV(I+N)*COV(I+3*N))/XL2 - DDB0(I)*(B0(I)/Q0)
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
      DO 15 I=1,N
      DO 15 J=1,N
C      
C   R1 contains Cov(X(T(I)),X'(T(J))|X'(0),X''(0),X(0)) 
C
      R1(J+(I-1)*N)=R1(J+(I-1)*N) -  COV(I+N)*(COV(J+2*N)/XL2) 
     1 -  (B0(I)*DB0(J)/Q0) -  (B1(I)*DB1(J)/Q1) 
      
C      
C   R2 contains Cov(X'(T(I)),X'(T(J))|X'(0),X''(0),X(0))
C
      R2(J+(I-1)*N) = -R2(J+(I-1)*N) - COV(I+2*N)*(COV(J+2*N)/XL2) 
     1   - DB0(I)*DB0(J)/Q0  - DB1(I)*(DB1(J)/Q1) 
C      
C   R3 contains Cov(X''(T(I)),X'(T(J))|X'(0),X''(0),X(0))
C
      R3(J+(I-1)*N) = R3(J+(I-1)*N) - COV(I+3*N)*(COV(J+2*N)/XL2) 
     1   - DB0(J)*(DDB0(I)/Q0)  - DDB1(I)*(DB1(J)/Q1) 
15    CONTINUE

C  The initiations are finished and we are beginning with 3 loops
C  on T=T(I), U=Ulevels(IU), V=Ulevels(IV), U>V.
      
      DO 20 I=1,N

           NNIT=NIT
           IF (Q(I).LE.EPS) GO TO 20

         DO 30 I1=1,I
            DB2(I1)=R1(I1+(I-1)*N) 

C     Cov(X'(T(I1)),X(T(i))|X'(0),X''(0),X(0)) 
C     DDB2(I) contains Cov(X''(T(i)),X(T(i))|X'(0),X''(0),X(0))

 30      CONTINUE
	
         DO 50 I3=1,I
            DBI(I3) = R3(I3+(I-1)*N) - (DDB2(I)*DB2(I3)/Q(I)) 
            BI(I3)  = R2(I3+(I-1)*N) - (DB2(I)*DB2(I3)/Q(I))
 50      CONTINUE
         DO 51 I3=1,I-1
            AI(I3)=0.0d0
            AI(I3+I-1)=DB0(I3)/SQ0 
            AI(I3+2*(I-1))=DB1(I3)/SQ1
            AI(I3+3*(I-1))=DB2(I3)/SQ(I)
 51      CONTINUE
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
            DO 40 I1=1,I-1
               DO 40 I2=1,I-1

C   R contains Cov(X'(T(I1)),X'(T(I2))|X'(0),X''(0),X(0),X(I)) 

      R(I2+(I1-1)*(I-1))=R2(I2+(I1-1)*N)-(DB2(I1)*DB2(I2)/Q(I)) 

 40         CONTINUE
 41         CONTINUE
         END IF

C  Here the covariance of the problem would be innitiated

            INF=0
            Print *,'   Laps to go:',N-I+1
         DO 80 IV=1,Nv
            V=VT(IV)
!            IF (ABS(V).GT.5.0D0) GO TO 80                  
            IF (Vdd(IV).LT.EPS0) GO TO 80
            DO 60 IU=1,Nu
                  U=UT(IU)
				IF (U.LE.V) go to 60
!                  IF (ABS(U).GT.5.0D0) GO TO 60                  
                  IF (Udd(IU).LT.EPS0) GO TO 60
                  BB(1)=0.0d0
                  BB(2)=U
                  BB(3)=V
!	if (IV.EQ.2.AND.IU.EQ.1) THEN	
!	fffff = 10
!	endif
                 
      CALL MREG(F,R,BI,DBI,AA,BB,AI,DAI,VDERI,3,I-1,NNIT,INF)
             INF=1
             UVdens(IU,IV) = UVdens(IU,IV) + Udd(IU)*Vdd(IV)*HHT(I)*F
!	if (F.GT.0.01.AND.U.GT.2.AND.V.LT.-2) THEN  
! 	if (N-I+1 .eq. 38.and.IV.EQ.26.AND.IU.EQ.16) THEN
!	if (IV.EQ.32.AND.IU.EQ.8.and.I.eq.11) THEN		
!          PRINT * ,' R:', R(1:I)
!	    PRINT * ,' BI:', BI(1:I)
!		PRINT * ,' DBI:', DBI(1:I)
!		PRINT * ,' DB2:', DB2(1:I)
!		PRINT * ,' DB0(1):', DB0(1)
!		PRINT * ,' DB1(1):', DB1(1)
!          PRINT * ,' DAI:', DAI
!		PRINT * ,' BB:', BB
!	    PRINT * ,' VDERI:', VDERI
!          PRINT * ,' F    :', F
!		PRINT * ,' UVDENS :',  UVdens(IU,IV)
!	    fffff = 10
!	endif

 60         CONTINUE
 80       continue
 20   CONTINUE
       hhhh=0.0d0
       do 90 Iu=1,Nu
       do 90 Iv=1,Nv
       WRITE(10,300) Ulev(iu),Vlev(iv),UVdens(iu,iv)
       hhhh=hhhh+UVdens(iu,iv)
 90    continue
      if (nu.gt.1.and.nv.gt.1) then
      write(11,*) 'SumSum f_uv *du*dv='
     1,(Ulev(2)-Ulev(1))*(Vlev(2)-Vlev(1))*hhhh
      end if
      
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

      write(11,*) 'Rate of calls RINDT0:',float(iii01)/float(III0)
      write(11,*) 'Rate of calls RINDT1:',float(iii11)/float(III0)
      write(11,*) 'Rate of calls RINDT2:',float(iii21)/float(III0)
      write(11,*) 'Rate of calls RINDT3:',float(iii31)/float(III0)
      write(11,*) 'Rate of calls RINDT4:',float(iii41)/float(III0)
      write(11,*) 'Rate of calls RINDT5:',float(iii51)/float(III0)
      write(11,*) 'Rate of calls RINDT6:',float(iii61)/float(III0)
      write(11,*) 'Rate of calls RINDT7:',float(iii71)/float(III0)
      write(11,*) 'Rate of calls RINDT8:',float(iii81)/float(III0)
      write(11,*) 'Rate of calls RINDT9:',float(iii91)/float(III0)
      write(11,*) 'Rate of calls RINDT10:',float(iii101)/float(III0)
      write(11,*) 'Number of calls of RINDT*',III0

      CLOSE(UNIT=8)
      CLOSE(UNIT=9)
      CLOSE(UNIT=10)
      CLOSE(UNIT=11)

 300  FORMAT(4(3X,F10.6))
      STOP
      END

      SUBROUTINE INITLEVELS(ULEVELS,NU,Vlevels,Nv,T,HT,N,TG,XG,NG)
      USE TBRMOD
      USE SIZEMOD
      IMPLICIT NONE
C	INTEGER, PARAMETER:: NMAX = 101, RDIM = 10201
C      DIMENSION ULEVELS(1),Vlevels(1),T(1),HT(1),TG(1),XG(1),HH(101)
      REAL*8, DIMENSION(NMAX), intent(inout) :: ULEVELS,Vlevels,T,HT
      REAL*8, DIMENSION(RDIM), intent(inout) :: TG,XG
      INTEGER, intent(inout) :: NG
      REAL*8 :: UMIN,UMAX,VMIN,VMAX, HU,HV
      integer :: N, I, NU, NV
C	REAL*8, DIMENSION(NMAX) :: HH
C      COMMON/TBR/HH
      OPEN(UNIT=2,FILE='transf.in')
      OPEN(UNIT=4,FILE='Mm.in')
      OPEN(UNIT=3,FILE='t.in')
      

      NG=1
 12   READ (2,*,END=11) TG(NG),XG(NG)
      NG=NG+1
      GO TO 12
 11   CONTINUE
      NG=NG-1
      IF (NG.GT.501) THEN
      PRINT *,'Vector defining transformation of data > 501, stop'
      STOP
      END IF


      N=1
 32   READ (3,*,END=31) T(N)
      N=N+1
      GO TO 32
 31   CONTINUE
      N=N-1      
      
      CLOSE(UNIT=3)
      
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
      DO 10 I=2,N-1
         HT(I)=0.5d0*(T(I+1)-T(I-1))
         HH(I)=-100.0d0
10    CONTINUE
      

      
      READ(4,*) Umin,Umax,NU
      READ(4,*) Vmin,Vmax,NV

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

      Ulevels(1)=Umax
      IF (NU.lt.2) go to 25
      HU=(Umax-Umin)/DBLE(NU-1)
      DO 20 I=1,NU-1
         ULEVELS(I+1)=Umax-DBLE(I)*HU
20    CONTINUE

 25   continue
      Vlevels(1)=Vmax
      IF (NV.lt.2) go to 35
      HV=(Vmax-Vmin)/DBLE(NV-1)
      DO 30 I=1,Nv-1
         VLEVELS(I+1)=Vmax-DBLE(I)*HV
30    CONTINUE
35    continue
      CLOSE(UNIT=4)
      RETURN
      END


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
      REAL*8, DIMENSION(RDIM), intent(in) :: A,TIMEV
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
      END

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
      REAL*8, DIMENSION(5*NMAX), INTENT(IN) :: A,TIMEV
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
      END



      SUBROUTINE COVG(XL0,XL2,XL4,COV,COV1,COV2,COV3,T,N)
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
      USE SIZEMOD
!	IMPLICIT NONE
C	INTEGER, PARAMETER:: NMAX = 101, RDIM = 10201
      REAL*8, PARAMETER:: ZERO = 0.0d0
      REAL*8, intent(inout) :: XL0,XL2,XL4
      REAL*8, DIMENSION(5*NMAX), intent(inout) :: COV
      REAL*8, DIMENSION(5*NMAX) :: A, TIMEV
      REAL*8, DIMENSION(RDIM), intent(inout) :: COV1,COV2,COV3
      REAL*8, DIMENSION(NMAX), intent(in) :: T
      INTEGER, intent(in) :: N
      integer :: NT, I, J, II
      REAL*8 :: TT, T0
      OPEN(UNIT=32,FILE='Cd0.in')
      OPEN(UNIT=33,FILE='Cd1.in')
      OPEN(UNIT=34,FILE='Cd2.in')
      OPEN(UNIT=35,FILE='Cd3.in')
      OPEN(UNIT=36,FILE='Cd4.in')
C
C   COV(Y(T),Y(0))
C

      NT=1
 12   READ (32,*,END=11) TIMEV(NT),A(NT)
      NT=NT+1
      GO TO 12
 11   CONTINUE
      NT=NT-1
      
      
      XL0=SPLE(NT,ZERO,A,TIMEV)

      DO 10 I=1,N
      COV(I)=SPLE(NT,T(I),A,TIMEV)
10    CONTINUE

C     
C    DERIVATIVE  COV(Y(T),Y(0))
C

      NT=1
 22   READ (33,*,END=21) TIMEV(NT),A(NT)
      NT=NT+1
      GO TO 22
 21   CONTINUE
      NT=NT-1      

      II=0
      DO 20 I=1,N
      COV(I+N)=SPLE(NT,T(I),A,TIMEV)
      DO 20 J=1,N
      II=II+1
      T0=T(J)-T(I)
      TT=ABS(T0)
      COV1(II)=SPLE(NT,TT,A,TIMEV)
      IF (T0.LT.0.0d0) COV1(II)=-COV1(II)
20    CONTINUE

C    2-DERIVATIVE  COV(Y(T),Y(0))

      NT=1
 32   READ (34,*,END=31) TIMEV(NT),A(NT)
      NT=NT+1
      GO TO 32
 31   CONTINUE
      NT=NT-1      

      II=0
      XL2=-SPLE(NT,ZERO,A,TIMEV)

      DO 30 I=1,N
      COV(I+2*N)=SPLE(NT,T(I),A,TIMEV)
      DO 30 J=1,N
      II=II+1
      TT=ABS(T(J)-T(I))
      COV2(II)=SPLE(NT,TT,A,TIMEV)
30    CONTINUE

C    3-DERIVATIVE  COV(Y(T),Y(0))

            NT=1
 42   READ (35,*,END=41) TIMEV(NT),A(NT)
      NT=NT+1
      GO TO 42
 41   CONTINUE
      NT=NT-1      

      
      II=0
      DO 40 I=1,N
      COV(I+3*N)=SPLE(NT,T(I),A,TIMEV)
      DO 40 J=1,N
      II=II+1
      T0=T(J)-T(I)
      TT=ABS(T0)
      COV3(II)=SPLE(NT,TT,A,TIMEV)
      IF (T0.LT.0.0d0) COV3(II)=-COV3(II)
40    CONTINUE



C    4-DERIVATIVE  COV(Y(T),Y(0))

      NT=1
 52   READ (36,*,END=51) TIMEV(NT),A(NT)
      NT=NT+1
      GO TO 52
 51   CONTINUE
      NT=NT-1
      
      XL4=SPLE(NT,ZERO,A,TIMEV)

      DO 50 I=1,N
      COV(I+4*N)=SPLE(NT,T(I),A,TIMEV)
50    CONTINUE
      CLOSE(UNIT=32)
      CLOSE(UNIT=33)
      CLOSE(UNIT=34)
      CLOSE(UNIT=35)
      CLOSE(UNIT=36)
      RETURN
      END

      SUBROUTINE INITINTEG(NIT)
      USE RINTMOD
      USE EPSMOD
      USE INFCMOD
      USE MREGMOD
!	IMPLICIT NONE
      INTEGER, intent(inout) :: NIT
!	INTEGER ISQ1
C      dimension  INF(10),INFO(10)
      
C      COMMON /RINT/   C,FC
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /INFC/   ISQ,INF,INFO
      OPEN(UNIT=1,FILE='accur.in')
      OPEN(UNIT=8,FILE='min.out')
      OPEN(UNIT=9,FILE='Max.out')
      OPEN(UNIT=10,FILE='Maxmin.out')
      OPEN(UNIT=11,FILE='Maxmin.log')

      READ(1,*) NIT,IAC,ISQ
      READ(1,*) EPS,EPSS,EPS0
      
      CLOSE (UNIT=1)
      
      FC=FI(C)-FI(-C)
      CEPSS=1.0d0-EPSS

      RETURN
      END

