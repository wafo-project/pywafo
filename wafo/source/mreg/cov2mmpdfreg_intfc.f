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

      SUBROUTINE INITINTEG(EPS_,EPSS_,EPS0_,C_,IAC_,ISQ_)
!      Initiation of all constants and integration nodes 'INITINTEG'
      USE RINTMOD
      USE EPSMOD
      USE INFCMOD
      USE MREGMOD
      REAL*8 :: EPS_,EPSS_,EPS0_,C_
      INTEGER :: IAC_,ISQ_
Cf2py real*8, optional :: EPS_ = 0.01
Cf2py real*8, optional :: EPSS_ = 0.00005
Cf2py real*8, optional  :: EPS0_ = 0.00005
Cf2py real*8, optional  :: C_ = 4.5
Cf2py integer, optional :: IAC_ = 1
Cf2py integer, optional :: ISQ_ = 0
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

      subroutine cov2mmpdfreg(UVdens,t,COV,ULev,VLev,Tg,Xg,Nt,Nu,Nv,Ng,
     &   NIT)
      USE SIZEMOD
      USE EPSMOD
      USE CHECKMOD
      USE MREGMOD
      USE INTFCMOD
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
      real*8 Q0,SQ0,Q1,SQ1, U,V, XL0, XL2, XL4
      REAL*8 VDERI, DER, F, HHHH, VALUE
C      REAL*8 VV, CDER,SDER, CONST1, FM
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
C     INTEGER ::  fffff
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
!      CALL INITINTEG()

!     OPEN(UNIT=8,FILE='min.out')
!      OPEN(UNIT=9,FILE='Max.out')
!      OPEN(UNIT=10,FILE='Maxmin.out')
!      OPEN(UNIT=11,FILE='Maxmin.log')
c
c   OBS. we are using the variables R,R1,R2 R3 as a temporary storage
C   for transformation  g  of the process.

      N = Nt
      CALL INITLEVELS(T,HHT,Nt,NU,Nv)
C      CALL INITLEVELS(Ulev,NU,Vlev,NV,T,HHT,Nt,R1,R2,NG)
      IF( Tg(1) .gt. Tg(ng))  then
       print *,'Error Tg must be strictly increasing'
       return
      end if
      if(abs(Tg(ng)-Tg(1))*abs(Xg(ng)-Xg(1)).lt.0.01d0) then
       print *,'The transformation  g is singular, stop'
       return
      end if

!      do IV=1,Nt
!      print *, 'Cov', COV(IV,:)
!      end do

      DO IV=1,Nv
         V=Vlev(IV)
         CALL TRANSF(NG,V,Xg,Tg,VALUE,DER)
         VT(IV)=VALUE
         Vdd(IV)=DER
      enddo
      DO IU=1,Nu
         U = Ulev(IU)
         CALL TRANSF(NG,U,Xg,Tg,VALUE,DER)
         UT(IU)  = VALUE
         Udd(IU) = DER
         do IV=1,Nv
             UVdens(IU,IV)=0.0d0
         enddo
      enddo

      CALL COVG(XL0,XL2,XL4,R1,R2,R3,COV,T,Nt)


      Q0=XL4
      IF (Q0.le.1.0D0+EPS) then
      Print *,'Covariance structure is singular, stop.'
      return
      end if
      SQ0 = SQRT(Q0)
      Q1  = XL0-XL2*XL2/XL4
      IF (Q1.le.EPS) then
      Print *,'Covariance structure is singular, stop.'
      return
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

c10    CONTINUE
      enddo
      DO I=1,Nt
      DO J=1,Nt
C
C   R1 contains Cov(X(T(I)),X'(T(J))|X'(0),X''(0),X(0))
C
      R1(J+(I-1)*N) = R1(J+(I-1)*N) -  COV(I,2)*(COV(J,3)/XL2)
     1   - (B0(I)*DB0(J)/Q0) -  (B1(I)*DB1(J)/Q1)

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
c15    CONTINUE
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

         enddo

         DO I3=1,I
            DBI(I3) = R3(I3+(I-1)*N) - (DDB2(I)*DB2(I3)/Q(I))
            BI(I3)  = R2(I3+(I-1)*N) - (DB2(I)*DB2(I3)/Q(I))
         enddo
         DO I3=1,I-1
            AI(I3)=0.0d0
            AI(I3+I-1)=DB0(I3)/SQ0
            AI(I3+2*(I-1))=DB1(I3)/SQ1
            AI(I3+3*(I-1))=DB2(I3)/SQ(I)
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

            enddo
            enddo
 41         CONTINUE
         END IF

C  Here the covariance of the problem would be initiated

            INF=0
            Print *,'   Laps to go:',Nt-I+1
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

      hhhh=0.0d0
      do Iu=1,Nu
      do Iv=1,Nv
!       WRITE(10,300) Ulev(iu),Vlev(iv),UVdens(iu,iv)
        hhhh=hhhh+UVdens(iu,iv)
      enddo
      enddo
      if (nu.gt.1.and.nv.gt.1) then
      VALUE = (Ulev(2)-Ulev(1))*(Vlev(2)-Vlev(1))*hhhh
      print *,'SumSum f_uv *du*dv=', VALUE
      end if

C      sder=sqrt(XL4-XL2*XL2/XL0)
C      cder=-XL2/sqrt(XL0)
C      const1=1/sqrt(XL0*XL4)
C      DO 95 IU=1,NU
C        U=UT(IU)
C        FM=Udd(IU)*const1*exp(-0.5*U*U/XL0)*PMEAN(-cder*U,sder)
C        WRITE(9,300) Ulev(IU),FM
C 95   continue
C      DO 105 IV=1,NV
C        V=VT(IV)
C        VV=cder*V
C        Fm=Vdd(IV)*const1*exp(-0.5*V*V/XL0)*PMEAN(VV,sder)
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
