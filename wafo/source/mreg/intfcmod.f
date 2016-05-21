      MODULE INTFCMOD
      IMPLICIT NONE
      PUBLIC :: INITLEVELS, TRANSF, COVG

      CONTAINS
      SUBROUTINE INITLEVELS(T,HT,N,NU,Nv)
      USE TBRMOD
      USE SIZEMOD
      IMPLICIT NONE
C     INTEGER, PARAMETER:: NMAX = 101, RDIM = 10201
C      DIMENSION ULEVELS(1),Vlevels(1),T(1),HT(1),TG(1),XG(1),HH(101)
      REAL*8, DIMENSION(:), intent(in) :: T
      REAL*8, DIMENSION(:), intent(out) :: HT
C      INTEGER, intent(in) :: NG
      REAL*8 :: UMIN,UMAX,VMIN,VMAX, HU,HV
      integer :: N, I, NU, NV
C     REAL*8, DIMENSION(NMAX) :: HH
C      COMMON/TBR/HH

C      IF (NG.GT.501) THEN
C      PRINT *,'Vector defining transformation of data > 501, stop'
C      STOP
C      END IF


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
c10    CONTINUE
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
      DO I=2,N
      IF (T.LT.TIMEV(I)) GO TO 10
      ENDDO
 10   I=I-1
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
      DO I=2,N
      IF (T.LT.TIMEV(I)) GO TO 10
      ENDDO
 10   I=I-1
      T1=T-TIMEV(I)
      SPLE=A(I)+T1*(A(I+1)-A(I))/(TIMEV(i+1)-TIMEV(I))
      RETURN
      END FUNCTION SPLE

      SUBROUTINE COVG(XL0,XL2,XL4,COV1,COV2,COV3,COV,T,N)
C
C  Covariance function and its four derivatives for a vector T of length N
C  is assumed in a vector COV; COV(1,...,N,1)=r(T), COV(1,...,N, 2)=r'(T), etc.
C  The vector COV should be of the shape N x 5.
C
C  COVG  Returns:
C  XL0,XL2,XL4 - spectral moments.
C
C  Covariance matrices COV1=r'(T-T), COV2=r''(T-T) and COV3=r'''(T-T)
C  Dimension of  COV1, COV2  should be atleast  N*N.
C
      USE SIZEMOD
!     IMPLICIT NONE
C     INTEGER, PARAMETER:: NMAX = 101, RDIM = 10201
      REAL*8, PARAMETER:: ZERO = 0.0d0
      REAL*8, intent(inout) :: XL0,XL2,XL4
      REAL*8, DIMENSION(N,5), intent(in) :: COV
      REAL*8, DIMENSION(N), intent(in) :: T
      REAL*8, DIMENSION(RDIM), intent(inout) :: COV1,COV2,COV3
      INTEGER, intent(in) :: N
      integer :: I, J, II
      REAL*8 :: TT, T0
C
C                  COV(Y(T),Y(0)) = COV(:,1)
C      DERIVATIVE  COV(Y(T),Y(0)) = COV(:,2)
C    2-DERIVATIVE  COV(Y(T),Y(0)) = COV(:,3)
C    3-DERIVATIVE  COV(Y(T),Y(0)) = COV(:,4)
C    4-DERIVATIVE  COV(Y(T),Y(0)) = COV(:,5)

      XL0 = COV(1,1)
      XL2 = -COV(1,3)
      XL4 = COV(1,5)
!     XL0 =  SPLE(NT, ZERO, COV(:,1), T)
!     XL2 = -SPLE(NT, ZERO, COV(:,3), T)
!     XL4 =  SPLE(NT, ZERO, COV(:,5), T)

      II=0
      DO I=1,N
      DO J=1,N
      II = II+1
      T0 = T(J)-T(I)
      TT = ABS(T0)
      COV1(II) = SPLE(N, TT, COV(:,2), T)
      COV2(II) = SPLE(N, TT, COV(:,3), T)
      COV3(II) = SPLE(N, TT, COV(:,4), T)
      IF (T0.LT.0.0d0) then
      COV1(II)=-COV1(II)
      COV3(II)=-COV3(II)
      endif
      enddo
      enddo
      RETURN
      END SUBROUTINE COVG
      END module intfcmod
