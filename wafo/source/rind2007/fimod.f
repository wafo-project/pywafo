      MODULE TRIVARIATEVAR
!     Global variables used in calculation of TRIVARIATE
!     normal and TRIVARIATE student T probabilties
      INTEGER :: NU
      DOUBLE PRECISION :: H1, H2, H3, R23, RUA, RUB, AR, RUC
      END MODULE TRIVARIATEVAR
!
! FIMOD contains functions for calculating 1D, 2D and 3D Normal and student T probabilites
!       and  1D expectations
      MODULE FIMOD
!      USE ERFCOREMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: NORMPRB, FI, FIINV, MVNLIMITS, MVNLMS, BVU,BVNMVN
      PUBLIC :: GAUSINT, GAUSINT2, EXLMS, EXINV
      PUBLIC :: STUDNT, BVTL, TVTL, TVNMVN

      INTERFACE NORMPRB
      MODULE PROCEDURE NORMPRB
      END INTERFACE

      INTERFACE FI
      MODULE PROCEDURE FI
      END INTERFACE

      INTERFACE FI2
      MODULE PROCEDURE FI2
      END INTERFACE

      INTERFACE FIINV
      MODULE PROCEDURE FIINV
      END INTERFACE

      INTERFACE  MVNLIMITS
      MODULE PROCEDURE  MVNLIMITS
      END INTERFACE

      INTERFACE  MVNLMS
      MODULE PROCEDURE  MVNLMS
      END INTERFACE

      INTERFACE BVU
      MODULE PROCEDURE BVU
      END INTERFACE

      INTERFACE BVNMVN
      MODULE PROCEDURE BVNMVN
      END INTERFACE

      INTERFACE STUDNT
      MODULE PROCEDURE STUDNT
      END INTERFACE

      INTERFACE BVTL
      MODULE PROCEDURE BVTL
      END INTERFACE

      INTERFACE TVTL
      MODULE PROCEDURE TVTL
      END INTERFACE

      INTERFACE GAUSINT
      MODULE PROCEDURE GAUSINT
      END INTERFACE

      INTERFACE GAUSINT2
      MODULE PROCEDURE GAUSINT2
      END INTERFACE

      INTERFACE EXLMS
      MODULE PROCEDURE EXLMS
      END INTERFACE

      INTERFACE EXINV
      MODULE PROCEDURE EXINV
      END INTERFACE

      CONTAINS
      FUNCTION FIINV(P) RESULT (VAL)
      IMPLICIT NONE
*
*     ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3
*
*     Produces the normal deviate Z corresponding to a given lower
*     tail area of P.
*       Absolute error less than 1e-13
*       Relative error less than 1e-15 for abs(VAL)>0.1
*
*     The hash sums below are the sums of the mantissas of the
*     coefficients.   They are included for use in checking
*     transcription.
*
      DOUBLE PRECISION, INTENT(in) :: P
      DOUBLE PRECISION :: VAL
!local variables
      DOUBLE PRECISION SPLIT1, SPLIT2, CONST1, CONST2, ONE, ZERO, HALF,
     &     A0, A1, A2, A3, A4, A5, A6, A7, B1, B2, B3, B4, B5, B6, B7,
     &     C0, C1, C2, C3, C4, C5, C6, C7, D1, D2, D3, D4, D5, D6, D7,
     &     E0, E1, E2, E3, E4, E5, E6, E7, F1, F2, F3, F4, F5, F6, F7,
     &     Q, R
      PARAMETER ( SPLIT1 = 0.425D0, SPLIT2 = 5.D0,
     &     CONST1 = 0.180625D0, CONST2 = 1.6D0,
     &     ONE = 1.D0, ZERO = 0.D0, HALF = 0.5D0 )
*
*     Coefficients for P close to 0.5
*
      PARAMETER (
     *     A0 = 3.38713 28727 96366 6080D0,
     *     A1 = 1.33141 66789 17843 7745D+2,
     *     A2 = 1.97159 09503 06551 4427D+3,
     *     A3 = 1.37316 93765 50946 1125D+4,
     *     A4 = 4.59219 53931 54987 1457D+4,
     *     A5 = 6.72657 70927 00870 0853D+4,
     *     A6 = 3.34305 75583 58812 8105D+4,
     *     A7 = 2.50908 09287 30122 6727D+3,
     *     B1 = 4.23133 30701 60091 1252D+1,
     *     B2 = 6.87187 00749 20579 0830D+2,
     *     B3 = 5.39419 60214 24751 1077D+3,
     *     B4 = 2.12137 94301 58659 5867D+4,
     *     B5 = 3.93078 95800 09271 0610D+4,
     *     B6 = 2.87290 85735 72194 2674D+4,
     *     B7 = 5.22649 52788 52854 5610D+3 )
*     HASH SUM AB    55.88319 28806 14901 4439
*
*     Coefficients for P not close to 0, 0.5 or 1.
*
      PARAMETER (
     *     C0 = 1.42343 71107 49683 57734D0,
     *     C1 = 4.63033 78461 56545 29590D0,
     *     C2 = 5.76949 72214 60691 40550D0,
     *     C3 = 3.64784 83247 63204 60504D0,
     *     C4 = 1.27045 82524 52368 38258D0,
     *     C5 = 2.41780 72517 74506 11770D-1,
     *     C6 = 2.27238 44989 26918 45833D-2,
     *     C7 = 7.74545 01427 83414 07640D-4,
     *     D1 = 2.05319 16266 37758 82187D0,
     *     D2 = 1.67638 48301 83803 84940D0,
     *     D3 = 6.89767 33498 51000 04550D-1,
     *     D4 = 1.48103 97642 74800 74590D-1,
     *     D5 = 1.51986 66563 61645 71966D-2,
     *     D6 = 5.47593 80849 95344 94600D-4,
     *     D7 = 1.05075 00716 44416 84324D-9 )
*     HASH SUM CD    49.33206 50330 16102 89036
*
*	Coefficients for P near 0 or 1.
*
      PARAMETER (
     *     E0 = 6.65790 46435 01103 77720D0,
     *     E1 = 5.46378 49111 64114 36990D0,
     *     E2 = 1.78482 65399 17291 33580D0,
     *     E3 = 2.96560 57182 85048 91230D-1,
     *     E4 = 2.65321 89526 57612 30930D-2,
     *     E5 = 1.24266 09473 88078 43860D-3,
     *     E6 = 2.71155 55687 43487 57815D-5,
     *     E7 = 2.01033 43992 92288 13265D-7,
     *     F1 = 5.99832 20655 58879 37690D-1,
     *     F2 = 1.36929 88092 27358 05310D-1,
     *     F3 = 1.48753 61290 85061 48525D-2,
     *     F4 = 7.86869 13114 56132 59100D-4,
     *     F5 = 1.84631 83175 10054 68180D-5,
     *     F6 = 1.42151 17583 16445 88870D-7,
     *     F7 = 2.04426 31033 89939 78564D-15 )
*     HASH SUM EF    47.52583 31754 92896 71629
*
      Q = ( P - HALF)
      IF ( ABS(Q) .LE. SPLIT1 ) THEN ! Central range.
         R = CONST1 - Q*Q
         VAL = Q*( ( ( ((((A7*R + A6)*R + A5)*R + A4)*R + A3)
     *                  *R + A2 )*R + A1 )*R + A0 )
     *            /( ( ( ((((B7*R + B6)*R + B5)*R + B4)*R + B3)
     *                  *R + B2 )*R + B1 )*R + ONE)
      ELSE ! near the endpoints
         R = MIN( P, ONE - P )
         IF  (R .GT.ZERO) THEN ! ( 2.d0*R .GT. CFxCutOff) THEN ! R .GT.0.d0
            R = SQRT( -LOG(R) )
            IF ( R .LE. SPLIT2 ) THEN
               R = R - CONST2
               VAL = ( ( ( ((((C7*R + C6)*R + C5)*R + C4)*R + C3)
     *                      *R + C2 )*R + C1 )*R + C0 )
     *                /( ( ( ((((D7*R + D6)*R + D5)*R + D4)*R + D3)
     *                      *R + D2 )*R + D1 )*R + ONE )
            ELSE
               R = R - SPLIT2
               VAL = ( ( ( ((((E7*R + E6)*R + E5)*R + E4)*R + E3)
     *                      *R + E2 )*R + E1 )*R + E0 )
     *                /( ( ( ((((F7*R + F6)*R + F5)*R + F4)*R + F3)
     *                      *R + F2 )*R + F1 )*R + ONE )
            END IF
         ELSE
            VAL = 37.0d0 !9.D0 !XMAX 9.d0
         END IF
         IF ( Q  <  ZERO ) VAL = - VAL
      END IF
      RETURN
      END FUNCTION FIINV
                                ! *********************************
      SUBROUTINE NORMPRB(Z, P, Q)
!      USE ERFCOREMOD
!      USE GLOBALDATA, ONLY : XMAX
! Normal distribution probabilities accurate to 18 digits between
! -XMAX and XMAX
!
! Z    = no. of standard deviations from the mean.
! P, Q = probabilities to the left & right of Z.   P + Q = 1.
!
! by pab 23.03.2003
!
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)            :: Z
      DOUBLE PRECISION, INTENT(OUT)           :: P
      DOUBLE PRECISION, INTENT(OUT), OPTIONAL ::  Q
!Local variables
      DOUBLE PRECISION            :: PP, ZABS
      DOUBLE PRECISION, PARAMETER :: ZERO  = 0.0D0
      DOUBLE PRECISION, PARAMETER :: ONE   = 1.0D0
      DOUBLE PRECISION, PARAMETER :: SQ2M1 = 0.70710678118655D0 !     1/SQRT(2)
      DOUBLE PRECISION, PARAMETER :: HALF  = 0.5D0
      DOUBLE PRECISION, PARAMETER :: XMAX  = 37D0
      ZABS = ABS(Z)
!
!     |Z| > 37  (or XMAX)
!
      IF ( ZABS .GT. XMAX ) THEN
         IF (Z > ZERO) THEN
            P = ONE
            IF (PRESENT(Q)) Q = ZERO
         ELSE
            P = ZERO
            IF (PRESENT(Q)) Q = ONE
         END IF
      ELSE
!
!     |Z| <= 37
!
         PP = DERFC(ZABS*SQ2M1)*HALF

         IF (Z  <  ZERO) THEN
            P = PP
            IF (PRESENT(Q)) Q = ONE - PP
         ELSE
            P = ONE - PP
            IF (PRESENT(Q)) Q = PP
         END IF
      END IF

      RETURN
      END SUBROUTINE NORMPRB
      FUNCTION FI( Z ) RESULT (VALUE)
!      USE ERFCOREMOD
!      USE GLOBALDATA, ONLY : XMAX
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: Z
      DOUBLE PRECISION :: VALUE
! Local variables
      DOUBLE PRECISION :: ZABS
      DOUBLE PRECISION, PARAMETER:: SQ2M1 = 0.70710678118655D0 !     1/SQRT(2)
      DOUBLE PRECISION, PARAMETER:: HALF = 0.5D0
      DOUBLE PRECISION, PARAMETER:: XMAX = 37.D0
      ZABS = ABS(Z)
*
*     |Z| > 37  (or XMAX)
*
      IF ( ZABS .GT. XMAX ) THEN
         IF (Z  <  0.0D0) THEN
            VALUE = 0.0D0
         ELSE
            VALUE = 1.0D0
         ENDIF
      ELSE
         VALUE = DERFC(-Z*SQ2M1)*HALF
      ENDIF
      RETURN
      END FUNCTION FI

      FUNCTION FI2( Z ) RESULT (VALUE)
!      USE GLOBALDATA, ONLY : XMAX
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: Z
      DOUBLE PRECISION :: VALUE
*
*     Normal distribution probabilities accurate to 1.e-15.
*     relative error less than 1e-8;
*     Z = no. of standard deviations from the mean.
*
*     Based upon algorithm 5666 for the error function, from:
*     Hart, J.F. et al, 'Computer Approximations', Wiley 1968
*
*     Programmer: Alan Miller
*
*     Latest revision - 30 March 1986
*
      DOUBLE PRECISION :: P0, P1, P2, P3, P4, P5, P6,
     *     Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7,XMAX,
     *     P, EXPNTL, CUTOFF, ROOTPI, ZABS, Z2
      PARAMETER(
     *     P0 = 220.20 68679 12376 1D0,
     *     P1 = 221.21 35961 69931 1D0,
     *     P2 = 112.07 92914 97870 9D0,
     *     P3 = 33.912 86607 83830 0D0,
     *     P4 = 6.3739 62203 53165 0D0,
     *     P5 = 0.70038 30644 43688 1D0,
     *     P6 = 0.035262 49659 98910 9D0 )
      PARAMETER(
     *     Q0 = 440.41 37358 24752 2D0,
     *     Q1 = 793.82 65125 19948 4D0,
     *     Q2 = 637.33 36333 78831 1D0,
     *     Q3 = 296.56 42487 79673 7D0,
     *     Q4 = 86.780 73220 29460 8D0,
     *     Q5 = 16.064 17757 92069 5D0,
     *     Q6 = 1.7556 67163 18264 2D0,
     *     Q7 = 0.088388 34764 83184 4D0 )
      PARAMETER( ROOTPI = 2.5066 28274 63100 1D0 )
      PARAMETER( CUTOFF = 7.0710 67811 86547 5D0 )
      PARAMETER( XMAX   = 37.D0 )
*
      ZABS = ABS(Z)
*
*     |Z| > 37  (or XMAX)
*
      IF ( ZABS .GT. XMAX ) THEN
         P = 0.d0
      ELSE
*
*     |Z| <= 37
*
         Z2 = ZABS * ZABS
         EXPNTL = EXP( -Z2 * 0.5D0 )
*
*     |Z| < CUTOFF = 10/SQRT(2)
*
         IF ( ZABS  <  CUTOFF ) THEN
            P = EXPNTL*( (((((P6*ZABS + P5)*ZABS + P4)*ZABS + P3)*ZABS
     *           + P2)*ZABS + P1)*ZABS + P0)/(((((((Q7*ZABS + Q6)*ZABS
     *           + Q5)*ZABS + Q4)*ZABS + Q3)*ZABS + Q2)*ZABS + Q1)*ZABS
     *           + Q0 )
*
*     |Z| >= CUTOFF.
*
         ELSE
            P = EXPNTL/( ZABS + 1.d0/( ZABS + 2.d0/( ZABS + 3.d0/( ZABS
     *                        + 4.d0/( ZABS + 0.65D0 ) ) ) ) )/ROOTPI
         END IF
      END IF
      IF ( Z .GT. 0.d0 ) P = 1.d0 - P
      VALUE = P
      RETURN
      END FUNCTION FI2

      SUBROUTINE MVNLIMITS( A, B, INFIN, AP, PRB, AQ)
! RETURN probabilities for being between A and B
!  WHERE
!  AP = FI(A), AQ = 1 - FI(A)
!  BP = FI(B), BQ = 1 - FI(B)
!  PRB = BP-AP IF BP+AP<1
!      = AQ-BQ OTHERWISE
!
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: A, B
      DOUBLE PRECISION, INTENT(out) :: AP
      DOUBLE PRECISION, INTENT(out),OPTIONAL :: PRB,AQ
      INTEGER,INTENT(in) :: INFIN
!     LOCAL VARIABLES
      DOUBLE PRECISION :: BP,AQQ, BQQ
      DOUBLE PRECISION, PARAMETER :: ONE=1.D0, ZERO = 0.D0

      SELECT CASE (infin)
      CASE (:-1)
         AP  = ZERO
!     BP  = ONE
         IF (PRESENT(PRB)) PRB = ONE
         IF (PRESENT(AQ)) AQ = ONE
!     IF (PRESENT(BQ)) BQ = ZERO
      CASE (0)
         AP  = ZERO
         CALL NORMPRB(B,BP)     !,BQQ)
         IF (PRESENT(PRB)) PRB = BP
         IF (PRESENT(AQ)) AQ = ONE
!     IF (PRESENT(BQ)) BQ = BQQ
      CASE (1)
!     BP = ONE
         CALL NORMPRB(A,AP,AQQ)
         IF (PRESENT(PRB)) PRB = AQQ
         IF (PRESENT(AQ)) AQ = AQQ
!     IF (PRESENT(BQ)) BQ = ZERO
      CASE (2:)
         CALL NORMPRB(A,AP,AQQ)
         CALL NORMPRB(B,BP,BQQ)
         IF (PRESENT(PRB)) THEN
            IF (AP+BP  <  ONE) THEN
               PRB = BP - AP
            ELSE
               PRB = AQQ - BQQ
            END IF
         ENDIF
         IF (PRESENT(AQ)) AQ = AQQ
!     IF (PRESENT(BQ)) BQ = BQQ
      END SELECT
      RETURN
      END SUBROUTINE MVNLIMITS


      SUBROUTINE MVNLMS( A, B, INFIN, LOWER, UPPER )
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: A, B
      DOUBLE PRECISION, INTENT(out) :: LOWER, UPPER
      INTEGER,INTENT(in) :: INFIN

      LOWER = 0.0D0
      UPPER = 1.0D0
      IF ( INFIN  <  0 ) RETURN
      IF ( INFIN .NE. 0 ) LOWER = FI(A)
      IF ( INFIN .NE. 1 ) UPPER = FI(B)
      RETURN
      END SUBROUTINE MVNLMS

      FUNCTION TVNMVN(A, B, INFIN, R, EPSI ) RESULT (VAL)
      IMPLICIT NONE
*
*     A function for computing trivariate normal probabilities.
*
*  Parameters
*
*     A  REAL, array of lower integration limits.
*     B  REAL, array of upper integration limits.
!     R  REAL, array of correlation coefficents
!         R = [r12 r13 r23]
!    EPSI = REAL tolerance
*     INFIN  INTEGER, array of integration limits flags:
*            if INFIN(I) = 0, Ith limits are (-infinity, B(I)];
*            if INFIN(I) = 1, Ith limits are [A(I), infinity);
*            if INFIN(I) = 2, Ith limits are [A(I), B(I)].
      DOUBLE PRECISION, DIMENSION(:), INTENT (IN) :: A, B , R
      DOUBLE PRECISION,               INTENT (IN) :: EPSI
      INTEGER,          DIMENSION(:), INTENT (IN) :: INFIN
      DOUBLE PRECISION :: VAL

      IF ( INFIN(1) .EQ. 2 ) THEN
         IF (  INFIN(2) .EQ. 2 ) THEN
            IF (INFIN(3) .EQ. 2 ) THEN !OK
               VAL =  TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( A(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( B(1), A(2), B(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( B(1), B(2), A(3),R(1),R(2),R(3),EPSI )
     &              + TVNL( A(1), A(2), B(3),R(1),R(2),R(3),EPSI )
     &              + TVNL( A(1), B(2), A(3),R(1),R(2),R(3),EPSI )
     $              + TVNL( B(1), A(2), A(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( A(1), A(2), A(3),R(1),R(2),R(3),EPSI )
            ELSE IF (INFIN(3) .EQ. 1 ) THEN ! B(3) = inf ok
               VAL = TVNL( B(1), B(2), -A(3),R(1),-R(2),-R(3),EPSI )
     &              - TVNL( A(1), B(2), -A(3),R(1),-R(2),-R(3),EPSI )
     &              - TVNL( B(1), A(2), -A(3),R(1),-R(2),-R(3),EPSI )
     &              + TVNL( A(1), A(2), -A(3),R(1),-R(2),-R(3),EPSI )
            ELSE IF (INFIN(3) .EQ. 0 ) THEN !OK A(3) = -inf
               VAL =  TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( A(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( B(1), A(2), B(3),R(1),R(2),R(3),EPSI )
     &              + TVNL( A(1), A(2), B(3),R(1),R(2),R(3),EPSI )
            ELSE ! INFIN(1:2)=2
               VAL = BVNMVN( A ,B, INFIN, R(1) )
            ENDIF
         ELSE IF (INFIN(2) .EQ. 1 ) THEN ! B(2) = inf
            IF (INFIN(3) .EQ. 2 ) THEN
               VAL = TVNL( B(1), -A(2), B(3),-R(1),R(2),-R(3),EPSI )
     &              - TVNL( A(1), -A(2), B(3),-R(1),R(2),-R(3),EPSI )
     &              - TVNL( B(1), -A(2), A(3),-R(1),R(2),-R(3),EPSI )
     &              + TVNL( A(1), -A(2), A(3),-R(1),R(2),-R(3),EPSI )
            ELSE IF (INFIN(3) .EQ. 1 ) THEN
               VAL = TVNL( B(1), -A(2), -A(3),-R(1),-R(2),R(3),EPSI )
     $              - TVNL( A(1), -A(2), -A(3),-R(1),-R(2),R(3),EPSI )
            ELSE IF (INFIN(3) .EQ. 0 ) THEN
               VAL = TVNL( B(1), -A(2), B(3),-R(1),R(2),-R(3),EPSI)
     $              - TVNL( A(1), -A(2), B(3),-R(1),R(2),-R(3),EPSI)
            ELSE
               VAL = BVNMVN( A ,B, INFIN, R(1) )
            ENDIF
         ELSE IF (INFIN(2) .EQ. 0 ) THEN
            SELECT CASE (INFIN(3))
            CASE (2:)           ! % % A(2)=-INF
               VAL = TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     $              - TVNL( A(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     $              - TVNL( B(1), B(2), A(3),R(1),R(2),R(3),EPSI )
     $              + TVNL( A(1), B(2), A(3),R(1),R(2),R(3),EPSI )
            CASE (1)            !%  % A(2)=-INF B(3) = INF
               VAL = TVNL( B(1), B(2), -A(3),R(1),-R(2),-R(3),EPSI)
     $              - TVNL(A(1), B(2), -A(3),R(1),-R(2),-R(3),EPSI)
            CASE (0)            ! % % A(2)=-INF A(3) = -INF
               VAL = TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     $              - TVNL( A(1), B(2), B(3),R(1),R(2),R(3),EPSI)
            CASE DEFAULT
               VAL = BVNMVN(A,B,INFIN,R(1))
            END SELECT
         ELSE
            VAL = BVNMVN(A(1:3:2),B(1:3:2),INFIN(1:3:2),R(2))
         ENDIF
      ELSE IF ( INFIN(1) .EQ. 1 ) THEN
         SELECT CASE (INFIN(2))
         CASE (2)
            SELECT CASE (INFIN(3))
            CASE (2)            !% B(1) = INF   %OK
               VAL =  TVNL(-A(1), B(2), B(3),-R(1),-R(2),R(3),EPSI )
     $              - TVNL(-A(1), B(2), A(3),-R(1),-R(2),R(3),EPSI )
     $              - TVNL(-A(1), A(2), B(3),-R(1),-R(2),R(3),EPSI )
     $              + TVNL(-A(1), A(2), A(3),-R(1),-R(2),R(3),EPSI )
            CASE (1)            ! % B(1) = INF   B(3) = INF %OK
               VAL = TVNL(-A(1), B(2), -A(3),-R(1),R(2),-R(3),EPSI )
     $              - TVNL(-A(1), A(2), -A(3),-R(1),R(2),-R(3),EPSI)
            CASE (0)            ! % B(1) = INF   A(3) = -INF %OK
               VAL =  TVNL(-A(1), B(2), B(3),-R(1),-R(2),R(3),EPSI )
     $              - TVNL(-A(1), A(2), B(3),-R(1),-R(2),R(3),EPSI)
            CASE (-1)
               VAL = BVNMVN(A,B,INFIN,R(1))
            END SELECT
         CASE (1)               !%B(2) = INF
            SELECT CASE (INFIN(3))
            CASE (2)            ! % B(1) = INF  B(2) = INF % OK
               VAL = TVNL( -A(1), -A(2), B(3),-R(1),R(2),-R(3),EPSI )
     &              - TVNL( -A(1), -A(2),A(3),-R(1),R(2),-R(3),EPSI )
            CASE (1)            ! % B(1:3) = INF %OK
               VAL = TVNL( -A(1), -A(2), -A(3),R(1),R(2),R(3),EPSI)
            CASE (0)            !  % B(1:2) = INF A(3) = -INF %OK
               VAL = TVNL( -A(1), -A(2), B(3),R(1),-R(2),-R(3),EPSI )
            CASE (:-1)
               VAL = BVNMVN(A,B,INFIN,R(1))
            END SELECT
         CASE (0) ! A(2) = -INF
            SELECT CASE ( INFIN(3))
            CASE (2)            ! B(1) = INF , A(2) = -INF %OK
               VAL = TVNL( -A(1), B(2), B(3),-R(1),R(2),-R(3),EPSI )
     &              - TVNL( -A(1), B(2),A(3),-R(1),R(2),-R(3),EPSI )
            CASE (1)            ! B(1) = INF , A(2) = -INF  B(3) = INF % OK
               VAL = TVNL( -A(1), B(2), -A(3),-R(1),-R(2),R(3),EPSI)
            CASE (0)            !% B(1) = INF , A(2:3) = -INF
               VAL = TVNL( -A(1), B(2), B(3),-R(1),-R(2),R(3),EPSI )
            CASE (:-1)
               VAL = BVNMVN(A,B,INFIN,R(1))
            END SELECT
         CASE DEFAULT
            VAL = BVNMVN(A(1:3:2),B(1:3:2),INFIN(1:3:2),R(2))
         END SELECT
      ELSE IF ( INFIN(1) .EQ. 0 ) THEN
         SELECT CASE (INFIN(2))
         CASE (2)
            SELECT CASE (INFIN(3))
            CASE (2:)            ! A(1) = -INF %OK
               VAL =  TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( B(1), B(2), A(3),R(1),R(2),R(3),EPSI)
     &              - TVNL( B(1), A(2), B(3),R(1),R(2),R(3),EPSI )
     &              + TVNL( B(1), A(2), A(3),R(1),R(2),R(3),EPSI )
            CASE (1) ! % A(1) = -INF , B(3) = INF %OK
               VAL =  TVNL( B(1), B(2), -A(3),R(1),-R(2),-R(3),EPSI )
     $              - TVNL( B(1), A(2), -A(3),R(1),-R(2),-R(3),EPSI )
            CASE (0)            ! A(1) = -INF , A(3) = -INF %OK
               VAL =  TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI )
     &              - TVNL( B(1), A(2), B(3),R(1),R(2),R(3),EPSI )
            CASE DEFAULT
               VAL = BVNMVN(A,B,INFIN,R(1))
            END SELECT
         CASE (1)               ! B(2) = INF
            SELECT CASE (INFIN(3))
            CASE (2:)            ! A(1) = -INF B(2) = INF %OK
               VAL =  TVNL( B(1), -A(2), B(3),-R(1),R(2),-R(3),EPSI)
     $              - TVNL( B(1), -A(2), A(3),-R(1),R(2),-R(3),EPSI)
            CASE (1)            ! A(1) = -INF B(2) = INF  B(3) = INF %OK
               VAL =  TVNL( B(1), -A(2), -A(3),-R(1),-R(2),R(3),EPSI)
            CASE (0)            ! % A(1) = -INF B(2) = INF  A(3) = -INF %OK
               VAL = TVNL(B(1), -A(2), B(3),-R(1),R(2),-R(3),EPSI)
            CASE DEFAULT
               VAL = BVNMVN(A,B,INFIN,R(1))
            END SELECT
         CASE (0)               ! A(2) = -INF
            SELECT CASE (INFIN(3))
            CASE (2:)            ! %  A(1:2) = -INF
               VAL =  TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI)
     $              - TVNL( B(1), B(2), A(3),R(1),R(2),R(3),EPSI)
            CASE (1)            ! A(1:2) = -INF B(3) = INF
               VAL =  TVNL( B(1), B(2), -A(3),R(1),-R(2),-R(3),EPSI)
            CASE (0)            !  % A(1:3) = -INF
               VAL =  TVNL( B(1), B(2), B(3),R(1),R(2),R(3),EPSI )
            CASE DEFAULT
               VAL = BVNMVN(A,B,INFIN,R(1))
            END  SELECT
         CASE DEFAULT
            VAL = BVNMVN(A(1:3:2),B(1:3:2),INFIN(1:3:2),R(2))
         END SELECT
      ELSE
         VAL = BVNMVN(A(2:3),B(2:3),INFIN(2:3),R(3))
      END IF
      CONTAINS
      DOUBLE PRECISION FUNCTION TVNL(H1,H2,H3, R12,R13,R23, EPSI )
      !Returns Trivariate Normal CDF
      DOUBLE PRECISION, INTENT(IN) :: R12,R13,R23
      DOUBLE PRECISION, INTENT(IN) :: H1,H2,H3, EPSI
!     Locals
      INTEGER, PARAMETER :: NU = 0
      DOUBLE PRECISION,DIMENSION(3) :: H,R
      H(:) = (/ H1, H2, H3 /)
      R(:) = (/ R12, R13, R23 /)
      TVNL = TVTL(NU,H,R,EPSI)
      END FUNCTION TVNL
      END  FUNCTION TVNMVN
      FUNCTION BVNMVN( LOWER, UPPER, INFIN, CORREL ) RESULT (VAL)
      IMPLICIT NONE
*
*     A function for computing bivariate normal probabilities.
*
*  Parameters
*
*     LOWER  REAL, array of lower integration limits.
*     UPPER  REAL, array of upper integration limits.
*     INFIN  INTEGER, array of integration limits flags:
*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
*     CORREL REAL, correlation coefficient.
*

      DOUBLE PRECISION, DIMENSION(:), INTENT (IN) :: LOWER, UPPER
      DOUBLE PRECISION,               INTENT (IN) :: CORREL
      INTEGER,          DIMENSION(:), INTENT (IN) :: INFIN
      DOUBLE PRECISION :: VAL
      DOUBLE PRECISION :: E
      SELECT CASE (INFIN(1))
      CASE (2:)
         SELECT CASE ( INFIN(2) )
         CASE (2:)
            VAL =  BVU ( LOWER(1), LOWER(2), CORREL )
     &           - BVU ( UPPER(1), LOWER(2), CORREL )
     &           - BVU ( LOWER(1), UPPER(2), CORREL )
     &           + BVU ( UPPER(1), UPPER(2), CORREL )

         CASE (1)
            VAL =  BVU ( LOWER(1), LOWER(2), CORREL )
     &           - BVU ( UPPER(1), LOWER(2), CORREL )
         CASE (0)
            VAL =  BVU ( -UPPER(1), -UPPER(2), CORREL )
     &           - BVU ( -LOWER(1), -UPPER(2), CORREL )
         CASE DEFAULT
             CALL MVNLIMITS(LOWER(1),UPPER(1),INFIN(1),E,VAL)
         END SELECT
      CASE (1)
         SELECT CASE ( INFIN(2))
         CASE ( 2: )
            VAL =  BVU ( LOWER(1), LOWER(2), CORREL )
     &           - BVU ( LOWER(1), UPPER(2), CORREL )
         CASE (1)
            VAL =  BVU ( LOWER(1), LOWER(2), CORREL )
         CASE (0)
            VAL =  BVU ( LOWER(1), -UPPER(2), -CORREL )
         CASE DEFAULT
            CALL MVNLIMITS(LOWER(2),UPPER(2),INFIN(2),E,VAL)
         END SELECT
      CASE (0)
         SELECT CASE ( INFIN(2))
         CASE ( 2: )
            VAL =  BVU ( -UPPER(1), -UPPER(2), CORREL )
     &           - BVU ( -UPPER(1), -LOWER(2), CORREL )
         CASE ( 1 )
            VAL =  BVU ( -UPPER(1), LOWER(2), -CORREL )
         CASE (0)
            VAL =  BVU ( -UPPER(1), -UPPER(2), CORREL )
         CASE DEFAULT
            CALL MVNLIMITS(LOWER(1),UPPER(1),INFIN(1),E,VAL)
         END SELECT
      CASE DEFAULT !ELSE  !INFIN(1)<0
         CALL MVNLIMITS(LOWER(2),UPPER(2),INFIN(2),E,VAL)
      END SELECT
      END  FUNCTION BVNMVN
      FUNCTION BVU( SH, SK, R ) RESULT (VAL)
!      USE GLOBALDATA, ONLY: XMAX
      IMPLICIT NONE
*
!     A function for computing bivariate normal probabilities.
!
!       Yihong Ge
!       Department of Computer Science and Electrical Engineering
!       Washington State University
!       Pullman, WA 99164-2752
!     and
!       Alan Genz
!       Department of Mathematics
!       Washington State University
!       Pullman, WA 99164-3113
!       Email : alangenz@wsu.edu
!
!    This function is based on the method described by
!        Drezner, Z and G.O. Wesolowsky, (1989),
!        On the computation of the bivariate normal integral,
!        Journal of Statist. Comput. Simul. 35, pp. 101-107,
!    with major modifications for double precision, and for |R| close to 1.
!
! BVU - calculate the probability that X > SH and Y > SK.
!       (to accuracy of 1e-16?)
!
! Parameters
!
!   SH  REAL, lower integration limit
!   SK  REAL, lower integration limit
!   R   REAL, correlation coefficient
!
!   LG  INTEGER, number of Gauss Rule Points and Weights
!
! Revised pab added check on XMAX
      DOUBLE PRECISION, INTENT(IN) :: SH, SK, R
      DOUBLE PRECISION  :: VAL
! Local variables
      DOUBLE PRECISION :: ZERO,ONE,FOUR
      DOUBLE PRECISION :: SQTWOPI ,TWOPI1,FOURPI1
      DOUBLE PRECISION :: HALF,ONETHIRD,ONEEIGHT,ONESIXTEEN
      DOUBLE PRECISION :: TWELVE, EXPMIN, XMAX
      INTEGER :: I, LG, NG
      PARAMETER ( ZERO = 0.D0,ONE=1.0D0,HALF=0.5D0)
      PARAMETER (FOUR = 4.0D0, TWELVE = 12.0D0)
      PARAMETER (EXPMIN =  -100.0D0)
      PARAMETER (ONESIXTEEN = 0.0625D0) !1/16
      PARAMETER (ONEEIGHT = 0.125D0 )  !1/8
      PARAMETER (ONETHIRD = 0.3333333333333333333333D0)
!      PARAMETER (TWOPI   = 6.283185307179586D0 )
      PARAMETER (TWOPI1  = 0.15915494309190D0 )  !1/(2*pi)
      PARAMETER (FOURPI1 = 0.0795774715459476D0 ) !/1/(4*pi)
      PARAMETER (SQTWOPI = 2.50662827463100D0) ! SQRT(2*pi)
      PARAMETER (XMAX    = 8.3D0)
      DOUBLE PRECISION, DIMENSION(10,3) :: X, W
      DOUBLE PRECISION :: AS, A, B, C, D, RS, XS
      DOUBLE PRECISION :: SN, ASR, H, K, BS, HS, HK
!     Gauss Legendre Points and Weights, N =  6
      DATA  ( W(I,1), X(I,1), I = 1,3) /
     *  0.1713244923791705D+00,-0.9324695142031522D+00,
     *  0.3607615730481384D+00,-0.6612093864662647D+00,
     *  0.4679139345726904D+00,-0.2386191860831970D+00/
!     Gauss Legendre Points and Weights, N = 12
      DATA ( W(I,2), X(I,2), I = 1,6) /
     *  0.4717533638651177D-01,-0.9815606342467191D+00,
     *  0.1069393259953183D+00,-0.9041172563704750D+00,
     *  0.1600783285433464D+00,-0.7699026741943050D+00,
     *  0.2031674267230659D+00,-0.5873179542866171D+00,
     *  0.2334925365383547D+00,-0.3678314989981802D+00,
     *  0.2491470458134029D+00,-0.1252334085114692D+00/
!     Gauss Legendre Points and Weights, N = 20
      DATA ( W(I,3), X(I,3), I = 1,10) /
     *  0.1761400713915212D-01,-0.9931285991850949D+00,
     *  0.4060142980038694D-01,-0.9639719272779138D+00,
     *  0.6267204833410906D-01,-0.9122344282513259D+00,
     *  0.8327674157670475D-01,-0.8391169718222188D+00,
     *  0.1019301198172404D+00,-0.7463319064601508D+00,
     *  0.1181945319615184D+00,-0.6360536807265150D+00,
     *  0.1316886384491766D+00,-0.5108670019508271D+00,
     *  0.1420961093183821D+00,-0.3737060887154196D+00,
     *  0.1491729864726037D+00,-0.2277858511416451D+00,
     *  0.1527533871307259D+00,-0.7652652113349733D-01/
      SAVE W, X
      VAL = ZERO
      HK = MIN(SH,SK)
      IF ( HK  < -XMAX) THEN    ! pab 24.05.2003
         VAL = FI(-MAX(SH,SK))
         RETURN
      ELSE IF ( XMAX  < MAX(SH,SK)) THEN
         RETURN
      ENDIF
      IF ( ABS(R)  <  0.3D0 ) THEN
         NG = 1
         LG = 3
      ELSE IF ( ABS(R)  <  0.75D0 ) THEN
         NG = 2
         LG = 6
      ELSE
         NG = 3
         LG = 10
      ENDIF
      H = SH
      K = SK
      HK = H*K

      IF ( ABS(R)  <  0.925D0 ) THEN
         IF (ABS(R) .GT. ZERO ) THEN
            HS = ( H*H + K*K )*HALF
            ASR = ASIN(R)
            DO I = 1, LG
               SN  = SIN(ASR*(ONE + X(I,NG))*HALF)
               VAL = VAL + W(I,NG)*EXP( ( SN*HK - HS )/( ONE - SN*SN ) )
               SN  = SIN(ASR*(ONE - X(I,NG))*HALF)
               VAL = VAL + W(I,NG)*EXP( ( SN*HK - HS )/( ONE - SN*SN ) )
            END DO
            VAL = VAL*ASR*FOURPI1
         ENDIF
         VAL = VAL + FI(-H)*FI(-K)
      ELSE
         IF ( R  <  ZERO ) THEN
            K  = -K
            HK = -HK
         ENDIF
         IF ( ABS(R)  <  ONE ) THEN
            AS  = ( ONE - R )*( ONE + R )
            A   = SQRT(AS)
            B   = ABS( H - K ) !**2
            BS  = B * B
            C   = ( FOUR - HK ) * ONEEIGHT !/8D0
            D   = ( TWELVE - HK ) * ONESIXTEEN !/16D0
            ASR =  -(BS/AS + HK)*HALF
            IF (ASR.GT.EXPMIN) THEN
               VAL = A*EXP( ASR ) *
     &              ( ONE - C*(BS - AS)*(ONE - D*BS*0.2D0)*ONETHIRD +
     &              C*D*AS*AS*0.2D0 )
            ENDIF
            IF ( HK .GT. EXPMIN ) THEN
               VAL = VAL - EXP(-HK*HALF)*SQTWOPI*FI(-B/A)*B
     +              *( ONE - C*BS*( ONE - D*BS*0.2D0 )*ONETHIRD )
            ENDIF
            A = A * HALF
            DO I = 1, LG
               XS  = ( A * (ONE + X(I,NG)) ) !**2
               XS  = XS * XS
               RS  = SQRT( ONE - XS )
               ASR = -(BS / XS + HK) * HALF
               IF (ASR.GT.EXPMIN) THEN
                  VAL = VAL + A*W(I,NG)*EXP( ASR )
     &                 * ( EXP( - HALF*HK*( ONE - RS )/( ONE + RS ) )/RS
     $                 -( ONE + C*XS*( ONE + D*XS ) ) )
               ENDIF
               XS  = ( A * (ONE - X(I,NG)) ) !**2
               XS  = XS * XS
               RS  = SQRT( ONE - XS )
               ASR = -(BS / XS + HK) * HALF
               IF (ASR.GT.EXPMIN) THEN
                  VAL = VAL + A*W(I,NG)*EXP( ASR )
     &                 *( EXP( - HALF*HK*( ONE - RS )/( ONE + RS ) )/RS-
     $                 ( ONE + C*XS*( ONE + D*XS ) ) )
               ENDIF
            END DO
            VAL = -VAL*TWOPI1
         ENDIF
         IF ( R .GT. ZERO ) THEN
            VAL =  VAL + FI( -MAX( H, K ) )
         ELSE
            VAL = -VAL
            IF ( H  <  K )  VAL = VAL +  FI(K)-FI(H)
         ENDIF
      ENDIF
      RETURN
      END FUNCTION BVU
      DOUBLE PRECISION FUNCTION STUDNT( NU, T )
      IMPLICIT NONE
!
!     Student t Distribution Function
!
!                       T
!         STUDNT = C   I  ( 1 + y*y/NU )**( -(NU+1)/2 ) dy
!                   NU -INF
!
      INTEGER, INTENT(IN) :: NU
      DOUBLE PRECISION, INTENT(IN) :: T
!     Locals
      INTEGER :: J
      DOUBLE PRECISION :: ZRO, ONE
      PARAMETER ( ZRO = 0.0D0, ONE = 1.0D0 )
      DOUBLE PRECISION, PARAMETER :: PI = 3.14159265358979D0
      DOUBLE PRECISION :: CSSTHE, SNTHE, POLYN, TT, TS, RN

      IF ( NU  <  1 ) THEN
         STUDNT = FI( T )
      ELSE IF ( NU .EQ. 1 ) THEN
         STUDNT = ( ONE + 2.0D0*ATAN(T)/PI )*0.5D0
      ELSE IF ( NU .EQ. 2 ) THEN
         STUDNT = ( ONE + T/SQRT( 2.0D0 + T*T ))*0.5D0
      ELSE
         RN = NU ! convert to double
         TT = T * T
         CSSTHE = ONE/( ONE + TT/RN )
         POLYN = 1
         DO J = NU-2, 2, -2
            POLYN = ONE + ( J - 1 )*CSSTHE*POLYN/J
         END DO

         IF ( MOD( NU, 2 ) .EQ. 1 ) THEN
            TS = T/SQRT(RN)
            STUDNT = ( ONE + 2.0D0*( ATAN(TS) +
     &           TS*CSSTHE*POLYN )/PI )*0.5D0
         ELSE
            SNTHE = T/SQRT( RN + TT )
            STUDNT = ( ONE + SNTHE*POLYN )*0.5D0
         END IF
         STUDNT = MAX( ZRO, MIN( STUDNT, ONE ) )
      ENDIF
      END FUNCTION STUDNT
      DOUBLE PRECISION FUNCTION BVTL( NU, DH, DK, R )
      IMPLICIT NONE
!*
!*     A function for computing bivariate t probabilities.
!*
!*       Alan Genz
!*       Department of Mathematics
!*       Washington State University
!*       Pullman, WA 99164-3113
!*       Email : alangenz@wsu.edu
!*
!*    This function is based on the method described by
!*        Dunnett, C.W. and M. Sobel, (1954),
!*        A bivariate generalization of Student's t-distribution
!*        with tables for certain special cases,
!*        Biometrika 41, pp. 153-169.
!*
!* BVTL - calculate the probability that X < DH and Y < DK.
!*
!* parameters
!*
!*   NU number of degrees of freedom (NOTE: NU = 0 gives bivariate normal prb)
!*   DH 1st lower integration limit
!*   DK 2nd lower integration limit
!*   R   correlation coefficient
!*
      INTEGER, INTENT(IN) ::NU
      DOUBLE PRECISION, INTENT(IN) :: DH, DK, R
!     Locals
      INTEGER :: J, HS, KS
      DOUBLE PRECISION ::  ORS, HRK, KRH, BVT
      DOUBLE PRECISION ::  DH2, DK2, SNU ,DNU, DHDK
!, BVND, STUDNT
      DOUBLE PRECISION :: GMPH, GMPK, XNKH, XNHK, QHRK, HKN, HPK, HKRN
      DOUBLE PRECISION :: BTNCKH, BTNCHK, BTPDKH, BTPDHK
      DOUBLE PRECISION :: ZERO, ONE, EPS, PI,TPI
      PARAMETER ( ZERO = 0.0D0, ONE = 1.0D0, EPS = 1.0D-15 )
      PARAMETER (PI =  3.14159265358979D0, TPI = 6.28318530717959D0)
      IF ( NU  <  1 ) THEN
         BVTL = BVU( -DH, -DK, R )
      ELSE IF ( ONE - R .LE. EPS .OR. 1.0D+16<MAX(ABS(DH),ABS(DK))) THEN
            BVTL = STUDNT( NU, MIN( DH, DK ) )
      ELSE IF ( R + ONE  .LE. EPS ) THEN
         IF ( DH .GT. -DK )  THEN
            BVTL = STUDNT( NU, DH ) - STUDNT( NU, -DK )
         ELSE
            BVTL = ZERO
         END IF
      ELSE
         !PI = ACOS(-ONE)
         !TPI = 2*PI
         DNU = NU           ! convert to double
         SNU = SQRT(DNU)
         ORS = ONE - R * R
         HRK = DH - R * DK
         KRH = DK - R * DH
         DK2 = DK * DK
         DH2 = DH * DH
         IF ( ABS(HRK) + ORS .GT. ZERO ) THEN
            XNHK = HRK**2/( HRK**2 + ORS*( DNU + DK2 ) )
            XNKH = KRH**2/( KRH**2 + ORS*( DNU + DH2 ) )
         ELSE
            XNHK = ZERO
            XNKH = ZERO
         END IF
         HS = INT(DSIGN( ONE, HRK)) !DH - R*DK )
         KS = INT(DSIGN( ONE, KRH)) !DK - R*DH )
         IF ( MOD( NU, 2 ) .EQ. 0 ) THEN
            BVT = ATAN2( SQRT(ORS), -R )/TPI
            GMPH = DH/SQRT( 16.0D0*( DNU + DH2 ) )
            GMPK = DK/SQRT( 16.0D0*( DNU + DK2 ) )
            BTNCKH = 2*ATAN2( SQRT( XNKH ), SQRT( ONE - XNKH ) )/PI
            BTPDKH = 2*SQRT( XNKH*( ONE - XNKH ) )/PI
            BTNCHK = 2*ATAN2( SQRT( XNHK ), SQRT( ONE - XNHK ) )/PI
            BTPDHK = 2*SQRT( XNHK*( ONE - XNHK ) )/PI
            DO J = 1, NU/2
               BVT = BVT + GMPH*( ONE + KS*BTNCKH )
               BVT = BVT + GMPK*( ONE + HS*BTNCHK )
               BTNCKH = BTNCKH + BTPDKH
               BTPDKH = 2*J*BTPDKH*( ONE - XNKH )/( 2*J + 1 )
               BTNCHK = BTNCHK + BTPDHK
               BTPDHK = 2*J*BTPDHK*( ONE - XNHK )/( 2*J + 1 )
               GMPH = GMPH*( 2*J - 1 )/( 2*J*( ONE + DH2/DNU ) )
               GMPK = GMPK*( 2*J - 1 )/( 2*J*( ONE + DK2/DNU ) )
            END DO
         ELSE  ! NU is ODD
            DHDK = DH*DK
            QHRK = SQRT( DH2 + DK2 - 2.0D0*R*DHDK + DNU*ORS )
            HKRN = DHDK + R*DNU
            HKN = DHDK - DNU
            HPK = DH + DK
            BVT = ATAN2( -SNU*( HKN*QHRK + HPK*HKRN ),
     &                          HKN*HKRN-DNU*HPK*QHRK )/TPI
            IF ( BVT  <  -EPS ) BVT = BVT + ONE
            GMPH = DH/( TPI*SNU*( ONE + DH2/DNU ) )
            GMPK = DK/( TPI*SNU*( ONE + DK2/DNU ) )
            BTNCKH = SQRT( XNKH )
            BTPDKH = BTNCKH
            BTNCHK = SQRT( XNHK )
            BTPDHK = BTNCHK
            DO J = 1, ( NU - 1 )/2
               BVT = BVT + GMPH*( ONE + KS*BTNCKH )
               BVT = BVT + GMPK*( ONE + HS*BTNCHK )
               BTPDKH = ( 2*J - 1 )*BTPDKH*( ONE - XNKH )/( 2*J )
               BTNCKH = BTNCKH + BTPDKH
               BTPDHK = ( 2*J - 1 )*BTPDHK*( ONE - XNHK )/( 2*J )
               BTNCHK = BTNCHK + BTPDHK
               GMPH = 2*J*GMPH/( ( 2*J + 1 )*( ONE + DH2/DNU ) )
               GMPK = 2*J*GMPK/( ( 2*J + 1 )*( ONE + DK2/DNU ) )
            END DO
         END IF
         BVTL = BVT
      END IF
      END FUNCTION BVTL

      DOUBLE PRECISION FUNCTION TVTL( NU1, H, R, EPSI )
      USE TRIVARIATEVAR !Block to transfer variables to TVTMFN
      IMPLICIT NONE
!
!     A function for computing trivariate normal and t-probabilities.
!     This function uses algorithms developed from the ideas
!     described in the papers:
!       R.L. Plackett, Biometrika 41(1954), pp. 351-360.
!       Z. Drezner, Math. Comp. 62(1994), pp. 289-294.
!     with adaptive integration from (0,0,1) to (0,0,r23) to R.
!
!      Calculate the probability that X(I) < H(I), for I = 1,2,3
!    NU   INTEGER degrees of freedom; use NU = 0 for normal cases.
!    H    REAL array of uppoer limits for probability distribution
!    R    REAL array of three correlation coefficients, R should
!         contain the lower left portion of the correlation matrix r.
!         R should contains the values r21, r31, r23 in that order.
!   EPSI  REAL required absolute accuracy; maximum accuracy for most
!          computations is approximately 1D-14
!
!    The software is based on work described in the paper
!     "Numerical Computation of Rectangular Bivariate and Trivariate
!      Normal and t Probabilities", by the code author:
!
!       Alan Genz
!       Department of Mathematics
!       Washington State University
!       Pullman, WA 99164-3113
!       Email : alangenz@wsu.edu
!
!      EXTERNAL TVTMFN
      INTEGER,                       INTENT(IN) :: NU1
      DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: H, R
      DOUBLE PRECISION,              INTENT(IN) ::  EPSI
!Locals
      DOUBLE PRECISION :: R12, R13,  TVT
      DOUBLE PRECISION :: ONE, ZERO, EPS,  PT
!      DOUBLE PRECISION RUA, RUB, AR, RUC,
!        BVTL, PHID, ADONET
      PARAMETER ( ZERO = 0.0D0, ONE = 1.0D0 )
      PARAMETER ( PT = 1.57079632679489661923132169163975D0 ) !pi/2

!      COMMON /TVTMBK/ H1, H2, H3, R23, RUA, RUB, AR, RUC, NU
      EPS = MAX( 1.0D-13, EPSI )
!      PT = ASIN(ONE)
      NU = NU1
      H1 = H(1)
      H2 = H(2)
      H3 = H(3)
      R12 = R(1)
      R13 = R(2)
      R23 = R(3)
!
!     Sort R's and check for special cases
!
      IF ( ABS(R12) .GT. ABS(R13) ) THEN
         H2 = H3
         H3 = H(2)
         R12 = R13
         R13 = R(1)
      END IF
      IF ( ABS(R13) .GT. ABS(R23) ) THEN
         H1 = H2
         H2 = H(1)
         R23 = R13
         R13 = R(3)
      END IF
      TVT = 0
      IF ( ABS(H1) + ABS(H2) + ABS(H3)  <  EPS ) THEN
         TVT = (ONE + (ASIN(R12) + ASIN(R13) + ASIN(R23))/PT )*0.125D0
      ELSE IF ( NU  <  1 .AND. ABS(R12) + ABS(R13)  <  EPS ) THEN
         TVT = FI(H1)*BVTL( NU, H2, H3, R23 )
      ELSE IF ( NU  <  1 .AND. ABS(R13) + ABS(R23)  <  EPS ) THEN
         TVT = FI(H3)*BVTL( NU, H1, H2, R12 )
      ELSE IF ( NU  <  1 .AND. ABS(R12) + ABS(R23)  <  EPS ) THEN
         TVT = FI(H2)*BVTL( NU, H1, H3, R13 )
      ELSE IF ( ONE - R23  <  EPS ) THEN
         TVT = BVTL( NU, H1, MIN( H2, H3 ), R12 )
      ELSE IF ( R23 + ONE  <  EPS ) THEN
         IF  ( H2 .GT. -H3 )
     &        TVT = BVTL( NU, H1, H2, R12 ) - BVTL( NU, H1, -H3, R12 )
      ELSE
!
!        Compute singular TVT value
!
         IF ( NU  <  1 ) THEN
            TVT = BVTL( NU, H2, H3, R23 )*FI(H1)
         ELSE IF ( R23 .GE. ZERO ) THEN
            TVT = BVTL( NU, H1, MIN( H2, H3 ), ZERO )
         ELSE IF ( H2 .GT. -H3 ) THEN
            TVT = BVTL( NU, H1, H2, ZERO ) - BVTL( NU, H1, -H3, ZERO )
         END IF
!
!        Use numerical integration to compute probability
!
!
         RUA = ASIN( R12 )
         RUB = ASIN( R13 )
         AR  = ASIN( R23)
         RUC = SIGN( PT, AR ) - AR
         TVT = TVT + ADONET(TVTMFN, ZERO, ONE, EPS )/( 4.D0*PT )
      END IF
      TVTL = MAX( ZERO, MIN( TVT, ONE ) )
      END FUNCTION TVTL
!      CONTAINS
!
      DOUBLE PRECISION FUNCTION TVTMFN( X )
      USE TRIVARIATEVAR
      IMPLICIT NONE
!
!     Computes Plackett formula integrands
!
      DOUBLE PRECISION, INTENT(IN) :: X
! Locals
      DOUBLE PRECISION R12, RR2, R13, RR3, R, RR, ZRO !, PNTGND
! Parameters transfeered from TRIVARIATEVAR
!      INTEGER :: NU
!      DOUBLE PRECISION :: H1, H2, H3, R23, RUA, RUB, AR, RUC
      PARAMETER ( ZRO = 0.0D0 )
!      COMMON /TVTMBK/ H1, H2, H3, R23, RUA, RUB, AR, RUC, NU
      TVTMFN = 0.0D0
      CALL SINCS( RUA*X, R12, RR2 )
      CALL SINCS( RUB*X, R13, RR3 )
      IF ( ABS(RUA) .GT. ZRO )
     &     TVTMFN = TVTMFN + RUA*PNTGND( NU, H1,H2,H3, R13,R23,R12,RR2 )
      IF ( ABS(RUB) .GT. ZRO )
     &     TVTMFN = TVTMFN + RUB*PNTGND( NU, H1,H3,H2, R12,R23,R13,RR3 )
      IF ( NU .GT. 0 ) THEN
         CALL SINCS( AR + RUC*X, R, RR )
         TVTMFN = TVTMFN - RUC*PNTGND( NU, H2, H3, H1, ZRO, ZRO, R, RR )
      END IF
      END FUNCTION TVTMFN
!
      SUBROUTINE SINCS( X, SX, CS )
!
!     Computes SIN(X), COS(X)^2, with series approx. for |X| near PI/2
!
      DOUBLE PRECISION, INTENT(IN) :: X
      DOUBLE PRECISION, INTENT(OUT) :: SX, CS
!Locals
      DOUBLE PRECISION :: EE, PT, KS, KC, ONE, SMALL, HALF, ONETHIRD
      PARAMETER (ONE = 1.0D0, SMALL = 5.0D-5, HALF = 0.5D0 )
      PARAMETER ( PT = 1.57079632679489661923132169163975D0 )
      PARAMETER ( KS = 0.0833333333333333333333333333333D0) !1/12
      PARAMETER ( KC = 0.1333333333333333333333333333333D0) !2/15
      PARAMETER ( ONETHIRD = 0.33333333333333333333333333333333D0) !1/3
      EE = ( PT - ABS(X) )
      EE = EE * EE
      IF ( EE  <  SMALL ) THEN
         SX = SIGN( ONE - EE*( ONE - EE*KS )*HALF, X )
         CS = EE *( ONE - EE*( ONE - EE*KC )*ONETHIRD)
      ELSE
         SX = SIN(X)
         CS = ONE - SX*SX
      END IF
      END SUBROUTINE SINCS
!
      DOUBLE PRECISION FUNCTION PNTGND( NU, BA, BB, BC, RA, RB, R, RR )
      IMPLICIT NONE
!
!     Computes Plackett formula integrand
!
      INTEGER, INTENT(IN) :: NU
      DOUBLE PRECISION, INTENT(IN) :: BA, BB, BC, RA, RB, R, RR
! Locals
      DOUBLE PRECISION :: DT, FT, BT,RAB2, BARB, rNU!, PHID, STUDNT
      PNTGND = 0.0D0
      FT   = ( RA - RB )
      RAB2 = FT*FT
      DT   = RR*( RR -  RAB2 - 2.0D0*RA*RB*( 1.D0 - R ) )
      IF ( DT .GT. 0.0D0 ) THEN
         BT   = ( BC*RR + BA*( R*RB - RA ) + BB*( R*RA -RB ) )/SQRT(DT)
         BARB = ( BA - R*BB )
         FT   = ( BARB * BARB ) / RR + BB * BB
         IF ( NU  <  1 ) THEN
            IF ( BT .GT. -10.0D0 .AND. FT  <  100.0D0 ) THEN
               PNTGND = EXP( -FT * 0.5D0 ) * FI( BT )
!               PNTGND = EXP( -FT*0.5D0)
!               IF ( BT  <  10.0D0 ) PNTGND = PNTGND * FI(BT)
            END IF
         ELSE
            rNU = NU
            FT  = SQRT( 1.0D0 + FT/rNU )
            PNTGND = STUDNT( NU, BT/FT )/FT**NU
         END IF
      END IF
      END  FUNCTION PNTGND
!
      DOUBLE PRECISION FUNCTION ADONET( F, A, B, TOL )
      IMPLICIT NONE
!
!     One Dimensional Globally Adaptive Integration Function
!
!      EXTERNAL F
      DOUBLE PRECISION, INTENT(IN) :: A, B, TOL
      INTEGER :: NL, I, IM, IP
      PARAMETER ( NL = 100 )
      DOUBLE PRECISION, DIMENSION(NL) ::  EI, AI, BI, FI
      DOUBLE PRECISION :: FIN, ERR !, KRNRDT
      DOUBLE PRECISION, PARAMETER :: ZERO = 0.0D0
      DOUBLE PRECISION, PARAMETER :: HALF = 0.5D0
      DOUBLE PRECISION, PARAMETER :: ONE  = 1.0D0
      DOUBLE PRECISION, PARAMETER :: FOUR = 4.0D0
      INTERFACE
         FUNCTION F(Z) RESULT (VAL)
         DOUBLE PRECISION, INTENT(IN) :: Z
         DOUBLE PRECISION :: VAL
         END FUNCTION F
      END INTERFACE
!      COMMON /ABLK/ ERR, IM
      AI(1) = A
      BI(1) = B
      ERR = ONE
      IP = 1
      IM = 1
      DO WHILE ( (ERR .GT. TOL) .AND. (IM  <  NL) )
         IM = IM + 1
         BI(IM) = BI(IP)
         AI(IM) = ( AI(IP) + BI(IP) ) * HALF
         BI(IP) = AI(IM)
         FI(IP) = KRNRDT( AI(IP), BI(IP), F, EI(IP) )
         FI(IM) = KRNRDT( AI(IM), BI(IM), F, EI(IM) )
         ERR = ZERO
         FIN = ZERO
         DO I = 1, IM
            IF ( EI(I) .GT. EI(IP) ) IP = I
            FIN = FIN + FI(I)
            ERR = ERR + EI(I)*EI(I)
         END DO
         ERR = FOUR * SQRT( ERR )
      END DO
      ADONET = FIN
      END FUNCTION ADONET
!
      DOUBLE PRECISION FUNCTION KRNRDT( A, B, F, ERR )
!
!     Kronrod Rule
!
      DOUBLE PRECISION, intent(in) :: A, B
      DOUBLE PRECISION, intent(out) :: ERR
      DOUBLE PRECISION T, CEN, FC, WID, RESG, RESK
!
!        The abscissae and weights are given for the interval (-1,1);
!        only positive abscissae and corresponding weights are given.
!
!        XGK    - abscissae of the 2N+1-point Kronrod rule:
!                 XGK(2), XGK(4), ...  N-point Gauss rule abscissae;
!                 XGK(1), XGK(3), ...  optimally added abscissae.
!        WGK    - weights of the 2N+1-point Kronrod rule.
!        WG     - weights of the N-point Gauss rule.
!
      INTEGER :: J, N
      PARAMETER ( N = 11 )
      DOUBLE PRECISION, PARAMETER :: HALF = 0.5D0
      DOUBLE PRECISION WG(0:(N+1)/2), WGK(0:N), XGK(0:N)
      SAVE WG, WGK, XGK
      INTERFACE
          FUNCTION F(Z) RESULT (VAL)
         DOUBLE PRECISION, INTENT(IN) :: Z
         DOUBLE PRECISION :: VAL
         END FUNCTION F
      END INTERFACE
      DATA WG( 0)/ 0.2729250867779007D+00/
      DATA WG( 1)/ 0.5566856711617449D-01/
      DATA WG( 2)/ 0.1255803694649048D+00/
      DATA WG( 3)/ 0.1862902109277352D+00/
      DATA WG( 4)/ 0.2331937645919914D+00/
      DATA WG( 5)/ 0.2628045445102478D+00/
!
      DATA XGK( 0)/ 0.0000000000000000D+00/
      DATA XGK( 1)/ 0.9963696138895427D+00/
      DATA XGK( 2)/ 0.9782286581460570D+00/
      DATA XGK( 3)/ 0.9416771085780681D+00/
      DATA XGK( 4)/ 0.8870625997680953D+00/
      DATA XGK( 5)/ 0.8160574566562211D+00/
      DATA XGK( 6)/ 0.7301520055740492D+00/
      DATA XGK( 7)/ 0.6305995201619651D+00/
      DATA XGK( 8)/ 0.5190961292068118D+00/
      DATA XGK( 9)/ 0.3979441409523776D+00/
      DATA XGK(10)/ 0.2695431559523450D+00/
      DATA XGK(11)/ 0.1361130007993617D+00/
!
      DATA WGK( 0)/ 0.1365777947111183D+00/
      DATA WGK( 1)/ 0.9765441045961290D-02/
      DATA WGK( 2)/ 0.2715655468210443D-01/
      DATA WGK( 3)/ 0.4582937856442671D-01/
      DATA WGK( 4)/ 0.6309742475037484D-01/
      DATA WGK( 5)/ 0.7866457193222764D-01/
      DATA WGK( 6)/ 0.9295309859690074D-01/
      DATA WGK( 7)/ 0.1058720744813894D+00/
      DATA WGK( 8)/ 0.1167395024610472D+00/
      DATA WGK( 9)/ 0.1251587991003195D+00/
      DATA WGK(10)/ 0.1312806842298057D+00/
      DATA WGK(11)/ 0.1351935727998845D+00/
!
!           Major variables
!
!           CEN  - mid point of the interval
!           WID  - half-length of the interval
!           RESG - result of the N-point Gauss formula
!           RESK - result of the 2N+1-point Kronrod formula
!
!           Compute the 2N+1-point Kronrod approximation to
!            the integral, and estimate the absolute error.
!
      WID = ( B - A ) * HALF
      CEN = ( B + A ) * HALF
      FC  = F(CEN)
      RESG = FC * WG(0)
      RESK = FC * WGK(0)
      DO J = 1, N
         T  = WID * XGK(J)
         FC = F( CEN - T ) + F( CEN + T )
         RESK = RESK + WGK(J) * FC
         IF( MOD( J, 2 ) .EQ. 0 ) RESG = RESG + WG(J/2) * FC
      END DO
      KRNRDT = WID * RESK
      ERR = ABS( WID * ( RESK - RESG ) )
      END FUNCTION KRNRDT
!      END FUNCTION TVTL

      FUNCTION GAUSINT (X1, X2, A, B, C, D) RESULT (value)
!      USE GLOBALDATA,ONLY:  xCutOff
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: X1,X2,A,B,C,D
      DOUBLE PRECISION             :: value
!Local variables
      DOUBLE PRECISION             :: Y1,Y2,Y3
      DOUBLE PRECISION, PARAMETER :: SQTWOPI1=3.9894228040143d-1 !=1/sqrt(2*pi)
      DOUBLE PRECISION, PARAMETER :: XMAX = 37.d0
      ! Let  X  be standardized Gaussian variable,
      ! i.e., X=N(0,1). The function calculate the
      !  following integral E[I(X1<X<X2)(A+BX)(C+DX)
      ! where I(X1<X<X2) is an indicator function of
      ! the set {X1<X<X2}.
      IF (X1.GE.X2) THEN
         value = 0.d0
         RETURN
      ENDIF
      IF (ABS (X1) .GT.XMAX) THEN
         Y1 = 0.d0
      ELSE
         Y1 = (A * D+B * C + X1 * B * D) * EXP ( - 0.5d0 * X1 * X1)
      ENDIF
      IF (ABS (X2) .GT.XMAX) THEN
         Y2 = 0.d0
      ELSE
         Y2 = (A * D+B * C + X2 * B * D) * EXP ( - 0.5d0 * X2 * X2)
      ENDIF
      Y3 = (A * C + B * D) * (FI (X2) - FI (X1) )
      value = Y3 + SQTWOPI1 * (Y1 - Y2)
      RETURN
      END FUNCTION GAUSINT


      FUNCTION GAUSINT2 (X1, X2, A, B) RESULT (value)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: X1,X2,A,B
      DOUBLE PRECISION             :: value
! Local variables
      DOUBLE PRECISION             :: X0,Y0,Y1,Y2
      DOUBLE PRECISION, PARAMETER :: SQTWOPI1=3.9894228040143d-1 !=1/sqrt(2*pi)
!     Let  X  be standardized Gaussian variable,
!     i.e., X=N(0,1). The function calculate the
!     following integral E[I(X1<X<X2)ABS(A+BX)]
!     where I(X1<X<X2) is an indicator function of
!     the set {X1<X<X2}.
      IF (X1.GE.X2) THEN
         value = 0.d0
         RETURN
      ENDIF
      IF (ABS(B).EQ.0.d0) THEN
         value = ABS(A)*(FI(X2)-FI(X1))
         RETURN
      ENDIF

      Y1 = -A*FI(X1)+SQTWOPI1*B*EXP(-0.5d0*X1*X1)
      Y2 = A*FI(X2)-SQTWOPI1*B*EXP(-0.5d0*X2*X2)
      IF ((B*X1 < -A).AND.(-A < B*X2))THEN
         X0 = -A/B
         Y0 = 2.d0*(A*FI(X0)-SQTWOPI1*B*EXP(-0.5d0*X0*X0))
         value=ABS(Y2-Y1-Y0)
      ELSE
         value=ABS(Y1+Y2)
      ENDIF
      RETURN
      END FUNCTION GAUSINT2

      SUBROUTINE EXLMS(A, X1, X2, INFIN, LOWER, UPPER, Ca,Pa)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: A, X1, X2
      DOUBLE PRECISION, INTENT(out) :: LOWER, UPPER,Ca,Pa
      INTEGER,INTENT(in) :: INFIN
      ! Local variables
      DOUBLE PRECISION :: P1
!      DOUBLE PRECISION, PARAMETER :: AMAX = 5D15

      ! Let  X  be standardized Gaussian variable,
      ! i.e., X=N(0,1). The function calculate the
      !  following integral E[I(X1<X<X2)ABS(A+X)]/C(A)
      ! where I(X1<X<X2) is an indicator function of
      ! the set {X1<X<X2} and C(A) is a normalization factor
      ! i.e. E(I(-inf<X<inf)*ABS(A+X)) = C(A)
      !
      ! Pa = probability at the inflection point of the CDF
      ! Ca = C(A) normalization parameter
      ! A  = location parameter of the inflection point
      ! X1,X2 = integration limits < infinity
      LOWER = 0.d0
      UPPER = 1.d0

      P1 = EXFUN(-A,A)
      Ca = A+2D0*P1
      Pa = P1/Ca
      IF ( INFIN  <  0 ) RETURN
      IF ( INFIN .NE. 0 ) THEN
         IF (X1.LE.-A) THEN
            LOWER = EXFUN(X1,A)/Ca
         ELSE
            LOWER = 1D0-EXFUN(-X1,-A)/Ca
         ENDIF
      ENDIF
      IF ( INFIN .NE. 1 ) THEN
         IF (X2.LE.-A) THEN
            UPPER = EXFUN(X2,A)/Ca
         ELSE
            UPPER = 1D0-EXFUN(-X2,-A)/Ca
         ENDIF
      ENDIF

      RETURN
      CONTAINS
      FUNCTION EXFUN(X,A) RESULT (P)
      DOUBLE PRECISION, INTENT(in) :: X,A
      DOUBLE PRECISION :: P
      DOUBLE PRECISION, PARAMETER :: SQTWOPI1=3.9894228040143d-1 !=1/sqrt(2*pi)
      P = EXP(-X*X*0.5D0)*SQTWOPI1-A*FI(X)
      END FUNCTION EXFUN
      END SUBROUTINE EXLMS


      FUNCTION EXINV(P,A,Ca,Pa) RESULT (VAL)
!EXINV calculates the inverse of the CDF of abs(x+A)*exp(-x^2/2)/SQRT(2*pi)/C(A)
!      where C(A) is a normalization parameter depending on A
!
!   CALL:  val   = exinv(P,A,Ca,Pa)
!
!     val  = quantiles
!     p    = probabilites
!     A    = location parameter
!     Ca   = normalization parameter
!     Pa   = excdf(-A,A) probability at the inflection point of the CDF
      double precision, intent(in) :: P,A,Ca,Pa
      double precision :: val
! local variables
      double precision, parameter :: amax = 5.D15
      double precision, parameter :: epsl = 5.D-15
      double precision, parameter :: xmax = 8
      double precision :: P1,Xk,Ak,Zk,SGN

!      if (P<0.D0.OR.P.GT.1.D0) PRINT *,'warning P<0 or P>1'

! The inverse cdf of 0 is -inf, and the inverse cdf of 1 is inf.
      if (P.LE.EPSL.OR.P+EPSL.GE.1.D0) THEN
         VAL = SIGN(xmax,P-0.5D0)
         return
      endif
      Ak = ABS(A)
      if (EPSL < Ak .AND. Ak < amax) THEN
         IF (ABS(p-Pa).LE.EPSL) THEN
            VAL = SIGN(MIN(Ak,xmax),-A)
            RETURN
         ENDIF
         IF (Ak < 1D-2) THEN   ! starting guess always less than 0.2 from the true value
            IF (P.GE.0.5D0) THEN
               xk = SQRT(-2D0*log(2D0*(1D0-P)))
            ELSE
               xk = -SQRT(-2D0*log(2D0*P))
            ENDIF
         ELSE
            xk = FIINV(P)       ! starting guess always less than 0.8 from the true value
            ! Modify starting guess if possible in order to speed up Newtons method
            IF (1D-3.LE.P.AND. P.LE.0.99D0.AND.
     &           3.5.LE.Ak.AND.Ak.LE.1D3 ) THEN
               SGN = SIGN(1.d0,-A)
               Zk = xk*SGN
               xk = SGN*(Zk+((1D0/(64.9495D0*Ak-178.3191D0)-0.02D0/Ak)*
     &              Zk+1D0/(-0.99679234298211D0*Ak-0.07195350071872D0))/
     &            (Zk/(-1.48430620263825D0*Ak-0.33340759016175D0)+1D0))
            ELSEIF ((P < 1D-3.AND.A.LE.-3.5D0).OR.
     &              (3.5D0.LE.A.AND.P.GT.0.99D0)) THEN
               SGN = SIGN(1.d0,-A)
               Zk = xk*SGN
               P1 = -2.00126182192701D0*Ak-2.57306603933111D0
               xk = SGN*Zk*(1D0+
     &              P1/((-0.99179258785909D0*Ak-0.21359746002397D0)*
     &              (Zk+P1)))
            ENDIF
         ENDIF
         ! Check if the starting guess is on the correct side of the inflection point
         IF (xk.LE.-A .AND. P.GT.Pa) xk = 1.D-2-A
         IF (xk.GE.-A .AND. P < Pa) xk = -1.D-2-A


         IF (P < Pa) THEN
            VAL = funca(xk,A,P*Ca)
         ELSE  ! exploit the symmetry of the CDF
            VAL = -funca(-xk,-A,(1.D0-P)*Ca)
         ENDIF

      ELSEIF (ABS(A).LE.EPSL) THEN
         IF (P>=0.5D0) THEN
            VAL = SQRT(-2D0*log(2D0*(1.D0-P)))
         ELSE
            VAL = -SQRT(-2D0*log(2D0*P))
         ENDIF
      ELSE  ! ABS(A) > AMAX
         VAL = FIINV(P)
      ENDIF
      !CALL EXLMS(A,0.d0,VAL,0,ak,P1,zk,sgn)
      !If (ABS(p-P1).GT.0.0001) PRINT *,'excdf(x,a)-p',p-P1
      RETURN

      CONTAINS

      function funca(xk0,ak,CaP) RESULT (xk)
      double precision, intent(in) :: xk0,ak,CaP ! =Ca*P
      DOUBLE PRECISION :: xk
!Local variables
      INTEGER,          PARAMETER :: ixmax = 25
      double precision, parameter :: crit = 7.1D-08 ! = sqrt(1e-15)
      double precision, parameter :: SQTWOPI1 = 0.39894228040143D0 !=1/SQRT(2*pi)
      double precision, parameter :: SQTWOPI = 2.50662827463100D0 !=SQRT(2*pi)
      INTEGER :: IX
      DOUBLE PRECISION :: H,H1,tmp0,tmp1,XNEW
      ! Newton's Method or Fixed point iteration to find the inverse of the EXCDF.
      ! Assumption: xk0 < -ak and xk < -ak
      ! Permit no more than IXMAX iterations.
      IX = 0
      H  = 1.D0
      xk = xk0    ! starting guess for the iteration


!     Break out of the iteration loop for the following:
!     1) The last update is very small (compared to x).
!     2) The last update is very small (compared to sqrt(eps)=crit).
!     3) There are more than 15 iterations. This should NEVER happen.
      IF (.TRUE..OR.ABS(ak) < 1.D-2) THEN
      ! Newton's method
      !~~~~~~~~~~~~~~~~~
      DO WHILE( ABS(H).GT.MIN(crit*ABS(xk),crit).AND.IX < IXMAX)

         IX = IX+1
                                !print *,'Iteration ',IX

         tmp0  = FI(xk)
         tmp1  = EXP(-xk*xk*0.5D0)*SQTWOPI1 ! =normpdf(x)
         H1 = (tmp1-ak*tmp0-CaP)/(ABS(xk+ak)*tmp1)
         H  = DSIGN(MIN(ABS(H1),0.7D0/DBLE(IX)),H1) ! Only allow smaller and smaller steps

         xnew = xk - H
                                ! Make sure that the current guess is less than -a.
                                ! When Newton's Method suggests steps that lead to -a guesses
                                ! take a step 9/10ths of the way to -a:
         IF (xnew.GT.-ak-crit) THEN
            xnew = (xk - 9.D0*ak)*1D-1
            H    = xnew - xk
         ENDIF
         xk = xnew
      END DO
      ELSE                      ! FIXED POINT iteration
                                !~~~~~~~~~~~~~~~~~~~~~~~
         DO WHILE (ABS(H).GT.MIN(crit*ABS(xk),crit).AND.IX < IXMAX)
            IX   = IX+1
            tmp0 = SQTWOPI1*EXP(-xk*xk*0.5D0)/FI(xk)
            tmp1 = -2.D0*LOG(SQTWOPI*CaP*tmp0/(tmp0-ak))
            SGN  = sign(1.D0,tmp1)
            xnew = -SQRT(SGN*tmp1)*SGN
            ! Make sure that the current guess is less than -a.
            ! When this method suggests steps that lead to -a guesses
            ! take a step 9/10ths of the way to -a:
            IF (xnew.GT.-ak-crit) xnew = (xk - 9.D0*ak)*1.D-1

            H  = xnew - xk
            xk = xnew
         END DO
      ENDIF

      !print *,'EXINV total number of iterations ',IX
      if (IX.GE.IXMAX) THEN
!         print *, 'Warning: EXINV did not converge. Cap=',Cap
!         print *, 'The last step was:  ', h, ' value=,',xk,' ak=',ak
      endif
      return
      END FUNCTION FUNCA
      END FUNCTION EXINV
      END MODULE FIMOD
