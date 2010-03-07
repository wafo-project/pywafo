C $ f2py -m erfcore -h erfcore.pyf erfcore.f 
C f2py erfcore.pyf erfcore.f -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
C $ f2py --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 -m erfcore  -c erfcore.f 
C
C--------------------------------------------------------------------
C
C DERF subprogram computes approximate values for erf(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, January 8, 1985
C
C--------------------------------------------------------------------
      FUNCTION DERF( X ) RESULT (VALUE)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)  :: X
      DOUBLE PRECISION   :: VALUE
      INTEGER, PARAMETER :: JINT = 0
      CALL CALERF(X,VALUE,JINT)
      RETURN
      END FUNCTION DERF
C--------------------------------------------------------------------
C
C DERFC subprogram computes approximate values for erfc(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, January 8, 1985
C
C--------------------------------------------------------------------
      FUNCTION DERFC( X ) RESULT (VALUE)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)  :: X
      DOUBLE PRECISION :: VALUE
      INTEGER, PARAMETER :: JINT = 1
      CALL CALERF(X,VALUE,JINT)
      RETURN
      END FUNCTION DERFC
C------------------------------------------------------------------
C
C DERFCX subprogram computes approximate values for exp(x*x) * erfc(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, March 30, 1987
C
C------------------------------------------------------------------
      FUNCTION DERFCX( X ) RESULT (VALUE)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)  :: X
      DOUBLE PRECISION   :: VALUE
      INTEGER, PARAMETER :: JINT = 2
      CALL CALERF(X,VALUE,JINT)
      RETURN
      END FUNCTION DERFCX

      SUBROUTINE CALERF(ARG,RESULT,JINT)
      IMPLICIT NONE
C------------------------------------------------------------------
C
C CALERF packet evaluates  erf(x),  erfc(x),  and  exp(x*x)*erfc(x)
C   for a real argument  x.  It contains three FUNCTION type
C   subprograms: ERF, ERFC, and ERFCX (or DERF, DERFC, and DERFCX),
C   and one SUBROUTINE type subprogram, CALERF.  The calling
C   statements for the primary entries are:
C
C                   Y=ERF(X)     (or   Y=DERF(X)),
C
C                   Y=ERFC(X)    (or   Y=DERFC(X)),
C   and
C                   Y=ERFCX(X)   (or   Y=DERFCX(X)).
C
C   The routine  CALERF  is intended for internal packet use only,
C   all computations within the packet being concentrated in this
C   routine.  The function subprograms invoke  CALERF  with the
C   statement
C
C          CALL CALERF(ARG,RESULT,JINT)
C
C   where the parameter usage is as follows
C
C      Function                     Parameters for CALERF
C       call              ARG                  Result          JINT
C
C     ERF(ARG)      ANY REAL ARGUMENT         ERF(ARG)          0
C     ERFC(ARG)     ABS(ARG) .LT. XBIG        ERFC(ARG)         1
C     ERFCX(ARG)    XNEG .LT. ARG .LT. XMAX   ERFCX(ARG)        2
C
C   The main computation evaluates near-minimax approximations
C   from "Rational Chebyshev approximations for the error function"
C   by W. J. Cody, Math. Comp., 1969, PP. 631-638.  This
C   transportable program uses rational functions that theoretically
C   approximate  erf(x)  and  erfc(x)  to at least 18 significant
C   decimal digits.  The accuracy achieved depends on the arithmetic
C   system, the compiler, the intrinsic functions, and proper
C   selection of the machine-dependent constants.
C
C*******************************************************************
C*******************************************************************
C
C Explanation of machine-dependent constants
C
C   XMIN   = the smallest positive floating-point number.
C   XINF   = the largest positive finite floating-point number.
C   XNEG   = the largest negative argument acceptable to ERFCX;
C            the negative of the solution to the equation
C            2*exp(x*x) = XINF.
C   XSMALL = argument below which erf(x) may be represented by
C            2*x/sqrt(pi)  and above which  x*x  will not underflow.
C            A conservative value is the largest machine number X
C            such that   1.0 + X = 1.0   to machine precision.
C   XBIG   = largest argument acceptable to ERFC;  solution to
C            the equation:  W(x) * (1-0.5/x**2) = XMIN,  where
C            W(x) = exp(-x*x)/[x*sqrt(pi)].
C   XHUGE  = argument above which  1.0 - 1/(2*x*x) = 1.0  to
C            machine precision.  A conservative value is
C            1/[2*sqrt(XSMALL)]
C   XMAX   = largest acceptable argument to ERFCX; the minimum
C            of XINF and 1/[sqrt(pi)*XMIN].
C
C   Approximate values for some important machines are:
C
C                          XMIN       XINF        XNEG     XSMALL
C
C    C 7600      (S.P.)  3.13E-294   1.26E+322   -27.220  7.11E-15
C  CRAY-1        (S.P.)  4.58E-2467  5.45E+2465  -75.345  7.11E-15
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)  1.18E-38    3.40E+38     -9.382  5.96E-8
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)  2.23D-308   1.79D+308   -26.628  1.11D-16
C  IBM 195       (D.P.)  5.40D-79    7.23E+75    -13.190  1.39D-17
C  UNIVAC 1108   (D.P.)  2.78D-309   8.98D+307   -26.615  1.73D-18
C  VAX D-Format  (D.P.)  2.94D-39    1.70D+38     -9.345  1.39D-17
C  VAX G-Format  (D.P.)  5.56D-309   8.98D+307   -26.615  1.11D-16
C
C
C                          XBIG       XHUGE       XMAX
C
C    C 7600      (S.P.)  25.922      8.39E+6     1.80X+293
C  CRAY-1        (S.P.)  75.326      8.39E+6     5.45E+2465
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)   9.194      2.90E+3     4.79E+37
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)  26.543      6.71D+7     2.53D+307
C  IBM 195       (D.P.)  13.306      1.90D+8     7.23E+75
C  UNIVAC 1108   (D.P.)  26.582      5.37D+8     8.98D+307
C  VAX D-Format  (D.P.)   9.269      1.90D+8     1.70D+38
C  VAX G-Format  (D.P.)  26.569      6.71D+7     8.98D+307
C
C*******************************************************************
C*******************************************************************
C
C Error returns
C
C  The program returns  ERFC = 0      for  ARG .GE. XBIG;
C
C                       ERFCX = XINF  for  ARG .LT. XNEG;
C      and
C                       ERFCX = 0     for  ARG .GE. XMAX.
C
C
C Intrinsic functions required are:
C
C     ABS, AINT, EXP
C
C
C  Author: W. J. Cody
C          Mathematics and Computer Science Division
C          Argonne National Laboratory
C          Argonne, IL 60439
C
C  Latest modification: March 19, 1990
C  Updated to F90 by pab 23.03.2003
C
C------------------------------------------------------------------
      DOUBLE PRECISION, INTENT(IN) :: ARG
      INTEGER, INTENT(IN)          :: JINT
      DOUBLE PRECISION, INTENT(INOUT):: RESULT
! Local variables
      INTEGER :: I
      DOUBLE PRECISION :: DEL,X,XDEN,XNUM,Y,YSQ
C------------------------------------------------------------------
C  Mathematical constants
C------------------------------------------------------------------
      DOUBLE PRECISION, PARAMETER :: ZERO   = 0.0D0
      DOUBLE PRECISION, PARAMETER :: HALF   = 0.05D0
      DOUBLE PRECISION, PARAMETER :: ONE    = 1.0D0
      DOUBLE PRECISION, PARAMETER :: TWO    = 2.0D0
      DOUBLE PRECISION, PARAMETER :: FOUR   = 4.0D0
      DOUBLE PRECISION, PARAMETER :: SIXTEN = 16.0D0
      DOUBLE PRECISION, PARAMETER :: SQRPI  = 5.6418958354775628695D-1
      DOUBLE PRECISION, PARAMETER :: THRESH = 0.46875D0
C------------------------------------------------------------------
C  Machine-dependent constants
C------------------------------------------------------------------
      DOUBLE PRECISION, PARAMETER :: XNEG   = -26.628D0
      DOUBLE PRECISION, PARAMETER :: XSMALL = 1.11D-16
      DOUBLE PRECISION, PARAMETER :: XBIG   = 26.543D0
      DOUBLE PRECISION, PARAMETER :: XHUGE  = 6.71D7
      DOUBLE PRECISION, PARAMETER :: XMAX   = 2.53D307
      DOUBLE PRECISION, PARAMETER :: XINF   = 1.79D308  
!---------------------------------------------------------------
!     Coefficents to the rational polynomials
!--------------------------------------------------------------
      DOUBLE PRECISION, DIMENSION(5) :: A, Q
      DOUBLE PRECISION, DIMENSION(4) :: B
      DOUBLE PRECISION, DIMENSION(9) :: C
      DOUBLE PRECISION, DIMENSION(8) :: D
      DOUBLE PRECISION, DIMENSION(6) :: P
C------------------------------------------------------------------
C  Coefficients for approximation to  erf  in first interval
C------------------------------------------------------------------
      PARAMETER (A = (/ 3.16112374387056560D00,
     &     1.13864154151050156D02,3.77485237685302021D02,
     &     3.20937758913846947D03, 1.85777706184603153D-1/))
      PARAMETER ( B = (/2.36012909523441209D01,2.44024637934444173D02,
     &       1.28261652607737228D03,2.84423683343917062D03/))
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in second interval
C------------------------------------------------------------------
      PARAMETER ( C=(/5.64188496988670089D-1,8.88314979438837594D0,
     1       6.61191906371416295D01,2.98635138197400131D02,
     2       8.81952221241769090D02,1.71204761263407058D03,
     3       2.05107837782607147D03,1.23033935479799725D03,
     4       2.15311535474403846D-8/))
      PARAMETER ( D =(/1.57449261107098347D01,1.17693950891312499D02, 
     1       5.37181101862009858D02,1.62138957456669019D03,
     2       3.29079923573345963D03,4.36261909014324716D03,
     3       3.43936767414372164D03,1.23033935480374942D03/))
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in third interval
C------------------------------------------------------------------
      PARAMETER ( P =(/3.05326634961232344D-1,3.60344899949804439D-1,
     1       1.25781726111229246D-1,1.60837851487422766D-2,
     2       6.58749161529837803D-4,1.63153871373020978D-2/))
      PARAMETER (Q =(/2.56852019228982242D00,1.87295284992346047D00,
     1       5.27905102951428412D-1,6.05183413124413191D-2,
     2       2.33520497626869185D-3/))
C------------------------------------------------------------------
      X = ARG
      Y = ABS(X)
      IF (Y .LE. THRESH) THEN
C------------------------------------------------------------------
C  Evaluate  erf  for  |X| <= 0.46875
C------------------------------------------------------------------
         !YSQ = ZERO
         IF (Y .GT. XSMALL) THEN
            YSQ = Y * Y
            XNUM = A(5)*YSQ
            XDEN = YSQ
            DO  I = 1, 3
               XNUM = (XNUM + A(I)) * YSQ
               XDEN = (XDEN + B(I)) * YSQ
            END DO
            RESULT = X * (XNUM + A(4)) / (XDEN + B(4))
         ELSE
            RESULT = X *  A(4) / B(4)
         ENDIF
         IF (JINT .NE. 0) RESULT = ONE - RESULT
         IF (JINT .EQ. 2) RESULT = EXP(YSQ) * RESULT
         GO TO 800
C------------------------------------------------------------------
C     Evaluate  erfc  for 0.46875 <= |X| <= 4.0
C------------------------------------------------------------------
      ELSE IF (Y .LE. FOUR) THEN
         XNUM = C(9)*Y
         XDEN = Y
         DO I = 1, 7
            XNUM = (XNUM + C(I)) * Y
            XDEN = (XDEN + D(I)) * Y
         END DO
         RESULT = (XNUM + C(8)) / (XDEN + D(8))
         IF (JINT .NE. 2) THEN
            YSQ = AINT(Y*SIXTEN)/SIXTEN
            DEL = (Y-YSQ)*(Y+YSQ)
            RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT
         END IF
C------------------------------------------------------------------
C     Evaluate  erfc  for |X| > 4.0
C------------------------------------------------------------------
      ELSE
         RESULT = ZERO
         IF (Y .GE. XBIG) THEN
            IF ((JINT .NE. 2) .OR. (Y .GE. XMAX)) GO TO 300
            IF (Y .GE. XHUGE) THEN
               RESULT = SQRPI / Y
               GO TO 300
            END IF
         END IF
         YSQ = ONE / (Y * Y)
         XNUM = P(6)*YSQ
         XDEN = YSQ
         DO I = 1, 4
            XNUM = (XNUM + P(I)) * YSQ
            XDEN = (XDEN + Q(I)) * YSQ
         ENDDO
         RESULT = YSQ *(XNUM + P(5)) / (XDEN + Q(5))
         RESULT = (SQRPI -  RESULT) / Y
         IF (JINT .NE. 2) THEN
            YSQ = AINT(Y*SIXTEN)/SIXTEN
            DEL = (Y-YSQ)*(Y+YSQ)
            RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT
         END IF
      END IF
C------------------------------------------------------------------
C     Fix up for negative argument, erf, etc.
C------------------------------------------------------------------
 300  IF (JINT .EQ. 0) THEN
         RESULT = (HALF - RESULT) + HALF
         IF (X .LT. ZERO) RESULT = -RESULT
      ELSE IF (JINT .EQ. 1) THEN
         IF (X .LT. ZERO) RESULT = TWO - RESULT
      ELSE
         IF (X .LT. ZERO) THEN
            IF (X .LT. XNEG) THEN
               RESULT = XINF
            ELSE
               YSQ = AINT(X*SIXTEN)/SIXTEN
               DEL = (X-YSQ)*(X+YSQ)
               Y = EXP(YSQ*YSQ) * EXP(DEL)
               RESULT = (Y+Y) - RESULT
            END IF
         END IF
      END IF
 800  RETURN
      END SUBROUTINE CALERF
      FUNCTION FIINV(P) RESULT (VAL)
      IMPLICIT NONE
*
*	ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3
*
*	Produces the normal deviate Z corresponding to a given lower
*	tail area of P.
*       Absolute error less than 1e-13
*       Relative error less than 1e-15 for abs(VAL)>0.1
*
*	The hash sums below are the sums of the mantissas of the
*	coefficients.   They are included for use in checking
*	transcription.
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
     &            CONST1 = 0.180625D0, CONST2 = 1.6D0,
     &				ONE = 1.D0, ZERO = 0.D0, HALF = 0.5D0 )
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
            VAL = 37.D0 !XMAX 9.d0
         END IF
         IF ( Q  <  ZERO ) VAL = - VAL
      END IF
      RETURN
      END FUNCTION FIINV  
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
      PARAMETER( XMAX   = 8.25D0 )
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

      FUNCTION FI( Z ) RESULT (VALUE)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: Z
      DOUBLE PRECISION :: VALUE
! Local variables
      DOUBLE PRECISION, PARAMETER:: SQ2M1 = 0.70710678118655D0 !     1/SQRT(2)
      DOUBLE PRECISION, PARAMETER:: HALF = 0.5D0
      VALUE = DERFC(-Z*SQ2M1)*HALF
      RETURN
      END FUNCTION FI