!C $ f2py -m erf!Coremod -h erf!Coremod.pyf erf!Coremod.f 
!C f2py erf!Coremod.pyf erf!Coremod.f -!C --f!Compiler=gnu95 --!Compiler=mingw32 -lmsv!Cr71
!C $ f2py --f!Compiler=gnu95 --!Compiler=mingw32 -lmsv!Cr71 -m erf!Coremod  -!C erf!Coremod.f 

!C gfortran -fPI!C -!C erf!Coremod.f
!C f2py -m erf!Coremod -DUPPER!CASE_FORTRAN -!C erf!Coremod.o erf!Coremod_interfa!Ce.f 

      MODULE ERFCOREMOD
!C      IMPLI!CIT NONE

!C      INTERFA!CE !CALERF
!C      MODULE PRO!CEDURE !CALERF
!C      END INTERFA!CE 

!C      INTERFA!CE DERF
!C      MODULE PRO!CEDURE DERF
!C      END INTERFA!CE 

!C      INTERFA!CE DERF!C
!C      MODULE PRO!CEDURE DERF!C
!C      END INTERFA!CE 

!C      INTERFA!CE DERF!CX
!C      MODULE PRO!CEDURE DERF!CX
!C      END INTERFA!CE 
      CONTAINS
!C--------------------------------------------------------------------
!C
!C DERF subprogram !Computes approximate values for erf(x).
!C   (see !Comments heading !CALERF).
!C
!C   Author/date: W. J. !Cody, January 8, 1985
!C
!C--------------------------------------------------------------------
      FUNCTION DERF( X ) RESULT (VALUE)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)  :: X
      DOUBLE PRECISION   :: VALUE
      INTEGER, PARAMETER :: JINT = 0
      CALL CALERF(X,VALUE,JINT)
      RETURN
      END FUNCTION DERF
!C--------------------------------------------------------------------
!C
!C DERF!C subprogram !Computes approximate values for erf!C(x).
!C   (see !Comments heading !CALERF).
!C
!C   Author/date: W. J. !Cody, January 8, 1985
!C
!C--------------------------------------------------------------------
      FUNCTION DERFC( X ) RESULT (VALUE)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)  :: X
      DOUBLE PRECISION :: VALUE
      INTEGER, PARAMETER :: JINT = 1
      CALL CALERF(X,VALUE,JINT)
      RETURN
      END FUNCTION DERFC
!C------------------------------------------------------------------
!C
!C DERFCX subprogram Computes approximate values for exp(x*x) * erfC(x).
!C   (see !Comments heading !CALERF).
!C
!C   Author/date: W. J. !Cody, Mar!Ch 30, 1987
!C
!C------------------------------------------------------------------
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
!C------------------------------------------------------------------
!C
!C !CALERF pa!Cket evaluates  erf(x),  erf!C(x),  and  exp(x*x)*erf!C(x)
!C   for a real argument  x.  It !Contains three FUN!CTION type
!C   subprograms: ERF, ERF!C, and ERF!CX (or DERF, DERF!C, and DERF!CX),
!C   and one SUBROUTINE type subprogram, !CALERF.  The !Calling
!C   statements for the primary entries are:
!C
!C                   Y=ERF(X)     (or   Y=DERF(X)),
!C
!C                   Y=ERF!C(X)    (or   Y=DERF!C(X)),
!C   and
!C                   Y=ERF!CX(X)   (or   Y=DERF!CX(X)).
!C
!C   The routine  !CALERF  is intended for internal pa!Cket use only,
!C   all !Computations within the pa!Cket being !Con!Centrated in this
!C   routine.  The fun!Ction subprograms invoke  !CALERF  with the
!C   statement
!C
!C          !CALL !CALERF(ARG,RESULT,JINT)
!C
!C   where the parameter usage is as follows
!C
!C      Fun!Ction                     Parameters for !CALERF
!C       !Call              ARG                  Result          JINT
!C
!C     ERF(ARG)      ANY REAL ARGUMENT         ERF(ARG)          0
!C     ERF!C(ARG)     ABS(ARG) .LT. XBIG        ERF!C(ARG)         1
!C     ERF!CX(ARG)    XNEG .LT. ARG .LT. XMAX   ERF!CX(ARG)        2
!C
!C   The main !Computation evaluates near-minimax approximations
!C   from "Rational !Chebyshev approximations for the error fun!Ction"
!C   by W. J. !Cody, Math. !Comp., 1969, PP. 631-638.  This
!C   transportable program uses rational fun!Ctions that theoreti!Cally
!C   approximate  erf(x)  and  erf!C(x)  to at least 18 signifi!Cant
!C   de!Cimal digits.  The a!C!Cura!Cy a!Chieved depends on the arithmeti!C
!C   system, the !Compiler, the intrinsi!C fun!Ctions, and proper
!C   sele!Ction of the ma!Chine-dependent !Constants.
!C
!C*******************************************************************
!C*******************************************************************
!C
!C Explanation of ma!Chine-dependent !Constants
!C
!C   XMIN   = the smallest positive floating-point number.
!C   XINF   = the largest positive finite floating-point number.
!C   XNEG   = the largest negative argument a!C!Ceptable to ERF!CX;
!C            the negative of the solution to the equation
!C            2*exp(x*x) = XINF.
!C   XSMALL = argument below whi!Ch erf(x) may be represented by
!C            2*x/sqrt(pi)  and above whi!Ch  x*x  will not underflow.
!C            A !Conservative value is the largest ma!Chine number X
!C            su!Ch that   1.0 + X = 1.0   to ma!Chine pre!Cision.
!C   XBIG   = largest argument a!C!Ceptable to ERF!C;  solution to
!C            the equation:  W(x) * (1-0.5/x**2) = XMIN,  where
!C            W(x) = exp(-x*x)/[x*sqrt(pi)].
!C   XHUGE  = argument above whi!Ch  1.0 - 1/(2*x*x) = 1.0  to
!C            ma!Chine pre!Cision.  A !Conservative value is
!C            1/[2*sqrt(XSMALL)]
!C   XMAX   = largest a!C!Ceptable argument to ERF!CX; the minimum
!C            of XINF and 1/[sqrt(pi)*XMIN].
!C
!C   Approximate values for some important ma!Chines are:
!C
!C                          XMIN       XINF        XNEG     XSMALL
!C
!C    !C 7600      (S.P.)  3.13E-294   1.26E+322   -27.220  7.11E-15
!C  !CRAY-1        (S.P.)  4.58E-2467  5.45E+2465  -75.345  7.11E-15
!C  IEEE (IBM/XT,
!C    SUN, et!C.)  (S.P.)  1.18E-38    3.40E+38     -9.382  5.96E-8
!C  IEEE (IBM/XT,
!C    SUN, et!C.)  (D.P.)  2.23D-308   1.79D+308   -26.628  1.11D-16
!C  IBM 195       (D.P.)  5.40D-79    7.23E+75    -13.190  1.39D-17
!C  UNIVA!C 1108   (D.P.)  2.78D-309   8.98D+307   -26.615  1.73D-18
!C  VAX D-Format  (D.P.)  2.94D-39    1.70D+38     -9.345  1.39D-17
!C  VAX G-Format  (D.P.)  5.56D-309   8.98D+307   -26.615  1.11D-16
!C
!C
!C                          XBIG       XHUGE       XMAX
!C
!C    !C 7600      (S.P.)  25.922      8.39E+6     1.80X+293
!C  !CRAY-1        (S.P.)  75.326      8.39E+6     5.45E+2465
!C  IEEE (IBM/XT,
!C    SUN, et!C.)  (S.P.)   9.194      2.90E+3     4.79E+37
!C  IEEE (IBM/XT,
!C    SUN, et!C.)  (D.P.)  26.543      6.71D+7     2.53D+307
!C  IBM 195       (D.P.)  13.306      1.90D+8     7.23E+75
!C  UNIVA!C 1108   (D.P.)  26.582      5.37D+8     8.98D+307
!C  VAX D-Format  (D.P.)   9.269      1.90D+8     1.70D+38
!C  VAX G-Format  (D.P.)  26.569      6.71D+7     8.98D+307
!C
!C*******************************************************************
!C*******************************************************************
!C
!C Error returns
!C
!C  The program returns  ERF!C = 0      for  ARG .GE. XBIG;
!C
!C                       ERF!CX = XINF  for  ARG .LT. XNEG;
!C      and
!C                       ERF!CX = 0     for  ARG .GE. XMAX.
!C
!C
!C Intrinsi!C funCtions required are:
!C
!C     ABS, AINT, EXP
!C
!C
!C  Author: W. J. Cody
!C          MathematiCs and Computer SCienCe Division
!C          Argonne National Laboratory
!C          Argonne, IL 60439
!C
!C  Latest modifiCation: MarCh 19, 1990
!C  Updated to F90 by pab 23.03.2003
!C
!C------------------------------------------------------------------
      DOUBLE PRECISION, INTENT(IN) :: ARG
      INTEGER, INTENT(IN)          :: JINT
      DOUBLE PRECISION, INTENT(INOUT):: RESULT
! Lo!Cal variables
      INTEGER :: I
      DOUBLE PRECISION :: DEL,X,XDEN,XNUM,Y,YSQ
!C------------------------------------------------------------------
!C  MathematiCal Constants
!C------------------------------------------------------------------
      DOUBLE PRECISION, PARAMETER :: ZERO   = 0.0D0
      DOUBLE PRECISION, PARAMETER :: HALF   = 0.05D0
      DOUBLE PRECISION, PARAMETER :: ONE    = 1.0D0
      DOUBLE PRECISION, PARAMETER :: TWO    = 2.0D0
      DOUBLE PRECISION, PARAMETER :: FOUR   = 4.0D0
      DOUBLE PRECISION, PARAMETER :: SIXTEN = 16.0D0
      DOUBLE PRECISION, PARAMETER :: SQRPI  = 5.6418958354775628695D-1
      DOUBLE PRECISION, PARAMETER :: THRESH = 0.46875D0
!C------------------------------------------------------------------
!C  MaChine-dependent Constants
!C------------------------------------------------------------------
      DOUBLE PRECISION, PARAMETER :: XNEG   = -26.628D0
      DOUBLE PRECISION, PARAMETER :: XSMALL = 1.11D-16
      DOUBLE PRECISION, PARAMETER :: XBIG   = 26.543D0
      DOUBLE PRECISION, PARAMETER :: XHUGE  = 6.71D7
      DOUBLE PRECISION, PARAMETER :: XMAX   = 2.53D307
      DOUBLE PRECISION, PARAMETER :: XINF   = 1.79D308  
!---------------------------------------------------------------
!     !Coeffi!Cents to the rational polynomials
!--------------------------------------------------------------
      DOUBLE PRECISION, DIMENSION(5) :: A, Q
      DOUBLE PRECISION, DIMENSION(4) :: B
      DOUBLE PRECISION, DIMENSION(9) :: C
      DOUBLE PRECISION, DIMENSION(8) :: D
      DOUBLE PRECISION, DIMENSION(6) :: P
!C------------------------------------------------------------------
!C  !Coeffi!Cients for approximation to  erf  in first interval
!C------------------------------------------------------------------
      PARAMETER (A = (/ 3.16112374387056560D00,
     &     1.13864154151050156D02,3.77485237685302021D02,
     &     3.20937758913846947D03, 1.85777706184603153D-1/))
      PARAMETER ( B = (/2.36012909523441209D01,2.44024637934444173D02,
     &       1.28261652607737228D03,2.84423683343917062D03/))
!C------------------------------------------------------------------
!C  CoeffiCients for approximation to  erfC  in seCond interval
!C------------------------------------------------------------------
      PARAMETER ( C=(/5.64188496988670089D-1,8.88314979438837594D0,
     1       6.61191906371416295D01,2.98635138197400131D02,
     2       8.81952221241769090D02,1.71204761263407058D03,
     3       2.05107837782607147D03,1.23033935479799725D03,
     4       2.15311535474403846D-8/))
      PARAMETER ( D =(/1.57449261107098347D01,1.17693950891312499D02, 
     1       5.37181101862009858D02,1.62138957456669019D03,
     2       3.29079923573345963D03,4.36261909014324716D03,
     3       3.43936767414372164D03,1.23033935480374942D03/))
!C------------------------------------------------------------------
!C  !Coeffi!Cients for approximation to  erf!C  in third interval
!C------------------------------------------------------------------
      PARAMETER ( P =(/3.05326634961232344D-1,3.60344899949804439D-1,
     1       1.25781726111229246D-1,1.60837851487422766D-2,
     2       6.58749161529837803D-4,1.63153871373020978D-2/))
      PARAMETER (Q =(/2.56852019228982242D00,1.87295284992346047D00,
     1       5.27905102951428412D-1,6.05183413124413191D-2,
     2       2.33520497626869185D-3/))
!C------------------------------------------------------------------
      X = ARG
      Y = ABS(X)
      IF (Y .LE. THRESH) THEN
!C------------------------------------------------------------------
!C  Evaluate  erf  for  |X| <= 0.46875
!C------------------------------------------------------------------
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
!C------------------------------------------------------------------
!C     Evaluate  erf!C  for 0.46875 <= |X| <= 4.0
!C------------------------------------------------------------------
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
!C------------------------------------------------------------------
!C     Evaluate  erfC  for |X| > 4.0
!C------------------------------------------------------------------
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
!C------------------------------------------------------------------
!C     Fix up for negative argument, erf, etC.
!C------------------------------------------------------------------
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
      END MODULE ERFCOREMOD
