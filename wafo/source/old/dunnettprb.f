
      MODULE ERFCOREMOD
      IMPLICIT NONE

      INTERFACE CALERF
      MODULE PROCEDURE CALERF
      END INTERFACE 

      INTERFACE DERF
      MODULE PROCEDURE DERF
      END INTERFACE 

      INTERFACE DERFC
      MODULE PROCEDURE DERFC
      END INTERFACE 

      INTERFACE DERFCX
      MODULE PROCEDURE DERFCX
      END INTERFACE 
      CONTAINS
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
      END MODULE ERFCOREMOD
      
      MODULE DUNNETMOD
      
      SUBROUTINE MVNPRD(A, B, BPD, EPS, N, INF, IERC, HINC, PROB,
     &     BOUND,IFAULT)
C     
C     ALGORITHM AS 251.1  APPL.STATIST. (1989), VOL.38, NO.3
C
C     FOR A MULTIVARIATE NORMAL VECTOR WITH CORRELATION STRUCTURE
C     DEFINED BY RHO(I,J) = BPD(I) * BPD(J), COMPUTES THE PROBABILITY
C     THAT THE VECTOR FALLS IN A RECTANGLE IN N-SPACE WITH ERROR
C     LESS THAN EPS.
C
      INTEGER NN
      PARAMETER (NN = 50)
      REAL A(*), B(*), BPD(*), ESTT(22), FV(5), FD(5), F1T(22),
     *  F2T(22), F3T(22), G1T(22), G3T(22), PSUM(22), H(NN), HL(NN),
     *  BB(NN)
      INTEGER INF(*), INFT(NN), LDIR(22)
      REAL ZERO, HALF, ONE, TWO, FOUR, SIX, PT1, PT24, ONEP5,
     *  X2880, SMALL, DXMIN, SQRT2, PROB, ERRL, BI, START,
     *  Z, HINC, ADDN, EPS2, EPS1, EPS, ZU, Z2, Z3, Z4, Z5, ZZ,
     *  ERFAC, EL, EL1, BOUND, PART0, PART2, PART3, FUNC0, FUNC2,
     *  FUNCN, WT, CONTRB, DLG, DX, DA, ESTL, ESTR, SUM, EXCESS, ERROR,
     *  PROB1, SAFE
      INTEGER N, IERC, IFAULT, I, NTM, NMAX, LVL, NR, NDIM
      REAL ALNORM, PPND7
      EXTERNAL ALNORM, PPND7
      DATA ZERO, HALF, ONE, TWO, FOUR, SIX /0.0, 0.5, 1.0, 2.0,
     *  4.0, 6.0/
      DATA PT1, PT24, ONEP5, X2880 /0.1, 0.24, 1.5, 2880.0/
      DATA SMALL, DXMIN, SQRT2 /1.0E-10, 0.0000001, 1.41421356237310/
C
C        CHECK FOR INPUT VALUES OUT OF RANGE.
C
      PROB = ZERO
      BOUND = ZERO
      IFAULT = 1
      IF (N .LT. 1 .OR. N .GT. NN) RETURN
      DO 10 I = 1, N
         BI = ABS(BPD(I))
         IFAULT = 2
         IF (BI .GE. ONE) RETURN
         IFAULT = 3
         IF (INF(I) .LT. 0 .OR. INF(I) .GT. 2) RETURN
         IFAULT = 4
         IF (INF(I) .EQ. 2 .AND. A(I) .LE. B(I)) RETURN
   10 CONTINUE
      IFAULT = 0
      PROB = ONE
C
C        CHECK WHETHER ANY BPD(I) = 0.
C
      NDIM = 0
      DO 20 I = 1, N
         IF (BPD(I) .NE. ZERO) THEN
            NDIM = NDIM + 1
            H(NDIM) = A(I)
            HL(NDIM) = B(I)
            BB(NDIM) = BPD(I)
            INFT(NDIM) = INF(I)
         ELSE
C
C        IF ANY BPD(I) = 0, THE CONTRIBUTION TO PROB FOR THAT
C        VARIABLE IS COMPUTED FROM A UNIVARIATE NORMAL.
C
            IF (INF(I) .LT. 1) THEN
               PROB = PROB * (ONE - ALNORM(B(I), .FALSE.))
            ELSE IF (INF(I) .EQ. 1) THEN
               PROB = PROB * ALNORM(A(I), .FALSE.)
            ELSE
               PROB = PROB * (ALNORM(A(I), .FALSE.) -
     *                ALNORM(B(I), .FALSE.))
            END IF
            IF (PROB .LE. SMALL) PROB = ZERO
         END IF
   20 CONTINUE
      IF (NDIM .EQ. 0 .OR. PROB .EQ. ZERO) RETURN
C
C        IF NOT ALL BPD(I) = 0, PROB IS COMPUTED BY SIMPSON'S RULE.
C        BUT FIRST, INITIALIZE THE VARIABLES.
C
      Z = ZERO
      IF (HINC .LE. ZERO) HINC = PT24
      ADDN = -ONE
      DO 30 I = 1, NDIM
         IF (INFT(I) .EQ. 2 .OR.
     *   (INFT(I) .NE. INFT(1) .AND. BB(I) * BB(1) .GT. ZERO) .OR.
     *   (INFT(I) .EQ. INFT(1) .AND. BB(I) * BB(1) .LT. ZERO))
     *       ADDN = ZERO
   30 CONTINUE
C
C        THE VALUE OF ADDN IS TO BE ADDED TO THE PRODUCT EXPRESSIONS IN
C        THE INTEGRAND TO INSURE THAT THE LIMITING VALUE IS ZERO.
C
      PROB1 = ZERO
      NTM = 0
      NMAX = 400
      IF (IERC .EQ. 0) NMAX = NMAX * 2
      CALL PFUNC (Z, H, HL, BB, NDIM, INFT, ADDN, SAFE, FUNC0, NTM,
     *  IERC, PART0)
      EPS2 = EPS * PT1 * HALF
C
C        SET UPPER BOUND ON Z AND APPORTION EPS.
C
      ZU = -PPND7(EPS2, IFAULT) / SQRT2
      IF (IFAULT .NE. 0) THEN
         IFAULT = 6
         RETURN
      END IF
      NR = IFIX(ZU / HINC) + 1
      ERFAC = ONE
      IF (IERC .NE. 0) ERFAC = X2880 / HINC ** 5
      EL = (EPS - EPS2) / FLOAT(NR) * ERFAC
      EL1 = EL
C
C        START COMPUTATIONS FOR THE INTERVAL (Z, Z + HINC).
C
   40 ERROR = ZERO
      LVL = 0
      FV(1) = PART0
      FD(1) = SAFE
      START = Z
      DA = HINC
      Z3 = START + HALF * DA
      CALL PFUNC(Z3, H, HL, BB, NDIM, INFT, ADDN, FD(3), FUNCN, NTM,
     *  IERC, FV(3))
      Z5 = START + DA
      CALL PFUNC(Z5, H, HL, BB, NDIM, INFT, ADDN, FD(5), FUNC2, NTM,
     *  IERC, FV(5))
      PART2 = FV(5)
      SAFE = FD(5)
      WT = DA / SIX
      CONTRB = WT * (FV(1) + FOUR * FV(3) + FV(5))
      DLG = ZERO
      IF (IERC .NE. 0) THEN
         CALL WMAX(FD(1), FD(3), FD(5), DLG)
         IF (DLG .LE. EL) GO TO 90
         DX = DA
         GO TO 60
      END IF
      LVL = 1
      LDIR(LVL) = 2
      PSUM(LVL) = ZERO
C
C        BISECT INTERVAL.  IF IERC = 1, COMPUTE ESTIMATE ON LEFT
C        HALF; IF IERC = 0, ON BOTH HALVES.
C
   50 DX = HALF * DA
      WT = DX / SIX
      Z2 = START + HALF * DX
      CALL PFUNC(Z2, H, HL, BB, NDIM, INFT, ADDN, FD(2), FUNCN, NTM,
     *  IERC,FV(2))
      ESTL = WT * (FV(1) + FOUR * FV(2) + FV(3))
      IF (IERC .EQ. 0) THEN
         Z4 = START + ONEP5 * DX
         CALL PFUNC(Z4, H, HL, BB, NDIM, INFT, ADDN, FD(4), FUNCN,
     *     NTM, IERC, FV(4))
         ESTR = WT * (FV(3) + FOUR * FV(4) + FV(5))
         SUM = ESTL + ESTR
         DLG = ABS(CONTRB - SUM)
         EPS1 = EL / TWO ** (LVL - 1)
         ERRL = DLG
      ELSE
         FV(3) = FV(2)
         FD(3) = FD(2)
         CALL WMAX(FD(1), FD(3), FD(5), DLG)
         ERRL = DLG / TWO ** (5 * LVL)
         SUM = ESTL
         EPS1 = EL * (TWO ** LVL) ** 4
      END IF
C
C        STOP SUBDIVIDING INTERVAL WHEN ACCURACY IS SUFFICIENT,
C        OR IF INTERVAL TOO NARROW OR SUBDIVIDED TOO OFTEN.
C
      IF (DLG .LE. EPS1 .OR. DLG .LT. SMALL) GO TO 70
      IF (IFAULT .EQ. 0 .AND. NTM .GE. NMAX) IFAULT = 5
      IF (ABS(DX) .LE. DXMIN .OR. LVL .GT. 21) IFAULT = 7
      IF (IFAULT .NE. 0) GO TO 70
C
C        RAISE LEVEL.  STORE INFORMATION FOR RIGHT HALF AND APPLY
C        SIMPSON'S RULE TO LEFT HALF.
C
   60 LVL = LVL + 1
      LDIR(LVL) = 1
      F1T(LVL) = FV(3)
      F3T(LVL) = FV(5)
      DA = DX
      FV(5) = FV(3)
      IF (IERC .EQ. 0) THEN
         F2T(LVL) = FV(4)
         ESTT(LVL) = ESTR
         CONTRB = ESTL
         FV(3) = FV(2)
      ELSE
         G1T(LVL) = FD(3)
         G3T(LVL) = FD(5)
         FD(5) = FD(3)
      END IF
      GO TO 50
C
C        ACCEPT APPROXIMATE VALUE FOR INTERVAL.
C        RESTORE SAVED INFORMATION TO PROCESS
C        RIGHT HALF INTERVAL.
C
   70 ERROR = ERROR + ERRL
   80 IF (LDIR(LVL) .EQ. 1) THEN
         PSUM(LVL) = SUM
         LDIR(LVL) = 2
         IF (IERC .EQ. 0) DX = DX * TWO
         START = START + DX
         DA = HINC / TWO ** (LVL - 1)
         FV(1) = F1T(LVL)
         IF (IERC .EQ. 0) THEN
            FV(3) = F2T(LVL)
            CONTRB = ESTT(LVL)
         ELSE
            FV(3) = F3T(LVL)
            FD(1) = G1T(LVL)
            FD(5) = G3T(LVL)
         END IF
         FV(5) = F3T(LVL)
         GO TO 50
      END IF
      SUM = SUM + PSUM(LVL)
      LVL = LVL - 1
      IF (LVL .GT. 0) GO TO 80
      CONTRB = SUM
      LVL = 1
      DLG = ERROR
   90 PROB1 = PROB1 + CONTRB
      BOUND = BOUND + DLG
      EXCESS = EL - DLG
      EL = EL1
      IF (EXCESS .GT. ZERO) EL = EL1 + EXCESS
      IF ((FUNC0 .GT. ZERO .AND. FUNC2 .LE. FUNC0) .OR.
     *    (FUNC0 .LT. ZERO .AND. FUNC2 .GE. FUNC0)) THEN
         ZZ = -SQRT2 * Z5
         PART3 = ABS(FUNC2) * ALNORM(ZZ, .FALSE.) + BOUND / ERFAC
         IF (PART3 .LE. EPS .OR. NTM .GE. NMAX .OR. Z5 .GE. ZU) GOTO 100
      END IF
      Z = Z5
      PART0 = PART2
      FUNC0 = FUNC2
      IF (Z .LT. ZU .AND. NTM .LT. NMAX) GO TO 40
  100 PROB = (PROB1 - ADDN * HALF) * PROB
      BOUND = PART3
      IF (NTM .GE. NMAX .AND. IFAULT .EQ. 0) IFAULT = 5
      IF (BOUND .GT. EPS .AND. IFAULT .EQ. 0) IFAULT = 8
      RETURN
      END
      SUBROUTINE PFUNC(Z, A, B, BPD, N, INF, ADDN, DERIV, FUNCN, NTM,
     *  IERC, RESULT)
C
C        ALGORITHM AS 251.2  APPL.STATIST. (1989), VOL.38, NO.3
C
C
C        COMPUTE FUNCTION IN INTEGRAND AND ITS 4TH DERIVATIVE.
C
      INTEGER NN
      PARAMETER (NN = 50)
      REAL A(*), B(*), BPD(*), FOU(NN), FOU1(4, NN), TMP(4), GOU(NN),
     *  GOU1(4, NN), FF(4), GF(4), TERM(4), GERM(4)
      INTEGER INF(*)
      REAL ZERO, ONE, TWO, THREE, FOUR, SIX, EIGHT, TWELVE, SIXTN,
     *  SMALL, Z, U, U1, U2, BI, HI, HLI, BP, ADDN, DERIV, FUNCN,
     *  RESULT, RSLT1, RSLT2, DEN, SQRT2, SQRTPI, PHI, PHI1, PHI2,
     *  PHI3, PHI4, FRM, GRM
      INTEGER N, NTM, IERC, INFI, I, J, K, M, L, IK
      REAL ALNORM
      EXTERNAL ALNORM
      DATA ZERO, ONE, TWO, THREE, FOUR, SIX, EIGHT, TWELVE, SIXTN,
     *  SMALL /0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 0.1E-12/
      DATA SQRT2, SQRTPI /1.41421356237310, 1.77245385090552/
      DERIV = ZERO
      NTM = NTM + 1
      RSLT1 = ONE
      RSLT2 = ONE
      BI = ONE
      HI = A(1) + ONE
      HLI = B(1) + ONE
      INFI = -1
      DO 60 I = 1, N
         IF (BPD(I) .EQ. BI .AND. A(I) .EQ. HI .AND. B(I) .EQ. HLI .AND.
     *      INF(I) .EQ. INFI) THEN
            FOU(I) = FOU(I - 1)
            GOU(I) = GOU(I - 1)
            DO 10 IK = 1, 4
               FOU1(IK, I) = FOU1(IK, I - 1)
               GOU1(IK, I) = GOU1(IK, I - 1)
   10       CONTINUE
      ELSE
         BI = BPD(I)
         HI = A(I)
         HLI = B(I)
         INFI = INF(I)
         IF (BI .EQ. ZERO) THEN
            IF (INFI .LT. 1) THEN
               FOU(I) = ONE - ALNORM(HLI, .FALSE.)
            ELSE IF (INFI .EQ. 1) THEN
               FOU(I) = ALNORM(HI, .FALSE.)
            ELSE
               FOU(I) = ALNORM(HI, .FALSE.) - ALNORM(HLI, .FALSE.)
            END IF
            GOU(I) = FOU(I)
            DO 20 IK = 1, 4
               FOU1(IK, I) = ZERO
               GOU1(IK, I) = ZERO
   20       CONTINUE
         ELSE
            DEN = SQRT(ONE - BI * BI)
            BP = BI * SQRT2 / DEN
            IF (INFI .LT. 1) THEN
               U = -HLI / DEN + Z * BP
               FOU(I) = ALNORM(U, .FALSE.)
               CALL ASSIGN (U, BP, FOU1(1, I))
               BP = -BP
               U = -HLI / DEN + Z * BP
               GOU(I) = ALNORM(U, .FALSE.)
               CALL ASSIGN (U, BP, GOU1(1, I))
            ELSE IF (INFI .EQ. 1) THEN
               U = HI / DEN + Z * BP
               GOU(I) = ALNORM(U, .FALSE.)
               CALL ASSIGN (U, BP, GOU1(1, I))
               BP = -BP
               U = HI / DEN + Z * BP
               FOU(I) = ALNORM(U, .FALSE.)
               CALL ASSIGN (U, BP, FOU1(1, I))
            ELSE
               U2 = -HLI / DEN + Z * BP
               CALL ASSIGN (U2, BP, FOU1(1, I))
               BP = -BP
               U1 = HI / DEN + Z * BP
               CALL ASSIGN (U1, BP, TMP(1))
               FOU(I) = ALNORM(U1, .FALSE.) + ALNORM(U2, .FALSE.) - ONE
               DO 30 IK = 1, 4
                  FOU1(IK, I) = FOU1(IK, I) + TMP(IK)
   30          CONTINUE
               IF (-HLI .EQ. HI) THEN
                  GOU(I) = FOU(I)
                  DO 40 IK = 1, 4
                     GOU1(IK, I) = FOU1(IK, I)
   40             CONTINUE
               ELSE
                  U2 = -HLI / DEN + Z * BP
                  CALL ASSIGN (U2, BP, GOU1(1, I))
                  BP = -BP
                  U1 = HI / DEN + Z * BP
                  GOU(I) = ALNORM(U1, .FALSE.) + ALNORM(U2, .FALSE.)-ONE
                  CALL ASSIGN (U1, BP, TMP(1))
                  DO 50 IK = 1, 4
                     GOU1(IK, I) = GOU1(IK, I) + TMP(IK)
   50             CONTINUE
               END IF
            END IF
         END IF
      END IF
      RSLT1 = RSLT1 * FOU(I)
      RSLT2 = RSLT2 * GOU(I)
      IF (RSLT1 .LE. SMALL) RSLT1 = ZERO
      IF (RSLT2 .LE. SMALL) RSLT2 = ZERO
   60 CONTINUE
      FUNCN = RSLT1 + RSLT2 + ADDN
      RESULT = FUNCN * EXP(-Z * Z) / SQRTPI
C
C        IF 4TH DERIVATIVE IS NOT WANTED, STOP HERE.
C        OTHERWISE, PROCEED TO COMPUTE 4TH DERIVATIVE.
C
      IF (IERC .EQ. 0) RETURN
      DO 70 IK = 1, 4
         FF(IK) = ZERO
         GF(IK) = ZERO
   70 CONTINUE
      DO 100 I = 1, N
         FRM = ONE
         GRM = ONE
         DO 80 J = 1, N
            IF (J .EQ. 1) GO TO 80
            FRM = FRM * FOU(J)
            GRM = GRM * GOU(J)
            IF (FRM .LE. SMALL) FRM = ZERO
            IF (GRM .LE. SMALL) GRM = ZERO
   80    CONTINUE
         DO 90 IK = 1, 4
            FF(IK) = FF(IK) + FRM * FOU1(IK, I)
            GF(IK) = GF(IK) + GRM * GOU1(IK, I)
   90    CONTINUE
  100 CONTINUE
      IF (N .LE. 2) GO TO 230
      DO 130 I = 1, N
         DO 120 J = I + 1, N
            TERM(2) = FOU1(1, I) * FOU1(1, J)
            GERM(2) = GOU1(1, I) * GOU1(1, J)
            TERM(3) = FOU1(2, I) * FOU1(1, J)
            GERM(3) = GOU1(2, I) * GOU1(1, J)
            TERM(4) = FOU1(3, I) * FOU1(1, J)
            GERM(4) = GOU1(3, I) * GOU1(1, J)
            TERM(1) = FOU1(2, I) * FOU1(2, J)
            GERM(1) = GOU1(2, I) * GOU1(2, J)
            DO 110 K = 1, N
               IF (K .EQ. I .OR. K .EQ. J) GO TO 110
               CALL TOOSML (1, TERM, FOU(K))
               CALL TOOSML (1, GERM, GOU(K))
  110       CONTINUE
            FF(2) = FF(2) + TWO * TERM(2)
            FF(3) = FF(3) + TWO * TERM(3) * THREE
            FF(4) = FF(4) + TWO * (TERM(4) * FOUR + TERM(1) * THREE)
            GF(2) = GF(2) + TWO * GERM(2)
            GF(3) = GF(3) + TWO * GERM(3) * THREE
            GF(4) = GF(4) + TWO * (GERM(4) * FOUR + GERM(1) * THREE)
  120    CONTINUE
  130 CONTINUE
      DO 170 I = 1, N
         DO 160 J = I + 1, N
            DO 150 K = J + 1, N
               TERM(3) = FOU1(1, I) * FOU1(1, J) * FOU1(1, K)
               TERM(4) = FOU1(2, I) * FOU1(1, J) * FOU1(1, K)
               GERM(3) = GOU1(1, I) * GOU1(1, J) * GOU1(1, K)
               GERM(4) = GOU1(2, I) * GOU1(1, J) * GOU1(1, K)
               IF (N .GT. 3) THEN
                  DO 140 M = 1, N
                  IF (M .EQ. I .OR. M .EQ. J .OR. M .EQ. K) GO TO  140
                  CALL TOOSML (3, TERM, FOU(M))
                  CALL TOOSML (3, GERM, GOU(M))
  140             CONTINUE
            END IF
            FF(3) = FF(3) + SIX * TERM(3)
            FF(4) = FF(4) + SIX * TERM(4) * SIX
            GF(3) = GF(3) + SIX * GERM(3)
            GF(4) = GF(4) + SIX * GERM(4) * SIX
  150       CONTINUE
  160    CONTINUE
  170 CONTINUE
      IF (N .LE. 3) GO TO 230
      DO 220 I = 1, N
         DO 210 J = I + 1, N
            DO 200 K = J + 1, N
               DO 190 M = K + 1, N
      TERM(4) = FOU1(1, I) * FOU1(1, J) * FOU1(1, K) * FOU1(1, M)
      GERM(4) = GOU1(1, I) * GOU1(1, J) * GOU1(1, K) * GOU1(1, M)
               IF (N .GT. 4) THEN
                  DO 180 L = 1, N
         IF (L .EQ. I .OR. L .EQ. J .OR. L .EQ. K .OR. L .EQ. M)GOTO 180
                     CALL TOOSML (4, TERM, FOU(L))
                     CALL TOOSML (4, GERM, GOU(L))
  180             CONTINUE
               END IF
               FF(4) = FF(4) + FOUR * SIX * TERM(4)
               GF(4) = GF(4) + FOUR * SIX * GERM(4)
  190          CONTINUE
  200       CONTINUE
  210    CONTINUE
  220 CONTINUE
C
  230 CONTINUE
      PHI = EXP(-Z * Z) / SQRTPI
      PHI1 = -TWO * Z * PHI
      PHI2 = (FOUR * Z ** 2 - TWO) * PHI
      PHI3 = (-EIGHT * Z ** 3 + TWELVE * Z) * PHI
      PHI4 = (SIXTN * Z ** 2 * (Z ** 2 - THREE) + TWELVE) * PHI
      DERIV = PHI * (FF(4) + GF(4)) + FOUR * PHI1 * (FF(3) + GF(3))
     *  + SIX * PHI2 * (FF(2) + GF(2)) + FOUR * PHI3 * (FF(1) + GF(1))
     *  + PHI4 * FUNCN
      RETURN
      END
      SUBROUTINE ASSIGN (U, BP, FF)
C
C        ALGORITHM AS 251.3  APPL.STATIST. (1989), VOL.38, NO.3
C
C
C        COMPUTE DERIVATIVES OF NORMAL CDF'S.
C
      REAL FF(4)
      REAL U, U2, BP, HALF, ONE, THREE, SQ2PI, T1, T2, T3
      INTEGER I
      DATA HALF, ONE, THREE, SQ2PI /0.5, 1.0, 3.0, 2.50662827463100/
      DATA ZERO, UMAX, SMALL /0.0,  8.0, 0.1E-07/
      IF (ABS(U) .GT. UMAX) THEN
         DO 10 I = 1, 4
            FF(I) = ZERO
   10    CONTINUE
      ELSE
         U2 = U * U
         T1 = BP * EXP(-HALF * U2) / SQ2PI
         T2 = BP * T1
         T3 = BP * T2
         FF(1) = T1
         FF(2) = -U * T2
         FF(3) = (U2 - ONE) * T3
         FF(4) = (THREE - U2) * U * BP * T3
         DO 20 I = 1, 4
            IF(ABS(FF(I)) .LT. SMALL) FF(I) = ZERO
   20    CONTINUE
      END IF
      RETURN
      END
      SUBROUTINE WMAX(W1, W2, W3, DLG)
C
C        ALGORITHM AS 251.4  APPL.STATIST. (1989), VOL.38, NO.3
C
C
C        LARGEST ABSOLUTE VALUE OF QUADRATIC FUNCTION FITTED
C        TO THREE POINTS.
C
      REAL W1, W2, W3, DLG, QUAD, QLIM, QMIN, ONE, TWO, B2C
      DATA ONE, TWO, QMIN /1.0, 2.0, 0.00001/
      DLG = MAX( ABS(W1), ABS(W3) )
      QUAD = W1 - W2 * TWO + W3
      QLIM = MAX( ABS(W1 - W3) / TWO , QMIN)
      IF (ABS(QUAD) .LE. QLIM) RETURN
      B2C = (W1 - W3) / QUAD / TWO
      IF (ABS(B2C) .GE. ONE) RETURN
      DLG = MAX( DLG, ABS(W2 - B2C * QUAD * B2C / TWO) )
      RETURN
      END
      SUBROUTINE TOOSML (N, FF, F)
C
C        ALGORITHM AS 251.5  APPL.STATIST. (1989), VOL.38, NO.3
C
C
C        MULTIPLY FF(I) BY F FOR I = N TO 4.  SET TO ZERO IF TOO SMALL.
C
      REAL FF(4), F, ZERO, SMALL
      INTEGER N, I
      DATA ZERO, SMALL /0.0, 0.1E-12/
      DO 10 I = N, 4
         FF(I) = FF(I) * F
         IF (ABS(FF(I)) .LE. SMALL) FF(I) = ZERO
   10 CONTINUE
      RETURN
      END
      REAL FUNCTION ALNORM(X, UPPER)
C
C        ALGORITHM AS 66  APPL. STATIST. (1973) VOL.22, P.424
C
C        EVALUATES THE TAIL AREA OF THE STANDARDIZED NORMAL CURVE
C        FROM X TO INFINITY IF UPPER IS .TRUE. OR
C        FROM MINUS INFINITY TO X IF UPPER IS .FALSE.
C
      REAL LTONE, UTZERO, ZERO, HALF, ONE, CON, A1, A2, A3,
     $  A4, A5, A6, A7, B1, B2, B3, B4, B5, B6, B7, B8, B9,
     $  B10, B11, B12, X, Y, Z, ZEXP
      LOGICAL UPPER, UP
C
C        LTONE AND UTZERO MUST BE SET TO SUIT THE PARTICULAR COMPUTER
C        (SEE INTRODUCTORY TEXT)
C
      DATA LTONE, UTZERO /7.0, 18.66/
      DATA ZERO, HALF, ONE, CON /0.0, 0.5, 1.0, 1.28/
      DATA           A1,             A2,            A3,
     $               A4,             A5,            A6,
     $               A7
     $  /0.398942280444, 0.399903438504, 5.75885480458,
     $    29.8213557808,  2.62433121679, 48.6959930692,
     $    5.92885724438/
      DATA           B1,            B2,             B3,
     $               B4,            B5,             B6,
     $               B7,            B8,             B9,
     $              B10,           B11,            B12
     $  /0.398942280385,     3.8052E-8,  1.00000615302,
     $    3.98064794E-4, 1.98615381364, 0.151679116635,
     $    5.29330324926,  4.8385912808,  15.1508972451,
     $   0.742380924027,  30.789933034,  3.99019417011/
C
      ZEXP(Z) = EXP(Z)
C
      UP = UPPER
      Z = X
      IF (Z .GE. ZERO) GOTO 10
      UP = .NOT. UP
      Z = -Z
   10 IF (Z .LE. LTONE .OR. UP .AND. Z .LE. UTZERO) GOTO 20
      ALNORM = ZERO
      GOTO 40
   20 Y = HALF * Z * Z
      IF (Z .GT. CON) GOTO 30
C
      ALNORM = HALF - Z * (A1 - A2 * Y / (Y + A3 - A4 / (Y + A5 +
     $  A6 / (Y + A7))))
      GOTO 40
C
   30 ALNORM = B1 * ZEXP(-Y) / (Z - B2 + B3 / (Z + B4 + B5 / (Z -
     $  B6 + B7 / (Z + B8 - B9 / (Z + B10 + B11 / (Z + B12))))))
C
   40 IF (.NOT. UP) ALNORM = ONE - ALNORM
      RETURN
      END
      REAL FUNCTION PPND7 (P, IFAULT)
C
C        ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3
C
C       PRODUCES THE NORMAL DEVIATE  Z  CORRESPONDING TO A GIVEN LOWER
C       TAIL AREA OF  P;  Z  IS ACCURATE TO ABOUT  1  PART IN 10**7.
C
C       THE HASH SUMS BELOW ARE THE SUMS OF THE MANTISSAS OF THE
C       COEFFICIENTS. THEY ARE INCLUDED FOR USE IN CHECKING
C       TRANSCRIPTION.
C
      INTEGER IFAULT
      REAL ZERO, ONE, HALF, SPLIT1, SPLIT2, CONST1, CONST2,
     *  A0, A1, A2, A3, B1, B2, B3, C0, C1, C2, C3, D1, D2,
     *  E0, E1, E2, E3, F1, F2, P, Q, R
      PARAMETER (ZERO = 0.0E0, ONE = 1.0E0, HALF = 0.5E0,
     *  SPLIT1 = 0.425E0,    SPLIT2 = 5.0E0,
     *  CONST1 = 0.180625E0, CONST2 = 1.6E0)
C
C     COEFFICIENTS FOR  P  CLOSE TO  1/2
      PARAMETER (A0 = 3.38713 27179E0,
     *           A1 = 5.04342 71938E1,
     *           A2 = 1.59291 13202E2,
     *           A3 = 5.91093 74720E1,
     *           B1 = 1.78951 69469E1,
     *           B2 = 7.87577 57664E1,
     *           B3 = 6.71875 63600E1)
C     HASH SUM AB    32.31845 77772
C
C     COEFFICIENTS FOR  P  NEITHER CLOSE TO  1/2  NOR  0 OR 1
      PARAMETER (C0 = 1.42343 72777E0,
     *           C1 = 2.75681 53900E0,
     *           C2 = 1.30672 84816E0,
     *           C3 = 1.70238 21103E-1,
     *           D1 = 7.37001 64250E-1,
     *           D2 = 1.20211 32975E-1)
C     HASH SUM CD    15.76149 29821
C
C     COEFFICIENTS FOR  P  NEAR  0 OR 1
      PARAMETER (E0 = 6.65790 51150E0,
     *           E1 = 3.08122 63860E0,
     *           E2 = 4.28682 94337E-1,
     *           E3 = 1.73372 03997E-2,
     *           F1 = 2.41978 94225E-1,
     *           F2 = 1.22582 02635E-2)
C     HASH SUM EF    19.40529 10204
C
      IFAULT = 0
      Q = P - HALF
      IF (ABS(Q) .LE. SPLIT1) THEN
         R = CONST1 - Q * Q
         PPND7 = Q * (((A3 * R + A2) * R + A1) * R + A0) /
     *               (((B3 * R + B2) * R + B1) * R + ONE)
         RETURN
      ELSE
         IF (Q .LT. 0) THEN
            R = P
         ELSE
            R = ONE - P
         ENDIF
         IF (R .LE. ZERO) THEN
            IFAULT = 1
            PPND7 = ZERO
            RETURN
         ENDIF
         R = SQRT(-LOG(R))
         IF (R .LE. SPLIT2) THEN
            R = R - CONST2
            PPND7 = (((C3 * R + C2) * R + C1) * R + C0) /
     *               ((D2 * R + D1) * R + ONE)
         ELSE
            R = R - SPLIT2
            PPND7 = (((E3 * R + E2) * R + E1) * R + E0) /
     *               ((F2 * R + F1) * R + ONE)
         ENDIF
         IF (Q .LT. 0) PPND7 = -PPND7
         RETURN
      ENDIF
      END
   
      SUBROUTINE SIMPSN (NDF,A,B,BPD,ERRB,N,INF,D,IERC,HNC,PROB,
     *   BND,IFLT)
C
C        STUDENTIZES A MULTIVARIATE INTEGRAL USING SIMPSON'S RULE.
C
      DIMENSION A(*),B(*),BPD(*),INF(*),D(*),
     *   FV(5),F1T(30),F2T(30),F3T(30),
     *   LDIR(30),PSUM(30),ESTT(30),ERRR(30),GV(5),G1T(30),G2T(30),
     *   G3T(30),GSUM(30)
      DATA ZERO,HALF,ONE,ONEP5,TWO,FOUR,SIX,DXMIN /0.0,0.5,1.0,1.5,
     *   2.0,4.0,6.0,0.000004/
      PROB    =  ZERO
      BOUNDA  =  ZERO
      BOUNDG  =  ZERO
      IFLAG   =  0
      IER     =  0
      START   =  -ONE
      DAX     =   ONE
      ERB2    =   ERRB * HALF
      EPS1    =   ERB2 * HALF
      CALL FUN (ZERO,NDF,A,B,BPD,ERB2,N,INF,D,F0,G0,IERC,HNC,IER)
   10 FV(1)   =  ZERO
      GV(1)   =  ZERO
      ERROR   =  ZERO
      DA      =  DAX
      LVL     =  1
      Z3      =  START + HALF*DA
      CALL FUN(Z3,NDF,A,B,BPD,ERB2,N,INF,D,FV(3),GV(3),IERC,HNC,IER)
      FV(5)   =  F0
      GV(5)   =  G0
      WT      =  ABS(DA) / SIX
      CONTRB  =  WT * (FV(1) + FOUR * FV(3) + FV(5))
      CONTRG  =  WT * (GV(1) + FOUR * GV(3) + GV(5))
      LDIR(LVL)  = 2
      PSUM(LVL)  = ZERO
      GSUM(LVL)  = ZERO
C
C        BISECT INTERVAL; COMPUTE ESTIMATES FOR EACH HALF.
C
   20 DX      =  HALF * DA
      WT      =  ABS(DX) / SIX
      Z2      =  START + HALF * DX
      CALL FUN(Z2,NDF,A,B,BPD,ERB2,N,INF,D,FV(2),GV(2),IERC,HNC,IER)
      Z4      =  START + ONEP5 * DX
      CALL FUN(Z4,NDF,A,B,BPD,ERB2,N,INF,D,FV(4),GV(4),IERC,HNC,IER)
      ESTL    =  WT * (FV(1) + FOUR * FV(2) + FV(3))
      ESTR    =  WT * (FV(3) + FOUR * FV(4) + FV(5))
      ESTGL   =  WT * (GV(1) + FOUR * GV(2) + GV(3))
      ESTGR   =  WT * (GV(3) + FOUR * GV(4) + GV(5))
      SUM     =  ESTL  +  ESTR
      SUMG    =  ESTGL +  ESTGR
      DLG     =  ABS(CONTRB - SUM)
      ERRL    =  DLG
C
C        STOP BISECTING WHEN ACCURACY SUFFICIENT, OR IF
C        INTERVAL TOO NARROW OR BISECTED TOO OFTEN.
C
   30 IF (DLG .LE. EPS1) GO TO 50
      IF (ABS(DX) .LE. DXMIN .OR. LVL .GE. 30) GO TO 40
C
C        RAISE LEVEL.  STORE INFORMATION FOR RIGHT HALF
C        AND APPLY SIMPSON'S RULE TO LEFT HALF.
C
      LVL     =  LVL + 1
      LDIR(LVL)  =  1
      F1T(LVL)   =  FV(3)
      F2T(LVL)   =  FV(4)
      F3T(LVL)   =  FV(5)
      G1T(LVL)   =  GV(3)
      G2T(LVL)   =  GV(4)
      G3T(LVL)   =  GV(5)
      DA      =  DX
      FV(5)   =  FV(3)
      FV(3)   =  FV(2)
      GV(5)   =  GV(3)
      GV(3)   =  GV(2)
      ESTT(LVL)  =  ESTR
      CONTRB  =  ESTL
      CONTRG  =  ESTGL
      EPS1    =  EPS1 * HALF
      ERRR(LVL)  =  EPS1
      GO TO 20
C
C        ACCEPT APPROXIMATE VALUE FOR INTERVAL.
C
   40 IFLAG   =  11
   50 ERROR   =  ERROR + ERRL
   60 IF (LDIR(LVL) .EQ. 1) GO TO 70
      SUM     =  SUM + PSUM(LVL)
      SUMG    =  SUMG + GSUM(LVL)
      LVL     =  LVL - 1
      IF (LVL .GT. 0) GO TO 60
      CONTRB  =  SUM
      CONTRG  =  SUMG
      LVL     =  1
      DLG     =  ERROR
      GO TO 80
C
C        RESTORE SAVED INFORMATION TO PROCESS RIGHT HALF.
C
   70 PSUM(LVL)  =  SUM
      GSUM(LVL)  =  SUMG
      LDIR(LVL)  =  2
      DA      =  DAX / TWO**(LVL-1)
      START      =  START + DX * TWO
      FV(1)      =  F1T(LVL)
      FV(3)      =  F2T(LVL)
      FV(5)      =  F3T(LVL)
      GV(1)      =  G1T(LVL)
      GV(3)      =  G2T(LVL)
      GV(5)      =  G3T(LVL)
      CONTRB     =  ESTT(LVL)
      EXCESS     =  EPS1 - DLG
      EPS1       =  ERRR(LVL)
      IF (EXCESS .GT. ZERO) EPS1 = EPS1 + EXCESS
      GO TO 20
   80 PROB       =  PROB + CONTRB
      BOUNDG     =  BOUNDG + CONTRG
      BOUNDA     =  BOUNDA + DLG
      IF (Z4 .LE. ZERO) GO TO 90
      IF (IFLT .EQ. 0) IFLT = IER
      IF (IFLT .EQ. 0) IFLT = IFLAG
      BOUNDA     =  BOUNDA + BOUNDG
      IF (BND .LT. BOUNDA) BND = BOUNDA
      RETURN
   90 EPS1       =  ERB2 * HALF
      EXCESS     =  EPS1 - BND
      IF (EXCESS .GT. ZERO) EPS1 = EPS1 + EXCESS
      START      =  ONE
      DAX        = -ONE
      GO TO 10
      END
      FUNCTION SDIST(Y,N)
C
C        COMPUTE Y**(N/2 - 1) EXP(-Y) / GAMMA(N/2)
C
C                (Revised: 1994-01-19)
C
      DATA ZERO, HALF, ONE, X23 / 0.0, 0.5, 1.0, -23.0 /
      DATA SQRTPI / 1.77245385090552 /
      SDIST      =  ZERO
      IF (Y .LE. ZERO) RETURN
      JJ         =  N/2 - 1
      JK         =  2 * JJ - N + 2
      JKP        =  JJ - JK
      SDIST      =  ONE
      IF (JK .LT. 0) SDIST = SDIST / SQRT(Y) / SQRTPI
      IF (JKP .EQ. 0) GO TO 20
      XN         =  FLOAT(N) * HALF
      TEST       =  ALOG(Y) - Y / FLOAT(JKP)
      IF ( TEST .LT. X23 ) THEN
         SDIST = ZERO
         RETURN
      ENDIF
      SDIST = ALOG ( SDIST )
      DO 10 J = 1, JKP
         XN    =   XN - ONE
         SDIST   =  SDIST + TEST - ALOG(XN)
   10 CONTINUE
      IF ( SDIST .LT. X23 ) THEN
          SDIST  =  ZERO
      ELSE
          SDIST  =  EXP( SDIST )
      ENDIF
      RETURN
   20 SDIST      =  SDIST * EXP(-Y)
      RETURN
      END
      SUBROUTINE FUN (Z,NDF,H,HL,BPD,ERB2,N,INF,D,F0,G0,IERC
     *   ,HNC,IER)
      INTEGER NN
      PARAMETER (NN=50)
      DIMENSION A(NN),B(NN),H(*),HL(*),BPD(*),INF(*),D(*)
      DATA  ZERO, ONE, TWO, SMALL / 0.0, 1.0, 2.0, 1.0E-08 /
      F0    =  ZERO
      G0    =  ZERO
      IF (Z .LE. -ONE .OR. Z .GE. ONE) RETURN
      DF    =  FLOAT(NDF)
      ARG   =  (ONE + Z) / (ONE - Z)
      TERM = ARG * DF * TWO / (ONE-Z)**2 * SDIST(DF/TWO*ARG*ARG,NDF)
      IF (TERM .LE. SMALL) RETURN
      DO 10 I = 1, N
         A(I) = ARG * H(I)   - D(I)
         B(I) = ARG * HL(I)  - D(I)
   10 CONTINUE
      CALL MVNPRD (A,B,BPD,ERB2,N,INF,IERC,HNC,PROB,BND,IFLT)
      IF (IER .EQ. 0) IER = IFLT
      G0  =  TERM * BND
      F0  =  TERM * PROB
      RETURN
      END

C * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
C Charles Dunnett
C Dept. of Mathematics and Statistics
C McMaster University
C Hamilton, Ontario L8S 4K1
C Canada
C E-mail: dunnett@mcmaster.ca
C Tel.: (905) 525-9140 (Ext. 27104)
C * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      END MODULE DUNNETMOD
