C f2py -m  -h deamod.pyf dea.f
C f2py integrationmod.pyf integration1Dmodule.f90 .f90 -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
! f2py --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 -m integrationmod  -c integration1Dmodule.f90 

      DOUBLE PRECISION FUNCTION D1MACH(I)
      implicit none
C 
C  Double-precision machine constants.
C
C  D1MACH( 1) = B**(EMIN-1), the smallest positive magnitude.
C  D1MACH( 2) = B**EMAX*(1 - B**(-T)), the largest magnitude.
C  D1MACH( 3) = B**(-T), the smallest relative spacing.
C  D1MACH( 4) = B**(1-T), the largest relative spacing.
C  D1MACH( 5) = LOG10(B)
C
C  Two more added much later:
C
C  D1MACH( 6) = Infinity.
C  D1MACH( 7) = Not-a-Number.
C
C  Reference:  Fox P.A., Hall A.D., Schryer N.L.,"Framework for a
C              Portable Library", ACM Transactions on Mathematical
C              Software, Vol. 4, no. 2, June 1978, PP. 177-188.
C     
      INTEGER , INTENT(IN) :: I
      DOUBLE PRECISION, SAVE :: DMACH(7)
      DOUBLE PRECISION :: B, EPS
      DOUBLE PRECISION :: ONE  = 1.0D0
      DOUBLE PRECISION :: ZERO = 0.0D0
      INTEGER :: EMAX,EMIN,T
      DATA DMACH /7*0.0D0/
!     First time through, get values from F90 INTRINSICS:
      IF (DMACH(1) .EQ. 0.0D0) THEN
         T        = DIGITS(ONE)
         B        = DBLE(RADIX(ONE))    ! base number
         EPS      = SPACING(ONE) 
         EMIN     = MINEXPONENT(ONE)
         EMAX     = MAXEXPONENT(ONE)
         DMACH(1) = B**(EMIN-1)               !TINY(ONE)
         DMACH(2) = (B**(EMAX-1)) * (B-B*EPS) !HUGE(ONE)
         DMACH(3) = EPS/B   ! EPS/B 
         DMACH(4) = EPS
         DMACH(5) = LOG10(B)  
         DMACH(6) = B**(EMAX+5)  !infinity
         DMACH(7) = ZERO/ZERO  !nan
      ENDIF
C
      D1MACH = DMACH(I)
      RETURN
      END FUNCTION D1MACH
      SUBROUTINE DEA(NEWFLG,SVALUE,LIMEXP,RESULT,ABSERR,EPSTAB,IERR)
C***BEGIN PROLOGUE  DEA
C***DATE WRITTEN   800101  (YYMMDD)
C***REVISION DATE  871208  (YYMMDD)
C***CATEGORY NO.  E5
C***KEYWORDS  CONVERGENCE ACCELERATION,EPSILON ALGORITHM,EXTRAPOLATION
C***AUTHOR  PIESSENS, ROBERT, APPLIED MATH. AND PROGR. DIV. -
C             K. U. LEUVEN
C           DE DONCKER-KAPENGA, ELISE,WESTERN MICHIGAN UNIVERSITY
C           KAHANER, DAVID K., NATIONAL BUREAU OF STANDARDS
C           STARKENBURG, C. B., NATIONAL BUREAU OF STANDARDS
C***PURPOSE  Given a slowly convergent sequence, this routine attempts
C            to extrapolate nonlinearly to a better estimate of the
C            sequence's limiting value, thus improving the rate of
C            convergence. Routine is based on the epsilon algorithm
C            of P. Wynn. An estimate of the absolute error is also
C            given.
C***DESCRIPTION
C
C              Epsilon algorithm. Standard fortran subroutine.
C              Double precision version.
C
C       A R G U M E N T S   I N   T H E   C A L L   S E Q U E N C E
C
C              NEWFLG - LOGICAL                       (INPUT and OUTPUT)
C                       On the first call to DEA set NEWFLG to .TRUE.
C                       (indicating a new sequence). DEA will set NEWFLG
C                       to .FALSE.
C
C              SVALUE - DOUBLE PRECISION                         (INPUT)
C                       On the first call to DEA set SVALUE to the first
C                       term in the sequence. On subsequent calls set
C                       SVALUE to the subsequent sequence value.
C
C              LIMEXP - INTEGER                                  (INPUT)
C                       An integer equal to or greater than the total
C                       number of sequence terms to be evaluated. Do not
C                       change the value of LIMEXP until a new sequence
C                       is evaluated (NEWFLG=.TRUE.).  LIMEXP .GE. 3
C
C              RESULT - DOUBLE PRECISION                        (OUTPUT)
C                       Best approximation to the sequence's limit.
C
C              ABSERR - DOUBLE PRECISION                        (OUTPUT)
C                       Estimate of the absolute error.
C
C              EPSTAB - DOUBLE PRECISION                        (OUTPUT)
C                       Workvector of DIMENSION at least (LIMEXP+7).
C
C              IERR   - INTEGER                                 (OUTPUT)
C                       IERR=0 Normal termination of the routine.
C                       IERR=1 The input is invalid because LIMEXP.LT.3.
C
C    T Y P I C A L   P R O B L E M   S E T U P
C
C   This sample problem uses the trapezoidal rule to evaluate the
C   integral of the sin function from 0.0 to 0.5*PI (value = 1.0). The
C   program implements the trapezoidal rule 8 times creating an
C   increasingly accurate sequence of approximations to the integral.
C   Each time the trapezoidal rule is used, it uses twice as many
C   panels as the time before. DEA is called to obtain even more
C   accurate estimates.
C
C      PROGRAM SAMPLE
C      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C      DOUBLE PRECISION EPSTAB(57)
CC                                     [57 = LIMEXP + 7]
C      LOGICAL NEWFLG
C      EXTERNAL F
C      DATA LIMEXP/50/
C      WRITE(*,*) ' NO. PANELS          TRAP. APPROX'
C     *           ,'        APPROX W/EA         ABSERR'
C      WRITE(*,*)
C      HALFPI = DASIN(1.0D+00)
CC                                     [UPPER INTEGRATION LIMIT = PI/2]
C      NEWFLG = .TRUE.
CC                                     [SET FLAG - 1ST DEA CALL]
C      DO 10 I = 0,7
C         NPARTS = 2 ** I
C         WIDTH = HALFPI/NPARTS
C         APPROX = 0.5D+00 * WIDTH * (F(0.0D+00) + F(HALFPI))
C         DO 11 J = 1,NPARTS-1
C            APPROX = APPROX + F(J * WIDTH) * WIDTH
C  11     CONTINUE
CC                                     [END TRAPEZOIDAL RULE APPROX]
C         SVALUE = APPROX
CC                                     [SVALUE = NEW SEQUENCE VALUE]
C         CALL DEA(NEWFLG,SVALUE,LIMEXP,RESULT,ABSERR,EPSTAB,IERR)
CC                                     [CALL DEA FOR BETTER ESTIMATE]
C         WRITE(*,12) NPARTS,APPROX,RESULT,ABSERR
C  12     FORMAT('   ',I4,T20,F16.13,T40,F16.13,T60,D11.4)
C  10  CONTINUE
C      STOP
C      END
C
C      DOUBLE PRECISION FUNCTION F(X)
C      DOUBLE PRECISION X
C      F = DSIN(X)
CC                                     [INTEGRAND]
C      RETURN
C      END
C
C   Output from the above program will be:
C
C  NO. PANELS          TRAP. APPROX        APPROX W/EA         ABSERR
C
C      1              .7853981633974      .7853981633974      .7854D+00
C      2              .9480594489685      .9480594489685      .9760D+00
C      4              .9871158009728      .9994567212570      .2141D+00
C      8              .9967851718862      .9999667417647      .3060D-02
C     16              .9991966804851      .9999998781041      .6094D-03
C     32              .9997991943200      .9999999981026      .5767D-03
C     64              .9999498000921      .9999999999982      .3338D-04
C    128              .9999874501175     1.0000000000000      .1238D-06
C
C-----------------------------------------------------------------------
C***REFERENCES  "Acceleration de la convergence en analyse numerique",
C                 C. Brezinski, "Lecture Notes in Math.", vol. 584,
C                 Springer-Verlag, New York, 1977.
C***ROUTINES CALLED  D1MACH,XERROR
C***END PROLOGUE  DEA
      double precision, dimension(LIMEXP+7), intent(inout) :: EPSTAB
      double precision, intent(out) :: RESULT !, ABSERR
      double precision, intent(inout) :: ABSERR
      double precision, intent(in) :: SVALUE
      INTEGER, INTENT(IN) :: LIMEXP
      INTEGER, INTENT(OUT) :: IERR
      LOGICAL, intent(INOUT) :: NEWFLG
      DOUBLE PRECISION :: DELTA1,DELTA2,DELTA3,DRELPR,DEPRN,
     1   ERROR,ERR1,ERR2,ERR3,E0,E1,E2,E3,RES,
     2   SS,TOL1,TOL2,TOL3
      double precision, dimension(3) :: RES3LA
      INTEGER I,IB,IB2,IE,IN,K1,K2,K3,N,NEWELM,NUM,NRES
C
C
C           LIMEXP  is the maximum number of elements the
C           epsilon table data can contain. The epsilon table
C           is stored in the first (LIMEXP+2) entries of EPSTAB.
C
C
C           LIST OF MAJOR VARIABLES
C           -----------------------
C           E0,E1,E2,E3 - DOUBLE PRECISION
C                         The 4 elements on which the computation of
C                         a new element in the epsilon table is based.
C           NRES   - INTEGER
C                    Number of extrapolation results actually
C                    generated by the epsilon algorithm in prior
C                    calls to the routine.
C           NEWELM - INTEGER
C                    Number of elements to be computed in the
C                    new diagonal of the epsilon table. The
C                    condensed epsilon table is computed. Only
C                    those elements needed for the computation of
C                    the next diagonal are preserved.
C           RES    - DOUBLE PRECISION
C                    New element in the new diagonal of the
C                    epsilon table.
C           ERROR  - DOUBLE PRECISION
C                    An estimate of the absolute error of RES.
C                    Routine decides whether RESULT=RES or
C                    RESULT=SVALUE by comparing ERROR with
C                    ABSERR from the previous call.
C           RES3LA - DOUBLE PRECISION
C                    Vector of DIMENSION 3 containing at most
C                    the last 3 results.
C
C
C            MACHINE DEPENDENT CONSTANTS
C            ---------------------------
C            DRELPR  is the largest relative spacing.
C
C***FIRST EXECUTABLE STATEMENT  DEA
      IF(LIMEXP.LT.3) THEN
        IERR = 1
!        CALL XERROR('LIMEXP IS LESS THAN 3',21,1,1)
        GO TO 110
      ENDIF
      IERR = 0
      RES3LA(1)=EPSTAB(LIMEXP+5)
      RES3LA(2)=EPSTAB(LIMEXP+6)
      RES3LA(3)=EPSTAB(LIMEXP+7)
      RESULT=SVALUE
      IF(NEWFLG) THEN
        N=1
        NRES=0
        NEWFLG=.FALSE.
        EPSTAB(N)=SVALUE
        ABSERR=ABS(RESULT)
        GO TO 100
      ELSE
        N=INT(EPSTAB(LIMEXP+3))
        NRES=INT(EPSTAB(LIMEXP+4))
        IF(N.EQ.2) THEN
          EPSTAB(N)=SVALUE
          ABSERR=.6D+01*ABS(RESULT-EPSTAB(1))
          GO TO 100
        ENDIF
      ENDIF
      EPSTAB(N)=SVALUE
      DRELPR=D1MACH(4)
      DEPRN=1.0D+01*DRELPR
      EPSTAB(N+2)=EPSTAB(N)
      NEWELM=(N-1)/2
      NUM=N
      K1=N
      DO 40 I=1,NEWELM
        K2=K1-1
        K3=K1-2
        RES=EPSTAB(K1+2)
        E0=EPSTAB(K3)
        E1=EPSTAB(K2)
        E2=RES
        DELTA2=E2-E1
        ERR2=ABS(DELTA2)
        TOL2=MAX(ABS(E2),ABS(E1))*DRELPR
        DELTA3=E1-E0
        ERR3=ABS(DELTA3)
        TOL3=MAX(ABS(E1),ABS(E0))*DRELPR
        IF(ERR2.GT.TOL2.OR.ERR3.GT.TOL3) GO TO 10
C
C           IF E0, E1 AND E2 ARE EQUAL TO WITHIN MACHINE
C           ACCURACY, CONVERGENCE IS ASSUMED.
C           RESULT=E2
C           ABSERR=ABS(E1-E0)+ABS(E2-E1)
C
        RESULT=RES
        ABSERR=ERR2+ERR3
        GO TO 50
   10   IF(I.NE.1) THEN
          E3=EPSTAB(K1)
          EPSTAB(K1)=E1
          DELTA1=E1-E3
          ERR1=ABS(DELTA1)
          TOL1=MAX(ABS(E1),ABS(E3))*DRELPR
C
C           IF TWO ELEMENTS ARE VERY CLOSE TO EACH OTHER, OMIT
C           A PART OF THE TABLE BY ADJUSTING THE VALUE OF N
C
          IF(ERR1.LE.TOL1.OR.ERR2.LE.TOL2.OR.ERR3.LE.TOL3) GO TO 20
          SS=0.1D+01/DELTA1+0.1D+01/DELTA2-0.1D+01/DELTA3
        ELSE
          EPSTAB(K1)=E1
          IF(ERR2.LE.TOL2.OR.ERR3.LE.TOL3) GO TO 20
          SS=0.1D+01/DELTA2-0.1D+01/DELTA3
        ENDIF
C
C           TEST TO DETECT IRREGULAR BEHAVIOUR IN THE TABLE, AND
C           EVENTUALLY OMIT A PART OF THE TABLE ADJUSTING THE VALUE
C           OF N
C
        IF(ABS(SS*E1).GT.0.1D-03) GO TO 30
   20   N=I+I-1
        IF(NRES.EQ.0) THEN
          ABSERR=ERR2+ERR3
          RESULT=RES
        ELSE IF(NRES.EQ.1) THEN
          RESULT=RES3LA(1)
        ELSE IF(NRES.EQ.2) THEN
          RESULT=RES3LA(2)
        ELSE
          RESULT=RES3LA(3)
        ENDIF
        GO TO 50
C
C           COMPUTE A NEW ELEMENT AND EVENTUALLY ADJUST
C           THE VALUE OF RESULT
C
   30   RES=E1+0.1D+01/SS
        EPSTAB(K1)=RES
        K1=K1-2
        IF(NRES.EQ.0) THEN
          ABSERR=ERR2+ABS(RES-E2)+ERR3
          RESULT=RES
          GO TO 40
        ELSE IF(NRES.EQ.1) THEN
          ERROR=.6D+01*(ABS(RES-RES3LA(1)))
        ELSE IF(NRES.EQ.2) THEN
          ERROR=.2D+01*(ABS(RES-RES3LA(2))+ABS(RES-RES3LA(1)))
        ELSE
          ERROR=ABS(RES-RES3LA(3))+ABS(RES-RES3LA(2))
     1          +ABS(RES-RES3LA(1))
        ENDIF
        IF(ERROR.GT.1.0D+01*ABSERR) GO TO 40
        ABSERR=ERROR
        RESULT=RES
   40 CONTINUE
C
C           COMPUTE ERROR ESTIMATE
C
        IF(NRES.EQ.1) THEN
          ABSERR=.6D+01*(ABS(RESULT-RES3LA(1)))
        ELSE IF(NRES.EQ.2) THEN
          ABSERR=.2D+01*ABS(RESULT-RES3LA(2))+ABS(RESULT-RES3LA(1))
        ELSE IF(NRES.GT.2) THEN
          ABSERR=ABS(RESULT-RES3LA(3))+ABS(RESULT-RES3LA(2))
     1          +ABS(RESULT-RES3LA(1))
        ENDIF
C
C           SHIFT THE TABLE
C
   50 IF(N.EQ.LIMEXP) N=2*(LIMEXP/2)-1
      IB=1
      IF((NUM/2)*2.EQ.NUM) IB=2
      IE=NEWELM+1
      DO 60 I=1,IE
        IB2=IB+2
        EPSTAB(IB)=EPSTAB(IB2)
        IB=IB2
   60 CONTINUE
      IF(NUM.EQ.N) GO TO 80
      IN=NUM-N+1
      DO 70 I=1,N
        EPSTAB(I)=EPSTAB(IN)
        IN=IN+1
   70 CONTINUE
C
C           UPDATE RES3LA
C
   80 IF(NRES.EQ.0) THEN
        RES3LA(1)=RESULT
      ELSE IF(NRES.EQ.1) THEN
        RES3LA(2)=RESULT
      ELSE IF(NRES.EQ.2) THEN
        RES3LA(3)=RESULT
      ELSE
        RES3LA(1)=RES3LA(2)
        RES3LA(2)=RES3LA(3)
        RES3LA(3)=RESULT
      ENDIF
   90 ABSERR=MAX(ABSERR,DEPRN*ABS(RESULT))
      NRES=NRES+1
  100 N=N+1
      EPSTAB(LIMEXP+3)=DBLE(N)
      EPSTAB(LIMEXP+4)=DBLE(NRES)
      EPSTAB(LIMEXP+5)=RES3LA(1)
      EPSTAB(LIMEXP+6)=RES3LA(2)
      EPSTAB(LIMEXP+7)=RES3LA(3)
  110 RETURN
      END subroutine DEA