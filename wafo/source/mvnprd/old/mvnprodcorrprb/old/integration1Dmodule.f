C f2py -m integrationmod -h integrationmod.pyf integration1Dmodule.f
C f2py integrationmod.pyf integration1Dmodule.f -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
! f2py --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 -m integrationmod  -c integration1Dmodule.f

!      module Integration1DModule
!      implicit none
!      interface AdaptiveSimpson
!      module procedure AdaptiveSimpson2, AdaptiveSimpsonWithBreaks
!      end interface

!      interface AdaptiveSimpson1
!      module procedure AdaptiveSimpson1
!      end interface

!      interface AdaptiveTrapz
!      module procedure AdaptiveTrapz1, AdaptiveTrapzWithBreaks
!      end interface

!     interface Romberg
!      module procedure Romberg1, RombergWithBreaks
!      end interface

!     INTERFACE DEA
!     MODULE PROCEDURE DEA
!     END INTERFACE
!     INTERFACE d1mach
!     MODULE PROCEDURE d1mach
!     END INTERFACE
!     contains
 
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
      subroutine dea3(E0,E1,E2,abserr,result1)
!***PURPOSE  Given a slowly convergent sequence, this routine attempts
!            to extrapolate nonlinearly to a better estimate of the
!            sequence's limiting value, thus improving the rate of
!            convergence. Routine is based on the epsilon algorithm
!            of P. Wynn. An estimate of the absolute error is also
!            given. 
      double precision, intent(in) :: E0,E1,E2
      double precision, intent(out) :: abserr, result1
      !locals
      double precision, parameter :: ten = 10.0d0
      double precision, parameter :: one = 1.0d0
      double precision :: small, delta2, delta1
      double precision :: tol2, tol1, err2, err1,ss
      small  = spacing(one)
      delta2 = E2 - E1
      delta1 = E1 - E0
      err2   = abs(delta2)
      err1   = abs(delta1)
      tol2   = max(abs(E2),abs(E1)) * small
      tol1   = max(abs(E1),abs(E0)) * small
      if ( ( err1 <= tol1 ) .or. err2 <= tol2) then
C           IF E0, E1 AND E2 ARE EQUAL TO WITHIN MACHINE
C           ACCURACY, CONVERGENCE IS ASSUMED.
         result1 = E2
         abserr = err1 + err2 + E2*small*ten
      else
         ss = one/delta2 - one/delta1
         if (abs(ss*E1) <= 1.0d-3) then
            result1 = E2
            abserr = err1 + err2 + E2*small*ten
         else
            result1 = E1 + one/ss
            abserr = err1 + err2 + abs(result1-E2)
         endif
      endif
      end subroutine dea3
      SUBROUTINE DEA(NEWFLG,SVALUE,LIMEXP,result1,ABSERR,EPSTAB,IERR)
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
C              result1 - DOUBLE PRECISION                        (OUTPUT)
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
C         CALL DEA(NEWFLG,SVALUE,LIMEXP,result1,ABSERR,EPSTAB,IERR)
CC                                     [CALL DEA FOR BETTER ESTIMATE]
C         WRITE(*,12) NPARTS,APPROX,result1,ABSERR
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
      double precision, dimension(*), intent(inout) :: EPSTAB
      double precision, intent(out) :: result1 
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
      result1=SVALUE
      IF(NEWFLG) THEN
        N=1
        NRES=0
        NEWFLG=.FALSE.
        EPSTAB(N)=SVALUE
        ABSERR=ABS(result1)
        GO TO 100
      ELSE
        N=INT(EPSTAB(LIMEXP+3))
        NRES=INT(EPSTAB(LIMEXP+4))
        IF(N.EQ.2) THEN
          EPSTAB(N)=SVALUE
          ABSERR=.6D+01*ABS(result1-EPSTAB(1))
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
C           result1=E2
C           ABSERR=ABS(E1-E0)+ABS(E2-E1)
C
        result1=RES
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
          result1=RES
        ELSE IF(NRES.EQ.1) THEN
          result1=RES3LA(1)
        ELSE IF(NRES.EQ.2) THEN
          result1=RES3LA(2)
        ELSE
          result1=RES3LA(3)
        ENDIF
        GO TO 50
C
C           COMPUTE A NEW ELEMENT AND EVENTUALLY ADJUST
C           THE VALUE OF result1
C
   30   RES=E1+0.1D+01/SS
        EPSTAB(K1)=RES
        K1=K1-2
        IF(NRES.EQ.0) THEN
          ABSERR=ERR2+ABS(RES-E2)+ERR3
          result1=RES
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
        result1=RES
   40 CONTINUE
C
C           COMPUTE ERROR ESTIMATE
C
        IF(NRES.EQ.1) THEN
          ABSERR=.6D+01*(ABS(result1-RES3LA(1)))
        ELSE IF(NRES.EQ.2) THEN
          ABSERR=.2D+01*ABS(result1-RES3LA(2))+ABS(result1-RES3LA(1))
        ELSE IF(NRES.GT.2) THEN
          ABSERR=ABS(result1-RES3LA(3))+ABS(result1-RES3LA(2))
     1          +ABS(result1-RES3LA(1))
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
        RES3LA(1)=result1
      ELSE IF(NRES.EQ.1) THEN
        RES3LA(2)=result1
      ELSE IF(NRES.EQ.2) THEN
        RES3LA(3)=result1
      ELSE
        RES3LA(1)=RES3LA(2)
        RES3LA(2)=RES3LA(3)
        RES3LA(3)=result1
      ENDIF
   90 ABSERR=MAX(ABSERR,DEPRN*ABS(result1))
      NRES=NRES+1
  100 N=N+1
      EPSTAB(LIMEXP+3)=DBLE(N)
      EPSTAB(LIMEXP+4)=DBLE(NRES)
      EPSTAB(LIMEXP+5)=RES3LA(1)
      EPSTAB(LIMEXP+6)=RES3LA(2)
      EPSTAB(LIMEXP+7)=RES3LA(3)
  110 RETURN
      END subroutine DEA

      subroutine AdaptiveIntWithBreaks(f,a,b,N,brks,epsi,iflg
     $     ,abserr, val)
      !use AdaptiveGaussKronrod
      implicit none
      double precision :: f
      integer,          intent(in) :: N
      double precision, intent(in) :: a,b,epsi
      double precision, dimension(:), intent(in) :: brks
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      external f
! Locals
      double precision, dimension(N+2) :: pts
      double precision :: LTol,tol, error, valk, excess, errorEstimate
      double precision :: delta, deltaK
      integer :: kflg, k, limit,neval
      limit    = 30
      pts(1)   = a
      pts(N+2) = b
      delta    = b - a
      do k = 2,N+1
         pts(k) = minval(brks(k-1:N))  !add user supplied break points
      enddo
      LTol   = epsi / delta
      abserr = 0.0d0
      val    = 0.0D0
      iflg   = 0
      do k = 1, N + 1
         deltaK =  pts(k+1) - pts(k)
         tol = LTol * deltaK
         if (deltaK < 0.5D0) then
            call AdaptiveSimpson(f,pts(k),pts(k+1),tol, kflg,error,valk)
!            call romberg(f,pts(k),pts(k+1),20,tol,kflg,error, valk)
         else
            call AdaptiveSimpson3(f,pts(k),pts(k+1),tol,kflg,error,valk)
!            call  dqagp(f,pts(k),pts(k+1),0,pts,tol,0.0D0,limit,valk,
!     *           error,neval,kflg)
            
         endif	
         abserr = abserr + abs(error)
         
         errorEstimate = abserr + (b - pts(k+1)) * LTol
         excess = epsi - errorEstimate
         if (excess < 0.0D0 ) then
            LTol = 0.1D0*LTol
         elseif (  epsi < 2.0D0 * excess ) then
            LTol = (epsi + excess*0.5D0) / delta
         endif
         val = val + valk
         if (kflg>0) iflg = IOR(iflg, kflg)
      end do
      if (epsi<abserr)  iflg = IOR(iflg, 3)
      end subroutine AdaptiveIntWithBreaks
      subroutine AdaptiveSimpsonWithBreaks(f,a,b,N,brks,epsi,iflg
     $     ,abserr, val)
      implicit none
      double precision :: f
      integer,          intent(in) :: N
      double precision, intent(in) :: a,b,epsi
      double precision, dimension(:), intent(in) :: brks
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      external f
! Locals
      double precision, dimension(N+2) :: pts
      double precision :: error, valk, excess, errorEstimate
      double precision :: Lepsi,LTol, tol, delta, deltaK,small
      integer :: kflg, k
      pts(1)   = a
      pts(N+2) = b
      delta    = (b - a)
      do k = 2, N+1
         pts(k) = minval(brks(k-1:N))  !add user supplied break points
      enddo
      small = spacing(1.0D0)
      Lepsi = max(epsi,small)
      LTol   = Lepsi / delta
      abserr = 0.0d0
      val    = 0.0D0
      iflg   = 0
      do k = 1, N + 1
         deltaK = pts(k+1)-pts(k)
         tol = LTol * deltaK
         call AdaptiveSimpson(f,pts(k),pts(k+1),tol, kflg,error, valk)
         abserr = abserr + abs(error)
         deltaK  = (b-pts(k+1))
         errorEstimate = abserr + deltaK * LTol
         excess = Lepsi - errorEstimate
         if (excess < 0.0D0 ) then
            if (deltaK>0.0d0 .and. Lepsi > abserr) then
              LTol = (Lepsi - abserr) / deltaK
            else
              LTol = 0.1D0 * LTol
           endif
         elseif ( Lepsi < 5D0 * excess  ) then
            LTol = (Lepsi + excess) / delta 
         endif
         val = val + valk
         if (kflg>0) iflg = IOR(iflg, kflg)
      end do
      if (epsi<abserr)  iflg = IOR(iflg, 4)
      end subroutine AdaptiveSimpsonWithBreaks
      subroutine AdaptiveSimpson1(f,a,b,epsi, iflg,abserr, val)
      implicit none
c     Numerical Analysis:
c     The Mathematics of Scientific Computing
c     D.R. Kincaid &amp; E.W. Cheney
c     Brooks/Cole Publ., 1990
C
! Revised by Per A. Brodtkorb 4 June 2003
! Added check on stepsize, i.e., hMin and hMax
      double precision :: f
      double precision, intent(in) :: a,b,epsi
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      integer, parameter :: stackLimit = 30
      double precision, dimension(6,stackLimit) :: v
      external f
! Locals
      double precision, parameter :: small    = 1.0D-16
      double precision, parameter :: hmin     = 1.0D-9
      double precision, parameter :: zero     = 0.0D0
      double precision, parameter :: zpz66666 = 0.06666666666666666666D0 !1/15
      double precision, parameter :: onethird = 0.33333333333333333333D0
      double precision, parameter :: half     = 0.5D0
      double precision, parameter :: one      = 1.0D0
      double precision, parameter :: two      = 2.0D0
      double precision, parameter :: three    = 3.0D0
      double precision, parameter :: four     = 4.0D0
      double precision :: delta,h,c,y,z,localError, correction
      double precision :: abar, bbar, cbar,ybar,vbar,zbar
      double precision :: hmax, S, Star, SStar, SSStar,Sdiff
      integer :: k
      logical :: acceptError, lastInStack
      logical :: stepSizeTooSmall, stepSizeOK

      hmax = 0.24D0
c
c     initialize everything, 
c     particularly the first column vector in the stack.
c
      val    = zero
      abserr = zero
      iflg   = 0

      delta  = b - a
      
      h      = half * delta  
      c      = half * ( a + b )
      k      = 1
      abar   = f(a)
      cbar   = f(c)
      bbar   = f(b)
      
      S      = (abar + four * cbar + bbar) * h * onethird
      v(1,1) = a
      v(2,1) = h
      v(3,1) = abar
      v(4,1) = cbar
      v(5,1) = bbar
      v(6,1) = S
c
      do while ((1<=k) .and. (k <= stackLimit))
c     
c     take the last column off the stack and process it.
c
         h     = half * v(2,k)
         y     = v(1,k) + h
         z     = v(1,k) + three * h 
         ybar  = f(y)
         zbar  = f(z)
         Star  = ( v(3,k) + four * ybar + v(4,k) ) * h * onethird
         SStar = ( v(4,k) + four * zbar + v(5,k) ) * h * onethird
         SSStar     =  Star + SStar
         Sdiff      =  (SSStar - v(6,k))
         correction =  Sdiff * zpz66666 !=0.066666... = 1/15.0D0
         localError = abs(Sdiff) * two 
!     acceptError is made conservative in order to avoid premature termination

         acceptError      = (localError * delta  <= two* epsi * h 
     &                      .or. localError < small)     
         lastInStack      = ( stackLimit <= k)
         stepSizeOK       = ( h < hMax )
         stepSizeTooSmall = ( h < hMin)
         if (lastInStack .or. (stepSizeOK.and.acceptError) 
     &        .or. stepSizeTooSmall ) then
!     Stop subdividing interval when
!     1) accuracy is sufficient, or
!     2) interval too narrow, or
!     3) subdivided too often. (stack limit reached)
            
!     Add partial integral and take a new vector from the bottom of the stack.

            abserr = abserr + localError 
            val    = val + SSStar + correction
            k      = k - 1
c
            if (.not.acceptError) then
               if (lastInStack)      iflg = IOR(iflg,1) ! stack limit reached
               if (stepSizeTooSmall) iflg = IOR(iflg,2) ! stepSize limit reached
            endif	
            if (k <= 0) then 
               return
            endif
         else
c     Subdivide the interval and create two new vectors in the stack, 
c     one of which  overwrites the vector just processed.
            vbar   = v(5,k)
            v(2,k) = h
            v(5,k) = v(4,k)
            v(4,k) = ybar
            v(6,k) = Star
c
            k = k + 1
            v(1,k) = v(1,k-1) + two * h
            v(2,k) = h
            v(3,k) = v(5,k-1)
            v(4,k) = zbar
            v(5,k) = vbar
            v(6,k) = SStar
         endif
      enddo ! while
      end subroutine AdaptiveSimpson1
      subroutine AdaptiveSimpson2(f,a,b,epsi, iflg,abserr, val)
      implicit none
! by Per A. Brodtkorb 4 June 2003
! based on  psudo code in chapter 7, Kincaid and Cheney (1991).
! Added check on stepsize, i.e., hMin and hMax
! Added an alternitive check on termination: this is more robust  
! Reference:  
!     D.R. Kincaid &  E.W. Cheney (1991)
!     "Numerical Analysis"
!     Brooks/Cole Publ., 1991
!
!  C. Brezinski (1977)
!   "Acceleration de la convergence en analyse numerique",
!   , "Lecture Notes in Math.", vol. 584,
!    Springer-Verlag, New York, 1977.


      double precision :: f
      double precision, intent(in) :: a,b,epsi
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      integer, parameter :: stackLimit = 300
      double precision, dimension(10,stackLimit) :: v
      external f
! Locals
      double precision, parameter :: zero     = 0.0D0
      double precision, parameter :: zpz66666 = 0.06666666666666666666D0             !1/15
      double precision, parameter :: zpz588   = 0.05882352941176D0                   !1/17
      double precision, parameter :: onethird = 0.33333333333333333333D0
      double precision, parameter :: half     = 0.5D0
      double precision, parameter :: one      = 1.0D0
      double precision, parameter :: two      = 2.0D0
      double precision, parameter :: three    = 3.0D0
      double precision, parameter :: four     = 4.0D0
      double precision, parameter :: six      = 6.0D0
      double precision, parameter :: eight    = 8.0D0
      double precision, parameter :: ten      = 10.0D0
      double precision, dimension(4) :: x, fx, Sn
      double precision, dimension(5) :: d4fx
      double precision, dimension(55) :: EPSTAB
      double precision :: small 
      double precision :: delta, h, h8, localError, correction
      double precision :: Sn1, Sn2, Sn4, Sn1e, Sn2e, Sn4e
      double precision :: Sn12, Sn24, Sn124, Sn12e, Sn24e
      double precision :: hmax, hmin, dhmin, val0
      double precision :: Lepsi,Ltol, excess, deltaK, errorEstimate
      integer :: k, kp1, i, j,ix, numExtrap, IERR 
      integer, parameter :: LIMEXP = 5
      logical :: acceptError, lastInStack
      logical :: stepSizeTooSmall, stepSizeOK
      logical :: NEWFLG 
      small = spacing(one)
!      useDEA      = .TRUE.
      Lepsi = max(epsi,small/ten)
      if (Lepsi < 1.0D-7) then
         numExtrap = 1
      else
         numExtrap = 0
      endif
      numExtrap = 1
      hmax  = one
      hmin  = 1.0D-9
      dhmin = 1.0D-1
c
c     initialize everything, 
c     particularly the first column vector in the stack.
c
      val    = zero
      abserr = zero
      iflg   = 0
	
      delta  = b - a
      h      = half * delta  
      Ltol   = Lepsi / delta
      
      x(1) = a
      x(3) = half * ( a + b )
      x(2) = half * ( a + x(3) )
      x(4) = half * ( x(3) + b )

      k    = 1
      do I = 1,4
         v(I,1) = f(x(I))
      enddo
      v(5,1) = f(b)
      Sn(1) = ( v(1,1) + four * v(3,1) + v(5,1) ) * h * onethird
      h = h * half
      Sn(2) = ( v(1,1) + four * v(2,1) + v(3,1) ) * h * onethird
      Sn(3) = ( v(3,1) + four * v(4,1) + v(5,1) ) * h * onethird

      v(6   ,1) = x(1)
      v(7   ,1) = h
      v(8:10,1) = Sn(1:3);
	

      do while ((1<=k) .and. (k <= stackLimit))
!     
!     take the last column off the stack and process it.
!
         h   = half * v(7,k)  
         do i = 1,4
            x(i)  = v(6,k) + dble(2*i-1)*h
            fx(i) = f(x(i))
            Sn(i) = ( v(i,k) + four * fx(i) + v(i+1,k) ) * h * onethird           
         enddo
         
         stepSizeOK       = ( h < hMax )
         lastInStack      = ( stackLimit <= k)       
         if (lastInStack .OR. stepSizeOK) then
            Sn1 = v(8,k) 
            Sn2 = ( v(9,k) + v(10,k) )   
            Sn4   = Sn(1) + Sn(2) + Sn(3) + Sn(4)
            if (numExtrap>0) then
               Sn12 = (Sn1 - Sn2)
               Sn24 = (Sn2 - Sn4)
                                !     Extrapolate Sn1 and Sn2:
               Sn1e = Sn2 -  Sn12 * zpz66666
               Sn2e = Sn4 -  Sn24 * zpz66666
               Sn12e = ( Sn1e - Sn2e )
               
               Sn24e = (Sn2e - Sn4)
!               Sn1e =  Sn2e -  Sn12e * zpz66666
!               Sn12e = (Sn1e - Sn2e)

               Sn124 = (Sn12e - Sn24)
               if ((abs(Sn124)<= hmin) .or.
     &              .false..and.(Sn24*Sn12e < zero)) then
!     Correction based on the assumption of slowly varying fourth derivative
                  correction = -Sn24 * zpz588 !
               else
!     Correction based on assumption that the termination error 
!     is of the form: C*h^q
                  correction = -Sn24 * Sn24 / Sn124
               endif
               Sn4e = Sn4 + correction
               
!               NEWFLG = .TRUE.
!               CALL DEA(NEWFLG,Sn1,LIMEXP,val0,localError,EPSTAB,IERR)
!               CALL DEA(NEWFLG,Sn2,LIMEXP,val0,localError,EPSTAB,IERR)
!               CALL DEA(NEWFLG,Sn1e,LIMEXP,val0,localError,EPSTAB,IERR)
!               CALL DEA(NEWFLG,Sn4,LIMEXP,val0,localError,EPSTAB,IERR)
!               CALL DEA(NEWFLG,Sn4e,LIMEXP,val0,localError,EPSTAB,IERR)
!     localError is made conservative in order to avoid premature
!     termination  
              CALL DEA3(Sn1e,Sn2e,Sn4e,localError,val0)
              !if (h>dhMin) then
              !localError = max(localError,abs(correction))
              !else
                 !val0 = Sn4e
                 !localError = abs(correction)*two
              !endif
            else
               CALL DEA3(Sn1,Sn2,Sn4,localError,val0)
            endif
            acceptError  = ( localError <= Ltol * h * eight
     &           .or. localError < small)
         else
            acceptError = .FALSE.
         endif
         
         stepSizeTooSmall = ( h < hMin)
         if (lastInStack  .or.
     &        ( stepSizeOK .and. acceptError ) .or.
     &        stepSizeTooSmall) then
!     Stop subdividing interval when
!     1) accuracy is sufficient, or
!     2) interval too narrow, or
!     3) subdivided too often. (stack limit reached)
            
!     Add partial integral and take a new vector from the bottom of the stack.
            
            abserr = abserr + max(localError, ten*small*val0)
            val    = val + val0           
            k      = k - 1
            if (.not.acceptError) then
               if (lastInStack)      iflg = IOR(iflg,1) !stack limit reached
               if (stepSizeTooSmall) iflg = IOR(iflg,2) !stepSize limit reached  
            endif	
            if (k <= 0) then 
               exit ! while loop
            endif
            deltaK        = (v(6,k+1)-a)
            errorEstimate = abserr + deltaK * Ltol
            excess = Lepsi - errorEstimate
            if (excess < zero ) then
               if (deltaK > zero .and. Lepsi > abserr) then
                  LTol = (Lepsi - abserr) / deltaK
               else
                  LTol = 0.1D0 * LTol
               endif
            elseif (.true..or. Lepsi < four * excess  ) then
               LTol = (Lepsi + 0.9D0 * excess) / delta 
            endif
         else
!     Subdivide the interval and create two new vectors in the stack, 
!     one of which  overwrites the vector just processed.
!
!	 v(:,k)  = [fx1,fx2,fx3,fx4,fx5,x1,h,S,SL,SR]
	    kp1 = k + 1;
!	Process right interval
	    v(1,kp1)    = v(3,k); !fx1R 
	    v(2,kp1)    = fx(3);  !fx2R
	    v(3,kp1)    = v(4,k); !fx3R
	    v(4,kp1)    = fx(4);  !fx4R
	    v(5,kp1)    = v(5,k); !fx5R
	    v(6,kp1)    = v(6,k) + four * h; ! x1R
	    v(7,kp1)    = h;
	    v(8,kp1)    = v(10,k); ! S
	    v(9:10,kp1) = Sn(3:4); ! SL, SR
!	Process left interval		
	    v(5,k)    = v(3,k); ! fx5L
	    v(4,k)    = fx(2);  ! fx4L
	    v(3,k)    = v(2,k); ! fx3L
	    v(2,k)    = fx(1);  ! fx2L
!		 v(1,k)  unchanged     fx1L
!		 v(6,k)  unchanged      x1L
	    v(7,k)    = h;
	    v(8,k)    = v(9,k); ! S
	    v(9:10,k) = Sn(1:2); ! SL, SR
	    k = kp1;
         endif
      enddo ! while
      if (epsi<abserr) iflg = IOR(iflg,4) 
      end subroutine AdaptiveSimpson2

      subroutine AdaptiveSimpson3(f,a,b,epsi, iflg,abserr, val)
      implicit none
! by Per A. Brodtkorb 4 June 2003
! based on  psudo code in chapter 7, Kincaid and Cheney (1991).
! Added check on stepsize, i.e., hMin and hMax
! Added an alternitive check on termination: this is more robust  
! Reference:  
!     D.R. Kincaid &  E.W. Cheney (1991)
!     "Numerical Analysis"
!     Brooks/Cole Publ., 1991
!
!  C. Brezinski (1977)
!   "Acceleration de la convergence en analyse numerique",
!   , "Lecture Notes in Math.", vol. 584,
!    Springer-Verlag, New York, 1977.


      double precision :: f
      double precision, intent(in) :: a,b,epsi
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      integer, parameter :: stackLimit = 10
      double precision, dimension(18,stackLimit) :: v
      external f
! Locals
      double precision, parameter :: zero     = 0.0D0
      double precision, parameter :: zp125    = 0.125D0                          ! 1/8
      double precision, parameter :: zpz66666 = 0.06666666666666666666D0         !1/15
      double precision, parameter :: zpz588   = 0.05882352941176D0               !1/17
      double precision, parameter :: onethird = 0.33333333333333333333D0
      double precision, parameter :: half     = 0.5D0
      double precision, parameter :: one      = 1.0D0
      double precision, parameter :: two      = 2.0D0
      double precision, parameter :: three    = 3.0D0
      double precision, parameter :: four     = 4.0D0
      double precision, parameter :: six      = 6.0D0
      double precision, parameter :: eight    = 8.0D0
      double precision, parameter :: ten      = 10.0D0
      double precision, parameter :: sixteen  = 16.0D0 
      double precision, dimension(8) ::  fx, Sn
      double precision, dimension(55) :: EPSTAB
      double precision :: small, x, a0 
      double precision :: delta, h, h8, localError, correction
      double precision :: Sn1, Sn2, Sn4, Sn8, Sn1e, Sn2e, Sn4e
      double precision :: Sn12, Sn24,Sn48,Sn124, Sn12e, Sn24e
      double precision :: hmax, hmin, dhmin, val0
      double precision :: Lepsi,Ltol, excess, deltaK, errorEstimate
      integer :: k, kp1, i, j,ix, numExtrap, IERR, Nrule, step
      integer, parameter :: LIMEXP = 5
      logical :: acceptError, lastInStack
      logical :: stepSizeTooSmall, stepSizeOK
      logical :: NEWFLG 
      Nrule = 9
      small = spacing(one)
!      useDEA      = .TRUE.
      Lepsi = max(epsi,small/ten)
      if (Lepsi < 1.0D-5) then
         numExtrap = 1
      else
         numExtrap = 0
      endif
      numExtrap = 0
      hmax  = one
      hmin  = 1.0D-9
      dhmin = 1.0D-1
c
c     initialize everything, 
c     particularly the first column vector in the stack.
c
      localError = one
      val    = zero
      abserr = zero
      iflg   = 0
	
      delta  = b - a
      h      = delta * zp125  
      Ltol   = Lepsi / delta
      

      k    = 1
      do I = 1,Nrule-1
         x = a + dble(i-1) * h
         v(I,1) = f(x)
      enddo
      v(Nrule,   1) = f(b)
      v(Nrule+1, 1) = a
      v(Nrule+2, 1) = h
      step = 8
      ix = Nrule + 2
      do I = 1,3
         step = step / 2
         do j = 1,Nrule-2,2*step
            ix = ix + 1
            v(ix,1) = ( v(j,1) + four * v(j+step,1) + v(j + 2*step,1) )*
     &           h * onethird * dble(step)
         enddo
      enddo
      	
      do while ((1<=k) .and. (k <= stackLimit))
!     
!     take the last column off the stack and process it.
!
         h   = half * v(Nrule+2,k)  
         a0  = v(Nrule + 1,k)
         Sn8 = zero
         do i = 1, Nrule - 1
            x  = a0 + dble(2*i-1)*h
            fx(i) = f(x)
            Sn(i) = ( v(i,k) + four * fx(i) + v(i+1,k) ) * h * onethird           
            Sn8   = Sn8 + Sn(i)
         enddo
         ix = Nrule + 3
         
         Sn1 = v(ix,k) 
         Sn2 = v(ix+1,k) + v(ix+2,k) 
         Sn4 = v(ix+3,k) + v(ix+4,k) + v(ix+5,k) + v(ix+6,k)
 
         stepSizeOK       = ( h < hMax )
         lastInStack      = ( stackLimit <= k )       
         if (lastInStack .OR. stepSizeOK) then
            if (numExtrap>0) then
               Sn12 = (Sn1 - Sn2)
               Sn24 = (Sn2 - Sn4)
               Sn48 = (Sn4 - Sn8)
                                !     Extrapolate Sn1 and Sn2:
               Sn1e = Sn2 -  Sn12 * zpz66666
               Sn2e = Sn4 -  Sn24 * zpz66666
               Sn4e = Sn8 -  Sn48 * zpz588
               Sn12e = (Sn1e - Sn2e)
               Sn24e = (Sn2e - Sn4e)

               Sn124 = (Sn12e - Sn24e)
               if ((abs(Sn124)<= hmin) .or.
     &              (Sn12e*Sn24e < zero)) then
!     Correction based on the assumption of slowly varying fourth derivative
                  correction = -Sn48*zpz588 !
               else
!     Correction based on assumption that the termination error 
!     is of the form: C*h^q
                  correction = -Sn24e * Sn24e / Sn124
                  !Sn4e = Sn4e + correction
               endif
               CALL DEA3(Sn1e,Sn2e,Sn4e,localError,val0)
!     localError is made conservative in order to avoid premature
!     termination  
!               localError = max(localError,abs(correction)*three)
!               localError = abs(correction)*three
            else
               !CALL DEA3(Sn1,Sn2,Sn4,localError,val0)
               NEWFLG = .TRUE.
               CALL DEA(NEWFLG,Sn1,LIMEXP,val0,localError,EPSTAB,IERR)
               CALL DEA(NEWFLG,Sn2,LIMEXP,val0,localError,EPSTAB,IERR)
               CALL DEA(NEWFLG,Sn4,LIMEXP,val0,localError,EPSTAB,IERR)
               CALL DEA(NEWFLG,Sn8,LIMEXP,val0,localError,EPSTAB,IERR)
            endif
            acceptError  = ( localError <= Ltol * h * sixteen
     &           .or. localError < small)
         else
            acceptError = .FALSE.
         endif
         
         stepSizeTooSmall = ( h < hMin)
         if (lastInStack  .or.
     &        ( stepSizeOK .and. acceptError ) .or.
     &        stepSizeTooSmall) then
!     Stop subdividing interval when
!     1) accuracy is sufficient, or
!     2) interval too narrow, or
!     3) subdivided too often. (stack limit reached)
            
!     Add partial integral and take a new vector from the bottom of the stack.
            
            abserr = abserr + max(localError, ten*small*val0)
            val    = val + val0           
            k      = k - 1
            if (.not.acceptError) then
               if (lastInStack)      iflg = IOR(iflg,1) !stack limit reached
               if (stepSizeTooSmall) iflg = IOR(iflg,2) !stepSize limit reached  
            endif	
            if (k <= 0) then 
               exit ! while loop
            endif
            deltaK        = (v(Nrule+1,k+1)-a)
            errorEstimate = abserr + deltaK * Ltol
            excess = Lepsi - errorEstimate
            if (excess < zero ) then
               if (deltaK > zero .and. Lepsi > abserr) then
                  LTol = (Lepsi - abserr) / deltaK
               else
                  LTol = 0.1D0 * LTol
               endif
            elseif (.TRUE..or. Lepsi < four * excess  ) then
               LTol = (Lepsi + 0.9D0 * excess) / delta 
            endif
         else
!     Subdivide the interval and create two new vectors in the stack, 
!     one of which  overwrites the vector just processed.
!
!	 v(:,k)  = [fx1,fx2,..,fx8,fx9,x1,h,S,SL,SR,SL1,SL2 SR1,SR2]
	    kp1 = k + 1;
!	Process right interval
            
	    v(1,kp1)    = v(5,k); !fx1R 
	    v(2,kp1)    = fx(5);  !fx2R
	    v(3,kp1)    = v(6,k); !fx3R
	    v(4,kp1)    = fx(6);  !fx4R
	    v(5,kp1)    = v(7,k); !fx5R
            v(6,kp1)    = fx(7);  !fx6R
	    v(7,kp1)    = v(8,k); !fx7R
            v(8,kp1)    = fx(8);  !fx8R
	    v(9,kp1)    = v(9,k); !fx9R

	    v(Nrule+1,kp1)    = v(Nrule+1,k) + eight * h ! x1R
	    v(Nrule+2,kp1)    = h;
	    v(Nrule+3,kp1)    = v(Nrule+5,k); ! S
            v(Nrule+4,kp1)    = v(Nrule+8,k); ! SL
            v(Nrule+5,kp1)    = v(Nrule+9,k); ! SR
	    v(Nrule+6:Nrule+9,kp1) = Sn(5:8); ! SL1,SL2,SR1, SR2
!	Process left interval		
	    v(9,k)    = v(5,k); ! fx9L
	    v(8,k)    = fx(4);  ! fx8L
	    v(7,k)    = v(4,k); ! fx7L
	    v(6,k)    = fx(3);  ! fx6L
	    v(5,k)    = v(3,k); ! fx5L
	    v(4,k)    = fx(2);  ! fx4L
            v(3,k)    = v(2,k); ! fx3L
	    v(2,k)    = fx(1);  ! fx2L
!	    v(1,k)    = v(1,k); ! fx1L
!           v(Nrule+1,k)  unchanged      x1L
	    v(Nrule+2,k)    = h;
	    v(Nrule+3,k)    = v(Nrule + 4,k); ! S
            v(Nrule+4,k)    = v(Nrule+6,k); ! SL
            v(Nrule+5,k)    = v(Nrule+7,k); ! SR
            v(Nrule+6:Nrule+9,k) = Sn(1:4); ! SL1,SL2,SR1, SR2
	    k = kp1;
         endif
      enddo ! while
      if (epsi<abserr) iflg = IOR(iflg,4) 
      end subroutine AdaptiveSimpson3
      subroutine AdaptiveTrapzWithBreaks(f,a,b,N,brks,epsi,iflg
     $     ,abserr, val)
      implicit none
      double precision :: f
      integer,          intent(in) :: N
      double precision, intent(in) :: a,b,epsi
      double precision, dimension(:), intent(in) :: brks
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      external f
! Locals
      double precision, dimension(N+2) :: pts
      double precision :: LTol,tol, error, valk, excess, errorEstimate
      integer :: kflg, k
      pts(1)   = a
      pts(N+2) = b
      do k = 2,N+1
         pts(k) = minval(brks(k-1:N))  !add user supplied break points
      enddo
      LTol   = epsi / dble( N+2 )
      tol    = LTol
      abserr = 0.0d0
      val    = 0.0D0
      iflg   = 0
      do k = 1, N + 1
         call AdaptiveTrapz(f,pts(k),pts(k+1),tol, kflg,error, valk)
         abserr = abserr + abs(error)
         excess = LTol - abs(error)
         errorEstimate = abserr + dble(N-k+1)*LTol
         if (epsi < errorEstimate ) then
            tol = max(0.1D0*LTol,2.0D-16)	
!         elseif (  LTol < excess / 10.0D0  ) then
!			tol = LTol + excess*0.5D0
		else	
            tol = LTol
         endif
         val = val + valk
         if (kflg>0) iflg = IOR(iflg, kflg)
      end do
      if (epsi<abserr)  iflg = IOR(iflg, 3)
      end subroutine AdaptiveTrapzWithBreaks
      subroutine AdaptiveTrapz1(f,a,b,epsi, iflg,abserr, val)
      implicit none
! by Per A. Brodtkorb 4 June 2003
! based on  psudo code in chapter 7, Kincaid and Cheney (1991).
! Added check on stepsize, i.e., hMin and hMax
! Added an alternitive check on termination: this is more robust  
! Reference:  
!     D.R. Kincaid &  E.W. Cheney (1991)
!     "Numerical Analysis"
!     Brooks/Cole Publ., 1991
!


      double precision :: f
      double precision, intent(in) :: a,b,epsi
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      integer, parameter :: stackLimit = 30
      double precision, dimension(8,stackLimit) :: v
      external f
! Locals
      double precision, parameter :: small    = 1.0D-16
      double precision, parameter :: hmin     = 1.0D-10
      double precision, parameter :: zero     = 0.0D0
      double precision, parameter :: zpz66666 = 0.06666666666666666666D0 !1/15
      double precision, parameter :: onethird = 0.33333333333333333333D0
      double precision, parameter :: half     = 0.5D0
      double precision, parameter :: one      = 1.0D0
      double precision, parameter :: two      = 2.0D0
      double precision, parameter :: three    = 3.0D0
      double precision, parameter :: four     = 4.0D0
      double precision, dimension(4) :: x, fx, Sn
      double precision :: delta,h,localError, correction
      double precision :: hmax, Sn1, Sn2, Sn4, Sn12, Sn24, Sn124
      integer :: k, kp1, i
      logical :: acceptError, lastInStack
      logical :: stepSizeTooSmall, stepSizeOK

      hmax = 0.24D0
c
c     initialize everything, 
c     particularly the first column vector in the stack.
c
      val    = zero
      abserr = zero
      iflg   = 0
	
      delta  = b - a
      h      = delta  
      
      x(1) = a
      x(2) = half * ( a + b )
      x(3) = b
	

      k      = 1
      do I = 1,3
         v(I,1) = f(x(I))
      enddo
	
      Sn(1) = ( v(1,1) + v(3,1) ) * h * half
      h = h * half
      Sn(2) = ( v(1,1) + v(2,1) ) * h * half
      Sn(3) = ( v(2,1) + v(3,1) ) * h * half

      v(4   ,1) = x(1)
      v(5   ,1) = h
      v(6:8 ,1) = Sn(1:3);
	

      do while ((1<=k) .and. (k <= stackLimit))
!     
!     take the last column off the stack and process it.
!
         h   = half * v(5,k)
         Sn1 =  v(6,k)
         Sn2 =  (v(7,k) + v(8,k))
         Sn4 = zero
         do i = 1,2
            x(i)      = v(4,k) + dble(2*i-1)*h
            fx(i)     = f(x(i))
            Sn(2*i-1) = ( v(i  ,k) + fx(i) ) * h * half 
            Sn(2*i  ) = ( v(i+1,k) + fx(i) ) * h * half 
            Sn4   = Sn4 + Sn(2*i-1) + Sn(2*i)
         enddo	
         
         Sn12  = (Sn1 - Sn2)
         Sn24  = (Sn2 - Sn4);
         Sn124 = (Sn12 - Sn24)
!	 Correction based on assumption that the termination error 
!	 is of the form: C*h^q
         if (Sn124==zero) then
            correction = zero !-Sn24 * Sn24 / sign(small,Sn124)
            
            if (Sn12 == zero) then 
               !round off error?
            endif
            
         else
            correction = -Sn24 * Sn24 / Sn124
         endif
         localError = max(abs(correction),abs(Sn24)*half)
         
!     acceptError is made conservative in order to avoid premature termination

         acceptError      = (localError * delta <= two * epsi * h 
     &                      .or. localError < small)     
         lastInStack      = ( stackLimit <= k)
         stepSizeOK       = ( h < hMax )
         stepSizeTooSmall = ( h < hMin)
         if (lastInStack .or. (stepSizeOK.and.acceptError) 
     &       .or. stepSizeTooSmall) then
!     Stop subdividing interval when
!     1) accuracy is sufficient, or
!     2) interval too narrow, or
!     3) subdivided too often. (stack limit reached)
            
!     Add partial integral and take a new vector from the bottom of the stack.

            abserr = abserr + localError 
            val    = val + Sn4 + correction
            k      = k - 1
            if (.not.acceptError) then
               if (lastInStack)      iflg = IOR(iflg,1) ! stack limit reached
               if (stepSizeTooSmall) iflg = IOR(iflg,2) ! stepSize limit reached
            endif	
            if (k <= 0) then 
               return
            endif
         else
!     Subdivide the interval and create two new vectors in the stack, 
!     one of which  overwrites the vector just processed.
!
!	 v(:,k)  = [fx1,fx2,fx3,x1,h,S,SL,SR]
	    kp1 = k + 1;
!	Process right interval
	    v(1,kp1)   = v(2,k); !fx1R 
	    v(2,kp1)   = fx(2); !fx2R
	    v(3,kp1)   = v(3,k); !fx3R
	    v(4,kp1)   = v(4,k) + two * h; ! x1R
	    v(5,kp1)   = h;
	    v(6,kp1)   = v(8,k); ! S
	    v(7:8,kp1) = Sn(3:4); ! SL, SR
!	Process left interval		
	    v(3,k)    = v(2,k); ! fx5L
	    v(2,k)    = fx(1);  ! fx4L
!           v(1,k)  unchanged     fx1L
!           v(4,k)  unchanged      x1L
	    v(5,k)    = h;
	    v(6,k)    = v(7,k); ! S
	    v(7:8,k)  = Sn(1:2); ! SL, SR
	    k = kp1;
         endif
      enddo ! while
      end subroutine AdaptiveTrapz1
      subroutine RombergWithBreaks(f,a,b,N,brks,epsi,iflg
     $     ,abserr, val) 
      implicit none
      double precision :: f
      integer,          intent(in) :: N
      double precision, intent(in) :: a,b,epsi
      double precision, dimension(:), intent(in) :: brks
      double precision, intent(out) :: abserr, val
      integer, intent(out) :: iflg
      external f
! Locals
      double precision, dimension(N+2) :: pts
      double precision :: tol, LTol, error, valk
      double precision :: excess,  errorEstimate
      double precision :: delta, deltaK
      integer :: kflg, k, deciDig
      pts(1)   = a
      pts(N+2) = b
      delta = (b - a) + 0.1D0
      do k = 2,N + 1
         pts(k) = minval(brks(k-1:N))  !add user supplied break points
      enddo
      LTol    = epsi / delta 
      decidig = max(abs(NINT(log10(epsi)))+5,3)
      abserr  = 0.0d0
      val     = 0.0D0
      iflg    = 0
      do k = 1, N+1
         deltaK = pts( k + 1 ) - pts( k )
         tol = LTol * deltaK
         call romberg(f,pts(k),pts(k+1),decidig,tol,kflg,error, valk)
         abserr = abserr + abs(error)
         errorEstimate = abserr + (b - pts( k + 1 ) )* LTol
         excess = epsi - errorEstimate
         if (excess < 0.0D0 ) then
            LTol = 0.1D0 * LTol
         elseif (  epsi < 2.0D0 * excess ) then
            LTol = (epsi + excess*0.5D0) / delta
         endif
         val = val + valk
         if (kflg>0) iflg = ior(iflg,kflg)
      end do
	  if (epsi<abserr)  iflg = IOR(iflg, 3)
      end subroutine RombergWithBreaks
      subroutine Romberg1(f,a,b,decdigs,abseps,errFlg,abserr,VAL)
      implicit none
      double precision :: f
      double precision, intent(in)  :: a,b, abseps
      integer,          intent(in)  :: decdigs                                    ! Relative number of decimal digits
      integer,          intent(out) :: errFlg
      double precision, intent(out) :: abserr, val
      external f
!     Locals
      double precision, dimension(decdigs) ::  rom1,rom2,fp
      double precision, dimension(decdigs + 7) :: EPSTAB
      integer :: LIMEXP
      double precision :: h, fourk, Un5, T1n, T2n, T4n, T12n, T24n
      double precision :: correction
      integer :: ipower, k, i, IERR
      logical :: stepSizeTooSmall, NEWFLG
      double precision, parameter :: four = 4.0D0
      double precision, parameter :: one  = 1.0D0
      double precision, parameter :: half = 0.5D0
      double precision, parameter :: zero = 0.0D0
      double precision, parameter :: hmin = 1.0d-10
      
      LIMEXP  = decdigs
      val     = zero
      errFlg  = 0

      rom1(:) = zero
      rom2(:) = zero
      fp(1)   = four;
      h       = ( b - a )
      rom1(1) = h * ( f(a) + f(b) ) * half
      ipower  = 1
      T1n     = zero
      T2n     = zero
      T4n     = rom1(1)
      NEWFLG = .TRUE.
      CALL DEA(NEWFLG,T4n,LIMEXP,val,abserr,EPSTAB,IERR)
      stepSizeTooSmall = ( h < hMin)
      do i = 2, decdigs
         h = h * half
         
         Un5 = zero
         do  k = 1, ipower      
            Un5 = Un5 + f(a + DBLE(2*k-1)*h)   
         enddo
!     trapezoidal approximations
         rom2(1) = half * rom1(1) + h * Un5  
                  
!     Richardson extrapolation
         do k = 1, i-1
            rom2(k+1) = ( fp(k)*rom2(k)-rom1(k) ) / ( fp(k) - one )
         enddo
         T1n = T2n
         T2n = T4n
         T4n = rom2(i)
         CALL DEA(NEWFLG,T4n,LIMEXP,val,abserr,EPSTAB,IERR)
         if (3<=i) then
!            T12n       = ( T1n - T2n )
!            T24n       = ( T2n - T4n )
!            correction = -T24n * T24n / ( T12n - T24n )
!            abserr = abs( correction )
            stepSizeTooSmall = ( h < hMin)
            if ( abserr < abseps .or. stepSizeTooSmall) then
               exit ! exit do loop
            endif
         endif
         rom1(1:i) = rom2(1:i)
         ipower    = ipower * 2
         fp(i)     = four * fp(i-1)
      enddo
      if (decdigs < 3) then
!         val    = T4n
         abserr = min(abs(T4n-T2n) * half, abserr)
      endif
      if (abseps < abserr) then
         if (stepSizeTooSmall) errFlg = ior(errFlg,2) ! step size limit reached
      endif
      end subroutine Romberg1
!      end module Integration1DModule