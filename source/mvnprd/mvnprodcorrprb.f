C Does not work: f2py -m mvnprdmod  -h mvnprdmod.pyf mvnprodcorrprb.f only: mvnprodcorrprb


C gfortran -fPIC -c mvnprodcorrprb.f
C f2py -m mvnprdmod  -c mvnprodcorrprb.o mvnprodcorrprb_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
C f2py -m mvnprdmod  -c mvnprodcorrprb.o mvnprodcorrprb_interface.f --build-dir tmp1 --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71


* This is a MEX-file for MATLAB.
* and contains a mex-interface to, mvnprodcorrprb a subroutine
* for computing multivariate normal probabilities with product
* correlation structure.
* The file should compile without errors on (Fortran90)
* standard Fortran compilers.
*
* The mex-interface and mvnprodcorrprb was written by
*     Per Andreas Brodtkorb
*     Norwegian Defence Research Establishment
*     P.O. Box 115m
*     N-3191 Horten
*     Norway
*     Email: Per.Brodtkorb@ffi.no
*
*
* MVNPRODCORRPRBMEX Computes multivariate normal probability
*                with product correlation structure.
*
*  CALL [value,error,inform]=mvnprodcorrprbmex(rho,A,B,abseps,releps,useBreakPoints);
*
*     RHO    REAL, array of coefficients defining the correlation
*            coefficient by:
*                correlation(I,J) =  RHO(I)*RHO(J) for J/=I
*            where
*                1 <= RHO(I) <= 1
*     A		 REAL, array of lower integration limits.
*     B		 REAL, array of upper integration limits.
*	       NOTE: any values greater the 10, are considered as
*                   infinite values.
*     ABSEPS REAL absolute error tolerance.
*     RELEPS REAL relative error tolerance.
*     USEBREAKPOINTS = 1 If extra integration points should be used
*                        around possible singularities
*                      0 If no extra
*
*     ERROR  REAL estimated absolute error, with 99% confidence level.
*     VALUE  REAL estimated value for the integral
*     INFORM INTEGER, termination status parameter:
*            if INFORM = 0, normal completion with ERROR < EPS;
*            if INFORM = 1, completion with ERROR > EPS and MAXPTS
*                           function vaules used; increase MAXPTS to
*                           decrease ERROR;
*
* MVNPRODCORRPRB calculates multivariate normal probability
* with product correlation structure for rectangular regions.
* The accuracy is up to almost double precision, i.e., about 1e-14.
*
* This file was successfully compiled for matlab 5.3
* using Compaq Visual Fortran 6.1, and Windows 2000.
* The example here uses Fortran77 source.
* First, you will need to modify your mexopts.bat file.
* To find it, issue the command prefdir(1) from the Matlab command line,
* the directory it answers with will contain your mexopts.bat file.
* Open it for editing. The first section will look like:
*
*rem ********************************************************************
*rem General parameters
*rem ********************************************************************
*set MATLAB=%MATLAB%
*set DF_ROOT=C:\Program Files\Microsoft Visual Studio
*set VCDir=%DF_ROOT%\VC98
*set MSDevDir=%DF_ROOT%\Common\msdev98
*set DFDir=%DF_ROOT%\DF98
*set PATH=%MSDevDir%\bin;%DFDir%\BIN;%VCDir%\BIN;%PATH%
*set INCLUDE=%DFDir%\INCLUDE;%DFDir%\IMSL\INCLUDE;%INCLUDE%
*set LIB=%DFDir%\LIB;%VCDir%\LIB
*
* then you are ready to compile this file at the matlab prompt using the
* following command:
*  mex -O mvnprodcorrprbmex.f
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
C  Revised pab Dec 2008
C   updated parameter statements in CALERF so that it works when
C   compiling with gfortran.
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
C      DOUBLE PRECISION, DIMENSION(5) :: A, Q
C      DOUBLE PRECISION, DIMENSION(4) :: B
C      DOUBLE PRECISION, DIMENSION(9) :: C
C      DOUBLE PRECISION, DIMENSION(8) :: D
C      DOUBLE PRECISION, DIMENSION(6) :: P
C------------------------------------------------------------------
C  Coefficients for approximation to  erf  in first interval
C------------------------------------------------------------------
      DOUBLE PRECISION, PARAMETER, DIMENSION(5) ::
     & A = (/ 3.16112374387056560D00,
     &     1.13864154151050156D02,3.77485237685302021D02,
     &     3.20937758913846947D03, 1.85777706184603153D-1/)
      DOUBLE PRECISION, PARAMETER, DIMENSION(4) ::
     & B = (/2.36012909523441209D01,2.44024637934444173D02,
     &       1.28261652607737228D03,2.84423683343917062D03/)
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in second interval
C------------------------------------------------------------------
      DOUBLE PRECISION, DIMENSION(9) ::
     &   C=(/5.64188496988670089D-1,8.88314979438837594D0,
     1       6.61191906371416295D01,2.98635138197400131D02,
     2       8.81952221241769090D02,1.71204761263407058D03,
     3       2.05107837782607147D03,1.23033935479799725D03,
     4       2.15311535474403846D-8/)
      DOUBLE PRECISION, DIMENSION(8) ::
     &  D =(/1.57449261107098347D01,1.17693950891312499D02,
     1       5.37181101862009858D02,1.62138957456669019D03,
     2       3.29079923573345963D03,4.36261909014324716D03,
     3       3.43936767414372164D03,1.23033935480374942D03/)
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in third interval
C------------------------------------------------------------------
      DOUBLE PRECISION, parameter,
     & DIMENSION(6) :: P =(/3.05326634961232344D-1,
     & 3.60344899949804439D-1,
     1       1.25781726111229246D-1,1.60837851487422766D-2,
     2       6.58749161529837803D-4,1.63153871373020978D-2/)
      DOUBLE PRECISION, parameter,
     & DIMENSION(5) :: Q =(/2.56852019228982242D00,
     & 1.87295284992346047D00,
     1       5.27905102951428412D-1,6.05183413124413191D-2,
     2       2.33520497626869185D-3/)
C------------------------------------------------------------------

      X = ARG
      Y = ABS(X)
      IF (Y .LE. THRESH) THEN
C------------------------------------------------------------------
C  Evaluate  erf  for  |X| <= 0.46875
C------------------------------------------------------------------
         YSQ = ZERO
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
      module functionInterface
      INTERFACE
         FUNCTION F(Z) result (VAL)
         DOUBLE PRECISION, INTENT(IN) :: Z
         DOUBLE PRECISION :: VAL
         END FUNCTION F
      END INTERFACE
      end module functionInterface
      module AdaptiveGaussKronrod
      implicit none
      private
      public :: dqagpe,dqagp

      INTERFACE dqagpe
      MODULE PROCEDURE dqagpe
      END INTERFACE

      INTERFACE dqagp
      MODULE PROCEDURE dqagp
      END INTERFACE

      INTERFACE dqelg
      MODULE PROCEDURE dqelg
      END INTERFACE

      INTERFACE dqpsrt
      MODULE PROCEDURE dqpsrt
      END INTERFACE

      INTERFACE dqk21
      MODULE PROCEDURE dqk21
      END INTERFACE

      INTERFACE dqk15
      MODULE PROCEDURE dqk15
      END INTERFACE

      INTERFACE dqk9
      MODULE PROCEDURE dqk9
      END INTERFACE

      INTERFACE d1mach
      MODULE PROCEDURE d1mach
      END INTERFACE

      contains
      subroutine dea3(E0,E1,E2,abserr,result)
!***PURPOSE  Given a slowly convergent sequence, this routine attempts
!            to extrapolate nonlinearly to a better estimate of the
!            sequence's limiting value, thus improving the rate of
!            convergence. Routine is based on the epsilon algorithm
!            of P. Wynn. An estimate of the absolute error is also
!            given.
      double precision, intent(in) :: E0,E1,E2
      double precision, intent(out) :: abserr, result
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
         result = E2
         abserr = err1 + err2 + E2*small*ten
      else
         ss = one/delta2 - one/delta1
         if (abs(ss*E1) <= 1.0d-3) then
            result = E2
            abserr = err1 + err2 + E2*small*ten
         else
            result = E1 + one/ss
            abserr = err1 + err2 + abs(result-E2)
         endif
      endif
      end subroutine dea3
       subroutine dqagp(f,a,b,npts,points,epsabs,epsrel,limit,result1,
     *   abserr,neval,ier)
!       use functionInterface
       implicit none
       integer,                          intent(in) :: npts,limit
       double precision,dimension(npts), intent(in) :: points
       double precision,  intent(in) :: a, b,  epsabs,epsrel
       double precision, intent(out) :: result1,abserr
       integer,          intent(out) :: neval,ier
       double precision :: f
!Locals
       double precision,dimension(limit)  :: alist, blist, rlist, elist
       double precision,dimension(npts+2) :: pts
       integer, dimension(limit)          :: iord, level
       integer, dimension(npts+2)         :: ndin
       integer ::last
       external f
       CALL dqagpe(f,a,b,npts,points,epsabs,epsrel,limit,result1,
     *      abserr,neval,ier,alist,blist,rlist,elist,pts,iord,level,ndin
     $      ,last)
       end subroutine dqagp
      subroutine dqagpe(f,a,b,npts,points,epsabs,epsrel,limit,result,
     *   abserr,neval,ier,alist,blist,rlist,elist,pts,iord,level,ndin,
     *   last)
!      use functionInterface
      implicit none
c***begin prologue  dqagpe
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a2a1
c***keywords  automatic integrator, general-purpose,
!             singularities at user specified points,
!             extrapolation, globally adaptive.
c***author  piessens,robert ,appl. math. & progr. div. - k.u.leuven
!           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a given
!            definite integral i = integral of f over (a,b), hopefully
!            satisfying following claim for accuracy abs(i-result).le.
!            max(epsabs,epsrel*abs(i)). break points of the integration
!            interval, where local difficulties of the integrand may
!            occur(e.g. singularities,discontinuities),provided by user.
c***description
!
!        computation of a definite integral
!        standard fortran subroutine
!        double precision version
!
!        parameters
!         on entry
!            f      - double precision
!                     function subprogram defining the integrand
!                     function f(x). the actual name for f needs to be
!                     declared e x t e r n a l in the driver program.
!
!            a      - double precision
!                     lower limit of integration
!
!            b      - double precision
!                     upper limit of integration
!
!            npts2  - integer
!                     number equal to two more than the number of
!                     user-supplied break points within the integration
!                     range, npts2.ge.2.
!                     if npts2.lt.2, the routine will end with ier = 6.
!
!            points - double precision
!                     vector of dimension npts2, the first (npts2-2)
!                     elements of which are the user provided break
!                     points. if these points do not constitute an
!                     ascending sequence there will be an automati!
!                     sorting.
!
!            epsabs - double precision
!                     absolute accuracy requested
!            epsrel - double precision
!                     relative accuracy requested
!                     if  epsabs.le.0
!                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
!                     the routine will end with ier = 6.
!
!            limit  - integer
!                     gives an upper bound on the number of subintervals
!                     in the partition of (a,b), limit.ge.npts2
!                     if limit.lt.npts2, the routine will end with
!                     ier = 6.
!
!         on return
!            result - double precision
!                     approximation to the integral
!
!            abserr - double precision
!                     estimate of the modulus of the absolute error,
!                     which should equal or exceed abs(i-result)
!
!            neval  - integer
!                     number of integrand evaluations
!
!            ier    - integer
!                     ier = 0 normal and reliable termination of the
!                             routine. it is assumed that the requested
!                             accuracy has been achieved.
!                     ier.gt.0 abnormal termination of the routine.
!                             the estimates for integral and error are
!                             less reliable. it is assumed that the
!                             requested accuracy has not been achieved.
!            error messages
!                     ier = 1 maximum number of subdivisions allowed
!                             has been achieved. one can allow more
!                             subdivisions by increasing the value of
!                             limit (and taking the according dimension
!                             adjustments into account). however, if
!                             this yields no improvement it is advised
!                             to analyze the integrand in order to
!                             determine the integration difficulties. if
!                             the position of a local difficulty can be
!                             determined (i.e. singularity,
!                             discontinuity within the interval), it
!                             should be supplied to the routine as an
!                             element of the vector points. if necessary
!                             an appropriate special-purpose integrator
!                             must be used, which is designed for
!                             handling the type of difficulty involved.
!                         = 2 the occurrence of roundoff error is
!                             detected, which prevents the requested
!                             tolerance from being achieved.
!                             the error may be under-estimated.
!                         = 3 extremely bad integrand behaviour occurs
!                             at some points of the integration
!                             interval.
!                         = 4 the algorithm does not converge.
!                             roundoff error is detected in the
!                             extrapolation table. it is presumed that
!                             the requested tolerance cannot be
!                             achieved, and that the returned result is
!                             the best which can be obtained.
!                         = 5 the integral is probably divergent, or
!                             slowly convergent. it must be noted that
!                             divergence can occur with any other value
!                             of ier.gt.0.
!                         = 6 the input is invalid because
!                             npts2.lt.2 or
!                             break points are specified outside
!                             the integration range or
!                             (epsabs.le.0 and
!                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
!                             or limit.lt.npts2.
!                             result, abserr, neval, last, rlist(1),
!                             and elist(1) are set to zero. alist(1) and
!                             blist(1) are set to a and b respectively.
!
!            alist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the left end points
!                     of the subintervals in the partition of the given
!                     integration range (a,b)
!
!            blist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the right end points
!                     of the subintervals in the partition of the given
!                     integration range (a,b)
!
!            rlist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the integral
!                     approximations on the subintervals
!
!            elist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the moduli of the
!                     absolute error estimates on the subintervals
!
!            pts    - double precision
!                     vector of dimension at least npts2, containing the
!                     integration limits and the break points of the
!                     interval in ascending sequence.
!
!            level  - integer
!                     vector of dimension at least limit, containing the
!                     subdivision levels of the subinterval, i.e. if
!                     (aa,bb) is a subinterval of (p1,p2) where p1 as
!                     well as p2 is a user-provided break point or
!                     integration limit, then (aa,bb) has level l if
!                     abs(bb-aa) = abs(p2-p1)*2**(-l).
!
!            ndin   - integer
!                     vector of dimension at least npts2, after first
!                     integration over the intervals (pts(i)),pts(i+1),
!                     i = 0,1, ..., npts2-2, the error estimates over
!                     some of the intervals may have been increased
!                     artificially, in order to put their subdivision
!                     forward. if this happens for the subinterval
!                     numbered k, ndin(k) is put to 1, otherwise
!                     ndin(k) = 0.
!
!            iord   - integer
!                     vector of dimension at least limit, the first k
!                     elements of which are pointers to the
!                     error estimates over the subintervals,
!                     such that elist(iord(1)), ..., elist(iord(k))
!                     form a decreasing sequence, with k = last
!                     if last.le.(limit/2+2), and k = limit+1-last
!                     otherwise
!
!            last   - integer
!                     number of subintervals actually produced in the
!                     subdivisions process
!
c***references  (none)
c***routines called  d1mach,dqelg,dqk21,dqpsrt
c***end prologue  dqagpe
       integer,                          intent(in) :: npts,limit
       double precision,dimension(npts), intent(in) :: points
       double precision,  intent(in) :: a, b,  epsabs,epsrel
       double precision, intent(out) :: result,abserr
       integer,          intent(out) :: neval,ier
       double precision,dimension(limit), intent(out)  :: alist, blist
       double precision,dimension(limit), intent(out)  :: rlist, elist
       double precision,dimension(npts+2),intent(out)  :: pts
       integer,         dimension(limit), intent(out)  :: iord, level
       integer,         dimension(npts+2), intent(out) :: ndin
       integer ::last
       double precision :: f
! locals
      double precision :: area,area1,area12,area2,a1,
     *  a2,b1,b2,correc,abseps,defabs,defab1,defab2,
     *  dres,epmach,erlarg,erlast,errbnd,
     *  errmax,error1,erro12,error2,errsum,ertest,oflow,
     *  resa,resabs,reseps,sign,temp,uflow, hSplit
      double precision, dimension(3)  :: res3la(3)
      double precision, dimension(52) :: rlist2(52)
      integer :: i,id,ierro,ind1,ind2,ip1,iroff1,iroff2,iroff3,j,
     *  jlow,jupbnd,k,ksgn,ktmin,levcur,levmax,maxerr,
     *  nint,nintp1,npts2,nres,nrmax,numrl2
      logical :: extrap,noext
      external f
!
!

!
!
!            the dimension of rlist2 is determined by the value of
!            limexp in subroutine epsalg (rlist2 should be of dimension
!            (limexp+2) at least).
!
!
!            list of major variables
!            -----------------------
!
!           alist     - list of left end points of all subintervals
!                       considered up to now
!           blist     - list of right end points of all subintervals
!                       considered up to now
!           rlist(i)  - approximation to the integral over
!                       (alist(i),blist(i))
!           rlist2    - array of dimension at least limexp+2
!                       containing the part of the epsilon table which
!                       is still needed for further computations
!           elist(i)  - error estimate applying to rlist(i)
!           maxerr    - pointer to the interval with largest error
!                       estimate
!           errmax    - elist(maxerr)
!           erlast    - error on the interval currently subdivided
!                       (before that subdivision has taken place)
!           area      - sum of the integrals over the subintervals
!           errsum    - sum of the errors over the subintervals
!           errbnd    - requested accuracy max(epsabs,epsrel*
!                       abs(result))
!           *****1    - variable for the left subinterval
!           *****2    - variable for the right subinterval
!           last      - index for subdivision
!           nres      - number of calls to the extrapolation routine
!           numrl2    - number of elements in rlist2. if an appropriate
!                       approximation to the compounded integral has
!                       been obtained, it is put in rlist2(numrl2) after
!                       numrl2 has been increased by one.
!           erlarg    - sum of the errors over the intervals larger
!                       than the smallest interval considered up to now
!           extrap    - logical variable denoting that the routine
!                       is attempting to perform extrapolation. i.e.
!                       before subdividing the smallest interval we
!                       try to decrease the value of erlarg.
!           noext     - logical variable denoting that extrapolation is
!                       no longer allowed (true-value)
!
!            machine dependent constants
!            ---------------------------
!
!           epmach is the largest relative spacing.
!           uflow is the smallest positive magnitude.
!           oflow is the largest positive magnitude.
!
c***first executable statement  dqagpe
      epmach = d1mach(4)
      uflow  = d1mach(1)
      oflow  = d1mach(2)
!
!            test on validity of parameters
!            -----------------------------
!
      hSplit  = 0.2D0
      ier     = 0
      neval   = 0
      last    = 0
      result  = 0.0d+00
      abserr  = 0.0d+00
      alist(1) = a
      blist(1) = b
      rlist(1) = 0.0d+00
      elist(1) = 0.0d+00
      iord(1)  = 0
      level(1) = 0
      npts2 = npts+2
      if((npts2.lt.2).or.(limit.le.npts).or.
     &     ((epsabs.le.0.0d+00).and.
     &     (epsrel.lt.dmax1(0.5d+02*epmach,0.5d-28)))) then
         ier = 6
         go to 999
      endif

      sign = 1.0d+00
      if(a.gt.b) then
         go to 999
      endif
      if (npts>0) then
         if(any(points(1:npts)<=a).or.any(b<=points(1:npts))) then
            ier = 6
            go to 999
         endif
      endif
!
!            if any break points are provided, sort them into an
!            ascending sequence.
!
      pts(1)      = a
      pts(npts+2) = b
      do i = 1,npts
        pts(i+1) = minval(points(i:npts))
      enddo
!
!            compute first integral and error approximations.
!            ------------------------------------------------
!
      nint   = npts+1;
      a1     = pts(1);
      resabs = 0.0d+00
      do  i = 1,nint
        b1 = pts(i+1)
        if (b1-a1 > hSplit) then
           call dqk21(f,a1,b1,area1,error1,defabs,resa)
           !call dqk15(f,a1,b1,area1,error1,defabs,resa)
        else
           call dqkl9(f,a1,b1,area1,error1,defabs,resa)
        endif
        abserr = abserr + error1
        result = result + area1
        ndin(i) = 0
        if(error1.eq.resa.and.error1.ne.0.0d+00) ndin(i) = 1
        resabs = resabs + defabs
        level(i) = 0
        elist(i) = error1
        alist(i) = a1
        blist(i) = b1
        rlist(i) = area1
        iord(i) = i
        a1 = b1
      enddo                     !50 continue
      errsum = 0.0d+00
      do  i = 1,nint
        if(ndin(i).eq.1) elist(i) = abserr
        errsum = errsum+elist(i)
      enddo                     !55 continue
!
!           test on accuracy.
!
      last   = nint
      neval  = 21*nint
      dres   = dabs(result)
      errbnd = dmax1(epsabs,epsrel*dres)
      if(abserr.le.0.1d+03*epmach*resabs.and.abserr.gt.errbnd) ier = 2
      if(nint.eq.1) go to 80
      do 70 i = 1,npts
        jlow = i+1
        ind1 = iord(i)
        do 60 j = jlow,nint
          ind2 = iord(j)
          if(elist(ind1).gt.elist(ind2)) go to 60
          ind1 = ind2
          k = j
   60   continue
        if(ind1.eq.iord(i)) go to 70
        iord(k) = iord(i)
        iord(i) = ind1
   70 continue
      if(limit.lt.npts2) ier = 1
 80   if(ier.ne.0.or.abserr.le.errbnd) go to 210

!
!           initialization
!           --------------
!
      rlist2(1) = result
      maxerr    = iord(1)
      errmax    = elist(maxerr)
      area      = result
      nrmax     = 1
      nres   = 0
      numrl2 = 1
      ktmin  = 0
      extrap = .false.
      noext  = .false.
      erlarg = errsum
      ertest = errbnd
      levmax = 1
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ierro  = 0
      abserr = oflow
      ksgn   = -1
      if(dres.ge.(0.1d+01-0.5d+02*epmach)*resabs) ksgn = 1
!
!           main do-loop
!           ------------
!
      do 160 last = npts2,limit
!
!           bisect the subinterval with the nrmax-th largest error
!           estimate.
!
        levcur = level(maxerr)+1
        a1 = alist(maxerr)
        b1 = 0.5d+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        if (b1-a1 > hSplit) then
           call dqk21(f,a1,b1,area1,error1,resa,defab1)
           call dqk21(f,a2,b2,area2,error2,resa,defab2)
           !call dqk15(f,a1,b1,area1,error1,resa,defab1)
           !call dqk15(f,a2,b2,area2,error2,resa,defab2)
        else

           call dqkl9(f,a1,b1,area1,error1,resa,defab1)
           call dqkl9(f,a2,b2,area2,error2,resa,defab2)
        endif
!
!           improve previous approximations to integral
!           and error and test for accuracy.
!
        neval = neval+42
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2) go to 95
        if(dabs(rlist(maxerr)-area12).gt.0.1d-04*dabs(area12)
     *  .or.erro12.lt.0.99d+00*errmax) go to 90
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   90   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   95   level(maxerr) = levcur
        level(last) = levcur
        rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = dmax1(epsabs,epsrel*dabs(area))
!
!           test for roundoff error and eventually set error flag.
!
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
!
!           set error flag in the case that the number of
!           subintervals equals limit.
!
        if(last.eq.limit) ier = 1
!
!           set error flag in the case of bad integrand behaviour
!           at a point of the integration range
!
        if(dmax1(dabs(a1),dabs(b2)).le.(0.1d+01+0.1d+03*epmach)*
     *  (dabs(a2)+0.1d+04*uflow)) ier = 4
!
!           append the newly-created intervals to the list.
!
        if(error2.gt.error1) go to 100
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 110
  100   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
!
!           call subroutine dqpsrt to maintain the descending ordering
!           in the list of error estimates and select the subinterval
!           with nrmax-th largest error estimate (to be bisected next).
!
  110   call dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
! ***jump out of do-loop
        if(errsum.le.errbnd) go to 190
! ***jump out of do-loop
        if(ier.ne.0) go to 170
        if(noext) go to 160
        erlarg = erlarg-erlast
        if(levcur+1.le.levmax) erlarg = erlarg+erro12
        if(extrap) go to 120
!
!           test whether the interval to be bisected next is the
!           smallest interval.
!
        if(level(maxerr)+1.le.levmax) go to 160
        extrap = .true.
        nrmax = 2
  120   if(ierro.eq.3.or.erlarg.le.ertest) go to 140
!
!           the smallest interval has the largest error.
!           before bisecting decrease the sum of the errors over
!           the larger intervals (erlarg) and perform extrapolation.
!
        id = nrmax
        jupbnd = last
        if(last.gt.(2+limit/2)) jupbnd = limit+3-last
        do 130 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
! ***jump out of do-loop
          if(level(maxerr)+1.le.levmax) go to 160
          nrmax = nrmax+1
  130   continue
!
!           perform extrapolation.
!
  140   numrl2 = numrl2+1
        rlist2(numrl2) = area
        if(numrl2.le.2) go to 155
        call dqelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1d-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 150
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = dmax1(epsabs,epsrel*dabs(reseps))
! ***jump out of do-loop
        if(abserr.lt.ertest) go to 170
!
!           prepare bisection of the smallest interval.
!
  150   if(numrl2.eq.1) noext = .true.
        if(ier.ge.5) go to 170
  155   maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        levmax = levmax + 1
        erlarg = errsum
  160 continue
!
!           set the final result.
!           ---------------------
!
!
  170 if(abserr.eq.oflow) go to 190
      if((ier+ierro).eq.0) go to 180
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.0.0d+00.and.area.ne.0.0d+00)go to 175
      if(abserr.gt.errsum)go to 190
      if(area.eq.0.0d+00) go to 210
      go to 180
  175 if(abserr/dabs(result).gt.errsum/dabs(area))go to 190
!
!           test on divergence.
!
  180 if(ksgn.eq.(-1).and.dmax1(dabs(result),dabs(area)).le.
     *  resabs*0.1d-01) go to 210
      if(0.1d-01.gt.(result/area).or.(result/area).gt.0.1d+03.or.
     *  errsum.gt.dabs(area)) ier = 6
      go to 210
!
!           compute global integral sum.
!
  190 result = 0.0d+00
      do 200 k = 1,last
        result = result+rlist(k)
  200 continue
      abserr = errsum
  210 if(ier.gt.2) ier = ier-1
      result = result*sign
  999 return
      end subroutine dqagpe
      subroutine dqk21(f,a,b,result,abserr,resabs,resasc)
!      use functionInterface
      implicit none
c***begin prologue  dqk21
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  21-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           double precision version
c
c           parameters
c            on entry
c              f      - double precision
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              a      - double precision
c                       lower limit of integration
c
c              b      - double precision
c                       upper limit of integration
c
c            on return
c              result - double precision
c                       approximation to the integral i
c                       result is computed by applying the 21-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 10-point gauss rule (resg).
c
c              abserr - double precision
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - double precision
c                       approximation to the integral j
c
c              resasc - double precision
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  d1mach
c***end prologue  dqk21
c
      double precision :: f, a,absc,abserr,b,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc,
     *  resg,resk,reskh,result,uflow,wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(10),fv2(10),wg(5),wgk(11),xgk(11)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 21-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 10-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 10-point gauss rule
c
c           wgk    - weights of the 21-point kronrod rule
c
c           wg     - weights of the 10-point gauss rule
c
c
c gauss quadrature weights and kronron quadrature abscissae and weights
c as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
c bell labs, nov. 1981.
c
      data wg  (  1) / 0.0666713443 0868813759 3568809893 332 d0 /
      data wg  (  2) / 0.1494513491 5058059314 5776339657 697 d0 /
      data wg  (  3) / 0.2190863625 1598204399 5534934228 163 d0 /
      data wg  (  4) / 0.2692667193 0999635509 1226921569 469 d0 /
      data wg  (  5) / 0.2955242247 1475287017 3892994651 338 d0 /
c
      data xgk (  1) / 0.9956571630 2580808073 5527280689 003 d0 /
      data xgk (  2) / 0.9739065285 1717172007 7964012084 452 d0 /
      data xgk (  3) / 0.9301574913 5570822600 1207180059 508 d0 /
      data xgk (  4) / 0.8650633666 8898451073 2096688423 493 d0 /
      data xgk (  5) / 0.7808177265 8641689706 3717578345 042 d0 /
      data xgk (  6) / 0.6794095682 9902440623 4327365114 874 d0 /
      data xgk (  7) / 0.5627571346 6860468333 9000099272 694 d0 /
      data xgk (  8) / 0.4333953941 2924719079 9265943165 784 d0 /
      data xgk (  9) / 0.2943928627 0146019813 1126603103 866 d0 /
      data xgk ( 10) / 0.1488743389 8163121088 4826001129 720 d0 /
      data xgk ( 11) / 0.0000000000 0000000000 0000000000 000 d0 /
c
      data wgk (  1) / 0.0116946388 6737187427 8064396062 192 d0 /
      data wgk (  2) / 0.0325581623 0796472747 8818972459 390 d0 /
      data wgk (  3) / 0.0547558965 7435199603 1381300244 580 d0 /
      data wgk (  4) / 0.0750396748 1091995276 7043140916 190 d0 /
      data wgk (  5) / 0.0931254545 8369760553 5065465083 366 d0 /
      data wgk (  6) / 0.1093871588 0229764189 9210590325 805 d0 /
      data wgk (  7) / 0.1234919762 6206585107 7958109831 074 d0 /
      data wgk (  8) / 0.1347092173 1147332592 8054001771 707 d0 /
      data wgk (  9) / 0.1427759385 7706008079 7094273138 717 d0 /
      data wgk ( 10) / 0.1477391049 0133849137 4841515972 068 d0 /
      data wgk ( 11) / 0.1494455540 0291690566 4936468389 821 d0 /
c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 10-point gauss formula
c           resk   - result of the 21-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  dqk21
      epmach = d1mach(4)
      uflow = d1mach(1)
c
      centr = 0.5d+00*(a+b)
      hlgth = 0.5d+00*(b-a)
      dhlgth = dabs(hlgth)
c
c           compute the 21-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      resg = 0.0d+00
      fc = f(centr)
      resk = wgk(11)*fc
      resabs = dabs(resk)
      do 10 j=1,5
        jtw = 2*j
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(dabs(fval1)+dabs(fval2))
   10 continue
      do 15 j = 1,5
        jtwm1 = 2*j-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(dabs(fval1)+dabs(fval2))
   15 continue
      reskh = resk*0.5d+00
      resasc = wgk(11)*dabs(fc-reskh)
      do 20 j=1,10
        resasc = resasc+wgk(j)*(dabs(fv1(j)-reskh)+dabs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = dabs((resk-resg)*hlgth)*10.0d0
      if(resasc.ne.0.0d+00.and.abserr.ne.0.0d+00) then
         abserr = resasc*dmin1(0.1d+01,
     &        (0.2d+03*abserr/resasc)**1.5d+00)
      endif
      if(resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1
     *  ((epmach*0.5d+02)*resabs,abserr)
      return
      end subroutine dqk21
      subroutine dqk15(f,a,b,result,abserr,resabs,resasc)
!      use functionInterface
      implicit none
c***begin prologue  dqk15
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  15-point gauss-kronrod rules
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           double precision version
c
c           parameters
c            on entry
c              f      - double precision
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              a      - double precision
c                       lower limit of integration
c
c              b      - double precision
c                       upper limit of integration
c
c            on return
c              result - double precision
c                       approximation to the integral i
c                       result is computed by applying the 21-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 10-point gauss rule (resg).
c
c              abserr - double precision
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - double precision
c                       approximation to the integral j
c
c              resasc - double precision
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  d1mach
c***end prologue  dqk15
c
      double precision :: f, a,absc,abserr,b,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc,
     *  resg,resk,reskh,result,uflow,wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(7),fv2(7),wg(4),wgk(8),xgk(8)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 15-point kronrod rule
c                    xgk(2), xgk(4), ...  abscissae of the 7-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 7-point gauss rule
c
c           wgk    - weights of the 15-point kronrod rule
c
c           wg     - weights of the 7-point gauss rule
c
c
c gauss quadrature weights and kronron quadrature abscissae and weights
c as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
c bell labs, nov. 1981.
c
      data wg  (  1) / 0.129484966168869693270611432679082d0 /
      data wg  (  2) / 0.279705391489276667901467771423780d0 /
      data wg  (  3) / 0.381830050505118944950369775488975d0 /
      data wg  (  4) / 0.417959183673469387755102040816327d0 /

      data xgk (  1) / 0.991455371120812639206854697526329d0 /
      data xgk (  2) / 0.949107912342758524526189684047851d0 /
      data xgk (  3) / 0.864864423359769072789712788640926d0 /
      data xgk (  4) / 0.741531185599394439863864773280788d0 /
      data xgk (  5) / 0.586087235467691130294144838258730d0 /
      data xgk (  6) / 0.405845151377397166906606412076961d0 /
      data xgk (  7) / 0.207784955007898467600689403773245d0 /
      data xgk (  8) / 0.000000000000000000000000000000000d0 /

      data wgk (  1) / 0.022935322010529224963732008058970d0/
      data wgk (  2) / 0.063092092629978553290700663189204d0 /
      data wgk (  3) / 0.104790010322250183839876322541518d0 /
      data wgk (  4) / 0.140653259715525918745189590510238d0 /
      data wgk (  5) / 0.169004726639267902826583426598550d0 /
      data wgk (  6) / 0.190350578064785409913256402421014d0 /
      data wgk (  7) / 0.204432940075298892414161999234649d0 /
      data wgk (  8) / 0.209482141084727828012999174891714d0 /

c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 10-point gauss formula
c           resk   - result of the 21-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  dqk15
      epmach = d1mach(4)
      uflow = d1mach(1)
c
      centr = 0.5d+00*(a+b)
      hlgth = 0.5d+00*(b-a)
      dhlgth = dabs(hlgth)
c
c           compute the 15-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      fc = f(centr)
      resk = wgk(8)*fc
      resg =  wg(4)*fc
      resabs = dabs(resk)
      do 10 j=1,3
        jtw = 2*j
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(dabs(fval1)+dabs(fval2))
   10 continue
      do 15 j = 1,4
        jtwm1 = 2*j-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(dabs(fval1)+dabs(fval2))
   15 continue
      reskh = resk*0.5d+00
      resasc = wgk(8)*dabs(fc-reskh)
      do 20 j=1,7
        resasc = resasc+wgk(j)*(dabs(fv1(j)-reskh)+dabs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = dabs((resk-resg)*hlgth)*10.0D0
      if(resasc.ne.0.0d+00.and.abserr.ne.0.0d+00) then
         abserr = resasc*dmin1(0.1d+01,
     &        (0.2d+03*abserr/resasc)**1.5d+00)
      endif
      if(resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1
     *  ((epmach*0.5d+02)*resabs,abserr)
      return
      end subroutine dqk15
      subroutine dqk9(f,a,b,result,abserr,resabs,resasc)
!      use functionInterface
      implicit none
c***begin prologue  dqk15
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  15-point gauss-kronrod rules extended from a 3 point gaus rule
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           double precision version
c
c           parameters
c            on entry
c              f      - double precision
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              a      - double precision
c                       lower limit of integration
c
c              b      - double precision
c                       upper limit of integration
c
c            on return
c              result - double precision
c                       approximation to the integral i
c                       result is computed by applying the 21-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 10-point gauss rule (resg).
c
c              abserr - double precision
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - double precision
c                       approximation to the integral j
c
c              resasc - double precision
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  d1mach
c***end prologue  dqk15
c
      double precision :: f, a,absc,abserr,b,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc,
     *  resg,resk0,resk,reskh,result,uflow,wg,wgk0,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(7),fv2(7),wg(2),wgk0(4),wgk(8),xgk(8)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 15-point kronrod rule
!                    xgk(4), xgk(8)  abscissae of the 3-point gauss rule
c                    xgk(2), xgk(4),xgk(6), xgk(8) ...  abscissae of the 7-point
c                    kronrod rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 7-point kronrod rule
c
c           wgk    - weights of the 15-point kronrod rule
!
!           wgk0   - weights of the 7-point kronrod rule
c
c           wg     - weights of the 3-point gauss rule
c
c
c gauss quadrature weights and kronrod quadrature abscissae and weights
c as evaluated in quadruple precision  by Patterson
c
      data wg  (  1) /  0.5555555555555555D+00/
      data wg  (  2) /  0.8888888888888889D+00/

      data wgk0  (  1) /  0.1046562260264673D+00/
      data wgk0  (  2) /  0.2684880898683335D+00/
      data wgk0  (  3) /  0.4013974147759622D+00/
      data wgk0  (  4) /  0.4509165386584741D+00/

      data xgk (  1) / 0.9938319632127550D+00/
      data xgk (  2) / 0.9604912687080203D+00/
      data xgk (  3) / 0.8884592328722570D+00 /
      data xgk (  4) / 0.7745966692414834D+00/
      data xgk (  5) / 0.6211029467372264D+00/
      data xgk (  6) / 0.4342437493468026D+00/
      data xgk (  7) / 0.2233866864289669D+00 /
      data xgk (  8) / 0.000000000000000000000000000000000d0 /

      data wgk (  1) / 0.1700171962994028D-01/
      data wgk (  2) / 0.5160328299707982D-01/
      data wgk (  3) / 0.9292719531512452D-01/
      data wgk (  4) / 0.1344152552437843D+00/
      data wgk (  5) / 0.1715119091363914D+00/
      data wgk (  6) / 0.2006285293769890D+00/
      data wgk (  7) / 0.2191568584015875D+00/
      data wgk (  8) / 0.2255104997982067D+00/

c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 10-point gauss formula
c           resk   - result of the 21-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  dqk15
      epmach = d1mach(4)
      uflow = d1mach(1)
c
      centr = 0.5d+00*(a+b)
      hlgth = 0.5d+00*(b-a)
      dhlgth = dabs(hlgth)
c
c           compute the 15-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      fc = f(centr)
      resk = wgk(8)*fc
      resk0 =  wgk0(4)*fc
      resabs = dabs(resk)
      do 10 j=1,3
        jtw = 2*j
        absc  = hlgth * xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum   = fval1 + fval2
        resk0  = resk0 + wgk0(j) * fsum
        resk   = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(dabs(fval1)+dabs(fval2))
   10 continue
      resg  =  wg(2)*fc + wg(1)*(fv1(4) + fv2(4))
      do 15 j = 1,4
        jtwm1 = 2*j-1
        absc  = hlgth * xgk(jtwm1)
        fval1 = f( centr - absc )
        fval2 = f( centr + absc )
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum   = fval1  + fval2
        resk   = resk   + wgk(jtwm1) * fsum
        resabs = resabs + wgk(jtwm1) * (dabs(fval1) + dabs(fval2))
   15 continue

      reskh = resk*0.5d+00
      resasc = wgk(8)*dabs(fc-reskh)
      do 20 j=1,7
        resasc = resasc+wgk(j)*(dabs(fv1(j)-reskh)+dabs(fv2(j)-reskh))
   20 continue
      resg   = resg   * hlgth
      resk0  = resk0  * hlgth
      resk   = resk   * hlgth
      resabs = resabs * dhlgth
      resasc = resasc * dhlgth
      result = resk
      call dea3(resg,resk0,resk,abserr,result)
      abserr = max((dabs(resk-resk0) +  dabs(resg-resk0))
     &     * 10.0D0, abserr)
      if(resasc.ne.0.0d+00.and.abserr.ne.0.0d+00) then
         abserr = resasc * dmin1(0.1d+01,
     &        (0.2d+03*abserr/resasc)**1.5d+00)
      endif
      if(resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1
     *  ((epmach*0.5d+02)*resabs,abserr)
      return

      end subroutine dqk9
      subroutine dqkl9(f,a,b,result,abserr,resabs,resasc)
!     use functionInterface
      implicit none
c***begin prologue  dqk15
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a1a2
c***keywords  15-point gauss-kronrod rules extended from a 3 point gaus rule
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f over (a,b), with error
c                           estimate
c                       j = integral of abs(f) over (a,b)
c***description
c
c           integration rules
c           standard fortran subroutine
c           double precision version
c
c           parameters
c            on entry
c              f      - double precision
c                       function subprogram defining the integrand
c                       function f(x). the actual name for f needs to be
c                       declared e x t e r n a l in the driver program.
c
c              a      - double precision
c                       lower limit of integration
c
c              b      - double precision
c                       upper limit of integration
c
c            on return
c              result - double precision
c                       approximation to the integral i
c                       result is computed by applying the 21-point
c                       kronrod rule (resk) obtained by optimal addition
c                       of abscissae to the 10-point gauss rule (resg).
c
c              abserr - double precision
c                       estimate of the modulus of the absolute error,
c                       which should not exceed abs(i-result)
c
c              resabs - double precision
c                       approximation to the integral j
c
c              resasc - double precision
c                       approximation to the integral of abs(f-i/(b-a))
c                       over (a,b)
c
c***references  (none)
c***routines called  d1mach
c***end prologue  dqk15
c
      double precision :: f, a,absc,abserr,b,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc,
     *  resg,resk0,resk,reskh,result,uflow,wg,wgk0,wgk,xgk
      integer j,jtw,jtwm1
      external f
c
      dimension fv1(7),fv2(7),wg(2),wgk0(3),wgk(5),xgk(5)
c
c           the abscissae and weights are given for the interval (-1,1).
c           because of symmetry only the positive abscissae and their
c           corresponding weights are given.
c
c           xgk    - abscissae of the 9-point Gauss-kronrod-lobatto rule
!                    xgk(1), xgk(5)  abscissae of the 3-point gauss-lobatto rule
c                    xgk(1), xgk(3),xgk(5)  abscissae of the 5-point
c                    kronrod rule
c                    xgk(2), xgk(4), ...  abscissae which are optimally
c                    added to the 5-point kronrod rule
c
c           wgk    - weights of the 9-point kronrod rule
!
!           wgk0   - weights of the 5-point kronrod rule
c
c           wg     - weights of the 3-point gauss rule
c
c
c gauss quadrature weights and kronrod quadrature abscissae and weights
c as evaluated in quadruple precision  by Patterson
c

      data wg  (  1) /  0.33333333333333333333333333333333333D+00/
      data wg  (  2) /  0.13333333333333333333333333333333333D+01/

      data wgk0  (  1) /  0.1000000000000000D+00/
      data wgk0  (  2) /  0.5444444444444445D+00/
      data wgk0  (  3) /  0.7111111111111111D+00/

      data xgk (  1) / 0.1000000000000000D+01/
      data xgk (  2) / 0.8904055275126688D+00/
      data xgk (  3) / 0.6546536707079772D+00/
      data xgk (  4) / 0.3409822659109930D+00/
      data xgk (  5) / 0.000000000000000000000000000000000d0 /

      data wgk (  1) / 0.3064373897707232D-01/
      data wgk (  2) / 0.1792626995532074D+00/
      data wgk (  3) / 0.2839787780481211D+00/
      data wgk (  4) / 0.3342337398164177D+00/
      data wgk (  5) / 0.3437620872103631D+00/

c
c
c           list of major variables
c           -----------------------
c
c           centr  - mid point of the interval
c           hlgth  - half-length of the interval
c           absc   - abscissa
c           fval*  - function value
c           resg   - result of the 10-point gauss formula
c           resk   - result of the 21-point kronrod formula
c           reskh  - approximation to the mean value of f over (a,b),
c                    i.e. to i/(b-a)
c
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  dqk15
      epmach = d1mach(4)
      uflow = d1mach(1)
c
      centr = 0.5d+00*(a+b)
      hlgth = 0.5d+00*(b-a)
      dhlgth = dabs(hlgth)
c
c           compute the 15-point kronrod approximation to
c           the integral, and estimate the absolute error.
c
      fc = f(centr)
      resk = wgk(5)*fc
      resk0 =  wgk0(3)*fc
      resabs = dabs(resk)
      do 10 j=1,2
        jtw = 2*j - 1
        absc  = hlgth * xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum   = fval1 + fval2
        resk0  = resk0 + wgk0(j) * fsum
        resk   = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(dabs(fval1)+dabs(fval2))
   10 continue
      resg  =  wg(2)*fc + wg(1)*(fv1(1) + fv2(1))
      do 15 j = 1,2
        jtwm1 = 2*j
        absc  = hlgth * xgk(jtwm1)
        fval1 = f( centr - absc )
        fval2 = f( centr + absc )
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum   = fval1  + fval2
        resk   = resk   + wgk(jtwm1) * fsum
        resabs = resabs + wgk(jtwm1) * (dabs(fval1) + dabs(fval2))
   15 continue

      reskh = resk*0.5d+00
      resasc = wgk(5)*dabs(fc-reskh)
      do 20 j=1,4
        resasc = resasc+wgk(j)*(dabs(fv1(j)-reskh)+dabs(fv2(j)-reskh))
   20 continue
      resg   = resg   * hlgth
      resk0   = resk0   * hlgth
      resk   = resk   * hlgth
      resabs = resabs * dhlgth
      resasc = resasc * dhlgth
      result = resk
      call dea3(resg,resk0,resk,abserr,result)
      abserr = max((dabs(resk-resk0) + dabs(resg-resk0))* 10.0D0,abserr)

      if(resasc.ne.0.0d+00.and.abserr.ne.0.0d+00) then
         abserr = resasc * dmin1(0.1d+01,
     &        (0.2d+03*abserr/resasc)**1.5d+00)
      endif
      if(resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1
     *  ((epmach*0.5d+02)*resabs,abserr)
      return
      end subroutine dqkl9
      subroutine dqpsrt(limit,last,maxerr,ermax,elist,iord,nrmax)
      implicit none
c***begin prologue  dqpsrt
c***refer to  dqage,dqagie,dqagpe,dqawse
c***routines called  (none)
c***revision date  810101   (yymmdd)
c***keywords  sequential sorting
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  this routine maintains the descending ordering in the
c            list of the local error estimated resulting from the
c            interval subdivision process. at each call two error
c            estimates are inserted using the sequential search
c            method, top-down for the largest error estimate and
c            bottom-up for the smallest error estimate.
c***description
c
c           ordering routine
c           standard fortran subroutine
c           double precision version
c
c           parameters (meaning at output)
c              limit  - integer
c                       maximum number of error estimates the list
c                       can contain
c
c              last   - integer
c                       number of error estimates currently in the list
c
c              maxerr - integer
c                       maxerr points to the nrmax-th largest error
c                       estimate currently in the list
c
c              ermax  - double precision
c                       nrmax-th largest error estimate
c                       ermax = elist(maxerr)
c
c              elist  - double precision
c                       vector of dimension last containing
c                       the error estimates
c
c              iord   - integer
c                       vector of dimension last, the first k elements
c                       of which contain pointers to the error
c                       estimates, such that
c                       elist(iord(1)),...,  elist(iord(k))
c                       form a decreasing sequence, with
c                       k = last if last.le.(limit/2+2), and
c                       k = limit+1-last otherwise
c
c              nrmax  - integer
c                       maxerr = iord(nrmax)
c
c***end prologue  dqpsrt
c
      double precision elist,ermax,errmax,errmin
      integer i,ibeg,ido,iord,isucc,j,jbnd,jupbn,k,last,limit,maxerr,
     *  nrmax
      dimension elist(last),iord(last)
c
c           check whether the list contains more than
c           two error estimates.
c
c***first executable statement  dqpsrt
      if(last.gt.2) go to 10
      iord(1) = 1
      iord(2) = 2
      go to 90
c
c           this part of the routine is only executed if, due to a
c           difficult integrand, subdivision increased the error
c           estimate. in the normal case the insert procedure should
c           start after the nrmax-th largest error estimate.
c
   10 errmax = elist(maxerr)
      if(nrmax.eq.1) go to 30
      ido = nrmax-1
      do 20 i = 1,ido
        isucc = iord(nrmax-1)
c ***jump out of do-loop
        if(errmax.le.elist(isucc)) go to 30
        iord(nrmax) = isucc
        nrmax = nrmax-1
   20    continue
c
c           compute the number of elements in the list to be maintained
c           in descending order. this number depends on the number of
c           subdivisions still allowed.
c
   30 jupbn = last
      if(last.gt.(limit/2+2)) jupbn = limit+3-last
      errmin = elist(last)
c
c           insert errmax by traversing the list top-down,
c           starting comparison from the element elist(iord(nrmax+1)).
c
      jbnd = jupbn-1
      ibeg = nrmax+1
      if(ibeg.gt.jbnd) go to 50
      do 40 i=ibeg,jbnd
        isucc = iord(i)
c ***jump out of do-loop
        if(errmax.ge.elist(isucc)) go to 60
        iord(i-1) = isucc
   40 continue
   50 iord(jbnd) = maxerr
      iord(jupbn) = last
      go to 90
c
c           insert errmin by traversing the list bottom-up.
c
   60 iord(i-1) = maxerr
      k = jbnd
      do 70 j=i,jbnd
        isucc = iord(k)
c ***jump out of do-loop
        if(errmin.lt.elist(isucc)) go to 80
        iord(k+1) = isucc
        k = k-1
   70 continue
      iord(i) = last
      go to 90
   80 iord(k+1) = last
c
c           set maxerr and ermax.
c
   90 maxerr = iord(nrmax)
      ermax = elist(maxerr)
      return
      end subroutine dqpsrt
      subroutine dqelg(n,epstab,result,abserr,res3la,nres)
      implicit none
c***begin prologue  dqelg
c***refer to  dqagie,dqagoe,dqagpe,dqagse
c***routines called  d1mach
c***revision date  830518   (yymmdd)
c***keywords  epsilon algorithm, convergence acceleration,
c             extrapolation
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math & progr. div. - k.u.leuven
c***purpose  the routine determines the limit of a given sequence of
c            approximations, by means of the epsilon algorithm of
c            p.wynn. an estimate of the absolute error is also given.
c            the condensed epsilon table is computed. only those
c            elements needed for the computation of the next diagonal
c            are preserved.
c***description
c
c           epsilon algorithm
c           standard fortran subroutine
c           double precision version
c
c           parameters
c              n      - integer
c                       epstab(n) contains the new element in the
c                       first column of the epsilon table.
c
c              epstab - double precision
c                       vector of dimension 52 containing the elements
c                       of the two lower diagonals of the triangular
c                       epsilon table. the elements are numbered
c                       starting at the right-hand corner of the
c                       triangle.
c
c              result - double precision
c                       resulting approximation to the integral
c
c              abserr - double precision
c                       estimate of the absolute error computed from
c                       result and the 3 previous results
c
c              res3la - double precision
c                       vector of dimension 3 containing the last 3
c                       results
c
c              nres   - integer
c                       number of calls to the routine
c                       (should be zero at first call)
c
c***end prologue  dqelg
c
      double precision abserr,dabs,delta1,delta2,delta3,dmax1,
     *  epmach,epsinf,epstab,error,err1,err2,err3,e0,e1,e1abs,e2,e3,
     *  oflow,res,result,res3la,ss,tol1,tol2,tol3
      integer i,ib,ib2,ie,indx,k1,k2,k3,limexp,n,newelm,nres,num
      dimension epstab(52),res3la(3)
c
c           list of major variables
c           -----------------------
c
c           e0     - the 4 elements on which the computation of a new
c           e1       element in the epsilon table is based
c           e2
c           e3                 e0
c                        e3    e1    new
c                              e2
c           newelm - number of elements to be computed in the new
c                    diagonal
c           error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
c           result - the element in the new diagonal with least value
c                    of error
c
c           machine dependent constants
c           ---------------------------
c
c           epmach is the largest relative spacing.
c           oflow is the largest positive magnitude.
c           limexp is the maximum number of elements the epsilon
c           table can contain. if this number is reached, the upper
c           diagonal of the epsilon table is deleted.
c
c***first executable statement  dqelg
      epmach = d1mach(4)
      oflow  = d1mach(2)
      nres   = nres+1
      abserr = oflow
      result = epstab(n)
      if(n.lt.3) go to 100
      limexp = 50
      epstab(n+2) = epstab(n)
      newelm = (n-1)/2
      epstab(n) = oflow
      num = n
      k1 = n
      do 40 i = 1,newelm
        k2 = k1-1
        k3 = k1-2
        res = epstab(k1+2)
        e0 = epstab(k3)
        e1 = epstab(k2)
        e2 = res
        e1abs = dabs(e1)
        delta2 = e2-e1
        err2 = dabs(delta2)
        tol2 = dmax1(dabs(e2),e1abs)*epmach
        delta3 = e1-e0
        err3 = dabs(delta3)
        tol3 = dmax1(e1abs,dabs(e0))*epmach
        if(err2.gt.tol2.or.err3.gt.tol3) go to 10
c
c           if e0, e1 and e2 are equal to within machine
c           accuracy, convergence is assumed.
c           result = e2
c           abserr = abs(e1-e0)+abs(e2-e1)
c
        result = res
        abserr = err2+err3
c ***jump out of do-loop
        go to 100
   10   e3 = epstab(k1)
        epstab(k1) = e1
        delta1 = e1-e3
        err1 = dabs(delta1)
        tol1 = dmax1(e1abs,dabs(e3))*epmach
c
c           if two elements are very close to each other, omit
c           a part of the table by adjusting the value of n
c
        if(err1.le.tol1.or.err2.le.tol2.or.err3.le.tol3) go to 20
        ss = 0.1d+01/delta1+0.1d+01/delta2-0.1d+01/delta3
        epsinf = dabs(ss*e1)
c
c           test to detect irregular behaviour in the table, and
c           eventually omit a part of the table adjusting the value
c           of n.
c
        if(epsinf.gt.0.1d-03) go to 30
   20   n = i+i-1
c ***jump out of do-loop
        go to 50
c
c           compute a new element and eventually adjust
c           the value of result.
c
   30   res = e1+0.1d+01/ss
        epstab(k1) = res
        k1 = k1-2
        error = err2+dabs(res-e2)+err3
        if(error.gt.abserr) go to 40
        abserr = error
        result = res
   40 continue
c
c           shift the table.
c
   50 if(n.eq.limexp) n = 2*(limexp/2)-1
      ib = 1
      if((num/2)*2.eq.num) ib = 2
      ie = newelm+1
      do 60 i=1,ie
        ib2 = ib+2
        epstab(ib) = epstab(ib2)
        ib = ib2
   60 continue
      if(num.eq.n) go to 80
      indx = num-n+1
      do 70 i = 1,n
        epstab(i)= epstab(indx)
        indx = indx+1
   70 continue
   80 if(nres.ge.4) go to 90
      res3la(nres) = result
      abserr = oflow
      go to 100
c
c           compute error estimate
c
   90 abserr = dabs(result-res3la(3))+dabs(result-res3la(2))
     *  +dabs(result-res3la(1))
      res3la(1) = res3la(2)
      res3la(2) = res3la(3)
      res3la(3) = result
  100 abserr = dmax1(abserr,0.5d+01*epmach*dabs(result))
      return
      end subroutine dqelg
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
      end module AdaptiveGaussKronrod

      module Integration1DModule
      implicit none
      interface AdaptiveSimpson
      module procedure AdaptiveSimpson2, AdaptiveSimpsonWithBreaks
      end interface

!      interface AdaptiveSimpson1
!      module procedure AdaptiveSimpson1
!      end interface

      interface AdaptiveTrapz
      module procedure AdaptiveTrapz1, AdaptiveTrapzWithBreaks
      end interface

      interface Romberg
      module procedure Romberg1, RombergWithBreaks
      end interface

      INTERFACE DEA
      MODULE PROCEDURE DEA
      END INTERFACE

      INTERFACE d1mach
      MODULE PROCEDURE d1mach
      END INTERFACE
      contains
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
      subroutine dea3(E0,E1,E2,abserr,result)
!***PURPOSE  Given a slowly convergent sequence, this routine attempts
!            to extrapolate nonlinearly to a better estimate of the
!            sequence's limiting value, thus improving the rate of
!            convergence. Routine is based on the epsilon algorithm
!            of P. Wynn. An estimate of the absolute error is also
!            given.
      double precision, intent(in) :: E0,E1,E2
      double precision, intent(out) :: abserr, result
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
         result = E2
         abserr = err1 + err2 + E2*small*ten
      else
         ss = one/delta2 - one/delta1
         if (abs(ss*E1) <= 1.0d-3) then
            result = E2
            abserr = err1 + err2 + E2*small*ten
         else
            result = E1 + one/ss
            abserr = err1 + err2 + abs(result-E2)
         endif
      endif
      end subroutine dea3
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
      double precision, dimension(*), intent(inout) :: EPSTAB
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

      subroutine AdaptiveIntWithBreaks(f,a,b,N,brks,epsi,iflg
     $     ,abserr, val)
      use AdaptiveGaussKronrod
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
!            call AdaptiveSimpson3(f,pts(k),pts(k+1),tol,kflg,error,valk)
            call  dqagp(f,pts(k),pts(k+1),0,pts,tol,0.0D0,limit,valk,
     *           error,neval,kflg)

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
      double precision, parameter :: zpz66666 = 0.06666666666666666666D0!1/15
      double precision, parameter :: zpz588   = 0.05882352941176D0 !1/17
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
      logical :: NEWFLG !, useDEA
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
!       v(:,k)  = [fx1,fx2,fx3,fx4,fx5,x1,h,S,SL,SR]
        kp1 = k + 1;
!    Process right interval
        v(1,kp1)    = v(3,k); !fx1R
        v(2,kp1)    = fx(3);  !fx2R
        v(3,kp1)    = v(4,k); !fx3R
        v(4,kp1)    = fx(4);  !fx4R
        v(5,kp1)    = v(5,k); !fx5R
        v(6,kp1)    = v(6,k) + four * h; ! x1R
        v(7,kp1)    = h;
        v(8,kp1)    = v(10,k); ! S
        v(9:10,kp1) = Sn(3:4); ! SL, SR
!    Process left interval
        v(5,k)    = v(3,k); ! fx5L
        v(4,k)    = fx(2);  ! fx4L
        v(3,k)    = v(2,k); ! fx3L
        v(2,k)    = fx(1);  ! fx2L
!        v(1,k)  unchanged     fx1L
!        v(6,k)  unchanged      x1L
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
      double precision, parameter :: zp125    = 0.125D0 ! 1/8
      double precision, parameter :: zpz66666 = 0.06666666666666666666D0!1/15
      double precision, parameter :: zpz588   = 0.05882352941176D0 !1/17
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
      logical :: NEWFLG !, useDEA
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
      integer,          intent(in)  :: decdigs  ! Relative number of decimal digits
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
      end module Integration1DModule

      module mvnProdCorrPrbMod
      implicit none
      private
      public :: mvnprodcorrprb
      double precision, parameter :: mINFINITY = 8.25D0 !
      ! Inputs to integrand
      integer mNdim ! # of mRho/=0 and mRho/=+/-1 and -inf<a or b<inf
      double precision, allocatable, dimension(:) ::  mRho, mDen
      double precision, allocatable, dimension(:) ::  mA,mB


      INTERFACE mvnprodcorrprb
      MODULE PROCEDURE mvnprodcorrprb
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
      INTERFACE GetBreakPoints
      MODULE PROCEDURE GetBreakPoints
      END INTERFACE

      INTERFACE NarrowLimits
      MODULE PROCEDURE NarrowLimits
      END INTERFACE

      INTERFACE GetTruncationError
      MODULE PROCEDURE GetTruncationError
      END INTERFACE

      INTERFACE integrand
      MODULE PROCEDURE integrand
      END INTERFACE

      INTERFACE integrand1
      MODULE PROCEDURE integrand1
      END INTERFACE
      contains
      SUBROUTINE SORTRE(rarray,indices)
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:), INTENT(inout) :: rarray
      INTEGER,          DIMENSION(:), OPTIONAL, INTENT(inout) :: indices
! local variables
      double precision :: tmpR
      INTEGER  :: i,im,j,k,m,n, tmpI

! diminishing increment sort as described by
! Donald E. Knuth (1973) "The art of computer programming,",
! Vol. 3, pp 84-  (sorting and searching)
      n = size(rarray)
!      if (present(indices)) then
        ! if the below is commented out then assume indices are already initialized
!         forall(i=1,n) indices(i) = i
!      endif
100   continue
      if (n.le.1) goto 800
      m=1
200   continue
      m=m+m
      if (m.lt.n) goto 200
      m=m-1
300   continue
      m=m/2
      if (m.eq.0) goto 800
      k=n-m
      j=1
400   continue
      i=j
500   continue
      im=i+m
      if (rarray(i).gt.rarray(im)) goto 700
600   continue
      j=j+1
      if (j.gt.k) goto 300
      goto 400
700   continue
      tmpR        = rarray(i)
      rarray(i)   = rarray(im)
      rarray(im)  = tmpR
      if (present(indices)) then
         tmpI        = indices(i)
         indices(i)  = indices(im)
         indices(im) = tmpI
      endif
      i=i-m
      if (i.lt.1) goto 600
      goto 500
800   continue
      RETURN
      END SUBROUTINE SORTRE

      subroutine mvnprodcorrprb(rho,a,b,abseps,releps,useBreakPoints,
     &     useSimpson,abserr,errFlg,prb)
      use AdaptiveGaussKronrod
      use Integration1DModule
!      use numerical_libraries
      implicit none
      double precision,dimension(:),intent(in) :: rho,a,b
      double precision,intent(in) :: abseps,releps
      logical,         intent(in) :: useBreakPoints,useSimpson
      double precision,intent(out) :: abserr,prb
      integer, intent(out) :: errFlg
!     Locals
      double precision, parameter :: ZERO    = 0.0D0
      double precision, parameter :: ZPT1    = 0.1D0
      double precision, parameter :: ZPTZ5   = 0.05D0
      double precision, parameter :: ZPTZZ1  = 0.001D0
      double precision, parameter :: ZPTZZZ1 = 0.0001D0
      double precision, parameter :: ONE     = 1.d0
      double precision :: small, LTol, val0,val, truncError
      double precision :: zCutOff, zlo, zup, As, Bs
      double precision, dimension(1000) :: breakPoints
      integer :: n, k , limit, Npts, neval
      logical :: isSingular, isLimitsNarrowed
      small = MAX(spacing(one),1.0D-16)
      isSingular = .FALSE.
      n     = size(a,DIM=1)

      LTol   = max(abseps,small)
      errFlg = 0
      prb    = ZERO
      abserr = small
      if ( any(b(:)<=a(:)).or.
     &     any(b(:)<=-mINFINITY) .or.
     &     any(mINFINITY<=a(:))) then
         goto 999  ! end program
      endif
      As      = - mInfinity
      Bs      =   mInfinity
      zCutOff = abs(max(FIINV(ZPTZ5*LTol),-mINFINITY));
      zlo     = - zCutOff
      zup     =   zCutOff

      allocate(mA(n),mB(n),mRho(n),mDen(n))
      do k = 1,n
        if (one <= abs(rho(k)) ) then
           mRho(k) = sign(one,rho(k))
           mDen(k) = zero
        else
          mRho(k) = rho(k)
          mDen(k) = sqrt(one - rho(k))*sqrt(one + rho(k))
        endif
      end do
!     See if we may narrow down the integration region: zlo, zup
      CALL NarrowLimits(zlo,zup,As,Bs,zCutOff,n,a,b,mRho,mDen)
      if (zup <= zlo) goto 999 ! end program

!     Move only significant variables to mA,mB, and mRho
!     (Note: If you scale them with mDen, the integrand must also be changed)
      mNdim = 0
      val0 = one
      do k = 1, n
         if (small < abs(mRho(k))) then
            if ( ONE <= abs(mRho(k))) then
!               rho(k) == 1
               isSingular = .TRUE.
            elseif ((-mINFINITY < a(k)) .OR. (b(k) < mINFINITY)) then
               mNdim       = mNdim + 1
               mA(mNdim)   =    a(k) / mDen(k)
               mB(mNdim)   =    b(k) / mDen(k)
               mRho(mNdim) = mRho(k) / mDen(k)
               mDen(mNdim) = mDen(k)
            endif
         else  ! independent variables which are evaluated separately
            val0 = val0 * ( FI( b(k) ) - FI( a(k) ) )
         endif
      enddo
      CALL GetTruncationError(zlo, zup, As, Bs, truncError)

      select case(mNdim)
      case (0)
         if (isSingular) then
            prb    = ( FI( zup ) - FI( zlo ) ) * val0
            abserr = sqrt(small) + truncError
         else
            prb    = val0;
            abserr = small+truncError;
         endif
         goto 999 ! end program
      case (1)
         if (.not.isSingular) then
            prb    = (FI(mB(1)*mDen(1))-FI(mA(1)*mDen(1))) * val0
            abserr = small + truncError
            goto 999 ! end program
         endif
      end select
      if (small < val0) then
         isLimitsNarrowed = ((-7.D0 < zlo) .or.  (zup < 7.D0))
         Npts = 0
         if (isLimitsNarrowed.AND.useBreakPoints) then
                                ! Provide abscissas for break points
            CALL GetBreakPoints(zlo,zup,mNdim,mA,mB,mRho,mDen,
     &           breakPoints,Npts)
         endif
         LTol = LTol - truncError
!
         if (useSimpson) then
            call AdaptiveSimpson(integrand,zlo,zup,Npts,breakPoints,LTol
     &           ,errFlg,abserr, val)

         else
            limit = 100
            call dqagp(integrand,zlo,zup,Npts,breakPoints,LTol,zero,
     &           limit,val,abserr,neval,errFlg)
         endif
         prb    = val * val0;
         abserr = (abserr + truncError)* val0;
      else
         prb    = zero
         abserr = small + truncError
      endif

 999  continue
      if (allocated(mDen)) deallocate(mDen)
      if (allocated(mA))   deallocate(mA,mB,mRho)

      return
      end subroutine mvnprodcorrprb

      subroutine GetTruncationError(zlo, zup, As, Bs, truncError)
      double precision, intent(in) :: zlo, zup, As, Bs
      double precision, intent(out) :: truncError
      double precision :: upError,loError
!     Computes the upper bound for the truncation error
      upError    = integrand1(zup) * abs( FI( Bs  ) - FI( zup ) )
      loError    = integrand1(zlo) * abs( FI( zlo ) - FI( As  ) )
      truncError = loError + upError
      end subroutine GetTruncationError

      subroutine GetBreakPoints(xlo,xup,n,a,b,rho,den,
     &     breakPoints,Npts)
      implicit none
      double precision,                 intent(in) :: xlo, xup
      double precision,dimension(:),    intent(in) :: a,b, rho,den
      integer,                          intent(in) ::  n
      double precision,dimension(:), intent(inout) :: breakPoints
      integer,                       intent(inout) :: Npts
!     Locals
      integer, dimension(2*n)          :: indices
      integer, dimension(4*n)          :: indices2
      double precision, dimension(2*n) :: brkPts
      double precision, dimension(4*n) :: brkPtsVal
      double precision, parameter :: zero = 0.0D0, brkSplit = 2.5D0
      double precision, parameter :: stepSize = 0.24
      double precision            :: brk,brk1,hMin,distance, xLow, dx
      double precision :: z1, z2, val1,val2
      integer :: j,k, kL,kU , Nprev, Nk
      hMin = 1.0D-5
      kL = 0
      Npts = 0
      if (.false.) then
         if (xup-xlo>stepSize) then
            Nk = floor((xup-xlo)/stepSize) + 1
            dx = (xup-xlo)/dble(Nk)
            do j=1, Nk -1
               Npts = Npts  + 1
               breakPoints(Npts) = xlo + dx * dble( j )
            enddo
         endif
      else
      ! Compute candidates for the breakpoints
      brkPts(1:2*n) = xup
      forall(k=1:n,rho(k) .ne. zero)
         indices(2*k-1) = k
         indices(2*k  ) = k
         brkPts(2*k-1) = a(k)/rho(k)
         brkPts(2*k  ) = b(k)/rho(k)
      end forall
      ! Sort the candidates
      call sortre(brkPts,indices)
      ! Make unique list of breakpoints

      do k = 1,2*n
         brk =  brkPts(k)
         if (xlo < brk) then
            if ( xup <= brk )  exit ! terminate do loop

!     if (Npts>0) then
!     xLow = max(xlo, breakPoints(Npts))
!     else
!     xLow = xlo
!     endif
!     if (brk-xLow>stepSize) then
!     Nk = floor((brk-xLow)/stepSize)
!     dx = (brk-xLow)/dble(Nk)
!     do j=1, Nk -1
!     Npts = Npts  + 1
!     breakPoints(Npts) = brk + dx * dble( j )
!     enddo
!     endif

            kU = indices(k)

                                !if ( xlo + distance < brk  .and. brk + distance < xup )
                                !then
            if ( den(kU) < 0.2) then
               distance = max(brkSplit*den(kU),hMin)
               z1 = brk + distance
               z2 = brk - distance
               if (Npts <= 0) then
                  if (xlo + distance < z1) then
                     Npts = Npts + 1
                     breakPoints(Npts) = z1
                     brkPtsVal(Npts) = integrand(z1)
                     indices2(Npts) = kU
                  endif
!     Nprev = Nprev + 1
!     breakPoints(Npts + Nprev) = brk
                  if ( z2 + distance < xup) then
                     Npts = Npts + 1
                     breakPoints(Npts) = z2
                     brkPtsVal(Npts)   = integrand(z2)
                     indices2(Npts) = kU
                  endif
                  kL = kU
               elseif (breakPoints(Npts)+ max(distance
     &                 ,brkSplit*den(kL)) < z1) then
                  if (breakPoints(Npts) + distance < z1) then
                     Npts = Npts + 1
                     breakPoints(Npts) = z1
                     brkPtsVal(Npts) = integrand(z1)
                     indices2(Npts) = kU
                     kL = kU
                  endif
!     Nprev = Nprev + 1
!     breakPoints(Npts + Nprev) = brk
                  if ( z2 + distance < xup) then
                     Npts = Npts + 1
                     breakPoints(Npts) = z2
                     brkPtsVal(Npts) = integrand(z2)
                     indices2(Npts) = kU
                     kL = kU
                  endif
               else
                  val1 = 0.0d0
                  val2 = 0.0d0
                  brkPts(Npts+1) = integrand(z1)
                  brkPts(Npts+2) = integrand(z2)
                  if ((xlo+ distance < z1) .and. (z1 + distance < xup))
     &                 val2 = brkPts(Npts +1)
                  if ((xlo+ distance < z2) .and. (z2 + distance < xup))
     &                 val2 = max(val2,brkPts(Npts +2))
                  val1 = breakPoints(Npts)
                  Nprev = 1
                  if (Npts>1) then
                     if (indices2(Npts-1)==kL) then
                        Nprev = 2
                        val1 = max(val1,breakPoints(Npts-1))
                     endif
                  endif
                  if (val1 <  val2) then
                                !overwrite previous candidate
                     Npts  = Npts - Nprev
                     if (Npts>0) then
                        val1 = breakPoints(Npts)+ distance
                     else
                        val1 = xlo+ distance
                     endif
                  if (val1 < z1) then
                     Npts = Npts + 1
                     breakPoints(Npts) = z1
                     brkPtsVal(Npts) = brkPtsVal(Npts+Nprev)
                     indices2(Npts) = kU
                  endif
!     Nprev = Nprev + 1
!     breakPoints(Npts + Nprev) = brk

                  if ((val1< z2) .and. (z2 + distance < xup)) then
                     Npts = Npts + 1
                     breakPoints(Npts) = z2
                     brkPtsVal(Npts) = integrand(z2)
                     indices2(Npts) = kU
                  endif
                  if (Npts>0) kL = indices2(Npts)
                  endif
               endif
            endif
         endif
      enddo
      endif
      end subroutine GetBreakPoints
      subroutine NarrowLimits(zMin,zMax,As,Bs,zCutOff,n,a,b,rho,den)
      implicit none
      double precision, intent(inout) :: zMin, zMax, As, Bs
      double precision,dimension(*),intent(in) :: rho,a,b,den
      double precision, intent(in) :: zCutOff
      integer, intent(in) :: n
!     Locals
      double precision, parameter :: zero = 0.0D0, one = 1.0D0
      integer :: k

!     Uses the regression equation to limit the
!     integration limits zMin and zMax

      do k = 1,n
         if (ZERO < rho(k)) then
            zMax = max(zMin, min(zMax,(b(k)+den(k)*zCutOff)/rho(k)))
            zMin = min(zMax, max(zMin,(a(k)-den(k)*zCutOff)/rho(k)))
            if ( one <= rho(k) ) then
               if ( b(k) < Bs   ) Bs = b(k)
               if ( As   < a(k) ) As = a(k)
            endif
         elseif (rho(k)< ZERO) then
            zMax = max(zMin,min(zMax,(a(k)-den(k)*zCutOff)/rho(k)))
            zMin = min(zMax,max(zMin,(b(k)+den(k)*zCutOff)/rho(k)))
            if ( rho(k) <= -one ) then
               if ( -a(k) <  Bs   ) Bs = -a(k)
               if ( As    < -b(k) ) As = -b(k)
            endif
         endif
      enddo
      As = min(As,Bs)
      end subroutine NarrowLimits

      function integrand(z) result (val)
      implicit none
      DOUBLE PRECISION, INTENT(IN)  :: Z
      DOUBLE PRECISION  :: VAL
      double precision, parameter :: sqtwopi1 =  0.39894228040143D0
      double precision, parameter :: half     = 0.5D0
      val = sqtwopi1 * exp(-half * z * z) * integrand1(z)
      return
      end function integrand

      function integrand1(z) result (val)
      implicit none
      double precision, intent(in) :: z
      double precision             :: val
      double precision             :: xUp,xLo,zRho
      double precision, parameter  :: one = 1.0D0, zero = 0.0D0
      integer :: I
      val = one
      do I = 1, mNdim
         zRho = z * mRho(I)
         ! Uncomment / mDen below if mRho, mA, mB is not scaled
         xUp  = ( mB(I) - zRho )  !/ mDen(I)
         xLo  = ( mA(I) - zRho )  !/ mDen(I)
         if (zero<xLo) then
            val = val * ( FI( -xLo ) - FI( -xUp ) )
         else
            val = val * ( FI( xUp ) - FI( xLo ) )
         endif
      enddo
      end function integrand1
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
     & ONE = 1.D0, ZERO = 0.D0, HALF = 0.5D0 )
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
      IF ( Z .GT. XMAX .OR. ZABS .GT. 37) THEN
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
      USE ERFCOREMOD
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(in) :: Z
      DOUBLE PRECISION :: VALUE
! Local variables
      DOUBLE PRECISION, PARAMETER:: SQ2M1 = 0.70710678118655D0 !     1/SQRT(2)
      DOUBLE PRECISION, PARAMETER:: HALF = 0.5D0
      VALUE = DERFC(-Z*SQ2M1)*HALF
      RETURN
      END FUNCTION FI
      end module mvnProdCorrPrbMod

