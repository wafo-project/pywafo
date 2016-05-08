!****************************************************************************
! if compilation complains about too many continuation lines extend it.
!
!
!  modules:   GLOBALDATA, QUAD, RIND71MOD    Version 1.0
!
! Programs available in module RIND71MOD :
! (NB! the GLOBALDATA and QUAD  module is also used to transport the inputs)
!
!
! SETDATA initializes global constants explicitly:
!
! CALL SETDATA(EPSS,REPS,EPS2,NIT,xCutOff,NINT,XSPLT)
!
!                   GLOBALDATA module :
!   EPSS,CEPSS = 1.d0 - EPSS , controlling the accuracy of indicator function
!        EPS2  = if conditional variance is less it is considered as zero
!                i.e., the variable is considered deterministic
!      xCutOff = 5 (standard deviations by default)
!
!                   QUAD module:
!     Nint1(i) = quadrature formulae used in integration of Xd(i)
!                implicitly determining # nodes
!
! INITDATA initializes global constants implicitly:
!
! CALL INITDATA (speed)
!
!        speed = 1,2,...,9 (1=slowest and most accurate,9=fastest,
!                           but less accurate)
!
! see the GLOBALDATA and QUAD module for other constants and default values
!
!
!RIND71 computes  E[Jacobian*Indicator|Condition]*f_{Xc}(xc(:,ix))
!
! where
!     "Indicator" = I{ H_lo(i) < X(i) < H_up(i), i=1:Nt+Nd }
!     "Jacobian"  = J(X(Nt+1),...,X(Nt+Nd+Nc)), special case is
!     "Jacobian"  = |X(Nt+1)*...*X(Nt+Nd)|=|Xd(1)*Xd(2)..Xd(Nd)|
!     "condition" = Xc=xc(:,ix),  ix=1,...,Nx.
!     X = [Xt; Xd ;Xc], a stochastic vector of Multivariate Gaussian
!         variables where Xt,Xd and Xc have the length Nt, Nd and Nc,
!         respectively.
!         (Recommended limitations Nx, Nt<101, Nd<7 and NIT,Nc<11)
! (RIND = Random Integration N Dimensions)
!
!CALL RIND71(E,S,m,xc,indI,Blo,Bup,xcScale);
!
!        E = expectation/density as explained above size 1 x Nx        (out)
!        S = Covariance matrix of X=[Xt;Xd;Xc] size N x N (N=Nt+Nd+Nc) (inout)
!            NB!: out=conditional sorted Covariance matrix
!        m = the expectation of X=[Xt;Xd;Xc]   size N x 1              (in)
!       xc = values to condition on            size Nc x Nx            (in)
!     indI = vector of indices to the different barriers in the        (in)
!            indicator function,  length NI, where   NI = Nb+1
!            (NB! restriction  indI(1)=0, indI(NI)=Nt+Nd )
!Blo,Bup = Lower and upper barrier coefficients used to compute the  (in)
!            integration limits Hlo and Hup, respectively.
!            size  Mb x Nb. If  Mb<Nc+1 then
!            Blo(Mb+1:Nc+1,:) is assumed to be zero. The relation
!            to the integration limits Hlo and Hup are as follows
!
!              Hlo(i)=Blo(1,j)+Blo(2:Mb,j).'*xc(1:Mb-1,ix),
!              Hup(i)=Bup(1,j)+Bup(2:Mb,j).'*xc(1:Mb-1,ix),
!
!            where i=indI(j-1)+1:indI(j), j=2:NI, ix=1:Nx
!            Thus the integration limits may change with the conditional
!            variables See example below.
! xcScale = REAL to scale the conditinal probability density, i.e.,
!             f_{Xc} = exp(-0.5*Xc*inv(Sxc)*Xc + XcScale) (Optional, default XcScale =0)
!
!Example:
! The indices, indI=[0 3 5], and coefficients Blo=[-inf 0], Bup=[0  inf]
!  gives   Hlo = [-inf -inf -inf 0 0]  Hup = [0 0 0 inf inf]
!
! The GLOBALDATA and QUAD  modules are used to transport the inputs:
!     SCIS = 0 Integrate by Gauss-Legendre quadrature (default) (Podgorski et al. 1999)
!            1 Integrate by SADAPT for Ndim<9 and by KRBVRC otherwise
!            2 Integrate by SADAPT for Ndim<19 and by KRBVRC otherwise
!            3 Integrate by KRBVRC by Genz (1993) (Fast Ndim<101)
!            4 Integrate by KROBOV by Genz (1992) (Fast Ndim<101)
!            5 Integrate by RCRUDE by Genz (1992) (Slow Ndim<1001)
!            6 Integrate by SOBNIED               (Fast Ndim<1041)
!            7 Integrate by DKBVRC by Genz (2003) (Fast Ndim<1001)
!      NIT = 0,1,2..., maximum # of iterations/integrations done by quadrature
!            to calculate the indicator function (default NIT=2)
!  NB!  the size information below must be set before calling RINDD
!       Nx = # different xc
!       Nt = length of Xt
!       Nd = length of Xd
!       Nc = length of Xc
!      Ntd = Nt+Nd
!     Ntdc = Nt+Nd+Nc
!       Mb
!       NI
!       Nj = # of variables in indicator integrated directly like the
!            variables in the jacobian (default 0)
!            The order of integration between Xd and Nj of  Xt is done in
!            decreasing order of conditional variance.
!      Njj = # of variables in indicator integrated directly like the
!            variables in the jacobian (default 0)
!            The Njj variables of Xt is integrated after Xd and Nj of Xt
!            also in decreasing order of conditional variance. (Not implemented yet)
!
! (Recommended limitations Nx,Nt<101, Nd<7 and NIT,Nc<11)
!
! if SCIS > 0 then you must initialize the random generator before you
!  call rindd by the following lines:
!
!      call random_seed(SIZE=seed_size)
!      allocate(seed(seed_size))
!      call random_seed(GET=seed(1:seed_size))  ! get current seed
!      seed(1)=seed1                            ! change seed
!      call random_seed(PUT=seed(1:seed_size))
!      deallocate(seed)
!
! For further description see the modules
!
!
! References
! Podgorski et. al. (1999)
! "Exact distributions for apparent waves in irregular seas"
! Ocean Engineering                                                    (RINDXXX)
!
! R. Ambartzumian, A. Der Kiureghian, V. Ohanian and H.
! Sukiasian (1998)
! "Multinormal probabilities by sequential conditioned
!  importance sampling: theory and application"             (RINDSCIS, MNORMPRB,MVNFUN,MVNFUN2)
! Probabilistic Engineering Mechanics, Vol. 13, No 4. pp 299-308
!
! Alan Genz (1992)
! 'Numerical Computation of Multivariate Normal Probabilites'
! J. computational Graphical Statistics, Vol.1, pp 141--149
!
! William H. Press, Saul Teukolsky,
! William T. Wetterling and Brian P. Flannery (1997)
! "Numerical recipes in Fortran 77", Vol. 1, pp 55-63, 299--305  (SVDCMP,SOBSEQ)
!
! Igor Rychlik and Georg Lindgren (1993)
! "Crossreg - A technique for first passage and wave density analysis" (RINDXXX)
! Probability in the Engineering and informational Sciences,
! Vol 7, pp 125--148
!
! Igor Rychlik (1992)
! "Confidence bands for linear regressions"                      (RIND2,RINDNIT)
! Commun. Statist. -simula., Vol 21,No 2, pp 333--352
!
!
! Donald E. Knuth (1973) "The art of computer programming,",
! Vol. 3, pp 84-  (sorting and searching)                               (SORTRE)

! Tested on:  DIGITAL UNIX Fortran90 compiler
!             PC pentium II with Lahey Fortran90 compiler
!             Solaris with SunSoft F90 compiler Version 1.0.1.0  (21229283)
! History:
! revised pab aug 2009
! -moved c1c2 to c1c2mod
! -removed rateLHD, useMIDP, FxCutOff, CFxCutOff from globaldata module
! revised pab July 2007
! -reordered integration methods (SCIS)
! revised pab 9 may 2004
! removed xcutoff2
! introduced XcScale to rindd
! revised pab 17.02.2003
! -new name rind71
! commented out all print statements
! revised pab 08.02.2001
! - New name rind70.f
! - moved the jacob function to a separate module.
! - jacobdef in module GLOBALDATA is now obsolete.
! revised pab 19.01.2001
! - added a NEW BVU function
! revised pab 06.11.2000
!  - added checks in condsort2, condsort3, condsort4 telling if the matrix is
!    negative definit
!  - changed the order of SCIS integration again.
! revised pab 07.09.2000
!   - To many continuation lines in QUAD module =>
!     broke them up and changed PARAMETER statements into DATA
!     statements instead.
! revised pab 22.05.2000
!  - changed order of SCIS integration: moved the less important SCIS
! revised pab 19.04.2000
!   - found a bug in THL when L<-1, now fixed
! revised pab 18.04.2000
!  new name rind60
!  New assumption of BIG for the conditional sorted variables:
!                         BIG(I,I)=sqrt(Var(X(I)|X(I+1)...X(N))=SQI
!                         BIG(1:I-1,I)=COV(X(1:I-1),X(I)|X(I+1)...X(N))/SQI
!      Otherwise
!                         BIG(I,I) = Var(X(I)|X(I+1)...X(N)
!                         BIG(1:I-1,I)=COV(X(1:I-1),X(I)|X(I+1)...X(N))
!  This also affects C1C2: SQ0=sqrt(Var(X(I)|X(I+1)...X(N)) is removed from input
!  =>  A lot of wasteful divisions are avoided
! revised pab 23.03.2000
!  - done some optimization in initdata
!  - added some things in THL + optimized THL
!  - fixed a bug in condsort and condsort0 when Nd+Nj=0
! revised pab 20.03.2000
!  - new name rind57
!  - added condsort0 and condsort4 which sort the covariance matrix using the shortest
!    expected integration interval => integration time is much shorter for all methods.
!    condsort and condsort3 sort by decreasing conditional variance
! revised pab 17.03.2000
!  - changed argp0 so that I0 and I1 really are the indices to the minimum and the second minimum
!  - changed rindnit so that norm2dprb is called whenever NITL<1 and Nsnew>=2
!  - changed default parameters for initdata for speed=7,8 and 9 to increase accuracy.
!  - Changed so that  xCutOff varies with speed => program is much faster without loosing any accuracy it seems
! revised pab 15.03.2000
!  - changed rindscis and mnormprb: moved the actual multidimensional integration
!              into separate module, rcrudemod.f (as a consequence SVDCMP,PYTHAG and SORTRE
!              are also moved into this module) => made the structure of the program simpler
!  - added the possibility to use adapt, krbvrc, krobov and ranmc to integrate
!  - Set NUGGET to 0 when Nc=0, since it is no longer needed
!  - added the module MVNFUNDATA
! revised pab 03.03.2000
!      - BIG are no longer changed when called by RINDD instead it is copied into a new variable
!      - new name rind55.f
!      - fixed the bug in THL, i.e. THL forgot to return a value in some cases giving floating invalid
!revised by I.R. 27.01.2000, Removed bugs in RINDNIT (There where some returns
!           without deallocating some variables. A misco error in THL, leading
!           to floating invalid on alpha has been repaired by seting value=zero.
!           Probably there is an error somehere making variable "value" to behave badly.
!Revised by IR. 03.01.2000 Bug in C1C2 fixed and deallocation of ind in RINDNIT.
!revised by I.R. 27.12.1999, New name RIND51.f
!          I have changed assumption about deterministic variables. Those have now
!          variances equal EPS2 not zero and have consequences for C1C2 and on some
!          places in RINDND. The effect is that barriers becomes fuzzy (not sharp)
!          and prevents for discountinuities due to numerical errors of order 1E-16.
!          The program RIND0 is removed making the structure of program simpler.
!          We have still a problem when variables in indicator become
!          deterministic before conditioning on derivatives in Xd - it needs to be solved.
!revised by  Igor Rychlik 01.12.1999  New name RIND49.f
!       - changed RINDNIT and ARGP0 in order to exclude
!         irrelevant variables (such that probability of beeing
!         between barriers is 1.) All computations related to NIT
!         are moved to RINDNIT (removing RIND2,RIND3). This caused some changes
!         in RIND0,RINDDND. Furthermore RINDD1 is removed and moved
!         some parts of it to RINDDND. This made program few seconds slower. The lower
!         bound in older ARGP0 programs contained logical error - corrected.
!revised by Per A. Brodtkorb 08.11.1999
!       - fixed a bug in rinddnd
!          new line: CmNew(Nst+1:Nsd-1)= Cm(Nst+1:Nsd-1)
!revised by Per A. Brodtkorb 28.10.1999
!       - fixed a bug in rinddnd
!       - changed rindscis, mnormprb
!       - added MVNFUN, MVNFUN2
!       - replaced CVaccept with RelEps
!revised by Per A. Brodtkorb 27.10.1999
!       - changed NINT to NINT1 due to naming conflict with an intrinsic of the same name
!revised by Per A. Brodtkorb 25.10.1999
!       - added an alternative FIINV for use in rindscis and mnormprb
!revised by Per A. Brodtkorb 13.10.1999
!       - added useMIDP for use in rindscis and mnormprb
!
!revised by Per A. Brodtkorb 22.09.1999
!       - removed all underscore letters due to
!         problems with  SunSoft F90 compiler
!         (i.e. changed  GLOBAL_DATA to GLOBALDATA etc.)
!revised by Per A. Brodtkorb 09.09.1999
!       - added sobseq: Sobol sequence (quasi random numbers)
!              an alternative to random_number in RINDSCIS and mnormprb
!revised by Per A. Brodtkorb 07.09.1999
!       - added pythag,svdcmp,sortre
!       - added RINDSCIS: evaluating multinormal integrals by SCIS
!              condsort3: prepares BIG for use with RINDSCIS and mnormprb
!revised by Per A. Brodtkorb 03.09.1999
!       - added mnormprb: evaluating multinormal probabilities by SCIS
!            See globaldata for SCIS
! revised by Per A. Brodtkorb 01.09.1999
!       - increased the default NUGGET from 1.d-12 to 1.d-8
!       - also set NUGGET depending on speed in INITDATA
! revised by Per A. Brodtkorb 27.08.1999
!       - changed rindnit,rind2:
!         enabled option to do the integration faster/(smarter?).
!         See GLOBALDATA for XSPLT
! revised by Per A. Brodtkorb 17.08.1999
!       - added THL, norm2dprb not taken in to use
!         due to some mysterious floating invalid
!         occuring from time to time in norm2dprb (on DIGITAL unix)
! revised by Per A. Brodtkorb 02.08.1999
!       - updated condsort
!       - enabled the use of C1C2 in rinddnd
! revised by Per A. Brodtkorb 14.05.1999
!       - updated to fortran90
!       - enabled recursive calls
!       - No limitations on size of the inputs
!       - fixed some bugs
!       - added some additonal checks
!       - added Hermite, Laguerre quadratures for alternative integration
!       - rewritten CONDSORT, conditional covariance matrix in upper
!         triangular.
!       - RINDXXX routines only work on the upper triangular
!         of the covariance matrix
!       - Added a Nugget effect to the covariance matrix in order
!         to ensure the conditioning is not corrupted by numerical errors
!       - added the option to condsort Nj variables of Xt, i.e.,
!         enabling direct integration like the integration of Xd
! by  Igor Rychlik 29.10.1998 (PROGRAM RIND11 --- Version 1.0)
!         which was a revision of program RIND from 3.9.1993 - the program that
!         is used in wave_t and wave_t2 programs.

!*********************************************************************

      MODULE GLOBALDATA
      IMPLICIT NONE
                      ! Constants determining accuracy of integration
                      !-----------------------------------------------
                      !if the conditional variance are less than:
      DOUBLE PRECISION :: EPS2=1.d-4    !- EPS2, the variable is
                                        !  considered deterministic
      DOUBLE PRECISION :: EPS = 1.d-2   ! SQRT(EPS2)
      DOUBLE PRECISION :: XCEPS2=1.d-16 ! if Var(Xc) is less return NaN
      DOUBLE PRECISION :: EPSS = 5.d-5  ! accuracy of Indicator
      DOUBLE PRECISION :: CEPSS=0.99995 ! accuracy of Indicator
      DOUBLE PRECISION :: EPS0 = 5.d-5 ! used in GAUSSLE1 to implicitly
                                       ! determ. # nodes
      DOUBLE PRECISION :: xcScale=0.d0
      DOUBLE PRECISION :: fxcEpss=1.d-20 ! if less do not compute E(...|Xc)
      DOUBLE PRECISION :: xCutOff=5.d0  ! upper/lower truncation limit of the
                                       ! normal CDF
                                ! Nugget>0: Adds a small value to diagonal
                                ! elements of the covariance matrix to ensure
                                ! that the inversion is  not corrupted by
                                ! round off errors.
                                ! Good choice might be 1e-8
      DOUBLE PRECISION :: NUGGET=1.d-8 ! Obs NUGGET must be smaller then EPS2

!parameters controlling the performance of RINDSCIS and MNORMPRB:
      INTEGER :: SCIS=0         !=0 integr. all by quadrature
                                !=1 Integrate all by SADAPT for Ndim<9 and by KRBVRC otherwise
                                !=2 Integrate all by SADAPT for Ndim<9 and by KROBOV otherwise
                                !=3 Integrate all by KRBVRC  (Fast and reliable)
                                !=4 Integrate all by KROBOV  (Fast and reliable)
                                !=5 Integrate all by RCRUDE  (Reliable)
                                !=6 Integrate all by SOBNIED (NDIM<1041)
                                !=7 Integrate all by DKBVRC (Ndim<1001)
      INTEGER :: NSIMmax = 1000 ! maximum number of simulations per stochastic dimension
      INTEGER :: NSIMmin = 10   ! minimum number of simulations per stochastic dimension
      INTEGER :: Ntscis  = 0    ! Ntscis=Nt-Nj-Njj when SCIS>0 otherwise Ntscis=0
      DOUBLE PRECISION :: RelEps = 0.001 ! Relative error, i.e. if
                                ! 3.0*STD(XIND)/XIND is less we accept the estimate
                                ! The following may be allocated outside RINDD
                                ! if one wants the coefficient of variation, i.e.
                                ! STDEV(XIND)/XIND when SCIS=2.  (NB: size Nx)
      DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: COV
      integer :: COVix ! counting variable for COV
      LOGICAL,PARAMETER :: useC1C2=.true. ! use C1C2 in rindscis,mnormprb
      LOGICAL,PARAMETER :: C1C2det=.true. ! use C1C2 only on the variables that becomes
                                ! deterministic after conditioning on X(N)
                                ! used in rinddnd rindd1 and rindscis mnormprb

!parameters controlling performance of quadrature integration:
                ! if Hup>=xCutOff AND Hlo<-XSPLT OR
                !    Hup>=XSPLT AND Hl0<=-xCutOff then
                !  do a different integration to increase speed
                ! in rind2 and rindnit. This give slightly different
                ! results
                ! DEFAULT 5 =xCutOff => do the same integration allways
                ! However, a resonable value is XSPLT=1.5
      DOUBLE PRECISION :: XSPLT = 5.d0 ! DEFAULT XSPLT= 5 =xCutOff
          ! weight between upper&lower limit returned by ARGP0
      DOUBLE PRECISION, PARAMETER :: Plowgth=0.d0 ! 0 => no weight to
                                !      lower limit
      INTEGER :: NIT=2          ! NIT=maximum # of iterations/integrations by
                                ! quadrature used to calculate the indicator function

                                ! size information of the covariance matrix BIG
                                ! Nt,Nd,....Ntd,Nx must be set before calling
                                ! RINDD.  NsXtmj, NsXdj is set in RINDD
      INTEGER :: Nt,Nd,Nc,Ntdc,Ntd,Nx
                                ! Constants determines how integration is done
      INTEGER :: Nj=0,Njj=0  ! Njj is not implemented yet
                                ! size information of indI, Blo,Bup
                                ! Blo/Bup size Mb x NI-1
                                ! indI vector of length NI
      INTEGER :: NI,Mb          ! must be set before calling RINDD

                                ! The following is allocated in RINDD
      DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: SQ
      DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: Hlo,Hup
      INTEGER,          DIMENSION(:), ALLOCATABLE :: index1,xedni,indXtd
      INTEGER,          DIMENSION(:), ALLOCATABLE :: NsXtmj, NsXdj

                                ! global constants
      DOUBLE PRECISION, PARAMETER :: SQTWOPI1=3.9894228040143d-1 !=1/sqrt(2*pi)
      DOUBLE PRECISION, PARAMETER :: SQPI1=5.6418958354776d-1    !=1/sqrt(pi)
      DOUBLE PRECISION, PARAMETER :: SQPI= 1.77245385090552d0    !=sqrt(pi)
      DOUBLE PRECISION, PARAMETER :: SQTWO=1.41421356237310d0    !=sqrt(2)
      DOUBLE PRECISION, PARAMETER :: SQTWO1=0.70710678118655d0   !=1/sqrt(2)
      DOUBLE PRECISION, PARAMETER :: PI1=0.31830988618379d0      !=1/pi
      DOUBLE PRECISION, PARAMETER :: PI= 3.14159265358979D0      !=pi
      DOUBLE PRECISION, PARAMETER :: TWOPI=6.28318530717958D0    !=2*pi
      END MODULE GLOBALDATA

      MODULE C1C2MOD
      IMPLICIT NONE
      INTERFACE C1C2
      MODULE PROCEDURE C1C2
      END INTERFACE
      CONTAINS
      SUBROUTINE C1C2(C1, C2, Cm, B1, SQ, ind)
! The regression equation for the conditional distr. of Y given X=x
! is equal  to the conditional expectation of Y given X=x, i.e.,
!
!       E(Y|X=x)=E(Y)+Cov(Y,X)/Var(X)[x-E(X)]
!
!  Let x1=(x-E(X))/SQRT(Var(X)) be zero mean, C1<x1<C2, B1(I)=COV(Y(I),X)/SQRT(Var(X)).
!  Then the process  Y(I) with mean Cm(I) can be written as
!
!       y(I)=Cm(I)+x1*B1(I)+Delta(I) for  I=1,...,N.
!
!  where SQ(I)=sqrt(Var(Y|X)) is the standard deviation of Delta(I).
!
!  Since we are truncating all Gaussian  variables to
!  the interval [-C,C], then if for any I
!
!  a) Cm(I)+x1*B1(I)-C*SQ(I)>Hup(I)  or
!
!  b) Cm(I)+x1*B1(I)+C*SQ(I)<Hlo(I)  then
!
!  (XIND|X1=x1) = 0 !!!!!!!!!
!
!  Consequently, for increasing the accuracy (by excluding possible
!  discontinuouities) we shall exclude such values for which (XIND|X1=x1) = 0.
!  Hence we assume that if C1<x<C2 any of the previous conditions are
!  satisfied
!
!  OBSERVE!!, C1, C2 has to be set to (the normalized) lower and upper bounds
!  of possible values for x1,respectively, i.e.,
!           C1=max((Hlo-E(X))/SQRT(Var(X)),-C), C2=min((Hup-E(X))/SQRT(Var(X)),C)
!  before calling C1C2 subroutine.
!
      USE GLOBALDATA, ONLY : Hup,Hlo,xCutOff,EPS2,EPS
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:), INTENT(in) :: Cm, B1, SQ
      INTEGER,          DIMENSION(:), INTENT(in) :: ind
      DOUBLE PRECISION,            INTENT(inout) :: C1,C2

      ! local variables

      DOUBLE PRECISION :: CC1, CC2,CSQ,HHup,HHlo,BdSQ0
      INTEGER :: N,I,I0            !,changedLimits=0

                                !ind contains indices to the varibles
                                !location in  Hlo and Hup
      IF (C1.GE.C2) GO TO 112

      N = SIZE(ind)
      IF (N.LT.1)  RETURN       !Not able to change integration limits

      DO I = N,1,-1             ! C=xCutOff
         CSQ  = xCutOff*SQ(I)
         I0   = ind(I)
         HHup = Hup (I0) - Cm (I)
         HHlo = Hlo (I0) - Cm (I)
                                !  If ABS(B1(I)) < EPS2 overflow may occur
                                !  and hence if
                                !  1) Cm(I) is so large or small so we can
                                !     surely assume that the probability
                                !     of staying between the barriers is 0,
                                !     consequently C1=C2=0
         BdSQ0 = B1 (I)
         !print *,'C1C1',C1,C2
         !print *,'I,HHup,HHlo,Bdsq0',I,HHup,HHlo,BdsQ0,CSQ
         IF (ABS (BdSQ0 ) .LT.EPS2 ) THEN
            IF (SQ(I).LT.EPS2) CSQ= xCutOff*EPS
            IF (HHlo.GT.CSQ.OR.HHup.LT. - CSQ) THEN
! print *,'C1C2:', I,BdSQ0,CSQ,HHlo,HHup, xCutOff*SQ(I)  !changedLimits=1
               GOTO 112
            ENDIF
         ELSE        !  In other cases this part follows
                     !  from the description of the problem.
!           IF (CSQ.GT.0) PRINT *,'c1c2:', I,BdSQ0,CSQ,HHlo,HHup, SQ(I)
            IF (BdSQ0.LT.0.d0) THEN
               CC2 = (HHlo - CSQ) / BdSQ0
               CC1 = (HHup + CSQ) / BdSQ0
            ELSE ! BdSQ0>0
               CC1 = (HHlo - CSQ) / BdSQ0
               CC2 = (HHup + CSQ) / BdSQ0
            ENDIF
            IF (C1.LT.CC1) THEN
               C1 = CC1         !changedLimits=1
               IF (C2.GT.CC2) C2 = CC2
               IF (C1.GE.C2) GO TO 112
            ELSEIF (C2.GT.CC2) THEN
               C2 = CC2         !changedLimits=1
               IF (C1.GE.C2) GO TO 112
            END IF
         ENDIF
      END DO
!IF (changedLimits.EQ.1) THEN
!  PRINT *,'C1C2=',C1,C2
!END IF
      RETURN
 112  continue
      C1 = -2D0*xCutOff
      C2 = -2D0*xCutOff

      RETURN
      END SUBROUTINE C1C2
      END MODULE C1C2MOD

!**************************************

      MODULE FUNCMOD
!     FUNCTION module containing constants transfeered to mvnfun and mvnfun2
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: BIG
      DOUBLE PRECISION, DIMENSION(:  ), ALLOCATABLE :: Cm,CmN,xd,xc
      DOUBLE PRECISION :: Pl1,Pu1

      INTERFACE  MVNFUN
      MODULE PROCEDURE MVNFUN
      END INTERFACE

      INTERFACE  MVNFUN2
      MODULE PROCEDURE MVNFUN2
      END INTERFACE

      CONTAINS
      function MVNFUN(Ndim,W) RESULT (XIND)
      USE FIMOD
      USE C1C2MOD
      USE JACOBMOD
      USE GLOBALDATA, ONLY : Hlo,Hup,xCutOff,Nt,Nd,Nj,Ntd,SQ,
     &     NsXtmj, NsXdj,indXtd,index1,useC1C2,C1C2det,EPS2
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)  :: W
      INTEGER,                           INTENT(in) :: Ndim
      DOUBLE PRECISION                              :: XIND
!local variables
      DOUBLE PRECISION :: Pl,Pu
      DOUBLE PRECISION :: X,Y,XMI,XMA,SQ0
      INTEGER          :: Nst,NstN,NsdN,Nst0,Nsd,Nsd0,K
      INTEGER          :: Ndleft,Ndjleft,Ntmj

!MVNFUN Multivariate Normal integrand function
! where the integrand is transformed from an integral
! having integration limits Hl0 and Hup to an
! integral having  constant integration limits i.e.
!   Hup                               1
! int jacob(xd,xc)*f(xd,xt)dxt dxd = int F2(W) dW
!Hlo                                 0
!
! W    - new transformed integration variables, valid range 0..1
!        The vector must have the length Ndim=Nst0+Ntd-Nsd0
! BIG  - conditional sorted covariance matrix (IN)
! Cm   = conditional mean of Xd and Xt given Xc, E(Xd,Xt|Xc)
! CmN  - local conditional mean
! xd   - variables to the jacobian variable, need no initialization
! xc   - conditional variables (IN)
! Pl1  = FI(XMI) for the first integration variable (IN)
! Pu1  = FI(XMA) ------||-------------------------------
!      print *,'MVNFUN, ndim', ndim, shape(W)
      CmN(1:Ntd) = Cm(1:Ntd) ! initialize conditional mean
      Nst = NsXtmj(Ntd+1) ! index to last stoch variable of Xt before conditioning on X(Ntd)
      Ntmj=Nt-Nj
      Nsd0=NsXdj(1)
      if (Nt.gt.Nj) then
         Nst0=NsXtmj(Ntmj)
      else
         Nst0=0
      endif
      Pl=Pl1
      Pu=Pu1
!      IF (NDIM.LT.Nst0+Ntd-Nsd0+1) PRINT *, 'MVNFUN NDIM,',NDIM
      Y=Pu-Pl
      if (Nd+Nj.EQ.0) then
         SQ0=SQ(1,1)
         goto 200
      endif
      Ndjleft=Nd+Nj
      Nsd = NsXdj(Ndjleft+1)    ! index to last stoch variable of Xd and Nj of Xt  before conditioning on X(Ntd)
      Ndleft=Nd
      SQ0=SQ(Ntd,Ntd)
                                !print *,'mvnfun,nst,nsd,nd,nj',nst,nsd,Nd,Nj
      !print *,'mvn start K loop'
      DO K=Ntd-1,Nsd0,-1
         X=FIINV(Pl+W(Ntd-K)*(Pu-Pl))
         IF (index1(K+1).GT.Nt) THEN ! isXd
            xd (Ndleft) =  CmN(K+1)+X*SQ0
            Ndleft=Ndleft-1
         END IF
         Nst    = NsXtmj(K+1)   ! # stoch. var. of Xt before conditioning on X(K)
         if (Nst.GT.0) CmN(1:Nst) =CmN(1:Nst)+X*BIG(1:Nst,K+1)     !/SQ0
         CmN(Nsd:K) =CmN(Nsd:K)+X*BIG(Nsd:K,K+1)     !/SQ0

         Ndjleft = Ndjleft-1
         Nsd      = NsXdj(Ndjleft+1)
         SQ0      = SQ(K,K)

         XMA = (Hup (K)-CmN(K))/SQ0
         XMI = (Hlo (K)-CmN(K))/SQ0

         if (useC1C2) then      ! see if we can narrow down sampling range
            XMI=max(XMI,-xCutOff)
            XMA=min(XMA,xCutOff)
            if (C1C2det) then
               NsdN = NsXdj(Ndjleft)
               NstN = NsXtmj(K)
               CALL C1C2(XMI,XMA,CmN(Nsd:NsdN-1),
     &              BIG(Nsd:NsdN-1,K),SQ(Nsd:NsdN-1,K),
     &              indXtd(Nsd:NsdN-1))
               CALL C1C2(XMI,XMA,CmN(NstN+1:Nst),
     &              BIG(NstN+1:Nst,K),SQ(NstN+1:Nst,K),
     &              indXtd(NstN+1:Nst))
            else
               CALL C1C2(XMI,XMA,CmN(Nsd:K-1),BIG(Nsd:K-1,K),
     &              SQ(Nsd:K-1,Ntmj+Ndjleft),indXtd(Nsd:K-1))
               CALL C1C2(XMI,XMA,CmN(1:Nst),BIG(1:Nst,K)
     &              ,SQ(1:Nst,Ntmj+Ndjleft),indXtd(1:Nst))
            endif
            IF (XMA.LE.XMI) goto 260
         endif
         Pl = FI(XMI)
         Pu = FI(XMA)
         Y=Y*(Pu-Pl)
      ENDDO                     ! K LOOP
      X   = FIINV(Pl+W(Ntd-Nsd0+1)*(Pu-Pl))
      Nst = NsXtmj(Nsd0)        ! # stoch. var. of Xt after conditioning on X(Nsd0)
                                ! and before conditioning on X(1)
!      CmN(1:Nst)=CmN(1:Nst)+X*BIG(1:Nst,Nsd0)    !/SQ0)
      if (Nd.gt.0) then
         CmN(Nsd:Nsd0-1) = CmN(Nsd:Nsd0-1)+X*BIG(Nsd:Nsd0-1,Nsd0)  !/SQ0
         if (Ndleft.gt.0) then
            if (index1(Nsd0).GT.Nt) then
               xd (Ndleft) =  CmN(Nsd0)+X*SQ0
               Ndleft=Ndleft-1
            endif
            K=Nsd0-1
            do while (Ndleft.gt.0)
               if ((index1(K).GT.Nt)) THEN ! isXd
                  xd (Ndleft) =  CmN(K)
                  Ndleft=Ndleft-1
               END IF
               K=K-1
            ENDDO
         endif                  ! Ndleft
         Y = Y*jacob ( xd,xc)   ! jacobian of xd,xc
      endif                     ! Nd>0
      if (Nst0.gt.0) then
         CmN(1:Nst)=CmN(1:Nst)+X*BIG(1:Nst,Nsd0) !/SQ0)
         SQ0 = SQ(1,1)
         XMA = MIN((Hup (1)-CmN(1))/SQ0,xCutOff)
         XMI = MAX((Hlo (1)-CmN(1))/SQ0,-xCutOff)

         if (C1C2det) then
            NstN = NsXtmj(1)    ! # stoch. var. after conditioning
            CALL C1C2(XMI,XMA,CmN(NstN+1:Nst),
     &           BIG(1,NstN+1:Nst),SQ(NstN+1:Nst,1),
     &           indXtd(NstN+1:Nst))
         else
            CALL C1C2(XMI,XMA,CmN(2:Nst),BIG(1,2:Nst),
     &           SQ(2:Nst,1),indXtd(2:Nst))
         endif
         IF (XMA.LE.XMI) GO TO 260
         Pl = FI(XMI)
         Pu = FI(XMA)
         Y  = Y*(Pu-Pl)
      endif
      !if (COVix.gt.2) then
      !print *,' mvnfun start K2 loop'
      !endif
 200  do K = 2,Nst0
         X   = FIINV(Pl+W(Ntd-Nsd0+K)*(Pu-Pl))
         Nst = NsXtmj(K-1)      ! index to last stoch. var. before conditioning on X(K)
         CmN(K:Nst)=CmN(K:Nst)+X*BIG(K-1,K:Nst)        !/SQ0
         SQ0 = SQ(K,K)
         XMA = MIN((Hup (K)-CmN(K))/SQ0,xCutOff)
         XMI = MAX((Hlo (K)-CmN(K))/SQ0,-xCutOff)

         if (C1C2det) then
            NstN = NsXtmj(K)    ! index to last stoch. var. after conditioning  X(K)
            CALL C1C2(XMI,XMA,CmN(NstN+1:Nst),
     &           BIG(K,NstN+1:Nst),SQ(NstN+1:Nst,K),
     &           indXtd(NstN+1:Nst))
         else
            CALL C1C2(XMI,XMA,CmN(K+1:Nst),BIG(K,K+1:Nst),
     &           SQ(K+1:Nst,K),indXtd(K+1:Nst))
         endif
         IF (XMA.LE.XMI) GO TO 260
         Pl = FI(XMI)
         Pu = FI(XMA)
         Y=Y*(Pu-Pl)
      enddo                     ! K loop
      XIND = Y
      RETURN
 260  XIND = 0.D0
      !if (Y.LT.0.d0) PRINT *,'MVNFUN NEGATIVE INTEGRAND'
      !print *,' mvnfun leaving'
      return
      END FUNCTION MVNFUN

      function MVNFUN2(Ndim,W) RESULT (XIND)
      USE FIMOD
      USE C1C2MOD
      USE GLOBALDATA, ONLY : Hlo,Hup,xCutOff,Njj,Nj,Ntscis,Ntd,SQ,
     &     NsXtmj, NsXdj,indXtd,index1,useC1C2,C1C2det,Nt,EPS2
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in) :: W
      INTEGER, INTENT(in)  :: Ndim
      DOUBLE PRECISION :: XIND
!local variables
      DOUBLE PRECISION :: Pl,Pu
      DOUBLE PRECISION :: X,Y,XMI,XMA,SQ0
      INTEGER          :: Nst,NstN,Nst0,K

!MVNFUN2 Multivariate Normal integrand function
! where the integrand is transformed from an integral
! having integration limits Hl0 and Hup to an
! integral having  constant integration limits i.e.
!   Hup                 1
! int f(xt)dxt      = int F2(W) dW
!Hlo                   0
!
! W   - new transformed integration variables, valid range 0..1
!       The vector must have the size Nst0
! BIG - conditional sorted covariance matrix (IN)
! CmN - Local conditional mean
! Cm  = Conditional mean E(Xt,Xd|Xc)
! Pl1 = FI(XMI) for the first integration variable
! Pu1 = FI(XMA) ------||-------------------------

      !print *,'MVNFUN2, ndim', ndim, shape(W)
      Nst0 = NsXtmj(Njj+Ntscis)

      if (Njj.GT.0) then
         Nst  = NsXtmj(Njj)
      else
         Nst  = NsXtmj(Ntscis+1)
      endif
!      IF (NDIM.LT.Nst0+Njj) PRINT *, 'MVNFUN2 NDIM,',NDIM
      ! initialize conditional mean
      CmN(1:Nst)=Cm(1:Nst)

      Pl = Pl1
      Pu = Pu1

      Y   = Pu-Pl
      SQ0 = SQ(1,1)

      do K = 2,Nst0
         X = FIINV(Pl+W(K-1)*(Pu-Pl))
         Nst = NsXtmj(K-1)      ! index to last stoch. var. before conditioning on X(K)
         CmN(K:Nst)=CmN(K:Nst)+X*BIG(K-1,K:Nst) !/SQ0
         SQ0 = SQ(K,K)
         XMA = MIN((Hup (K)-CmN(K))/SQ0,xCutOff)
         XMI = MAX((Hlo (K)-CmN(K))/SQ0,-xCutOff)

         if (C1C2det) then
            NstN=NsXtmj(K)      ! index to last stoch. var. after conditioning on X(K)
            CALL C1C2(XMI,XMA,CmN(NstN+1:Nst),
     &           BIG(K,NstN+1:Nst),SQ(NstN+1:Nst,K),
     &           indXtd(NstN+1:Nst))
         else
            CALL C1C2(XMI,XMA,CmN(K+1:Nst),BIG(K,K+1:Nst),
     &           SQ(K+1:Nst,K),indXtd(K+1:Nst))
         endif
         IF (XMA.LE.XMI) GO TO 260
         Pl = FI(XMI)
         Pu = FI(XMA)
         Y  = Y*(Pu-Pl)
      enddo                     ! K loop
      XIND = Y
      RETURN
 260  XIND = 0.d0
      return
      END FUNCTION MVNFUN2
      END MODULE FUNCMOD

      MODULE QUAD
      IMPLICIT NONE         ! Quadratures available: Legendre,Hermite,Laguerre
      INTEGER            :: I
      INTEGER, PARAMETER :: PMAX=24       ! maximum # nodes
      INTEGER, PARAMETER :: sizNint=13    ! size of Nint1
      INTEGER            :: minQNr=1      ! minimum quadrature number
                                          ! used in GaussLe1, Gaussle2
      INTEGER            :: Le2QNr=8      ! quadr. number used in  rind2,rindnit
      INTEGER, DIMENSION(sizNint) :: Nint1 ! use quadr. No. Nint1(i) in
                                                  ! integration of Xd(i)

                                ! # different quadratures stored for :
                                !-------------------------------------
      INTEGER,PARAMETER  :: NLeW=13 ! Legendre
      INTEGER,PARAMETER  :: NHeW=13 ! Hermite
      INTEGER,PARAMETER  :: NLaW=13 ! Laguerre
                                ! Quadrature Number stored for :
                                !-------------------------------------
      INTEGER, DIMENSION(NLeW) :: LeQNr    ! Legendre
      INTEGER, DIMENSION(NHeW) :: HeQNr    ! Hermite
      INTEGER, DIMENSION(NLaW) :: LaQNr    ! Laguerre
      PARAMETER (LeQNr=(/ 2,3,4,5,6,7, 8, 9, 10, 12, 16, 20, 24 /))
      PARAMETER (HeQNr=(/ 2,3,4,5,6,7, 8, 9, 10, 12, 16, 20, 24 /))
      PARAMETER (LaQNr=(/ 2,3,4,5,6,7, 8, 9, 10, 12, 16, 20, 24 /))


                            ! The indices to the weights & nodes stored for:
                            !------------------------------------------------
      INTEGER, DIMENSION(NLeW+1) :: LeIND  !Legendre
      INTEGER, DIMENSION(NHeW+1) :: HeIND  !Hermite
      INTEGER, DIMENSION(NLaW+1) :: LaIND  !Laguerre

      PARAMETER (LeIND=(/0,2,5,9,14,20,27,35,44,54,66,82,102,126/)) !Legendre
      PARAMETER (HeIND=(/0,2,5,9,14,20,27,35,44,54,66,82,102,126/)) !Hermite
      PARAMETER (LaIND=(/0,2,5,9,14,20,27,35,44,54,66,82,102,126/)) !Laguerre

                            !------------------------------------------------
      DOUBLE PRECISION,  DIMENSION(126) :: LeBP,LeWF,HeBP,HeWF
      DOUBLE PRECISION,  DIMENSION(126) :: LaBP0,LaWF0,LaBP5,LaWF5

!The Hermite Quadrature integrates an integral of the form
!        inf                         n
!       Int (exp(-x^2) F(x)) dx  =  Sum  wf(j)*F( bp(j) )
!       -Inf                        j=1
!The Laguerre Quadrature integrates an integral of the form
!        inf                               n
!       Int (x^alpha exp(-x) F(x)) dx  =  Sum  wf(j)*F( bp(j) )
!         0                               j=1
! weights stored here are for alpha=0 and alpha=-0.5

                             ! initialize Legendre weights, wf,  and nodes, bp
!PARAMETER ( LeWF = (
      DATA ( LeWF(I), I = 1, 78 )
     *     / 1.d0, 1.d0, 0.555555555555556d0,
     *     0.888888888888889d0, 0.555555555555556d0,
     *     0.347854845137454d0, 0.652145154862546d0,
     *     0.652145154862546d0, 0.347854845137454d0,
     *     0.236926885056189d0, 0.478628670499366d0,
     *     0.568888888888889d0, 0.478628670499366d0,
     *     0.236926885056189d0, 0.171324492379170d0,
     *     0.360761573048139d0, 0.467913934572691d0,
     *     0.467913934572691d0, 0.360761573048139d0,
     *     0.171324492379170d0, 0.129484966168870d0,
     *     0.279705391489277d0, 0.381830050505119d0,
     *     0.417959183673469d0, 0.381830050505119d0,
     *     0.279705391489277d0, 0.129484966168870d0,
     *     0.101228536290376d0, 0.222381034453374d0,
     *     0.313706645877887d0, 0.362683783378362d0,
     *     0.362683783378362d0, 0.313706645877887d0,
     *     0.222381034453374d0, 0.101228536290376d0,
     *     0.081274388361574d0, 0.180648160694857d0,
     *     0.260610696402935d0, 0.312347077040003d0,
     *     0.330239355001260d0, 0.312347077040003d0,
     *     0.260610696402935d0, 0.180648160694857d0,
     *     0.081274388361574d0, 0.066671344308688d0,
     *     0.149451349150581d0, 0.219086362515982d0,
     *     0.269266719309996d0, 0.295524224714753d0,
     *     0.295524224714753d0, 0.269266719309996d0,
     *     0.219086362515982d0, 0.149451349150581d0,
     *     0.066671344308688d0, 0.047175336386512d0,
     *     0.106939325995318d0, 0.160078328543346d0,
     *     0.203167426723066d0, 0.233492536538355d0,
     *     0.249147048513403d0, 0.249147048513403d0,
     *     0.233492536538355d0,
     *     0.203167426723066d0,       0.160078328543346d0,
     *     0.106939325995318d0,       0.047175336386512d0,
     *     0.027152459411754094852d0, 0.062253523938647892863d0,
     *     0.095158511682492784810d0, 0.124628971255533872052d0,
     *     0.149595988816576732081d0, 0.169156519395002538189d0,
     *     0.182603415044923588867d0, 0.189450610455068496285d0,
     *     0.189450610455068496285d0, 0.182603415044923588867d0,
     *     0.169156519395002538189d0, 0.149595988816576732081d0/
        DATA ( LeWF(I), I = 79, 126 )
     *     / 0.124628971255533872052d0, 0.095158511682492784810d0,
     *     0.062253523938647892863d0, 0.027152459411754094852d0,
     *     0.017614007139152118312d0, 0.040601429800386941331d0,
     *     0.062672048334109063570d0, 0.083276741576704748725d0,
     *     0.101930119817240435037d0, 0.118194531961518417312d0,
     *     0.131688638449176626898d0, 0.142096109318382051329d0,
     *     0.149172986472603746788d0, 0.152753387130725850698d0,
     *     0.152753387130725850698d0, 0.149172986472603746788d0,
     *     0.142096109318382051329d0, 0.131688638449176626898d0,
     *     0.118194531961518417312d0, 0.101930119817240435037d0,
     *     0.083276741576704748725d0, 0.062672048334109063570d0,
     *     0.040601429800386941331d0, 0.017614007139152118312d0,
     *     0.012341229799987199547d0, 0.028531388628933663181d0,
     *     0.044277438817419806169d0, 0.059298584915436780746d0,
     *     0.073346481411080305734d0, 0.086190161531953275917d0,
     *     0.097618652104113888270d0, 0.107444270115965634783d0,
     *     0.115505668053725601353d0, 0.121670472927803391204d0,
     *     0.125837456346828296121d0, 0.127938195346752156974d0,
     *     0.127938195346752156974d0, 0.125837456346828296121d0,
     *     0.121670472927803391204d0, 0.115505668053725601353d0,
     *     0.107444270115965634783d0, 0.097618652104113888270d0,
     *     0.086190161531953275917d0, 0.073346481411080305734d0,
     *     0.059298584915436780746d0, 0.044277438817419806169d0,
     *     0.028531388628933663181d0, 0.012341229799987199547d0 /
!      PARAMETER
      DATA ( LeBP(I), I=1,77)
     *       / -0.577350269189626d0,0.577350269189626d0,
     *    -0.774596669241483d0, 0.d0,
     *     0.774596669241483d0, -0.861136311594053d0,
     *    -0.339981043584856d0,  0.339981043584856d0,
     *     0.861136311594053d0, -0.906179845938664d0,
     *    -0.538469310105683d0,  0.d0,
     *     0.538469310105683d0,  0.906179845938664d0,
     *    -0.932469514203152d0, -0.661209386466265d0,
     *    -0.238619186083197d0,  0.238619186083197d0,
     *     0.661209386466265d0,  0.932469514203152d0,
     *    -0.949107912342759d0, -0.741531185599394d0,
     *    -0.405845151377397d0,  0.d0,
     *     0.405845151377397d0,  0.741531185599394d0,
     *     0.949107912342759d0, -0.960289856497536d0,
     *    -0.796666477413627d0, -0.525532409916329d0,
     *    -0.183434642495650d0,  0.183434642495650d0,
     *     0.525532409916329d0,  0.796666477413627d0,
     *     0.960289856497536d0, -0.968160239507626d0,
     *    -0.836031107326636d0, -0.613371432700590d0,
     *    -0.324253423403809d0,  0.d0,
     *     0.324253423403809d0,  0.613371432700590d0,
     *     0.836031107326636d0,  0.968160239507626d0,
     *    -0.973906528517172d0, -0.865063366688985d0,
     *    -0.679409568299024d0, -0.433395394129247d0,
     *    -0.148874338981631d0,  0.148874338981631d0,
     *     0.433395394129247d0,  0.679409568299024d0,
     *     0.865063366688985d0,  0.973906528517172d0,
     *    -0.981560634246719d0, -0.904117256370475d0,
     *    -0.769902674194305d0, -0.587317954286617d0,
     *    -0.367831498198180d0, -0.125233408511469d0,
     *     0.125233408511469d0, 0.367831498198180d0,
     *     0.587317954286617d0, 0.769902674194305d0,
     *     0.904117256370475d0,  0.981560634246719d0,
     *    -0.989400934991649932596d0,
     *    -0.944575023073232576078d0, -0.865631202387831743880d0,
     *    -0.755404408355003033895d0, -0.617876244402643748447d0,
     *    -0.458016777657227386342d0, -0.281603550779258913230d0,
     *    -0.095012509837637440185d0,  0.095012509837637440185d0,
     *     0.281603550779258913230d0,  0.458016777657227386342d0/
      DATA ( LeBP(I), I=78,126)
     *     / 0.617876244402643748447d0,  0.755404408355003033895d0,
     *     0.865631202387831743880d0,  0.944575023073232576078d0,
     *     0.989400934991649932596d0, -0.993128599185094924786d0,
     *    -0.963971927277913791268d0, -0.912234428251325905868d0,
     *    -0.839116971822218823395d0, -0.746331906460150792614d0,
     *    -0.636053680726515025453d0, -0.510867001950827098004d0,
     *    -0.373706088715419560673d0, -0.227785851141645078080d0,
     *     -0.076526521133497333755d0,  0.076526521133497333755d0,
     *      0.227785851141645078080d0,  0.373706088715419560673d0,
     *      0.510867001950827098004d0,  0.636053680726515025453d0,
     *      0.746331906460150792614d0,  0.839116971822218823395d0,
     *      0.912234428251325905868d0,
     *      0.963971927277913791268d0,  0.993128599185094924786d0,
     *     -0.995187219997021360180d0, -0.974728555971309498198d0,
     *     -0.938274552002732758524d0, -0.886415527004401034213d0,
     *     -0.820001985973902921954d0, -0.740124191578554364244d0,
     *     -0.648093651936975569252d0, -0.545421471388839535658d0,
     *     -0.433793507626045138487d0, -0.315042679696163374387d0,
     *     -0.191118867473616309159d0, -0.064056892862605626085d0,
     *      0.064056892862605626085d0,  0.191118867473616309159d0,
     *      0.315042679696163374387d0,  0.433793507626045138487d0,
     *      0.545421471388839535658d0,  0.648093651936975569252d0,
     *      0.740124191578554364244d0,  0.820001985973902921954d0,
     *      0.886415527004401034213d0,  0.938274552002732758524d0,
     *      0.974728555971309498198d0,  0.995187219997021360180d0 /

                                ! initialize Hermite weights in HeWF and
                                ! nodes in HeBP
                                ! NB! the relative error of these numbers
                                ! are less than 10^-15
!      PARAMETER
      DATA (HeWF(I),I=1,78) / 8.8622692545275816d-1,
     *     8.8622692545275816d-1,
     *     2.9540897515091930d-1,   1.1816359006036770d0,
     *     2.9540897515091930d-1,   8.1312835447245310d-2,
     *     8.0491409000551251d-1,   8.0491409000551295d-1,
     *     8.1312835447245213d-2,   1.9953242059045910d-2,
     *     3.9361932315224146d-1,   9.4530872048294134d-1,
     *     3.9361932315224102d-1,   1.9953242059045962d-2,
     *     4.5300099055088378d-3,   1.5706732032285636d-1,
     *     7.2462959522439319d-1,   7.2462959522439241d-1,
     *     1.5706732032285681d-1,   4.5300099055088534d-3,
     *     9.7178124509952175d-4,   5.4515582819126975d-2,
     *     4.2560725261012805d-1,   8.1026461755680768d-1,
     *     4.2560725261012783d-1,   5.4515582819126975d-2,
     *     9.7178124509951828d-4,   1.9960407221136729d-4,
     *     1.7077983007413571d-2,   2.0780232581489183d-1,
     *     6.6114701255824082d-1,   6.6114701255824138d-1,
     *     2.0780232581489202d-1,   1.7077983007413498d-2,
     *     1.9960407221136775d-4,   3.9606977263264446d-5,
     *     4.9436242755369411d-3,   8.8474527394376654d-2,
     *     4.3265155900255586d-1,   7.2023521560605108d-1,
     *     4.3265155900255559d-1,   8.8474527394376543d-2,
     *     4.9436242755369350d-3,   3.9606977263264324d-5,
     *     7.6404328552326139d-6,   1.3436457467812229d-3,
     *     3.3874394455481210d-2,   2.4013861108231502d-1,
     *     6.1086263373532623d-1,   6.1086263373532546d-1,
     *     2.4013861108231468d-1,   3.3874394455480884d-2,
     *     1.3436457467812298d-3,   7.6404328552325919d-6,
     *     2.6585516843562997d-7,   8.5736870435879089d-5,
     *     3.9053905846291028d-3,   5.1607985615883860d-2,
     *     2.6049231026416092d-1,   5.7013523626247820d-1,
     *     5.7013523626248030d-1,   2.6049231026416109d-1,
     *     5.1607985615883846d-2,   3.9053905846290530d-3,
     *     8.5736870435878506d-5,   2.6585516843562880d-7,
     *     2.6548074740111735d-10,  2.3209808448651987d-7,
     *     2.7118600925379007d-5,   9.3228400862418819d-4,
     *     1.2880311535509989d-2,   8.3810041398985652d-2,
     *     2.8064745852853318d-1,   5.0792947901661278d-1,
     *     5.0792947901661356d-1,   2.8064745852853334d-1,
     *     8.3810041398985735d-2,   1.2880311535510015d-2/
      DATA (HeWF(I),I=79,126) /
     *     9.3228400862418407d-4,   2.7118600925378956d-5,
     *     2.3209808448651966d-7,   2.6548074740111787d-10,
     *     2.2293936455342015d-13,  4.3993409922730765d-10,
     *     1.0860693707692910d-7,   7.8025564785320463d-6,
     *     2.2833863601635403d-4,   3.2437733422378719d-3,
     *     2.4810520887463536d-2,   1.0901720602002360d-1,
     *     2.8667550536283382d-1,   4.6224366960061047d-1,
     *     4.6224366960061070d-1,   2.8667550536283398d-1,
     *     1.0901720602002325d-1,   2.4810520887463588d-2,
     *     3.2437733422378649d-3,   2.2833863601635316d-4,
     *     7.8025564785321005d-6,   1.0860693707692749d-7,
     *     4.3993409922731370d-10,  2.2293936455342167d-13,
     *     1.6643684964891124d-16,  6.5846202430781508d-13,
     *     3.0462542699875022d-10,  4.0189711749413878d-8,
     *     2.1582457049023452d-6,   5.6886916364043773d-5,
     *     8.2369248268841073d-4,   7.0483558100726748d-3,
     *     3.7445470503230736d-2,   1.2773962178455966d-1,
     *     2.8617953534644325d-1,   4.2693116386869828d-1,
     *     4.2693116386869912d-1,   2.8617953534644286d-1,
     *     1.2773962178455908d-1,   3.7445470503230875d-2,
     *     7.0483558100726844d-3,   8.2369248268842027d-4,
     *     5.6886916364044037d-5,   2.1582457049023460d-6,
     *     4.0189711749414963d-8,   3.0462542699876118d-10,
     *     6.5846202430782225d-13,  1.6643684964889408d-16 /

                                !hermite nodes
!      PARAMETER (HeBP = (
      DATA  (HeBP(I),I=1,79)  /  -7.07106781186547572d-1,
     *     7.0710678118654752d-1,   -1.2247448713915894d0,
     *     0.d0,                     1.2247448713915894d0,
     *     -1.6506801238857845d0,   -5.2464762327529035d-1,
     *     5.2464762327529035d-1,    1.6506801238857845d0,
     *     -2.0201828704560869d0,   -9.5857246461381806d-1,
     *     0.d0,                     9.5857246461381851d-1,
     *     2.0201828704560860d0,    -2.3506049736744918d0,
     *     -1.3358490740136963d0,   -4.3607741192761629d-1,
     *     4.3607741192761657d-1,    1.3358490740136963d0,
     *     2.3506049736744927d0,    -2.6519613568352334d0,
     *     -1.6735516287674728d0,   -8.1628788285896470d-1,
     *     0.d0,                     8.1628788285896470d-1,
     *     1.6735516287674705d0,     2.6519613568352325d0,
     *     -2.9306374202572423d0,   -1.9816567566958434d0,
     *     -1.1571937124467806d0,   -3.8118699020732233d-1,
     *     3.8118699020732211d-1,    1.1571937124467804d0,
     *     1.9816567566958441d0,     2.9306374202572423d0,
     *     -3.1909932017815290d0,   -2.2665805845318436d0,
     *     -1.4685532892166682d0,   -7.2355101875283812d-1,
     *     0.d0,                     7.2355101875283756d-1,
     *     1.4685532892166657d0,     2.2665805845318405d0,
     *     3.1909932017815281d0,    -3.4361591188377387d0,
     *     -2.5327316742327906d0,   -1.7566836492998805d0,
     *     -1.0366108297895140d0,   -3.4290132722370548d-1,
     *     3.4290132722370464d-1,    1.0366108297895136d0,
     *     1.7566836492998834d0,     2.5327316742327857d0,
     *     3.4361591188377396d0,    -3.8897248978697796d0,
     *     -3.0206370251208856d0,   -2.2795070805010567d0,
     *     -1.5976826351526050d0,   -9.4778839124016290d-1,
     *     -3.1424037625435908d-1,   3.1424037625435935d-1,
     *     9.4778839124016356d-1,    1.5976826351526054d0,
     *     2.2795070805010602d0,     3.0206370251208905d0,
     *     3.8897248978697831d0,    -4.6887389393058214d0,
     *     -3.8694479048601251d0,   -3.1769991619799582d0,
     *     -2.5462021578474765d0,   -1.9517879909162541d0,
     *     -1.3802585391988809d0,   -8.2295144914465523d-1,
     *     -2.7348104613815177d-1,   2.7348104613815244d-1,
     *     8.2295144914465579d-1,    1.3802585391988802d0,
     *     1.9517879909162534d0,     2.5462021578474801d0/
      DATA (HeBP(I),I=80,126) /
     *     3.1769991619799565d0,     3.8694479048601265d0,
     *     4.6887389393058196d0,    -5.3874808900112274d0,
     *     -4.6036824495507513d0,   -3.9447640401156296d0,
     *     -3.3478545673832154d0,   -2.7888060584281300d0,
     *     -2.2549740020892721d0,   -1.7385377121165839d0,
     *     -1.2340762153953209d0,   -7.3747372854539361d-1,
     *     -2.4534070830090124d-1,   2.4534070830090149d-1,
     *     7.3747372854539439d-1,    1.2340762153953226d0,
     *     1.7385377121165866d0,     2.2549740020892770d0,
     *     2.7888060584281282d0,     3.3478545673832105d0,
     *     3.9447640401156230d0,     4.6036824495507398d0,
     *     5.3874808900112274d0,    -6.0159255614257390d0,
     *     -5.2593829276680442d0,   -4.6256627564237904d0,
     *     -4.0536644024481472d0,   -3.5200068130345219d0,
     *     -3.0125461375655647d0,   -2.5238810170114276d0,
     *     -2.0490035736616989d0,   -1.5842500109616944d0,
     *     -1.1267608176112460d0,   -6.7417110703721150d-1,
     *     -2.2441454747251538d-1,   2.2441454747251532d-1,
     *     6.7417110703721206d-1,    1.1267608176112454d0,
     *     1.5842500109616939d0,     2.0490035736616958d0,
     *     2.5238810170114281d0,     3.0125461375655687d0,
     *     3.5200068130345232d0,     4.0536644024481499d0,
     *     4.6256627564237816d0,     5.2593829276680353d0,
     *     6.0159255614257550d0 /
                          !initialize Laguerre weights and nodes (basepoints)
                          ! for alpha=0
                          ! NB! the relative error of these numbers
                          ! are less than 10^-15
!      PARAMETER
      DATA (LaWF0(I),I=1,75) /  8.5355339059327351d-1,
     *     1.4644660940672624d-1, 7.1109300992917313d-1,
     *     2.7851773356924092d-1,  1.0389256501586137d-2,
     *     6.0315410434163386d-1,
     *     3.5741869243779956d-1,  3.8887908515005364d-2,
     *     5.3929470556132730d-4,  5.2175561058280850d-1,
     *     3.9866681108317570d-1,  7.5942449681707588d-2,
     *     3.6117586799220489d-3,  2.3369972385776180d-5,
     *     4.5896467394996360d-1,  4.1700083077212080d-1,
     *     1.1337338207404497d-1,  1.0399197453149061d-2,
     *     2.6101720281493249d-4,  8.9854790642961944d-7,
     *     4.0931895170127397d-1,  4.2183127786171964d-1,
     *     1.4712634865750537d-1,
     *     2.0633514468716974d-2,  1.0740101432807480d-3,
     *     1.5865464348564158d-5,  3.1703154789955724d-8,
     *     3.6918858934163773d-1,  4.1878678081434328d-1,
     *     1.7579498663717152d-1,  3.3343492261215649d-2,
     *     2.7945362352256712d-3,  9.0765087733581999d-5,
     *     8.4857467162725493d-7,  1.0480011748715038d-9,
     *     3.3612642179796304d-1,  4.1121398042398466d-1,
     *     1.9928752537088576d0,   4.7460562765651609d-2,
     *     5.5996266107945772d-3,  3.0524976709321133d-4,
     *     6.5921230260753743d-6,  4.1107693303495271d-8,
     *     3.2908740303506941d-11,
     *     3.0844111576502009d-1,  4.0111992915527328d-1,
     *     2.1806828761180935d-1,  6.2087456098677683d-2,
     *     9.5015169751810902d-3,  7.5300838858753855d-4,
     *     2.8259233495995652d-5,  4.2493139849626742d-7,
     *     1.8395648239796174d-9,  9.9118272196090085d-13,
     &     2.6473137105544342d-01,
     &     3.7775927587313773d-01, 2.4408201131987739d-01,
     &     9.0449222211681030d-02, 2.0102381154634138d-02,
     &     2.6639735418653122d-03, 2.0323159266299895d-04,
     &     8.3650558568197802d-06, 1.6684938765409045d-07,
     &     1.3423910305150080d-09, 3.0616016350350437d-12,
     &     8.1480774674261369d-16, 2.0615171495780091d-01,
     &     3.3105785495088480d-01, 2.6579577764421392d-01,
     &     1.3629693429637740d-01, 4.7328928694125222d-02,
     &     1.1299900080339390d-02, 1.8490709435263156d-03,
     &     2.0427191530827761d-04, 1.4844586873981184d-05/
      DATA (LaWF0(I),I=76,126) /
     &     6.8283193308711422d-07, 1.8810248410796518d-08,
     &     2.8623502429738514d-10, 2.1270790332241105d-12,
     &     6.2979670025179594d-15, 5.0504737000353956d-18,
     &     4.1614623703728548d-22, 1.6874680185111446d-01,
     &     2.9125436200606764d-01, 2.6668610286700062d-01,
     &     1.6600245326950708d-01, 7.4826064668792408d-02,
     &     2.4964417309283247d-02, 6.2025508445722223d-03,
     &     1.1449623864769028d-03, 1.5574177302781227d-04,
     &     1.5401440865224898d-05, 1.0864863665179799d-06,
     &     5.3301209095567054d-08, 1.7579811790505857d-09,
     &     3.7255024025122967d-11, 4.7675292515782048d-13,
     &     3.3728442433624315d-15, 1.1550143395004071d-17,
     &     1.5395221405823110d-20, 5.2864427255691140d-24,
     &     1.6564566124989991d-28, 1.4281197333478154d-01,
     &     2.5877410751742391d-01, 2.5880670727286992d-01,
     &     1.8332268897777793d-01, 9.8166272629918963d-02,
     &     4.0732478151408603d-02, 1.3226019405120104d-02,
     &     3.3693490584783083d-03, 6.7216256409355021d-04,
     &     1.0446121465927488d-04, 1.2544721977993268d-05,
     &     1.1513158127372857d-06, 7.9608129591336357d-08,
     &     4.0728589875500037d-09, 1.5070082262925912d-10,
     &     3.9177365150584634d-12, 6.8941810529581520d-14,
     &     7.8198003824593093d-16, 5.3501888130099474d-18,
     &     2.0105174645555229d-20, 3.6057658645531092d-23,
     &     2.4518188458785009d-26, 4.0883015936805334d-30,
     &     5.5753457883284229d-35 /
!      PARAMETER (LaBP0=(/
      DATA (LaBP0(I),I=1,78) /5.8578643762690485d-1,
     *     3.4142135623730949d+00,  4.1577455678347897d-1,
     *     2.2942803602790409d0,    6.2899450829374803d0,
     *     3.2254768961939217d-1,  1.7457611011583465d0,
     *     4.5366202969211287d0,    9.3950709123011364d0,
     *     2.6356031971814076d-1,  1.4134030591065161d0,
     *     3.5964257710407206d0,    7.0858100058588356d0,
     *     1.2640800844275784d+01,  2.2284660417926061d-1,
     *     1.1889321016726229d0,    2.9927363260593141d+00,
     *     5.7751435691045128d0,    9.8374674183825839d0,
     *     1.5982873980601699d+01,  1.9304367656036231d-1,
     *     1.0266648953391919d0,    2.5678767449507460d0,
     *     4.9003530845264844d0,    8.1821534445628572d0,
     *     1.2734180291797809d+01,  1.9395727862262543d+01,
     *     1.7027963230510107d-1,  9.0370177679938035d-1,
     *     2.2510866298661316d0,    4.2667001702876597d0,
     *     7.0459054023934673d0,    1.0758516010180994d+01,
     *     1.5740678641278004d+01,  2.2863131736889272d+01,
     *     1.5232222773180798d-1,  8.0722002274225590d-1,
     *     2.0051351556193473d0,    3.7834739733312328d0,
     *     6.2049567778766175d0,    9.3729852516875773d0,
     *     1.3466236911092089d+01,  1.8833597788991703d+01,
     *     2.6374071890927389d+01,  1.3779347054049221d-1,
     *     7.2945454950317090d-1,  1.8083429017403163d0,
     *     3.4014336978548996d0,
     *     5.5524961400638029d0,    8.3301527467644991d0,
     *     1.1843785837900066d+01,  1.6279257831378107d+01,
     *     2.1996585811980765d+01,  2.9920697012273894d+01 ,
     &     1.1572211735802050d-01, 6.1175748451513112d-01,
     &     1.5126102697764183d+00, 2.8337513377435077d+00,
     &     4.5992276394183476d+00, 6.8445254531151809d+00,
     &     9.6213168424568707d+00, 1.3006054993306348d+01,
     &     1.7116855187462260d+01, 2.2151090379397019d+01,
     &     2.8487967250983996d+01, 3.7099121044466933d+01,
     &     8.7649410478926978d-02, 4.6269632891508106d-01,
     &     1.1410577748312269d+00, 2.1292836450983796d+00,
     &     3.4370866338932058d+00, 5.0780186145497677d+00,
     &     7.0703385350482320d+00, 9.4383143363919331d+00,
     &     1.2214223368866158d+01, 1.5441527368781616d+01,
     &     1.9180156856753147d+01, 2.3515905693991915d+01/
      DATA  (LaBP0(I),I=79,126) /
     &     2.8578729742882153d+01,
     &     3.4583398702286622d+01, 4.1940452647688396d+01,
     &     5.1701160339543350d+01, 7.0539889691989419d-02,
     &     3.7212681800161185d-01, 9.1658210248327376d-01,
     &     1.7073065310283420d+00, 2.7491992553094309d+00,
     &     4.0489253138508827d+00, 5.6151749708616148d+00,
     &     7.4590174536710663d+00, 9.5943928695810943d+00,
     &     1.2038802546964314d+01, 1.4814293442630738d+01,
     &     1.7948895520519383d+01, 2.1478788240285009d+01,
     &     2.5451702793186907d+01, 2.9932554631700611d+01,
     &     3.5013434240478986d+01, 4.0833057056728535d+01,
     &     4.7619994047346523d+01, 5.5810795750063903d+01,
     &     6.6524416525615763d+01, 5.9019852181507730d-02,
     &     3.1123914619848325d-01, 7.6609690554593646d-01,
     &     1.4255975908036129d+00, 2.2925620586321909d+00,
     &     3.3707742642089964d+00, 4.6650837034671726d+00,
     &     6.1815351187367655d+00, 7.9275392471721489d+00,
     &     9.9120980150777047d+00, 1.2146102711729766d+01,
     &     1.4642732289596671d+01, 1.7417992646508978d+01,
     &     2.0491460082616424d+01, 2.3887329848169724d+01,
     &     2.7635937174332710d+01, 3.1776041352374712d+01,
     &     3.6358405801651635d+01, 4.1451720484870783d+01,
     &     4.7153106445156347d+01, 5.3608574544695017d+01,
     &     6.1058531447218698d+01, 6.9962240035105026d+01,
     &     8.1498279233948850d+01/

                                !Laguerre nodes for alpha=-0.5
!      PARAMETER (LaBP5 = (/
      DATA (LaBP5(I),I=1,79) /2.7525512860841095e-01,
     &      2.7247448713915889e+00, 1.9016350919348812e-01,
     &      1.7844927485432514e+00, 5.5253437422632619e+00,
     &      1.4530352150331699e-01, 1.3390972881263605e+00,
     &      3.9269635013582880e+00, 8.5886356890120332e+00,
     &      1.1758132021177792e-01, 1.0745620124369035e+00,
     &      3.0859374437175511e+00, 6.4147297336620337e+00,
     &      1.1807189489971735e+01, 9.8747014068480951e-02,
     &      8.9830283456961701e-01, 2.5525898026681721e+00,
     &      5.1961525300544675e+00, 9.1242480375311814e+00,
     &      1.5129959781108084e+01, 8.5115442997593743e-02,
     &      7.7213792004277715e-01, 2.1805918884504596e+00,
     &      4.3897928867310174e+00, 7.5540913261017897e+00,
     &      1.1989993039823887e+01, 1.8528277495852500e+01,
     &      7.4791882596818141e-02, 6.7724908764928937e-01,
     &      1.9051136350314275e+00, 3.8094763614849056e+00,
     &      6.4831454286271679e+00, 1.0093323675221344e+01,
     &      1.4972627088426393e+01, 2.1984272840962646e+01,
     &      6.6702230958194261e-02, 6.0323635708174905e-01,
     &      1.6923950797931777e+00, 3.3691762702432655e+00,
     &      5.6944233429577471e+00, 8.7697567302685968e+00,
     &      1.2771825354869195e+01, 1.8046505467728977e+01,
     &      2.5485979166099078e+01, 6.0192063149587700e-02,
     &      5.4386750029464592e-01, 1.5229441054044432e+00,
     &      3.0225133764515753e+00, 5.0849077500985240e+00,
     &      7.7774392315254426e+00, 1.1208130204348663e+01,
     &      1.5561163332189356e+01, 2.1193892096301536e+01,
     &      2.9024950340236231e+01, 5.0361889117293709e-02,
     &      4.5450668156378027e-01, 1.2695899401039612e+00,
     &      2.5098480972321284e+00, 4.1984156448784127e+00,
     &      6.3699753880306362e+00, 9.0754342309612088e+00,
     &      1.2390447963809477e+01, 1.6432195087675318e+01,
     &      2.1396755936166095e+01, 2.7661108779846099e+01,
     &      3.6191360360615583e+01, 3.7962914575312985e-02,
     &      3.4220015601094805e-01, 9.5355315539086472e-01,
     &      1.8779315076960728e+00, 3.1246010507021431e+00,
     &      4.7067267076675874e+00, 6.6422151797414388e+00,
     &      8.9550013377233881e+00, 1.1677033673975952e+01,
     &      1.4851431341801243e+01, 1.8537743178606682e+01,
     &      2.2821300693525199e+01, 2.7831438211328681e+01/
      DATA (LaBP5(I),I=80,126) /
     &      3.3781970488226136e+01, 4.1081666525491165e+01,
     &      5.0777223877537075e+01, 3.0463239279482423e-02,
     &      2.7444471579285024e-01, 7.6388755844391365e-01,
     &      1.5018014976681033e+00, 2.4928301451213657e+00,
     &      3.7434180412162927e+00, 5.2620558537883513e+00,
     &      7.0596277357415627e+00, 9.1498983120306470e+00,
     &      1.1550198286442805e+01, 1.4282403685210406e+01,
     &      1.7374366975199074e+01, 2.0862075185437845e+01,
     &      2.4793039892463458e+01, 2.9231910157093431e+01,
     &      3.4270428925039589e+01, 4.0046815790245596e+01,
     &      4.6788846392124952e+01, 5.4931555621020564e+01,
     &      6.5589931990639684e+01, 2.5437996585689085e-02,
     &      2.2910231649262403e-01, 6.3729027873266897e-01,
     &      1.2517406323627462e+00, 2.0751129098523808e+00,
     &      3.1110524551477146e+00, 4.3642830769353065e+00,
     &      5.8407332713236055e+00, 7.5477046800234531e+00,
     &      9.4940953300264859e+00, 1.1690695926056069e+01,
     &      1.4150586187285759e+01, 1.6889671928527100e+01,
     &      1.9927425875242456e+01, 2.3287932824879903e+01,
     &      2.7001406056472355e+01, 3.1106464709046559e+01,
     &      3.5653703516328221e+01, 4.0711598185543110e+01,
     &      4.6376979557540103e+01, 5.2795432527283602e+01,
     &      6.0206666963057259e+01, 6.9068601975304347e+01,
     &      8.0556280819950416e+01/

!      PARAMETER (LaWF5 = (/
      DATA (LaWF5(I),I=1,79) / 1.6098281800110255e+00,
     &      1.6262567089449037e-01, 1.4492591904487846e+00,
     &      3.1413464064571323e-01, 9.0600198110176913e-03,
     &      1.3222940251164819e+00, 4.1560465162978422e-01,
     &      3.4155966014826969e-02, 3.9920814442273529e-04,
     &      1.2217252674706509e+00, 4.8027722216462992e-01,
     &      6.7748788910962143e-02, 2.6872914935624635e-03,
     &      1.5280865710465251e-05, 1.1402704725249586e+00,
     &      5.2098462052832328e-01, 1.0321597123176789e-01,
     &      7.8107811692581406e-03, 1.7147374087175731e-04,
     &      5.3171033687126004e-07, 1.0728118194241802e+00,
     &      5.4621121812849427e-01, 1.3701106844693015e-01,
     &      1.5700109452915889e-02, 7.1018522710384658e-04,
     &      9.4329687100378043e-06, 1.7257182336250307e-08,
     &      1.0158589580332265e+00, 5.6129491705706813e-01,
     &      1.6762008279797133e-01, 2.5760623071019968e-02,
     &      1.8645680172483614e-03, 5.4237201850757696e-05,
     &      4.6419616897304271e-07, 5.3096149480223697e-10,
     &      9.6699138945091101e-01, 5.6961457133995952e-01,
     &      1.9460349528263074e-01, 3.7280084775089407e-02,
     &      3.7770452605368474e-03, 1.8362253735858719e-04,
     &      3.6213089621868382e-06, 2.0934411591584102e-08,
     &      1.5656399544231742e-11, 9.2448733920121973e-01,
     &      5.7335101072566907e-01, 2.1803441204004675e-01,
     &      4.9621041774927162e-02, 6.4875466844757246e-03,
     &      4.5667727203270848e-04, 1.5605112957064066e-05,
     &      2.1721387415385585e-07, 8.7986819845463701e-10,
     &      4.4587872910682818e-13, 8.5386232773739834e-01,
     &      5.7235907069288550e-01, 2.5547924356911883e-01,
     &      7.4890941006461639e-02, 1.4096711620145414e-02,
     &      1.6473849653768340e-03, 1.1377383272808749e-04,
     &      4.3164914098046565e-06, 8.0379423498828602e-08,
     &      6.0925085399751771e-10, 1.3169240486156312e-12,
     &      3.3287369929782692e-16, 7.5047670518560539e-01,
     &      5.5491628460505815e-01, 3.0253946815328553e-01,
     &      1.2091626191182542e-01, 3.5106857663146820e-02,
     &      7.3097806533088429e-03, 1.0725367310559510e-03,
     &      1.0833168123639965e-04, 7.3011702591247581e-06,
     &      3.1483355850911864e-07, 8.1976643295418016e-09,
     &      1.1866582926793190e-10, 8.4300204226528705e-13/
      DATA (LaWF5(I),I=80,126) /
     &      2.3946880341857530e-15, 1.8463473073036743e-18,
     &      1.4621352854768128e-22, 6.7728655485117817e-01,
     &      5.3145650375475362e-01, 3.2675746542654360e-01,
     &      1.5694921173080897e-01, 5.8625131072344717e-02,
     &      1.6921776016516312e-02, 3.7429936591959084e-03,
     &      6.2770718908266166e-04, 7.8738679621849850e-05,
     &      7.2631523013860402e-06, 4.8222883273410492e-07,
     &      2.2424721664551585e-08, 7.0512415827308280e-10,
     &      1.4313056105380569e-11, 1.7611415290432366e-13,
     &      1.2016717578981511e-15, 3.9783620242330409e-18,
     &      5.1351867308233644e-21, 1.7088113927550770e-24,
     &      5.1820874276942667e-29, 6.2200206075592535e-01,
     &      5.0792308532951769e-01, 3.3840894389128295e-01,
     &      1.8364459415856996e-01, 8.0959353969207851e-02,
     &      2.8889923149962169e-02, 8.3060098239550965e-03,
     &      1.9127846396388331e-03, 3.5030086360234562e-04,
     &      5.0571980554969836e-05, 5.6945173834697106e-06,
     &      4.9373179873395243e-07, 3.2450282717915824e-08,
     &      1.5860934990330932e-09, 5.6305930756763865e-11,
     &      1.4093865163091798e-12, 2.3951797309583852e-14,
     &      2.6303192453168292e-16, 1.7460319202373756e-18,
     &      6.3767746470103704e-21, 1.1129154937804721e-23,
     &      7.3700721603011131e-27, 1.1969225386627985e-30,
     &      1.5871102921547987e-35 /

      INTERFACE GAUSSLA0
      MODULE PROCEDURE GAUSSLA0
      END INTERFACE

      INTERFACE GAUSSLE0
      MODULE PROCEDURE GAUSSLE0
      END INTERFACE

      INTERFACE GAUSSHE0
      MODULE PROCEDURE GAUSSHE0
      END INTERFACE


      INTERFACE GAUSSLE1
      MODULE PROCEDURE GAUSSLE1
      END INTERFACE

      INTERFACE GAUSSLE2
      MODULE PROCEDURE GAUSSLE2
      END INTERFACE

      INTERFACE GAUSSQ
      MODULE PROCEDURE GAUSSQ
      END INTERFACE

      CONTAINS
      SUBROUTINE GAUSSLE1 (N,WFout,BPOUT,XMI,XMA)
      USE GLOBALDATA,ONLY : EPS0
      USE FIMOD
!      USE QUAD , ONLY: LeBP,LeWF,LeIND,NLeW,minQnr
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:),INTENT(out) :: BPOUT, WFout
      DOUBLE PRECISION,             INTENT(in) :: XMI,XMA
      INTEGER,                     INTENT(inout) :: N
      ! local variables
      DOUBLE PRECISION :: Z1,SDOT, SDOT1, DIFF1
      DOUBLE PRECISION,PARAMETER :: SQTWOPI1 = 0.39894228040143D0 !=1/sqrt(2*pi)
      INTEGER :: NN,I,J,k

      ! The subroutine picks the lowest Gauss-Legendre
      ! quadrature needed to integrate the test function
      ! gaussint to the specified accuracy, EPS0.
      ! The nodes and weights between the  integration
      !  limits XMI and XMA (all normalized) are returned.
      ! Note that the weights are multiplied with
      ! 1/sqrt(2*pi)*exp(.5*bpout^2)

      IF (XMA.LE.XMI) THEN
!         PRINT * , 'Warning XMIN>=XMAX in GAUSSLE1 !',XMI,XMA
         RETURN
      ENDIF

      DO  I = minQnr, NLeW
         NN = N  !initialize
         DO J = LeIND(I)+1, LeIND(I+1)
            BPOUT (NN+1) = 0.5d0*(LeBP(J)*(XMA-XMI)+XMA+XMI)
            Z1 = BPOUT (NN+1) * BPOUT (NN+1)
                                !IF (Z1.LE.xCutOff2) THEN
            NN=NN+1
            WFout (NN) = 0.5d0 * SQTWOPI1 * (XMA - XMI) *
     &              LeWF(J) *EXP ( - 0.5d0* Z1  )
                                !ENDIF
         ENDDO

         SDOT = GAUSINT (XMI, XMA, - 2.5d0, 2.d0, 2.5d0, 2.d0)
         SDOT1 = 0.d0

         DO  k = N+1, NN
            SDOT1 = SDOT1+WFout(k)*(-2.5d0+2.d0*BPOUT(k) )*
     &           (2.5d0 + 2.d0 * BPOUT (k) )
         ENDDO
         DIFF1 = ABS (SDOT - SDOT1)

         IF (EPS0.GT.DIFF1) THEN
            N=NN
!            PRINT * ,'gaussle1, XMI,XMA,NN',XMI,XMA,NN
            RETURN
         END IF
      END DO
      RETURN
      END SUBROUTINE GAUSSLE1

      SUBROUTINE GAUSSLE0 (N, wfout, bpout, XMI, XMA, N0)
      USE GLOBALDATA, ONLY : EPSS
!      USE QUAD,        ONLY : LeBP,LeWF,NLeW,LeIND
      IMPLICIT NONE
      INTEGER, INTENT(in)                         :: N0
      INTEGER, INTENT(inout)                      :: N
      DOUBLE PRECISION, DIMENSION(:), INTENT(out) :: wfout,bpout
      DOUBLE PRECISION,                INTENT(in) :: XMI,XMA
! Local variables
       DOUBLE PRECISION,PARAMETER :: SQTWOPI1 = 0.39894228040143D0 !=1/sqrt(2*pi)
      DOUBLE PRECISION                            :: Z1
      INTEGER                                     :: J
      ! The subroutine computes Gauss-Legendre
      ! nodes and weights between
      ! the (normalized) integration limits XMI and XMA
      ! Note that the weights are multiplied with
      ! 1/sqrt(2*pi)*exp(.5*bpout^2) so that
      !  b
      ! int f(x)*exp(-x^2/2)/sqrt(2*pi)dx=sum f(bp(j))*wf(j)
      !  a                                 j

      IF (XMA.LE.XMI) THEN
         !PRINT * , 'Warning XMIN>=XMAX in GAUSSLE0 !',XMI,XMA
         RETURN  ! no more nodes added
      ENDIF
      IF ((XMA-XMI).LT.EPSS) THEN
         N=N+1
         BPout (N) = 0.5d0 * (XMA + XMI)
         Z1 = BPOUT (N) * BPOUT (N)
         WFout (N) = SQTWOPI1 * (XMA - XMI) *EXP ( - 0.5d0* Z1  )
         RETURN
      ENDIF
      IF (N0.GT.NLeW) THEN
         !PRINT * , 'error in GAUSSLE0, quadrature not available'
         STOP
      ENDIF
      !print *, 'GAUSSLE0',N0

      !print *, N
      DO  J = LeIND(N0)+1, LeIND(N0+1)

         BPout (N+1) = 0.5d0 * (LeBP(J) * (XMA - XMI) + XMA + XMI)
         Z1 = BPOUT (N+1) * BPOUT (N+1)
                     !         IF (Z1.LE.xCutOff2) THEN
         N=N+1            ! add a new node and weight
         WFout (N) = 0.5d0 * SQTWOPI1 * (XMA - XMI) *
     &        LeWF(J) *EXP ( - 0.5d0* Z1  )
                                !         ENDIF
      ENDDO
       !print *,BPout
      RETURN
      END SUBROUTINE GAUSSLE0

      SUBROUTINE GAUSSLE2 (N, wfout, bpout, XMI, XMA, N0)
      USE GLOBALDATA, ONLY : xCutOff,EPSS
!      USE QUAD,        ONLY : LeBP,LeWF,NLeW,LeIND,minQNr
      IMPLICIT NONE
      INTEGER, INTENT(in)                         :: N0
      INTEGER, INTENT(inout)                      :: N
      DOUBLE PRECISION, DIMENSION(:), INTENT(out) :: wfout,bpout
      DOUBLE PRECISION,                INTENT(in) :: XMI,XMA
! Local variables
      DOUBLE PRECISION                            :: Z1
      INTEGER                                     :: J,N1
      DOUBLE PRECISION,PARAMETER :: SQTWOPI1 = 0.39894228040143D0 !=1/sqrt(2*pi)
      ! The subroutine computes Gauss-Legendre
      ! nodes and weights between
      ! the (normalized) integration limits XMI and XMA
      ! This procedure select number of nodes
      ! depending on the length of the integration interval.
      ! Note that the weights are multiplied with
      ! 1/sqrt(2*pi)*exp(.5*bpout^2) so that
      !  b
      ! int f(x)*exp(-x^2/2)/sqrt(2*pi)dx=sum f(bp(j))*wf(j)
      !  a                                 j

      IF (XMA.LE.XMI) THEN
         !PRINT * , 'Warning XMIN>=XMAX in GAUSSLE2 !',XMI,XMA
         RETURN  ! no more nodes added
      ENDIF
!      IF (XMA.LT.XMI+EPSS) THEN
!         N=N+1
!         BPout (N) = 0.65d0 * (XMA + XMI)
!         Z1 = BPOUT (N) * BPOUT (N)
!         WFout (N) = SQTWOPI1 * (XMA - XMI) *EXP ( - 0.5d0* Z1  )
!         RETURN
!      ENDIF
      IF (N0.GT.NLeW) THEN
         !PRINT * , 'Warning in GAUSSLE2, quadrature not available'
      ENDIF
      !print *, 'GAUSSLE2',N0

      !print *, N
      N1=CEILING(0.5d0*(XMA-XMI)*DBLE(N0)/xCutOff) !0.65d0
      N1=MAX(MIN(N1,NLew),minQNr)

      DO  J = LeIND(N1)+1, LeIND(N1+1)

         BPout (N+1) = 0.5d0 * (LeBP(J) * (XMA - XMI) + XMA + XMI)
         Z1 = BPOUT (N+1) * BPOUT (N+1)
                                !         IF (Z1.LE.xCutOff2) THEN
         N=N+1                  ! add a new node and weight
         WFout (N) = 0.5d0 * SQTWOPI1 * (XMA - XMI) *
     &        LeWF(J) *EXP ( - 0.5d0* Z1  )
                                !         ENDIF
      ENDDO
      !PRINT * ,'gaussle2, XMI,XMA,N',XMI,XMA,N
       !print *,BPout
      RETURN
      END SUBROUTINE GAUSSLE2

      SUBROUTINE GAUSSHE0 (N, WFout, BPout, XMI, XMA, N0)
!      USE QUAD, ONLY : HeBP,HeWF,HeIND,NHeW
      IMPLICIT NONE
      INTEGER,                         INTENT(in) :: N0
      INTEGER,                      INTENT(inout) :: N
      DOUBLE PRECISION, DIMENSION(:), INTENT(out) :: wfout,bpout
      DOUBLE PRECISION,                INTENT(in) :: XMI,XMA
! Local variables
      DOUBLE PRECISION, PARAMETER :: SQPI1= 5.6418958354776D-1 !=1/sqrt(pi)
      DOUBLE PRECISION, PARAMETER :: SQTWO= 1.41421356237310D0   !=sqrt(2)
      INTEGER :: J
      ! The subroutine returns modified Gauss-Hermite
      ! nodes and weights between
      ! the integration limits XMI and XMA
      ! for the chosen number of nodes
      ! implicitly assuming that the integrand
      !  goes smoothly towards zero as its approach XMI or XMA
      ! Note that the nodes and weights are modified
      ! according to
      !  Inf
      ! int f(x)*exp(-x^2/2)/sqrt(2*pi)dx=sum f(bp(j))*wf(j)
      ! -Inf                               j

      IF (XMA.LE.XMI) THEN
         !PRINT * , 'Warning XMIN>=XMAX in GAUSSHE0 !',XMI,XMA
         RETURN ! no more nodes added
      ENDIF
      IF (N0.GT.NHeW) THEN
         !PRINT * , 'error in GAUSSHE0, quadrature not available'
         STOP
      ENDIF

      DO  J = HeIND(N0)+1, HeIND(N0+1)
         BPout (N+1) = HeBP (J) * SQTWO
         IF (BPout (N+1).GT.XMA) THEN
            RETURN
         END IF
         IF (BPout (N+1).GE.XMI) THEN
            N=N+1  ! add the node
            WFout (N) = HeWF (J) * SQPI1
         END IF
      ENDDO
      RETURN
      END SUBROUTINE GAUSSHE0

      SUBROUTINE GAUSSLA0 (N, WFout, BPout, XMI, XMA, N0)
      USE GLOBALDATA, ONLY :  SQPI1
!      USE QUAD, ONLY : LaBP5,LaWF5,LaIND,NLaW
      IMPLICIT NONE
      INTEGER, INTENT(in) :: N0
      INTEGER, INTENT(inout) :: N
      DOUBLE PRECISION, DIMENSION(:), INTENT(out) :: wfout,bpout
      DOUBLE PRECISION, INTENT(in) :: XMI, XMA
      INTEGER :: J
      ! The subroutine returns modified Gauss-Laguerre
      ! nodes and weights for alpha=-0.5 between
      ! the integration limits XMI and XMA
      ! for the chosen number of nodes
      ! implicitly assuming the integrand
      !  goes smoothly towards zero as its approach XMI or XMA
      ! Note that the nodes and weights are modified
      ! according to
      !  Inf
      ! int f(x)*exp(-x^2/2)/sqrt(2*pi)dx=sum f(bp(j))*wf(j)
      !  0                                 j

      IF (XMA.LE.XMI) THEN
         !PRINT * , 'Warning XMIN>=XMAX in GAUSSLA0 !',XMI,XMA
         RETURN !no more nodes added
      ENDIF
      IF (N0.GT.NLaW) THEN
         !PRINT * , 'error in GAUSSLA0, quadrature not available'
         STOP
      ENDIF

      DO  J = LaIND(N0)+1, LaIND(N0+1)
         IF (XMA.LE.0.d0) THEN
            BPout (N+1) = -SQRT(2.d0*LaBP5(J))
         ELSE
            BPout (N+1) = SQRT(2.d0*LaBP5(J))
         END IF
         IF (BPout (N+1).GT.XMA) THEN
            RETURN
         END IF
         IF (BPout (N+1).GE.XMI) THEN
            N=N+1 ! add the node
            WFout (N) = LaWF5 (J)*0.5d0*SQPI1
         END IF
      ENDDO
      !PRINT *,'gaussla0, bp',LaBP5(LaIND(N0)+1:LaIND(N0+1))
      !PRINT *,'gaussla0, wf',LaWF5(LaIND(N0)+1:LaIND(N0+1))
      RETURN
      END SUBROUTINE GAUSSLA0

      SUBROUTINE GAUSSQ(N, WF, BP, XMI, XMA, N0)
      USE GLOBALDATA, ONLY : xCutOff
!      USE QUAD       , ONLY : minQNr
      IMPLICIT NONE
      INTEGER,                         INTENT(in) :: N0
      INTEGER,                      INTENT(inout) :: N
      DOUBLE PRECISION, DIMENSION(:), INTENT(out) :: wf,bp
      DOUBLE PRECISION,                INTENT(in) :: XMI,XMA
      INTEGER                                     :: N1
      ! The subroutine returns
      ! nodes and weights between
      ! the integration limits XMI and XMA
      ! for the chosen number of nodes
      ! Note that the nodes and weights are modified
      ! according to
      !  Inf
      ! int f(x)*exp(-x^2/2)/sqrt(2*pi)dx=sum f(bp(j))*wf(j)
      !  0                                 j

      !IF (XMA.LE.XMI) THEN
      !   PRINT * , 'Warning XMIN>=XMAX in GAUSSQ !',XMI,XMA
      !   RETURN !no more nodes added
      !ENDIF
      CALL GAUSSLE0(N,WF,BP,XMI,XMA,N0)
      RETURN
      IF ((XMA.GE.xCutOff).AND.(XMI.LE.-xCutOff)) THEN
         CALL GAUSSHE0(N,WF,BP,XMI,XMA,N0)
      ELSE
         CALL GAUSSLE2(N,WF,BP,XMI,XMA,N0)
         RETURN
         IF (((XMA.LT.xCutOff).AND.(XMI.GT.-xCutOff)).OR.(.TRUE.)
     &        .OR.(XMI.GT.0.d0).OR.(XMA.LT.0.d0)) THEN
                                ! Grid by Gauss-LegENDre quadrature
            CALL GAUSSLE2(N,WF,BP,XMI,XMA,N0)
         ELSE
            ! this does not work well
            !PRINT *,'N0',N0,N
            N1=CEILING(DBLE(N0)/2.d0)
            IF (XMA.GE.xCutOff) THEN
               IF (XMI.LT.0.d0) THEN
                  CALL GAUSSLE2 (N, WF, BP,XMI ,0.d0,N0)
               ENDIF
               CALL GAUSSLA0 (N, WF, BP,0.d0, XMA, N1)
            ELSE
               IF (XMA.GT.0.d0) THEN
                  CALL GAUSSLE2 (N, WF,BP,0.d0,XMA,N0)
               ENDIF
               CALL GAUSSLA0 (N, WF,BP,XMI,0.d0, N1)
            END IF
         END IF
      ENDIF
      !PRINT *,'gaussq, wf',wf(1:N)
      !PRINT *,'gaussq, bp',bp(1:N)
      RETURN
      END SUBROUTINE GAUSSQ
      END MODULE QUAD

      MODULE RIND71MOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: RIND71, INITDATA, SETDATA,ECHO

      INTERFACE
         FUNCTION MVNFUN(N,Z) result (VAL)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         DOUBLE PRECISION :: VAL
         END FUNCTION MVNFUN
      END INTERFACE

      INTERFACE
         FUNCTION MVNFUN2(N,Z) result (VAL)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         DOUBLE PRECISION :: VAL
         END FUNCTION MVNFUN2
      END INTERFACE

      INTERFACE
         FUNCTION FI( Z ) RESULT (VALUE)
         DOUBLE PRECISION, INTENT(in) :: Z
         DOUBLE PRECISION :: VALUE
         END FUNCTION FI
      END INTERFACE

      INTERFACE
         FUNCTION FIINV( Z ) RESULT (VALUE)
         DOUBLE PRECISION, INTENT(in) :: Z
         DOUBLE PRECISION :: VALUE
         END FUNCTION FIINV
      END INTERFACE

      INTERFACE
         FUNCTION JACOB(XD,XC) RESULT (VALUE)
         DOUBLE PRECISION, DIMENSION(:), INTENT(in) :: XD,XC
         DOUBLE PRECISION :: VALUE
         END FUNCTION JACOB
      END INTERFACE

      INTERFACE RIND71
      MODULE PROCEDURE RIND71
      END INTERFACE

      INTERFACE SETDATA
      MODULE PROCEDURE SETDATA
      END INTERFACE

      INTERFACE INITDATA
      MODULE PROCEDURE INITDATA
      END INTERFACE

      INTERFACE ARGP0
      MODULE PROCEDURE ARGP0
      END INTERFACE

      INTERFACE  RINDDND
      MODULE PROCEDURE RINDDND
      END INTERFACE

      INTERFACE  RINDSCIS
      MODULE PROCEDURE RINDSCIS
      END INTERFACE

      INTERFACE RINDNIT
      MODULE  PROCEDURE RINDNIT
      END INTERFACE

      INTERFACE  BARRIER
      MODULE PROCEDURE BARRIER
      END INTERFACE

      INTERFACE echo
      MODULE PROCEDURE echo
      END INTERFACE

      INTERFACE swapRe
      MODULE PROCEDURE swapRe
      END INTERFACE

      INTERFACE swapint
      MODULE PROCEDURE swapint
      END INTERFACE

      INTERFACE getdiag
      MODULE PROCEDURE getdiag
      END INTERFACE

      INTERFACE CONDSORT0
      MODULE PROCEDURE CONDSORT0
      END INTERFACE

      INTERFACE CONDSORT
      MODULE PROCEDURE CONDSORT
      END INTERFACE


      INTERFACE CONDSORT2
      MODULE PROCEDURE CONDSORT2
      END INTERFACE

      INTERFACE CONDSORT3
      MODULE PROCEDURE CONDSORT3
      END INTERFACE

      INTERFACE CONDSORT4
      MODULE PROCEDURE CONDSORT4
      END INTERFACE

      CONTAINS
      SUBROUTINE SETDATA(method,scale, dEPSS,dREPS,dEPS2,
     &     dNIT,dXc, dNINT,dXSPLT)
      USE GLOBALDATA
      USE FIMOD
      USE QUAD, ONLY: sizNint,Nint1,minQnr,Le2Qnr
      IMPLICIT NONE
      DOUBLE PRECISION , INTENT(in) :: scale, dEPSS,dREPS
	DOUBLE PRECISION , INTENT(in) :: dEPS2,dXc, dXSPLT
      !INTEGER, DIMENSION(:), INTENT(in) :: dNINT
      INTEGER, INTENT(in) :: method,dNINT,dNIT
      INTEGER :: N=1

      !N=SIZE(dNINT)
      IF (sizNint.LT.N) THEN
         !PRINT *,'Error in setdata, Nint too large'
         N=sizNint
      ENDIF
      NINT1(1:N)=dNINT !(1:N)  ! quadrature formulae for the Xd variables
      IF (N.LT.sizNint) THEN
         NINT1(N:sizNint)=NINT1(N)
      END IF
      minQnr = 1
      Le2Qnr = NINT1(1)

      SCIS = method
      XcScale = scale
      RelEps   = dREPS
      EPSS     = dEPSS          ! accuracy of integration
      CEPSS    = 1.d0 - EPSS
      EPS2     = dEPS2          ! Constants controlling
      EPS      = SQRT(EPS2)
      xCutOff  = dXc
      XSPLT = dXSPLT
      NIT = dNIT

      IF (Nc.LT.1)  NUGGET=0.d0        ! Nugget is not needed when Nc=0

      IF (EPSS.LE.1e-4)      NsimMax=2000
      IF (EPSS.LE.1e-5)      NsimMax=4000
      IF (EPSS.LE.1e-6)      NsimMax=8000
      RETURN
      IF (.FALSE.) THEN
      print *,'Requested parameters :'
      SELECT CASE (SCIS)
      CASE (:0)
         PRINT *,'NIT = ',NIT,' integration by quadrature'
      CASE (1)
         PRINT *,'SCIS = 1 SADAPT if NDIM<9 otherwise by KRBVRC'
      CASE (2)
         PRINT *,'SCIS = 2 SADAPT if NDIM<20 otherwise by KRBVRC'
      CASE (3)
         PRINT *,'SCIS = 3 KRBVRC (Ndim<101)'
      CASE (4)
         PRINT *,'SCIS = 4 KROBOV (Ndim<101)'
      CASE (5)
          PRINT *,'SCIS = 5 RCRUDE (Ndim<1001)'
      CASE (6)
        PRINT *,'SCIS = 6 SOBNIED (Ndim<1041)'
      CASE (7:)
         PRINT *,'SCIS = 7 DKBVRC (Ndim<1001)'
      END SELECT
      PRINT *,'EPSS = ', EPSS, ' RELEPS = ' ,RELEPS
      PRINT *,'EPS2 = ',EPS2, ' xCutOff = ',xCutOff
      PRINT *,'NsimMax = ',NsimMax             !,FIINV(EPSS)
      ENDIF
      RETURN
      END SUBROUTINE SETDATA

      SUBROUTINE INITDATA (speed)
      USE GLOBALDATA
      USE FIMOD
      USE QUAD, ONLY: sizNint,Nint1,minQnr,Le2Qnr
      IMPLICIT NONE
      INTEGER , INTENT(in) :: speed
      SELECT CASE (speed)
      CASE (9:)
         NINT1 (1) = 2
         NINT1 (2) = 3
         NINT1 (3) = 4
      CASE (8)
         NINT1 (1) = 3
         NINT1 (2) = 4
         NINT1 (3) = 5
      CASE (7)
         NINT1 (1) = 4
         NINT1 (2) = 5
         NINT1 (3) = 6
      CASE (6)
         NINT1 (1) = 5
         NINT1 (2) = 6
         NINT1 (3) = 7
      CASE (5)
         NINT1 (1) = 6
         NINT1 (2) = 7
         NINT1 (3) = 8
      CASE (4)     ! quadrature formulae for the Xd variables
         NINT1 (1) = 7      ! use quadr. form. No. 6 in integration of Xd(1)
         NINT1 (2) = 8      ! use quadr. form. No. 7 in integration of Xd(2)
         NINT1 (3) = 9      ! use quadr. form. No. 8 in integration of Xd(3)
      CASE (3)
         NINT1 (1) = 8
         NINT1 (2) = 9
         NINT1 (3) = 10
      CASE (2)
         NINT1 (1) = 9
         NINT1 (2) = 10
         NINT1 (3) = 11
      CASE (:1)
         NINT1 (1) = 11
         NINT1 (2) = 12
         NINT1 (3) = 13
      END SELECT
      NsimMax=1000*abs(10-min(speed,9))
      NsimMin=0
      SELECT case (speed)
      CASE (11:)
         EPSS = 1d-1
      CASE (10)
         EPSS = 1d-2
      CASE (7:9)
         EPSS = 1d-3
      CASE (4:6)
         EPSS = 1d-4
      CASE (:3)
         EPSS = 1d-5
      END SELECT


      EPSS=EPSS*1d-1
      RELEPS = MIN(EPSS ,1.d-2)
      EPS2=EPSS*1.d1
      !EPS2*1.d+1
      !EPS2=1.d-10
      !xCutOff=MIN(MAX(ABS(FIINV(EPSS)),3.5d0),5.d0)
      !xCutOff=ABS(FIINV(EPSS*1.d-1))  ! this is good
      xCutOff=ABS(FIINV(EPSS))
      !xCutOff=ABS(FIINV(EPSS*5.d-1))
      if (SCIS.gt.0) then
         xCutOff= MIN(MAX(xCutOff+0.5d0,4.d0),5.d0)
! This gives approximately the same accuracy as when using RINDDND and RINDNIT
         EPSS=EPSS*1.d+2
         !EPS2=1.d-10
      endif
      NINT1(1:sizNint)=NINT1(3)
      Le2Qnr=NINT1(1)
      minQnr=1                  ! minimum quadrature No. used in GaussLe1,Gaussle2

      NUGGET = EPS2*1.d-1
      IF (Nc.LT.1) NUGGET=0.d0               ! Nugget is not needed when Nc=0
      EPS = SQRT(EPS2)
      CEPSS = 1.d0 - EPSS

! If SCIS=0 then the absolute error is usually less than EPSS*100
! otherwise absolute error is less than EPSS

      return
      IF (.FALSE.) THEN
      print *,'Requested parameters :'
      SELECT CASE (SCIS)
      CASE (:0)
         PRINT *,'NIT = ',NIT,' integration by quadrature'
      CASE (1)
         PRINT *,'SCIS = 1 SADAPT if NDIM<9 otherwise by KRBVRC'
      CASE (2)
         PRINT *,'SCIS = 2 SADAPT if NDIM<19 otherwise by KRBVRC'
      CASE (3)
         PRINT *,'SCIS = 3 KRBVRC (Ndim<101)'
      CASE (4)
         PRINT *,'SCIS = 4 KROBOV (Ndim<101)'
      CASE (5)
          PRINT *,'SCIS = 5 RCRUDE (Ndim<1001)'
      CASE (6)
        PRINT *,'SCIS = 6 SOBNIED (Ndim<1041)'
      CASE (7:)
         PRINT *,'SCIS = 7 DKBVRC (Ndim<1001)'
      END SELECT
      PRINT *,'EPSS = ', EPSS, ' RELEPS = ' ,RELEPS
      PRINT *,'EPS2 = ',EPS2, ' xCutOff = ',xCutOff
      PRINT *,'NsimMax = ',NsimMax             !,FIINV(EPSS)
      ENDIF
      RETURN
      END SUBROUTINE INITDATA

      SUBROUTINE ECHO(array)
      INTEGER ::j
      DOUBLE PRECISION,DIMENSION(:,:)::array
      DO j=1,size(array,1)
         PRINT 111,j,array(j,:)
111      FORMAT (i2,':',10F10.5)
      END DO
      END SUBROUTINE ECHO

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!******************* RIND71 - the main program *********************!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE RIND71(fxind,BIG1,Ex,xc1,Nt1,indI,Blo,Bup)
      USE FUNCMOD, ONLY : BIG, Cm,CmN,xd,xc
      USE GLOBALDATA, ONLY :Nt,Nj,Njj,Nd,Nc,Nx,Ntd,Ntdc,NsXtmj,NsXdj,
     &     indXtd,index1,xedni,SQ,Hlo,Hup,fxcepss,EPS2,XCEPS2,NIT,
     &     SQTWOPI1,xCutOff,SCIS,Ntscis,COVix,EPS, xcScale
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: BIG1
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: xc1
      DOUBLE PRECISION, DIMENSION(:), INTENT(in) :: Ex
      DOUBLE PRECISION, DIMENSION(:), INTENT(out):: fxind
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: Blo, Bup
      INTEGER,          DIMENSION(:), INTENT(in) :: indI
      INTEGER, INTENT(IN) :: Nt1
! local variables
      INTEGER          :: J,ix,Ntdcmj,Nst,Nsd,INFORM
      DOUBLE PRECISION :: xind,SQ0,xx,fxc,quant
!      IF (.NOT.PRESENT(xcScale)) THEN
!         xcScale = 0.0d0
!      ENDIF
      Nt =Nt1
      !print *,'rindd SCIS',SCIS
      Nc = size(xc1,dim=1)
      Nx = MAX(size(xc1,dim=2),1)
      Ntdc = size(BIG1,dim=1)
      IF (Nt+Nc.GT.Ntdc) Nt=Ntdc-Nc  ! make sure it does not exceed Ntdc-Nc
      Nd = Ntdc - Nt - Nc
      Ntd = Nt + Nd

                                !Initialization
                                !Call Initdata(speed)
      Nj = MIN(Nj,MAX(Nt,0))      ! make sure Nj<=Nt
      Njj = MIN(Njj,MAX(Nt-Nj,0)) ! make sure Njj<=Nt-Nj
      ALLOCATE(xc(1:Nc))
      IF (Nd.GT.0) THEN
         ALLOCATE(xd(1:Nd))
         xd = 0.d0
      END IF

      If (SCIS.GT.0) then
         Ntscis=Nt-Nj-Njj
         ALLOCATE(SQ(1:Ntd,1:Ntd)) ! Cond. stdev's
         ALLOCATE(NsXtmj(1:Ntd+1)) ! indices to stoch. var. See condsort
      else
         Ntscis=0
         ALLOCATE(SQ(1:Ntd,1:max(Njj+Nj+Nd,1)) ) ! Cond. stdev's
         ALLOCATE(NsXtmj(1:Nd+Nj+Njj+1)) ! indices to stoch. var. See condsort
      endif
      ALLOCATE(BIG(Ntdc,Ntdc))
      ALLOCATE(Cm(Ntdc),CmN(Ntd)) !Cond. mean which has the same order as local
      Cm = 0.d0                !covariance matrices (after sorting) or excluding
                               !irrelevant variables.

      ALLOCATE(index1(Ntdc))   ! indices to the var. original place in BIG
      index1=(/(J,J=1,Ntdc)/)  ! (before sort.)
      ALLOCATE(xedni(Ntdc))    ! indices to var. new place (after sorting),
      xedni=index1             ! eg. the point xedni(1) is the original position
                               ! of variable with conditional mean CM(1).
      ALLOCATE(Hlo(Ntd))       ! lower and upper integration limits are computed
                               ! in the new order that is the same as CM.
                               ! This convention is expressed in the vector indXTD.
      Hlo = 0.d0               ! However later on some variables will be exluded
                               ! since those are irrelevant and hence CMnew(1)
                               ! does not to be conditional mean of the same variable
                               ! as CM(1) is from the beginning. Consequently
      ALLOCATE(Hup(Ntd))       ! the order of Hup, Hlo will be unchanged. So we need
      Hup=0.d0                 ! to know where the relevant variables bounds are
                               ! This will be given in the subroutines by a vector indS.

      ALLOCATE(NsXdj(Nd+Nj+1))  ! indices to stoch. var. See condsort
      NsXdj=0
      ALLOCATE(indXtd(Ntd))     ! indices to Xt and Xd as they are
      indXtd=(/(J,J=1,Ntd)/)    ! sorted in Hlo and Hup


      BIG = BIG1(1:Ntdc,1:Ntdc)   !conditional covariance matrix BIG

      IF (.TRUE.) THEN   ! sort by shortest expected int. interval
         Cm = Ex (1:Ntdc)
         !xc = SUM(xc1(1:Nc,1:Nx),DIM=2)/DBLE(Nx) ! average of all xc's
         xc = xc1(1:Nc,max(Nx/2,1))       ! Or select the one in the middle
         CALL BARRIER(xc,indI,Blo,Bup) ! compute average integrationlimits

 !     print *,'rindd,xcmean:',xc
 !     print *,'rindd,Hup:',Hup
 !     print *,'rindd,Hlo:',Hlo

         CALL CONDSORT0(BIG,Cm,xc,SQ,index1,xedni,NsXtmj,NsXdj,INFORM)
      ELSE                        ! sort by decreasing cond. variance
         CALL CONDSORT (BIG,SQ,index1,xedni,NsXtmj,NsXdj,INFORM)
      ENDIF
      IF (INFORM.GT.0) GOTO 110 !Degenerated case the density can not computed

!      PRINT *, 'index=', index1
!      PRINT *,(sqrt(BIG(J,J)),J=1,Ntdc)
!      PRINT *, 'BIG'
!      CALL ECHO(BIG(1:Ntdc,1:MIN(Ntdc,10)))
      !PRINT *, 'xedni=', xedni
      !print *,'NsXtmj=',NsXtmj
      !print *,'NsXdj=',NsXdj

      fxind  = 0.d0             ! initialize
                                ! Now the loop over all different values of
                                ! variables Xc (the one one is conditioning on)
      DO  ix = 1, Nx            ! is started. The density f_{Xc}(xc(:,ix))
         COVix = ix             ! will be computed and denoted by  fxc.
         xind = 0.d0
         fxc = 1.d0
!         Cm  = Ex (1:Ntdc)
!         index1=(/(J,J=1,Ntdc)/)
!         xedni=index1
!         BIG = BIG1(1:Ntdc,1:Ntdc)
!         CALL BARRIER(xc1(1:Nc,ix),indI,Blo,Bup) ! integrationlimits
!         CALL CONDSORT0 (BIG,Cm,xc1(:,ix),SQ, index1,
!     &        xedni, NsXtmj,NsXdj)

                                ! Set the original means of the variables
         Cm  =Ex (index1(1:Ntdc)) !   Cm(1:Ntdc)  =Ex (index1(1:Ntdc))
         quant = 0.0d0
         DO J = 1, Nc           !Recursive conditioning on the last Nc variables
            Ntdcmj=Ntdc-J
            SQ0 = BIG(Ntdcmj+1,Ntdcmj+1) ! SQRT(var(X(i)|X(i+1),X(i+2),...,X(Ntdc)))
                                       ! i=Ntdc-J+1 (J=1 var(X(Ntdc))

            xx = (xc1(index1(Ntdcmj+1)-Ntd,ix)-Cm(Ntdcmj+1))/SQ0
                          !Trick to calculate
                          !fxc = fxc*SQTWPI1*EXP(-0.5*(XX**2))/SQ0
            quant = quant - 0.5d0 * xx * xx + LOG(SQTWOPI1) - LOG(SQ0)

                                ! conditional mean (expectation)
                                ! E(X(1:i-1)|X(i),X(i+1),...,X(Ntdc))
            Cm(1:Ntdcmj) = Cm(1:Ntdcmj)+xx*BIG (1:Ntdcmj,Ntdcmj+1)
         ENDDO
! fxc probability density for i=Ntdc-J+1,
! fXc=f(X(i)|X(i+1),X(i+2)...X(Ntdc))*
               !     f(X(i+1)|X(i+2)...X(Ntdc))*..*f(X(Ntdc))

         fxc = EXP(QUANT+XcScale)
         !print *,'density',fxc                 ! J
         !PRINT *, 'Rindd, Cm=',Cm(xedni(max(1,Nt-5):Ntdc))
         !PRINT *, 'Rindd, Cm=',Cm(xedni(1:Ntdc))

         !IF (fxc .LT.fxcEpss)  print *,'small, fxc=',fxc
         IF (fxc .LT.fxcEpss) GOTO 100 ! Small probability don't bother calculating it

                          !set the global integration limits Hlo,Hup
         CALL BARRIER(xc1(1:Nc,ix),indI,Blo,Bup)



         Nst = NsXtmj(Ntscis+Njj+Nd+Nj+1)
         Nsd = NsXdj(Nd+Nj+1)
         IF (any((Cm(Nst+1:Nsd-1) .GT.Hup(Nst+1:Nsd-1)+EPS ).OR.
     *     (Cm (Nst+1:Nsd-1)+EPS .LT.Hlo (Nst+1:Nsd-1)))) GO TO 100  !degenerate case
                                !mean of deterministic variable(s) is
                                ! outside the barriers

        !PRINT *,'RINDD SCIS',SCIS
        IF (SCIS.GE.1.AND.SCIS.LE.9) then     ! integrate all by SCIS
           XIND=RINDSCIS(xc1(:,ix))
           GO TO 100
        endif

        SELECT CASE (Nd+Nj)
        CASE (:0)
           IF (SCIS.NE.0) then  ! integrate all by SCIS
              XIND=MNORMPRB(Cm(1:Nst))
           ELSE
              XIND=RINDNIT(BIG,SQ(1:Nst,1),Cm,indXtd(1:Nst),NIT)
           END IF
        CASE (1:)
           xind=RINDDND(BIG,Cm,xd,xc1(:,ix),Nd,Nj)
        END SELECT
 100    fxind(ix)=xind*fxc
        !IF (fxc .LT.fxcEpss)  print *,'small, fxc, xind',fxc,xind
        !PRINT *, 'Rindd, Cm=',Cm(xedni(1:Ntdc))
      ENDDO                     !ix
!      PRINT *, 'Rindd, Cm=',Cm(xedni(1:Ntdc))
 110  CONTINUE
      IF (ALLOCATED(xc)) DEALLOCATE(xc)
      IF (ALLOCATED(xd)) DEALLOCATE(xd)
      IF (ALLOCATED(SQ)) DEALLOCATE(SQ)
      IF (ALLOCATED(NsXtmj)) DEALLOCATE(NsXtmj)
      IF (ALLOCATED(Cm)) DEALLOCATE(Cm)
      IF (ALLOCATED(CmN)) DEALLOCATE(CmN)
      IF (ALLOCATED(BIG)) DEALLOCATE(BIG)
      IF (ALLOCATED(index1)) DEALLOCATE(index1)
      IF (ALLOCATED(xedni)) DEALLOCATE(xedni)
!      print *,'before dealocation',Ntd,size(Hup),size(Hlo)
      IF (ALLOCATED(Hlo)) DEALLOCATE(Hlo)
      IF (ALLOCATED(Hup)) DEALLOCATE(Hup)
      IF (ALLOCATED(NsXdj)) DEALLOCATE(NsXdj)
      IF (ALLOCATED(indXtd)) DEALLOCATE(indXtd)
      RETURN
      END SUBROUTINE RIND71

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!*************************** ARGP0 *********************************!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE ARGP0 (I0,I1,P0,Plo,SQ,Cm,indS,ind,Nind)
      USE FIMOD
      USE GLOBALDATA, ONLY : Hlo,Hup,xCutOff,EPSS,EPS2,EPS
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:), INTENT(in)  :: SQ , Cm !stdev./mean
      INTEGER,          DIMENSION(:), INTENT(in)  :: indS
      INTEGER,          DIMENSION(:), INTENT(out) :: ind
      DOUBLE PRECISION,               INTENT(out) :: P0,Plo
      INTEGER,                        INTENT(out) :: I0, I1
      INTEGER,                        INTENT(out) :: Nind
      DOUBLE PRECISION                            :: P1,Prb
      DOUBLE PRECISION                            :: Xup, Xlo
      INTEGER                                     :: I, Nstoc
                                ! indS contains the indices to the limits
      Nstoc = SIZE(indS)        ! in Hlo/Hup of variables in the indicator
                                ! ind contains indices to the relevant
                                ! variables which are Nind<=Nstoc.
                                ! We wish to compute P(Hlo<X<Hup) but
                                ! only have lower and upper bounds Plo,P0, resp.
                                ! I0 is the position of the minimal
                                ! probability in the vector ind, i.e.
                                ! P0=P(Hlo<X(indS(ind(I0)))<Hup)
                                ! I1 is the second minimum.
      P0   = 2.d0
      P1   = 2.d0
      I0   = 1
      I1   = 1
      Plo  = 0.d0
      Nind = 0

      DO I = 1,Nstoc,1
         Xup = xCutOff
         Xlo =-xCutOff
         IF (SQ(I).GE.EPS2) THEN
            Xup = MIN( (Hup (indS(I)) - Cm (I))/ SQ(I),Xup)
            Xlo = MAX( (Hlo (indS(I)) - Cm (I))/ SQ(I),Xlo)
          ELSE
            IF (Hup(indS(I))+EPS.LT.Cm (I)) Xup = Xlo
            IF (Hlo(indS(I)).GT.Cm (I)+EPS) Xlo = Xup
            !PRINT *,'argpo',Xlo,Xup
         END IF
         IF (Xup.LE.Xlo+EPSS) THEN  ! +EPSS
            P0     = 0.d0
            Plo    = 0.d0
            ind(1) = I
            I0     = 1
            Nind   = 1
            RETURN
         ENDIF

       IF ((Xup+EPSS.LT.xCutOff).or.(Xlo+xCutOff.GT.EPSS)) THEN
         Nind      = Nind+1
         ind(Nind) = I
                                ! this procedure calculates
         Prb = FI(Xup)-FI(Xlo)
         Plo = Plo+Prb
         IF (Prb.LT.P0) THEN
            I1 = I0
            I0 = Nind
            P1 = P0             ! Prob(I0)=Prob(XMA>X(i0)>XMI)=
            P0 = Prb            !     min Prob(Hup(i)> X(i)>Hlo(i))
            IF (P0.LT.EPSS) THEN
               Plo=0.d0
               RETURN
            ENDIF
         ELSEIF (Prb.LT.P1) THEN
            I1 = Nind
            P1 = Prb
         ENDIF
       ENDIF
      ENDDO

      Plo = MAX(0.d0,1.d0-DBLE(Nind)+Plo)
      P0 = MIN(1.d0,P0)
!      print *,'ARGP0',Nstoc,Nind,P0,Plo,I0,I1,CM(ind(I0))
      RETURN
      END SUBROUTINE ARGP0



!Ntmj is the number of elements in indicator
!since Nj points of process valeus (Nt) have
!been moved to the jacobian.
!index1 contains the original
!positions of variables in the
!covaraince matrix before condsort
!and that why if index(Ntmj+1)>Nt
!it means the variable to conditon on
!is a derivative isXd=1

!= # stochastic variables before
!conditioning on X(Ntmj+1). This
!I still not checked why.



! ******************* RINDDND ****************************************
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      RECURSIVE FUNCTION RINDDND (BIG,Cm,xd,xc,Ndleft,Njleft)
     &      RESULT (xind)
      USE JACOBMOD
      USE GLOBALDATA, ONLY :SQPI1, SQTWOPI1,Hup,Hlo,Nt,Nj,Njj,Nd,
     &     NsXtmj,NsXdj,EPS2,NIT,xCutOff,EPSS,CEPSS,index1,
     &     indXtd,SQ,SQTWO,SQTWO1,SCIS,Ntscis,C1C2det,EPS
      USE FIMOD
      USE C1C2MOD
      USE QUAD
      IMPLICIT NONE
      INTEGER,INTENT(in) :: Ndleft,Njleft ! # DIMENSIONs to integrate
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: BIG
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in) :: Cm ! conditional mean
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(inout) :: xd ! integr. variables
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)    :: xc ! conditional values
!local variables
      DOUBLE PRECISION                                :: xind
      DOUBLE PRECISION                                :: xind1
      DOUBLE PRECISION, DIMENSION(PMAX)             :: WXdi, Xdi !weights/nodes
      DOUBLE PRECISION, DIMENSION(:  ), ALLOCATABLE :: CmNEW
      INTEGER  :: Nrr, Nr, J, N,Ndleft1,Ndjleft,Ntmj,isXd
      INTEGER :: Nst,Nstn,Nsd,NsdN
      DOUBLE PRECISION     :: SQ0,fxd,XMA,XMI

      Ntmj=Nt-Nj
      Ndjleft= Ndleft+Njleft
      N=Ntmj+Ndjleft

      IF (index1(N).GT.Nt) THEN
         isXd=1
      ELSE
         isXd=0
      END IF

      XIND = 0.d0
      SQ0  = BIG (N, N)
! index to last stoch. variable of Xt before conditioning on X(N)
      Nst  = NsXtmj(Ntscis+Njj+Ndjleft+1)

!********************************************************************************
!** Here Starts the degenerated case the remaining variables are deterministic **
!********************************************************************************

      IF (SQ0.LT.EPS2) THEN
                                !Next is the check for the special situation
                                !that after conditioning on Xc all derivatives are
                                !singular and not satisfying the limitations
                                !(so something is generally wrong)
         IF (any((Cm(Nst+1:N).GT.Hup(Nst+1:N)+EPS ).OR.
     &         (Cm(Nst+1:N)+EPS.LT.Hlo(Nst+1:N)))) THEN
            RETURN               !the mean of Xd or Xt is too extreme
         ENDIF
                                !Here we are putting in all conditional expectations
                                !for the values of the "deterministic" derivatives.
         IF (Nd.GT.0) THEN
            Ndleft1=Ndleft
            DO WHILE (Ndleft1.GT.0)
               IF (index1(N).GT.Nt) THEN  ! isXd
                  xd (Ndleft1) =  Cm (N)
                  Ndleft1=Ndleft1-1
               END IF
               N=N-1
            ENDDO
            fxd = jacob (xd,xc) ! jacobian of xd,xc
         ELSE
            fxd = 1.d0 !     XIND = FxCutOff???
         END IF

         XIND=fxd
         IF (Nst.le.0) RETURN
         IF (SCIS.ne.0) then
             XIND=fxd*MNORMPRB(Cm(1:Nst))
         ELSE
            XIND=fxd*RINDNIT(BIG,SQ(:,Ntscis+Njj+1),
     &         Cm,indXtd(1:Nst),NIT)
         END IF
         RETURN
      ENDIF

!***** Here Starts the conditioning on the last variable (nondeterministic) *
!****************************************************************************

      !      SQ0 = SQ(N,Ntscis+Njj+Ndjleft) !SQRT (SS0)

      !print *,'RINDD SQO', SQ0,SQ(N,Ntscis+Njj+Ndjleft)  !SQ(1:N,Ndjleft)

      XMA=MIN((Hup (indXtd(N))-Cm (N))/SQ0, xCutOff)
      XMI=MAX((Hlo (indXtd(N))-Cm (N))/SQ0,-xCutOff)

         ! See if we can narrow down integration range
      ! index to first stoch. variable of Xd before conditioning on X(N)
      Nsd  = NsXdj(Ndjleft+1)
      ! index to last stoch. variable of Xt after cond. on X(N)
      NstN = NsXtmj(Ntscis+Njj+Ndjleft)

      !PRINT *,xmi,xma
!      print *,Ntscis+Njj+Ndjleft
!      print *,'CM=',Cm(1:N-1)
!      print *,'SQ=', SQ(1:N-1,Ntscis+Njj+Ndjleft)
      if (C1C2det) then    ! checking only on the variables that becomes deterministic
!        index to first stoch. variable of Xd after conditioning on X(N)
         NsdN    = NsXdj(Ndjleft)
         CALL C1C2(XMI,XMA,Cm(Nsd:NsdN-1),BIG(Nsd:NsdN-1,N),
     &        SQ(Nsd:NsdN-1,Ntscis+Njj+Ndjleft),indXtd(Nsd:NsdN-1))
         CALL C1C2(XMI,XMA,Cm(NstN+1:Nst),BIG(NstN+1:Nst,N),
     &        SQ(NstN+1:Nst,Ntscis+Njj+Ndjleft),indXtd(NstN+1:Nst))
      else                      ! check on all variables
         CALL C1C2(XMI,XMA,Cm(Nsd:N-1),BIG(Nsd:N-1,N),
     &        SQ(Nsd:N-1,Ntscis+Njj+Ndjleft),indXtd(Nsd:N-1))
         CALL C1C2(XMI,XMA,Cm(1:Nst),BIG(1:Nst,N),
     &        SQ(1:Nst,Ntscis+Njj+Ndjleft),indXtd(1:Nst))
      endif
!      CALL C1C2(XMI,XMA,Cm(1:N-1),BIG(1:N-1,N),
!     &        SQ(1:N-1,Ntscis+Njj+Ndjleft),SQ0,indXtd(1:N-1))
      !PRINT *,xmi,xma
!      if (Ndleft<2) stop
      IF (XMA.LE.XMI) THEN
         XIND=0.d0
         RETURN
      ENDIF
      Nrr = NINT1 (MIN(Ndjleft,sizNint))
      Nr=0 ! initialize # of nodes
      !print *, 'rinddnd Nrr',Nrr
                        !Grid the interval [XMI,XMA] by  GAUSS quadr.
      CALL GAUSSLE2(Nr, WXdi, Xdi,XMI,XMA, Nrr)
                                !print *, 'Xdi',Xdi
      ALLOCATE(CmNEW(1:N-1))
                                ! The following variables are independent of X(N)
                                ! because BIG(Nst+1:Nsd-1,N) is set to 0 in condsrort.
                                ! Thus the mean is not changed for these variables
                                ! in order to avoid numerical problems
      ! The following if test is necessary on Solaris F90 compiler.
      if (Nst+1.LT.Nsd) CmNEW(Nst+1:Nsd-1)=Cm(Nst+1:Nsd-1)
!     print *,Ndjleft,N,NstN+1,Nsd-1
!     print *,BIG(Nst+1:Nsd-1,N)
!     print *,'Cm=',Cm(NstN+1:Nsd-1)
      DO J = 1, Nr
!        IF (Wxdi(J).GT.(CFxCutOff)) GO TO 100      !THEN ! EPSS???
         IF (isXd.EQ.1) xd (Ndleft) =  Xdi (J)*SQ0 + Cm (N)

                                       !  Here we start with the case when there
                                       !  some derivatives left to integrate.
         ! The following if test is necessary on Solaris F90 compiler.
         if (1.LE.Nst) CmNEW(1:Nst) = Cm(1:Nst)+Xdi(J)*BIG(1:Nst,N)
         if (Nsd.LT.N) CmNEW(Nsd:(N-1)) = Cm(Nsd:(N-1))+
     &        Xdi(J)*BIG(Nsd:(N-1),N)
            !print *,'CmNew=',N-1,Ndjleft,CmNew(1:N-1)
         fxd = Wxdi(J)
         IF  (Ndjleft.GT.1) THEN
            XIND1=RINDDND(BIG,CmNEW,xd,xc,Ndleft-isXd,Njleft-1+isXd)
         ELSE                   !  Here all is conditioned on
                                !  and we wish to compute the
                                !  conditional probability that
                                !  variables in indicator stays between barriers.
            XIND1 = 1.d0
                                !if there are derivatives we need
                                !to compute the jacobian, jacob(xd,xc)
            IF (Nd.GT.0) fxd = fxd *jacob(xd(1:Nd),xc)
                                !If there are no derivatives
                                !then we assume that jacob(xc)=1

            IF (NstN.LT.1) GOTO 100 !Here there are no points in indicator
                                !left to integrate and hence XIND1=1.

                                         !integrate by Monte Carlo - SCIS
            IF (SCIS.NE.0) XIND1 = MNORMPRB(CmNEW)
                                         !integrate by quadrature
            IF (SCIS.EQ.0) XIND1 = RINDNIT(BIG,
     &             SQ(:,Ntscis+Njj+1),CmNEW,indXtd(1:NstN),NIT)
                                !print *,'jacobian',xind,xind1,xind+fxd*xind1
         END IF
 100     CONTINUE
         XIND = XIND+XIND1 * fxd                               !END IF
      ENDDO

      DEALLOCATE(CmNEW)
      RETURN
      END FUNCTION RINDDND


! ******************* RINDNIT ****************************************
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                                ! old procedure rind2-6
      RECURSIVE FUNCTION RINDNIT(R,SQ,Cm,indS,NITL) RESULT (xind)
      USE GLOBALDATA, ONLY : Hlo,Hup,EPS2, EPSS,CEPSS
     &         ,xCutOff,Plowgth,XSPLT
      USE FIMOD
      USE C1C2MOD
      USE QUAD
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in)    :: R
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)    :: SQ
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)    :: Cm
      DOUBLE PRECISION                                :: xind
      INTEGER,          DIMENSION(:  ), INTENT(in)    :: indS
      INTEGER,                          INTENT(in)    :: NITL
! local variables
      DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE   :: RNEW
      DOUBLE PRECISION, DIMENSION(:  ), ALLOCATABLE   :: B,SQnew
      DOUBLE PRECISION, DIMENSION(:  ), ALLOCATABLE   :: CmNEW
      INTEGER,          DIMENSION(:  ), ALLOCATABLE   :: indSNEW,ind
      INTEGER                                         :: I0,I1

      DOUBLE PRECISION, DIMENSION(PMAX)               :: H1, XX1
      DOUBLE PRECISION               :: XIND1,XIND2,SQ0,SQ1,SS0,SS1,SS
      DOUBLE PRECISION, DIMENSION(2) :: XMI, XMA
      INTEGER, DIMENSION(2)          :: INFIN
      DOUBLE PRECISION               :: SGN,P0,Plo,rho
      INTEGER      :: Ns,Nsnew,row,r1,r2,J,N1

! Assumption is that there is at least one variable X in the indicator,
! LNIT nonegative integer.
! If  LNIT=0 or the number of relevant variables is less then 3, the recursion
! stops. It gives exact value if after removing irrelevant variables there
! are maximum 2 variables left in the indicator. The program is not using
! RIND2 function any more. IR. 28 XI 1999 - Indianapolis.
!
! explanation to variables (above):
! R       = cov. matr.
! B       = R(I,I0) I=1:Ns
! SQ      = SQRT(R(I,I)) I=1:Ns
! Cm      = cond. mean
! indS    = indices to the stochastic variables as they are stored in
!           the global variables Hlo and Hup
! Ns      = size of indS =# of variables in indicator before conditioning
! Nsnew   = # of relevant variables in indicator before conditioning
! I0,I1   = indicies to minimum prob. and next minimal, respectively
! ..NEW   = the var. above after conditioning on X(I0) or used in recursion
! ind     = temp. variable storing indices

      Ns=SIZE(indS)       !=# stochastic variables before conditioning
      XIND=1.d0

      if (Ns.lt.1) return

      ALLOCATE(ind(1:Ns))
      CALL ARGP0(I0,I1,P0,Plo,SQ,Cm,indS,ind,NSnew)
!      print *,'NSnew,P0,Plo=',NSnew,P0,Plo
                           !The probability of being between barriers is one
                           !since there are no relevant variables.

!      print *,'NIT',NITl,P0,Plo,Ns,Nsnew
      IF (NSnew.lt.1) GOTO 300
      XIND=(P0*DBLE(NSnew)+Plowgth*Plo)/(DBLE(NSnew)+Plowgth)
                         !Lower bound Plo and upper bound P0 are close
                         !or all variables are close to be irrelevant,
                         !e.g. Nsnew=1.
      IF ((P0.LT.Plo+EPSS).OR.(P0.GT.CEPSS)) GOTO 300

! Now CEPSS>P0>EPSS+Plo and there are more than one relevant variable (NSnew>1)
! Those have indices ind(I0), ind(I1).
! Hence we have nondegenerated case.

      SS0 = R (ind(I0) ,ind(I0))
      SQ0 = SQRT(SS0)
      r1=indS(ind(I0))
!      print *,'P0-Plo,SS0,Sq0',P0-Plo,SS0,Sq0
      XMA(1) = MIN((Hup (r1)-Cm (ind(I0)))/SQ0,xCutOff)
      XMI(1) = MAX((Hlo (r1)-Cm (ind(I0)))/SQ0,-xCutOff)

!If NSnew = 2 then we can compute the probability exactly and recursion stops.
      IF ((NSnew.EQ.2).OR.(NITL.LT.1)) THEN !.OR.(NITL.LT.1)
! Not necessary any longer:
!         I1=2
!         if (I0.eq.2) I1=1
!         if (I0.eq.I1) print *,'rindnit, I1,I0:',I1,I0
         SS1 = R (ind(I1) ,ind(I1))
         SQ1 = SQRT(SS1)

         IF (ind(I0).LT.ind(I1)) THEN
            SS=R(ind(I0),ind(I1))
         ELSE
            SS=R(ind(I1),ind(I0))
         ENDIF
         rho= SS/(SQ0*SQ1)

         r2=indS(ind(I1))
         XMA(2) = MIN((Hup (r2)-Cm (ind(I1)))/SQ1,xCutOff)
         XMI(2) = MAX((Hlo (r2)-Cm (ind(I1)))/SQ1,-xCutOff)
         IF (ABS(rho).gt.1.d0+EPSS) THEN
            !print *,'rindnit, Correlation > 1, rho=',rho
            IF (ABS(rho).gt.1.d0+EPSS) GO TO 300
            rho = sign(1.D0,rho)
!            print *,'rindnit, P0,Plo',P0,Plo,XIND
!            print *,'rindnit I0,I1:',I0,I1
!            print *,'rindnit XMI,XMA,XMI1,XMA1:',XMI(1),XMA(1),
!     &           XMI(2),XMA(2)
!            print *,'rindnit cov(I1,I0):',R(ind(I1),ind(I0))
!            print *,'rindnit cov(I0,I1):',R(ind(I0),ind(I1))
!            print *,'rindnit SS,SS1,SS0:',SS,SS1,SS0
!            print *,'rindnit ind:',ind(1:NSnew)
         ENDIF
!         print *,XMA1,XMI1,XMA,XMI,rho
!         XIND = NORM2DPRB(XMI(1),XMA(1),XMI(2),XMA(2),rho)
!         GO TO 300
*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
!         INFIN = 2
         IF (XMI(1).LE.-xCutOff) INFIN(1)=0
         IF (XMI(2).LE.-xCutOff) INFIN(2)=0
         IF (XMA(1).GE. xCutOff) INFIN(1)=1
         IF (XMA(2).GE. xCutOff) INFIN(2)=1

         !print *,'rindnit, xind,xind2=', XIND, BVNMVN(XMI,XMA,INFIN,rho)
          XIND = BVNMVN(XMI,XMA,INFIN,rho)
!         print *,xind
         GOTO 300
      END IF
        !If  NITL=0 which means computations without conditioning on X(ind(I0))
      IF(NITL.lt.1) GOTO 300

!We have NITL>0 and at least 3 variables in the indicator, ie.
!we will condition on X(ind(I0)).
!First we check whether one can use XSPLIT variant of integration.

      if ((XMA(1).GE.xCutOff).AND.(XMI(1).LT.-XSPLT)) THEN  ! (.FALSE.).AND.
         XMA(1)=XMI(1)
         XMI(1)=-xCutOff
         SGN=-1.d0
      elseif ((XMA(1).GT.XSPLT).AND.(XMI(1).LE.-xCutOff)) THEN
         XMI(1)=XMA(1)
         XMA(1)=xCutOff
         SGN=-1.d0
      else
         SGN=1.d0
         XIND2=0.d0
      endif

         ! Must allocate several variables to recursively
         ! transfer them to rindnit: Rnew, SQnew, CMnew, indSnew
         ! The variable B is used in computations of conditional mean and cov.
         ! The size is NSnew-1 (the relevant variables minus X(ind(I0)).

      ALLOCATE(indSNEW(1:NSnew-1))
      ALLOCATE(RNEW(NSnew-1,NSnew-1))
      ALLOCATE(CMnew(1:NSnew-1))
      ALLOCATE(SQnew(1:NSnew-1))
      ALLOCATE(B(1:NSnew-1))
                                !This DO loop is divided in two parts in order
                                !to only work on the upper triangular of R
      DO row=1,I0-1
         r1=ind(row)
         Rnew(row,row:I0-1)=R(r1,ind(row:I0-1))
         ! The if test below is required on Solaris F90 compiler
         IF (I0.LT.Nsnew) Rnew(row,I0:NSnew-1)=R(r1,ind(I0+1:NSnew))
         B(row)=R(r1,ind(I0))/SQ0
      enddo
      DO row=I0+1,NSnew
         r1=ind(row)
         Rnew(row-1,row-1:NSnew-1) = R(r1,ind(row:NSnew))
         B(row-1)=R(ind(i0),r1)/SQ0
      enddo
      DO row=I0+1,NSnew
         ind(row-1)=ind(row)
      enddo


      CMnew=CM(ind(1:NSnew-1))
      SQnew=SQ(ind(1:NSnew-1))
      indSnew=indS(ind(1:NSnew-1))

                                !USE the  XSPLIT variant
      IF (SGN.LT.0.d0) XIND2 = RINDNIT(Rnew,SQnew,CMnew,indSnew,NITL-1)

                                ! Perform conditioning on X(I0)
      NSnew=NSnew-1
      N1=0
      DO row = 1, NSnew
         Rnew(row,row:NSnew) = Rnew(row,row:NSnew) -
     &        B(row)*B(row:NSnew)        !/SS0)
         SS = RNEW(row,row)
         IF (SS.GE.EPS2) then
            SQNEW (row) = SQRT (SS)
         ELSE
            SQNEW(row) = 0.d0
            N1=N1+1             ! count number of deterministic variables
         END IF
      ENDDO

                                !See if we can Narrow down the limits
      CALL C1C2(XMI(1),XMA(1),CmNew,B,SQNEW,indSnew)
      XIND = (FI (XMA(1)) - FI (XMI(1)))
                                ! if Nsnew<=N1 then PRB = XIND almost always
                                ! if this check is not performed then
                                ! the numerical integration may currupt the answer due
                                ! to the limited number of nodes used in the integration
      IF (XIND.LT.EPSS.OR.Nsnew.LT.N1+1) GOTO 200

                                !      print *,'rindnit gaussle2'
      N1=0                      !  computing nodes for num. integration.
      CALL GAUSSLE2 (N1, H1, XX1, XMI(1), XMA(1),LE2Qnr)
                                ! new conditional covariance

      XIND = 0.d0
!      print *,'rindnit for loop',N1
      DO   J = 1, N1
                                !IF (H1(J).GT.CFxCutOff) THEN
         CMnew=Cm(ind(1:NSnew)) + XX1(J)*B !/ SQ0)
         XIND1=RINDNIT(Rnew,SQnew,CMnew,indSnew,NITL-1)
         XIND = XIND+XIND1 * H1 (J)
                                !END IF
      ENDDO
200   CONTINUE
       XIND=XIND2+SGN*XIND
!       Print *,'XIND, XIND2',XIND,XIND2
!       Print *,'XMI',XMI
!       Print *,'XMA',XMA
!      Print *,'xind,nit', xind,nitl,shape(indsnew),shape(ind)
                           !fix up round off errors and make sure 0=<xind<=1
      if (XIND.GT.1.d0) THEN
         XIND=1.D0
      elseif (XIND.LT.0.D0) THEN
         XIND=0.d0
      endif
 300  continue
      if (allocated(INDSNEW)) DEALLOCATE(INDSNEW)
      if (allocated(RNEW))    DEALLOCATE(RNEW)
      if (allocated(CmNEW))   DEALLOCATE(CmNEW)
      if (allocated(SQNEW))   DEALLOCATE(SQNEW)
      if (allocated(B))       DEALLOCATE(B)
      if (allocated(ind))     DEALLOCATE(ind)
!      print *,'rindnit leaving end'
      RETURN
      END FUNCTION  RINDNIT

      SUBROUTINE BARRIER(xc,indI,Blo,Bup)
      USE GLOBALDATA, ONLY : Hup,Hlo,xedni,Ntd,index1
      IMPLICIT NONE
      INTEGER,          DIMENSION(:  ), INTENT(in) :: indI
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: Blo,Bup
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in) :: xc
      INTEGER                                      :: I, J, K, L
      INTEGER :: Mb, Nb, NI, Nc
!this procedure set Hlo,Hup according to Blo/Bup
      Mb=size(Blo,DIM=1)
      Nb=size(Blo,DIM=2)
      NI=size(indI,DIM=1)
      Nc=size(xc,DIM=1)


      DO J = 2, NI
         DO I =indI (J - 1) + 1 , indI (J)
            L=xedni(I)
            Hlo (L) = Blo (1, J - 1)
            Hup (L) = Bup (1, J - 1)
            DO K = 1, Mb-1
               Hlo(L) = Hlo(L)+Blo(K+1,J-1)*xc(K)
               Hup(L) = Hup(L)+Bup(K+1,J-1)*xc(K)
            ENDDO ! K
         ENDDO ! I
      ENDDO ! J
      !print * ,'barrier hup:'
      !print * ,size(Hup),Hup(xedni(1:Ntd))
      !print * ,'barrier hlo:'
      !print * ,size(Hlo),Hlo(xedni(1:Ntd))
      RETURN
      END SUBROUTINE BARRIER

      function MNORMPRB(Cm1) RESULT (VALUE)
      USE ADAPTMOD
      USE KRBVRCMOD
      USE KROBOVMOD
      USE RCRUDEMOD
      USE DKBVRCMOD
      USE SSOBOLMOD
      USE FUNCMOD
      USE FIMOD
      USE C1C2MOD
      USE GLOBALDATA, ONLY : Hlo,Hup,xCutOff,NUGGET,EPSS,EPS2,
     &     RelEps,NSIMmax,NSIMmin,Nt,Nd,Nj,Ntd,SQ,
     &     Njj,Ntscis,NsXtmj, indXtd,index1,
     &     useC1C2,C1C2det,COV,COVix
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)    :: Cm1  ! conditional mean
      DOUBLE PRECISION                                :: VALUE
      DOUBLE PRECISION :: XMI,XMA,SQ0
      INTEGER          :: Nst,Nst0,Nlhd
      INTEGER          :: Ndim,DEF
      INTEGER :: MINPTS,MAXPTS, INFORM
      DOUBLE PRECISION  :: ABSEPS, ERROR

!MNORMPRB Multivariate Normal integrals by SCIS or LHSCIS
!  SCIS   = Sequential conditioned importance sampling
!  LHSCIS = Latin Hypercube Sequential Conditioned Importance Sampling
!
! !  NB!!: R must be conditional sorted by condsort3
!        works on the upper triangular part of R
!
! References
! R. Ambartzumian, A. Der Kiureghian, V. Ohanian and H.
! Sukiasian (1998)
! Probabilistic Engineering Mechanics, Vol. 13, No 4. pp 299-308
!
! Alan Genz (1992)
! 'Numerical Computation of Multivariate Normal Probabilities'
! J. computational Graphical Statistics, Vol.1, pp 141--149


      !print *,'enter mnormprb'
      Nst0 = NsXtmj(Njj+Ntscis)
      if (Njj.GT.0) then
         Nst  = NsXtmj(Njj)
      else
         Nst  = NsXtmj(Ntscis+1)
      endif
      !Nst=size(Cm)
      if (Nst.lt.Njj+1) then
        VALUE=1.d0
        if (allocated(COV)) then ! save the coefficient of variation in COV
           COV(COVix)=0.d0
        endif
        return
      endif

      if (Nst.lt.Njj+1) then
         if (allocated(COV)) then ! save the coefficient of variation in COV
            COV(COVix)=0.d0
         endif

         VALUE=1.d0
         return
      endif
       !print *,' mnormprb start calculat'
      VALUE=0.d0
      Cm(1:Nst-Njj)=Cm1(Njj+1:Nst) ! initialize conditional mean
      SQ0 = SQ(Njj+1,Njj+1)
      XMA = MIN((Hup (Njj+1)-Cm1(Njj+1))/SQ0,xCutOff)
      XMI = MAX((Hlo (Njj+1)-Cm1(Njj+1))/SQ0,-xCutOff)

      if (useC1C2) then         ! see if we can narrow down sampling range
         CALL C1C2(XMI,XMA,Cm1(Njj+2:Nst),BIG(1,2:Nst),
     &        SQ(2:Nst,1),indXtd(2:Nst))
      endif
      IF (XMA.LE.XMI)  RETURN
      Pl1    = FI(XMI)
      Pu1    = FI(XMA)
      Ndim   = Nst0-Njj
      MAXPTS = NSIMmax*Ndim
      MINPTS = NSIMmin*Ndim
      ABSEPS = EPSS
      DEF = 1  ! krbvrc is fastest
      SELECT CASE (DEF)
      CASE (:1)
       !print * ,'RINDSCIS: Ndim',Ndim
         IF (NDIM.lt.9) THEN
            CALL SADAPT(Ndim,MAXPTS,MVNFUN2,ABSEPS,
     &     RELEPS,ERROR,VALUE,INFORM)
         ELSE
            CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN2, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
         ENDIF
      CASE (2)
       !print * ,'RINDSCIS: Ndim',Ndim
         IF (NDIM.lt.19) THEN
            ! Call the subregion adaptive integration subroutine
            CALL SADAPT(Ndim,MAXPTS,MVNFUN2,ABSEPS,
     &     RELEPS,ERROR,VALUE,INFORM)
         ELSE
           CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN2, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
         ENDIF
      CASE (3)
         CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN2, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
      CASE (4)
          CALL KROBOV( NDIM, MINPTS, MAXPTS, MVNFUN2, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
      CASE (5)     ! Call Crude Monte Carlo integration procedure
         CALL RANMC( NDIM, MAXPTS, MVNFUN2, ABSEPS,
     &        RELEPS, ERROR, VALUE, INFORM )
      CASE (6)    ! Call the scrambled Sobol sequence rule integration procedure
          CALL SOBNIED( NDIM, MINPTS, MAXPTS, MVNFUN2, ABSEPS, RELEPS,
     &           ERROR, VALUE, INFORM )
      CASE (7:)
          CALL DKBVRC( NDIM, MINPTS, MAXPTS, MVNFUN2, ABSEPS, RELEPS,
     &           ERROR, VALUE, INFORM )
      END SELECT

      if (allocated(COV)) then  ! save the coefficient of variation in COV
         if ((VALUE.gt.0.d0))  COV(COVix)=ERROR/VALUE/3.0d0
      endif

      !print *,'mnormprb, error, inform,',error,inform
      !print *,'leaving mnormprb'
      return
      END FUNCTION MNORMPRB

      FUNCTION RINDSCIS(xc1) result(VALUE)

!RINDSCIS Multivariate Normal integrals by SCIS
!  SCIS   = Sequential conditioned importance sampling
!  The points can be sampled using Lattice rules, Latin Hypercube samples,
!  uniformly distributed, or using an adaptive algorithm
!
! References
! R. Ambartzumian, A. Der Kiureghian, V. Ohanian and H.
! Sukiasian (1998)
! Probabilistic Engineering Mechanics, Vol. 13, No 4. pp 299-308
!
! Alan Genz (1992)
! 'Numerical Computation of Multivariate Normal Probabilities'
! J. computational Graphical Statistics, Vol.1, pp 141--149
      USE ADAPTMOD
      USE KRBVRCMOD
      USE KROBOVMOD
      USE RCRUDEMOD
      USE DKBVRCMOD
      USE SSOBOLMOD
      USE FUNCMOD
      USE FIMOD
      USE C1C2MOD
      USE JACOBMOD
      USE GLOBALDATA, ONLY : Hlo,Hup,xCutOff,NUGGET,EPSS,EPS2,
     &     RelEps,NSIMmax,NSIMmin,Nt,Nd,Nj,Ntd,SQ,Nc,
     &     NsXtmj, NsXdj,indXtd,index1,
     &     useC1C2,C1C2det,COV,COVix,SCIS
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)    :: xc1 ! conditional values
      DOUBLE PRECISION                                :: VALUE
      DOUBLE PRECISION :: XMI,XMA,SQ0
      INTEGER          :: Nst,Nst0,Nsd,Nsd0,K
      INTEGER          :: Ndim,Ndleft,Ntmj,NLHD
      INTEGER :: MINPTS,MAXPTS, INFORM
      DOUBLE PRECISION  :: ABSEPS, ERROR

      VALUE = 0.d0

!      print *,'enter rindscis'
      Nst  = NsXtmj(Ntd+1)
      Ntmj=Nt-Nj
      if (Ntmj.GT.0) then
         Nst0 = NsXtmj(Ntmj)
      else
         Nst0 = 0
      endif
      Nsd  = NsXdj(Nd+Nj+1)
      Nsd0 = NsXdj(1)
      Ndim = Nst0+Ntd-Nsd0+1      ! # dim. we treat stochastically
      MAXPTS = NSIMmax*Ndim
      MINPTS = NSIMmin*Ndim
      ABSEPS = EPSS
      IF (Nc.GT.0)  xc=xc1



      if (Nd+Nj.gt.0) then
         IF ( BIG(Ntd,Ntd).LT.EPS2) THEN  !degenerate case
            IF (Nd.GT.0) THEN
               Ndleft=Nd;K=Ntd
               DO WHILE (Ndleft.GT.0)
                  IF (index1(K).GT.Nt) THEN ! isXd
                     xd (Ndleft) =  Cm (K)
                     Ndleft=Ndleft-1
                  END IF
                  K=K-1
               ENDDO
               VALUE = jacob (xd,xc) ! jacobian of xd,xc
            ELSE
               VALUE = 1.d0      !     VALUE = FxCutOff???
            END IF
            !print *,'jacob,xd',VALUE,xd
            IF (Nst.LT.1) then
               if (allocated(COV)) then ! save the coefficient of variation in COV
                  COV(COVix)=0.d0
               endif
               RETURN
            endif
            !print *,'RINDSCIS calling MNORMPRB '
            VALUE=VALUE*MNORMPRB(Cm(1:Nst))
            !print *,'leaving rindscis'
            RETURN
         ENDIF
      elseif (Nst.lt.1) then
         if (allocated(COV)) then ! save the coefficient of variation in COV
            COV(COVix)=0.d0
         endif

         VALUE=1.d0
         return
      endif

      if (Nd+Nj.gt.0) then
         SQ0=SQ(Ntd,Ntd)
         XMA = MIN((Hup (Ntd)-Cm(Ntd))/SQ0,xCutOff)
         XMI = MAX((Hlo (Ntd)-Cm(Ntd))/SQ0,-xCutOff)

         if (useC1C2) then ! see if we can narrow down sampling range
            CALL C1C2(XMI,XMA,Cm(1:Ntd-1),BIG(1:Ntd-1,Ntd),
     &           SQ(1:Ntd-1,Ntd),indXtd(1:Ntd-1))
         endif
      else
         SQ0=SQ(1,1)
         XMA = MIN((Hup (1)-Cm(1))/SQ0,xCutOff)
         XMI = MAX((Hlo (1)-Cm(1))/SQ0,-xCutOff)

         if (useC1C2) then ! see if we can narrow down sampling range
            CALL C1C2(XMI,XMA,Cm(2:Nst),BIG(1,2:Nst),
     &           SQ(2:Nst,1),indXtd(2:Nst))
         endif
      endif
      IF (XMA.LE.XMI) return    !PQ= Y=0 for all return
      Pl1 = FI(XMI)
      Pu1 = FI(XMA)
      IF ( Ndim .GT. 20. AND. SCIS.EQ.3) THEN
         !print *, 'Ndim to large for SADMVN.  Calling KRBVRC instead'
         SCIS=4
      ENDIF
      !print * ,'RINDSCIS: Ndim',Ndim
      SELECT CASE (SCIS)
      CASE (:1)
       !print * ,'RINDSCIS: Ndim',Ndim
         IF (NDIM.lt.9) THEN
            CALL SADAPT(Ndim,MAXPTS,MVNFUN,ABSEPS,
     &     RELEPS,ERROR,VALUE,INFORM)
         ELSE
            CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
         ENDIF
      CASE (2)
       !print * ,'RINDSCIS: Ndim',Ndim
         IF (NDIM.lt.19) THEN
!   Call the subregion adaptive integration subroutine
            CALL SADAPT(Ndim,MAXPTS,MVNFUN,ABSEPS,
     &     RELEPS,ERROR,VALUE,INFORM)
         ELSE
           CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
         ENDIF
      CASE (3)
         CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
      CASE (4)
         CALL KROBOV( NDIM, MINPTS, MAXPTS, MVNFUN, ABSEPS, RELEPS,
     &     ERROR, VALUE, INFORM )
      CASE (5)  ! Call Crude Monte Carlo integration procedure
         CALL RANMC( NDIM, MAXPTS, MVNFUN, ABSEPS,
     &        RELEPS, ERROR, VALUE, INFORM )
      CASE (6)  ! Call the scrambled Sobol sequence rule integration procedure
         CALL SOBNIED( NDIM, MINPTS, MAXPTS, MVNFUN, ABSEPS, RELEPS,
     &           ERROR, VALUE, INFORM )
      CASE (7:)
         CALL DKBVRC( NDIM, MINPTS, MAXPTS, MVNFUN, ABSEPS, RELEPS,
     &           ERROR, VALUE, INFORM )
      END SELECT
      if (allocated(COV)) then  ! save the coefficient of variation in COV
         if ((VALUE.gt.0.d0))  COV(COVix)=ERROR/VALUE/3.0d0
      endif
      IF (inform.gt.0.and.error.gt.10.*epss) then
         !print *,'rindscis, error', error,'inform,',inform
      endif
      !print *,'rindscis, Ndim,MINPTS, error',Ndim,MINPTS,error
      END FUNCTION  RINDSCIS

!********************************************************************

      SUBROUTINE CONDSORT0 (R,Cm,xcmean,CSTD,index1,xedni,NsXtmj,NsXdj
     &     ,INFORM)
      USE GLOBALDATA, ONLY : Nt,Nj,Njj,Nd,Nc,Ntdc,Ntd,EPS2,Nugget,
     &    XCEPS2,SCIS,Ntscis,SQTWOPI1,Hlo,Hup,xCutOff,EPSS
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(inout) :: Cm
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)    :: xcmean
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(out)   :: CSTD
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: index1
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: xedni
      INTEGER,          DIMENSION(:  ), INTENT(out)   :: NsXtmj
      INTEGER,          DIMENSION(:  ), INTENT(out)   :: NsXdj
      INTEGER,                          INTENT(out)   :: INFORM
! local variables
      DOUBLE PRECISION, DIMENSION(:  ), allocatable   :: SQ
      DOUBLE PRECISION, DIMENSION(:,:), allocatable   :: CSTD2
      INTEGER,          DIMENSION(1  )                :: m
      INTEGER,          DIMENSION(:  ), allocatable   :: ind
      DOUBLE PRECISION  :: P0,P1,XMI,XMA,SQ0,XX
      INTEGER           :: I0,I1
      INTEGER           :: Nstoc,Ntmp,NstoXd   !,degenerate
      INTEGER           :: changed,m1,r1,c1,r2,c2,ix,iy,Njleft,Ntmj

! R         = Input: Cov(X) where X=[Xt Xd Xc] is stochastic vector
!            Output: sorted Conditional Covar. matrix   Shape N X N  (N=Nt+Nd+Nc)
! CSTD      = SQRT(Var(X(1:I-1)|X(I:N)))
!            conditional standard deviation.             Shape Ntd X max(Nd+Nj,1)
! index1    = indices to the variables original place.   Size  Ntdc
! xedni     = indices to the variables new place.        Size  Ntdc
! NsXtmj(I) = indices to the last stochastic variable
!            among Nt-Nj first of Xt after conditioning on
!            X(Nt-Nj+I).                                 Size  Nd+Nj+Njj+Ntscis+1
! NsXdj(I)  = indices to the first stochastic variable
!            among Xd+Nj of Xt after conditioning on
!            X(Nt-Nj+I).                                 Size  Nd+Nj+1
!
! R=Cov([Xt,Xd,Xc]) is a covariance matrix of the stochastic vector X=[Xt Xd Xc]
! where the variables Xt, Xd and Xc have the size Nt, Nd and Nc, respectively.
! Xc is (are) the conditional variable(s).
! Xd and Xt are the variables to integrate.
! Xd + Nj variables of Xt are integrated directly by the RindDXX
! subroutines in the order of shortest expected integration interval.
! The remaining Nt-Nj variables of Xt are integrated in
! increasing order of the marginal probabilities by the RindXX subroutines.
! CONDSORT prepare and rearrange the covariance matrix
! in a special way to accomodate this strategy:
!
! After conditioning and sorting, the first Nt-Nj x Nt-Nj block of R
! will contain the conditional covariance matrix
! of Xt(1:Nt-Nj) given Xt(Nt-Nj+1:Nt)  Xd and Xc, i.e.,
! Cov(Xt(1:Nt-Nj),Xt(1:Nt-Nj)|Xt(Nt-Nj+1:Nt), Xd,Xc)
! NB! for Nj>0 the order of Xd and Xt(Nt-Nj+1:Nt) may be mixed.
! The covariances, Cov(X(1:I-1),X(I)|X(I+1:N)), needed for computation of the
! conditional expectation, E(X(1:I-1)|X(I:N), are saved in column I of R
! for I=Nt-Nj+1:Ntdc.
!
! IF any of the variables have variance less than EPS2. They will be
! be treated as deterministic and not stochastic variables by the
! RindXXX subroutines. The deterministic variables are moved to
! middle in the order they became deterministic in order to
! keep track of them. Their variance and covariance with
! the remaining stochastic variables are set to zero in
! order to avoid numerical difficulties.
!
! NsXtmj(I) is the number of variables  among the Nt-Nj
! first we treat stochastically after conditioning on X(Nt-Nj+I).
! The covariance matrix is sorted so that all variables with indices
! from 1 to NsXtmj(I) are stochastic after conditioning
! on X(Nt-Nj+I).  Thus NsXtmj(I) may also be considered
! as the index to the last stochastic variable after conditioning
! on X(Nt-Nj+I). In other words NsXtmj keeps track of the deterministic
! and stochastic variables among the Nt-Nj first variables in each
! conditioning step.
!
! Similarly  NsXdj(I)  keeps track of the deterministic and stochastic
! variables among the Nd+Nj following variables in each conditioning step.
! NsXdj(I) is the index to the first stochastic variable
! among the Nd+Nj following variables after conditioning on X(Nt-Nj+I).
! The covariance matrix is sorted so that all variables with indices
! from NsXdj(I+1) to NsXdj(I)-1 are  deterministic conditioned on
! X(Nt-Nj+I).
!

! Var(Xc(1))>Var(Xc(2)|Xc(1))>...>Var(Xc(Nc)|Xc(1),Xc(2),...,Xc(Nc)).
! If Nj=0 then
! Var(Xd(1)|Xc)>Var(Xd(2)|Xd(1),Xc)>...>Var(Xd(Nd)|Xd(1),Xd(2),...,Xd(Nd),Xc).
!
! NB!! Since R is symmetric, only the upper triangular contains the
! sorted conditional covariance. The whole matrix
! is easily obtained by copying elements of the upper triangle to
! the lower or by uncommenting some lines in the end of this subroutine
!
! revised pab 18.04.2000
!  new name rind60
!  New assumption of BIG for the conditional sorted variables:
!                         BIG(I,I)=sqrt(Var(X(I)|X(I+1)...X(N))=SQI
!                         BIG(1:I-1,I)=COV(X(1:I-1),X(I)|X(I+1)...X(N))/SQI
!      Otherwise
!                         BIG(I,I) = Var(X(I)|X(I+1)...X(N)
!                         BIG(1:I-1,I)=COV(X(1:I-1),X(I)|X(I+1)...X(N))
!  This also affects C1C2: SQ0=sqrt(Var(X(I)|X(I+1)...X(N)) is removed from input
!  =>  A lot of wasteful divisions are avoided


! Using SQ to temporarily store the diagonal of R
! Adding a nugget effect to ensure the the inversion is
! not corrupted by round off errors
! good choice for nugget might be 1e-8
                             !call getdiag(SQ,R)
      INFORM = 0
      ALLOCATE(SQ(1:Ntdc))
      ALLOCATE(ind(1:Ntdc))
      IF (Nd+Nj+Njj+Ntscis.GT.0) THEN
         ALLOCATE(CSTD2(1:Ntd,1:Nd+Nj+Njj+Ntscis))
         CSTD2=0.d0             ! initialize CSTD
      ENDIF
      !CALL ECHO(R,Ntdc)
      DO ix = 1, Ntdc
         R(ix,ix) = R(ix,ix)+Nugget
         SQ(ix) = R(ix,ix)
         index1 (ix) = ix       ! initialize index1
      ENDDO

      Ntmj   = Nt-Nj
      Njleft = Nj
      NstoXd = Ntmj+1
      Nstoc  = Ntmj


      DO ix = 1, Nc             ! Condsort Xc
         r1=Ntdc-ix
         m=r1+2-MAXLOC(SQ(r1+1:Ntd+1:-1))
         IF (SQ(m(1)).LT.XCEPS2) THEN
            INFORM = 1
            !PRINT *,'Condsort0, degenerate Xc'
                                !degenerate=1
            GOTO 200            ! RETURN    !degenerate case
         ENDIF
         m1 = index1(m(1))
         CALL swapint(index1(m(1)),index1(r1+1))
         CALL swapre(Cm(m(1)),Cm(r1+1))
         SQ(r1+1) = SQRT(SQ(m(1)))
         R(index1(1:r1+1),m1) = R(index1(1:r1+1),m1)/SQ(r1+1)
         R(m1,index1(1:r1))   = R(index1(1:r1),m1)

                                ! Calculate the conditional mean
         Cm(1:r1)=Cm(1:r1)+(xcmean(index1(r1+1)-Ntd)-Cm(r1+1))*
     &        R(index1(1:r1),m1)         !/SQ(r1+1)
                                ! sort and calculate conditional covariances
         CALL CONDSORT2(R,SQ,index1,Nstoc,NstoXd,Njleft,m1,r1)
      ENDDO                     ! ix
      ! index to first stochastic variable of Xd and Nj of Xt
      NsXdj(Nd+Nj+1)  = NstoXd
      ! index to last stochastic variable of Nt-Nj of Xt
      NsXtmj(Nd+Nj+Njj+Ntscis+1) = Nstoc
      !print *, 'condsort index1', index1
      !print *, 'condsort Xd'
      !call echo(R,Ntdc)

      DO ix = 1, Nd+Nj          ! Condsort Xd +  Nj of Xt
         CALL ARGP0(I1,r2,P1,XX,SQRT(SQ(NstoXd:Ntd-ix+1)),
     &        Cm(NstoXd:Ntd-ix+1),index1(NstoXd:Ntd-ix+1),ind,r1)
         IF (r1.NE.0) I1=ind(I1)
         m = MIN(NstoXd+I1-1,Ntd-ix+1)
         IF (Njleft.GT.0) THEN

            CALL ARGP0(I0,r2,P0,XX,SQRT(SQ(1:Nstoc)),
     &           Cm(1:Nstoc),index1(1:Nstoc),ind,r1)
            IF (r1.NE.0) I0=ind(I0)
!            m=Ntd-ix+2-MAXLOC(SQ(Ntd-ix+1:1:-1))
            IF (P0.LT.P1.AND.r1.GT.0) THEN
               m  = I0
               P1 = P0
            END IF
            Ntmp = NstoXd+Njleft-1
            IF (((NstoXd.LE.m(1)).AND.(m(1).LE.Ntmp))
     &           .OR.(m(1).LE.Nstoc)) THEN
               CALL swapint(index1(m(1)),index1(Ntmp))
               CALL swapRe(SQ(m(1)),SQ(Ntmp))
               CALL swapre(Cm(m(1)),Cm(Ntmp))
               m(1)=Ntmp
               Njleft=Njleft-1
            END IF
         END IF  ! Njleft
         IF (SQ(m(1)).LT.EPS2) THEN
                                !PRINT *,'Condsort, degenerate Xd'
            Ntmp = Nd+Nj+1-ix
            NsXtmj(Ntscis+Njj+1:Ntmp+Ntscis+Njj+1) = Nstoc
            NsXdj(1:Ntmp+1) = NstoXd
            IF (ix.EQ.1) THEN
               DO iy = 1,Ntd      !sqrt(VAR(X(I)|X(Ntd-ix+1:Ntdc))
                  r1 = index1(iy)
                  CSTD2(r1,Ntscis+Njj+1:Ntmp+Ntscis+Njj)=SQRT(SQ(iy))
               ENDDO
            ELSE
               DO iy=ix,Nd+Nj
                  CSTD2(:,Nd+Nj+Ntscis+Njj+1-iy)=
     &                 CSTD2(:,Ntmp+Ntscis+Njj+1)
               ENDDO
            ENDIF
            GOTO 200            ! degenerate case
         END IF
         r1 = Ntd-ix
         m1 = index1(m(1));
         CALL swapint(index1(m(1)),index1(r1+1))
         CALL swapre(Cm(m(1)),Cm(r1+1))
          !  CALL swapre(SQ(r1+1),SQ(m(1)))
         SQ0 = SQRT(SQ(m(1)))
         SQ(r1+1) = SQ0
         CSTD2(m1,Nd+Nj+Ntscis+Njj+1-ix)=SQ0

         R(index1(1:r1+1),m1) = R(index1(1:r1+1),m1)/SQ0
         R(m1,index1(1:r1)) = R(index1(1:r1),m1)

         XMA = MIN( (Hup (index1(r1+1)) - Cm (r1+1))/ SQ0,xCutOff)
         XMA = MAX(XMA,-xCutOff)
         XMI = MAX( (Hlo (index1(r1+1)) - Cm (r1+1))/ SQ0,-xCutOff)
         XMI = MIN(XMI,xCutOff)

! There is something wrong with XX
         IF (P1.GT.  EPSS ) THEN
                                ! Calculate the normalized expected mean without the jacobian
            XX = SQTWOPI1*(EXP(-0.5d0*XMI*XMI)-EXP(-0.5d0*XMA*XMA))/P1
         ELSE
            IF ( XMI .LE. -xCutOff ) XX = XMA
            IF ( XMA .GE. xCutOff )  XX = XMI
            IF (XMI.GT.-xCutOff.AND.XMA.LT.xCutOff) XX=(XMI+XMA)*0.5d0
         END IF

                                ! Calculate the conditional expected mean
         Cm(1:r1) = Cm(1:r1)+XX*R(index1(1:r1),m1)

                                ! Calculating conditional variances
         CALL CONDSORT2(R,SQ,index1,Nstoc,NstoXd,Njleft,m1,Ntd-ix)
                                ! saving indices
         NsXtmj(Nd+Nj+Njj+Ntscis+1-ix)=Nstoc
         NsXdj(Nd+Nj+1-ix)=NstoXd

                                ! Calculating standard deviations non-deterministic variables
         DO r2=1,Nstoc
            r1=index1(r2)
            CSTD2(r1,Nd+Nj+Njj+Ntscis+1-ix)=SQRT(SQ(r2)) !sqrt(VAR(X(I)|X(Ntd-ix+1:Ntdc))
         ENDDO
         DO r2=NstoXd,Ntd-ix
            r1=index1(r2)
            CSTD2(r1,Nd+Nj+Ntscis+Njj+1-ix)=SQRT(SQ(r2)) !sqrt(VAR(X(I)|X(Ntd-ix+1:Ntdc))
         ENDDO
      ENDDO                     ! ix


 200  IF ((SCIS.GT.0).OR. (Njj.gt.0)) THEN ! check on Njj instead
          ! Calculating conditional variances and sort for Nstoc of Xt
         CALL CONDSORT4(R,Cm,CSTD2,SQ,index1,NsXtmj,Nstoc)
         !Nst0=Nstoc
      ENDIF
      IF (Nd+Nj+Njj+Ntscis.GT.0) THEN
         DO r2=1,Ntd           ! sorting CSTD according to index1
            r1=index1(r2)
            CSTD(r2,:)= CSTD2(r1,:)
         END DO
         DEALLOCATE(CSTD2)
      ELSE
         IF (Nc.EQ.0) THEN
            ix=1; Nstoc=Ntmj
            DO WHILE (ix.LE.Nstoc)
               IF (SQ(ix).LT.EPS2) THEN
                  DO WHILE ((SQ(Nstoc).LT.EPS2).AND.(ix.LT.Nstoc))
                     SQ(Nstoc)=0.d0 !MAX(0.d0,SQ(Nstoc))
                     Nstoc=Nstoc-1
                  END DO
                  CALL swapint(index1(ix),index1(Nstoc)) ! swap indices
                  !CALL swapre(SQ(ix),SQ(Nstoc))
                  SQ(ix)=SQ(Nstoc);SQ(Nstoc)=0.d0
                  Nstoc=Nstoc-1
               ENDIF
               ix=ix+1
            END DO
         ENDIF
         CSTD(1:Nt,1)=SQRT(SQ(1:Nt))
         NsXtmj(1)=Nstoc
      ENDIF

      changed=0
      DO r2=Ntdc,1,-1          ! sorting the upper triangular of the
         r1=index1(r2)         ! covariance matrix according to index1
         xedni(r1)=r2
         !PRINT *,'condsort,xedni',xedni
         !PRINT *,'condsort,r1,r2',r1,r2
         IF ((r1.NE.r2).OR.(changed.EQ.1)) THEN
            changed=1
            R(r2,r2) = SQ(r2)
            DO c2=r2+1,Ntdc
               c1=index1(c2)
               IF (c1.GT.r1) THEN
                  R(r2,c2)=R(c1,r1)
               ELSE
                  R(r2,c2)=R(r1,c1)
               END IF
            END DO
         END IF
      END DO
                                ! you may sort the lower triangular according
                                ! to index1 also, but it is not needed
                                ! since R is symmetric.  Uncomment the
                                ! following if the whole matrix is needed
      DO c2=1,Ntdc
         DO r2=c2+1,Ntdc
            R(r2,c2)=R(c2,r2) ! R symmetric
         END DO
      END DO
!      IF (degenerate.EQ.1) THEN
!         PRINT *,'condsort,R='
!         call echo(R,Ntdc)
!         PRINT *,'condsort,SQ='
!         call echo(CSTD,Ntd)
!         PRINT *,'index=',index1
!         PRINT *,'xedni=',xedni
!      ENDIF
!      PRINT * , 'big'
!600   FORMAT(4F8.4)
!      PRINT 600, R
!      PRINT 600, SQ
      DEALLOCATE(SQ)
      IF (ALLOCATED(ind)) DEALLOCATE(ind)
      RETURN
      END SUBROUTINE CONDSORT0



      SUBROUTINE CONDSORT4(R,Cm,CSTD2,SQ,index1,NsXtmj,Nstoc)
      USE GLOBALDATA, ONLY : EPS2,Njj,Ntscis,SQTWOPI1,Hlo,Hup,
     &     xCutOff,EPSS
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R,CSTD2
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(inout) :: Cm
      DOUBLE PRECISION, DIMENSION(:),   INTENT(inout) :: SQ ! diag. of R
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: index1,NsXtmj
      INTEGER, INTENT(inout)   :: Nstoc
! local variables
      DOUBLE PRECISION :: P0,Plo,XMI,XMA,SQ0,XX
      INTEGER           :: I0
      INTEGER,          DIMENSION(1) :: m
      INTEGER,          DIMENSION(:), ALLOCATABLE :: ind
      INTEGER  :: m1
      INTEGER  :: Nsold
      INTEGER  :: r1,c1,row,col,iy,ix
! This function condsort all the Xt variables for use with RINDSCIS and
! MNORMPRB

      !Nsoold=Nstoc
      ix=1
      ALLOCATE(ind(1:Nstoc))
      DO WHILE ((ix.LE.Nstoc).and.(ix.LE.(Ntscis+Njj)))
         CALL ARGP0(I0,c1,P0,Plo,SQRT(SQ(ix:Nstoc)),
     &           Cm(ix:Nstoc),index1(ix:Nstoc),ind,r1)
         IF (r1.NE.0) I0=ind(I0)
         m = ix-1+max(I0-1,1)
!         m=ix-1+MAXLOC(SQ(ix:Nstoc))

         IF (SQ(m(1)).LT.EPS2) THEN
            !PRINT *,'Condsort3, error degenerate X'
            NsXtmj(1:Njj+Ntscis)=0
            Nstoc=0       !degenerate=1
            RETURN    !degenerate case
         ENDIF
         m1=index1(m(1));
         CALL swapint(index1(m(1)),index1(ix))
         CALL swapre(SQ(ix),SQ(m(1)))
         SQ0=SQRT(SQ(ix))
         CSTD2(m1,ix)=SQ0

         R(index1(ix:Nstoc),m1) = R(index1(ix:Nstoc),m1)/SQ0
         R(m1,index1(ix+1:Nstoc)) = R(index1(ix+1:Nstoc),m1)
         CALL swapre(Cm(m(1)),Cm(ix))


         XMA = MIN( (Hup (index1(ix)) - Cm (ix))/ SQ0,xCutOff)
         XMI = MAX( (Hlo (index1(ix)) - Cm (ix))/ SQ0,-xCutOff)
         XMA = MAX(XMA,-xCutOff)
         XMI = MIN(XMI,xCutOff)
         IF (P0.GT.  EPSS ) THEN
                                ! Calculate the expected mean
            XX= SQTWOPI1*(EXP(-0.5d0*XMI*XMI)-EXP(-0.5d0*XMA*XMA))/P0
         ELSE
            IF ( XMI .LE. -xCutOff ) XX = XMA
            IF ( XMA .GE. xCutOff )  XX = XMI
            IF (XMI.GT.-xCutOff.AND.XMA.LT.xCutOff) XX=(XMI+XMA)*0.5d0
         END IF

                                ! Calculate the conditional expected mean
         Cm(ix+1:Nstoc)=Cm(ix+1:Nstoc)+XX*
     &        R(m1,index1(ix+1:Nstoc))


                                ! Calculating conditional variances for the
                                ! first Nstoc variables.
                                ! variables with variance less than EPS2
                                ! will be treated as deterministic and not
                                ! stochastic variables and are therefore moved
                                ! to the end among these variables.
                                ! Nstoc is the # of variables we treat
                                ! stochastically
         iy=ix+1;Nsold=Nstoc
         DO WHILE (iy.LE.Nstoc)
            r1=index1(iy)
            SQ(iy)=R(r1,r1)-R(r1,m1)*R(m1,r1)        !/R(m1,m1)
            IF (SQ(iy).LT.EPS2) THEN
!              IF (SQ(iy).LT.-EPS2) THEN
!                 PRINT *,'Cndsrt4,Error Covariance negative definit'
!              ENDIF
               IF (iy.LT.Nstoc) THEN
                  r1=index1(Nstoc)
                  SQ(Nstoc)=R(r1,r1)-R(r1,m1)*R(m1,r1)  !/R(m1,m1)
                  DO WHILE ((SQ(Nstoc).LT.EPS2).AND.(iy.LT.Nstoc))
!                   IF (SQ(Nstoc).LT.-EPS2) THEN
!                     PRINT *,'Cndsrt4,Error Covariance negative definit'
!                   ENDIF
                     SQ(Nstoc)=0.d0 !MAX(0.d0,SQ(Nstoc))
                     Nstoc=Nstoc-1
                     r1=index1(Nstoc)
                     SQ(Nstoc)=R(r1,r1)-R(r1,m1)*R(m1,r1) !/R(m1,m1)
                  END DO
                  CALL swapint(index1(iy),index1(Nstoc)) ! swap indices
                                !CALL swapre(SQ(iy),SQ(Nstoc))          ! swap values
                  SQ(iy)=SQ(Nstoc);
               ENDIF
               SQ(Nstoc)=0.d0
               Nstoc=Nstoc-1
            ENDIF
            iy=iy+1
         END DO
         NsXtmj(ix)=Nstoc ! saving index to last stoch. var. after conditioning
             ! Calculating Covariances for non-deterministic variables
         DO row=ix+1,Nstoc
            r1=index1(row)
            R(r1,r1)=SQ(row)
            CSTD2(r1,ix)=SQRT(SQ(row)) ! saving stdev after conditioning on ix
            DO col=row+1,Nstoc
               c1=index1(col)
               R(c1,r1)=R(r1,c1)-R(r1,m1)*R(m1,c1)      !/R(m1,m1)
               R(r1,c1)=R(c1,r1)
            ENDDO
         ENDDO
            ! similarly for deterministic values
         DO row=Nstoc+1,Nsold
            r1=index1(row)
            SQ(row)  = 0.d0 !MAX(0.d0,SQ(row))
            CSTD2(r1,ix)=0.d0 !SQRT(SQ(row)) ! saving stdev after conditioning on ix
            R(r1,r1) = SQ(row)
            DO col=ix+1,Nsold   !row-1
               c1=index1(col)
               R(c1,r1)=0.d0
               R(r1,c1)=0.d0
            ENDDO
         ENDDO
         ix=ix+1
      ENDDO
      if (Nstoc.LT.Njj+Ntscis) THEN
         ! This test is necessary on Solaris F90 compiler.
         NsXtmj(Nstoc+1:Njj+Ntscis) = Nstoc
!      else
!         PRINT *,'Condsort4'
!         PRINT *,'Nstoc,Njj, Ntscis',Nstoc,Njj,Ntscis
      endif
      IF (ALLOCATED(ind)) DEALLOCATE(ind)
      RETURN
      END SUBROUTINE CONDSORT4

      SUBROUTINE CONDSORT (R,CSTD,index1,xedni,NsXtmj,NsXdj,INFORM)
      USE GLOBALDATA, ONLY : Nt,Nj,Njj,Nd,Nc,Ntdc,Ntd,EPS2,Nugget,
     &    XCEPS2,SCIS,Ntscis
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(out)   :: CSTD
      INTEGER,          DIMENSION(:  ), INTENT(out)   :: index1
      INTEGER,          DIMENSION(:  ), INTENT(out)   :: xedni
      INTEGER,          DIMENSION(:  ), INTENT(out)   :: NsXtmj
      INTEGER,          DIMENSION(:  ), INTENT(out)   :: NsXdj
      INTEGER,     INTENT(out)   :: INFORM
! local variables
      DOUBLE PRECISION, DIMENSION(:  ), allocatable   :: SQ
      DOUBLE PRECISION, DIMENSION(:,:), allocatable   :: CSTD2
      INTEGER,          DIMENSION(1  )                :: m
      INTEGER       :: Nstoc,Ntmp,NstoXd   !,degenerate
      INTEGER       :: changed,m1,r1,c1,row,col,ix,iy,Njleft,Ntmj

! R         = Input: Cov(X) where X=[Xt Xd Xc] is stochastic vector
!            Output: sorted Conditional Covar. matrix   Shape N X N  (N=Nt+Nd+Nc)
! CSTD      = SQRT(Var(X(1:I-1)|X(I:N)))
!            conditional standard deviation.             Shape Ntd X max(Nd+Nj,1)
! index1    = indices to the variables original place.   Size  Ntdc
! xedni     = indices to the variables new place.        Size  Ntdc
! NsXtmj(I) = indices to the last stochastic variable
!            among Nt-Nj first of Xt after conditioning on
!            X(Nt-Nj+I).                                 Size  Nd+Nj+Njj+Ntscis+1
! NsXdj(I)  = indices to the first stochastic variable
!            among Xd+Nj of Xt after conditioning on
!            X(Nt-Nj+I).                                 Size  Nd+Nj+1
!
! R=Cov([Xt,Xd,Xc]) is a covariance matrix of the stochastic vector X=[Xt Xd Xc]
! where the variables Xt, Xd and Xc have the size Nt, Nd and Nc, respectively.
! Xc is (are) the conditional variable(s).
! Xd and Xt are the variables to integrate.
! Xd + Nj variables of Xt are integrated directly by the RindDXX
! subroutines in the order of decreasing conditional variance.
! The remaining Nt-Nj variables of Xt are integrated in
! increasing order of the marginal probabilities by the RindXX subroutines.
! CONDSORT prepare and rearrange the covariance matrix
! by decreasing order of conditional variances in a special way
! to accomodate this strategy:
!
! After conditioning and sorting, the first Nt-Nj x Nt-Nj block of R
! will contain the conditional covariance matrix
! of Xt(1:Nt-Nj) given Xt(Nt-Nj+1:Nt)  Xd and Xc, i.e.,
! Cov(Xt(1:Nt-Nj),Xt(1:Nt-Nj)|Xt(Nt-Nj+1:Nt), Xd,Xc)
! NB! for Nj>0 the order of Xd and Xt(Nt-Nj+1:Nt) may be mixed.
! The covariances, Cov(X(1:I-1),X(I)|X(I+1:N)), needed for computation of the
! conditional expectation, E(X(1:I-1)|X(I:N), are saved in column I of R
! for I=Nt-Nj+1:Ntdc.
!
! IF any of the variables have variance less than EPS2. They will be
! be treated as deterministic and not stochastic variables by the
! RindXXX subroutines. The deterministic variables are moved to
! middle in the order they became deterministic in order to
! keep track of them. Their variance and covariance with
! the remaining stochastic variables are set to zero in
! order to avoid numerical difficulties.
!
! NsXtmj(I) is the number of variables  among the Nt-Nj
! first we treat stochastically after conditioning on X(Nt-Nj+I).
! The covariance matrix is sorted so that all variables with indices
! from 1 to NsXtmj(I) are stochastic after conditioning
! on X(Nt-Nj+I).  Thus NsXtmj(I) may also be considered
! as the index to the last stochastic variable after conditioning
! on X(Nt-Nj+I). In other words NsXtmj keeps track of the deterministic
! and stochastic variables among the Nt-Nj first variables in each
! conditioning step.
!
! Similarly  NsXdj(I)  keeps track of the deterministic and stochastic
! variables among the Nd+Nj following variables in each conditioning step.
! NsXdj(I) is the index to the first stochastic variable
! among the Nd+Nj following variables after conditioning on X(Nt-Nj+I).
! The covariance matrix is sorted so that all variables with indices
! from NsXdj(I+1) to NsXdj(I)-1 are  deterministic conditioned on
! X(Nt-Nj+I).
!

! Var(Xc(1))>Var(Xc(2)|Xc(1))>...>Var(Xc(Nc)|Xc(1),Xc(2),...,Xc(Nc)).
! If Nj=0 then
! Var(Xd(1)|Xc)>Var(Xd(2)|Xd(1),Xc)>...>Var(Xd(Nd)|Xd(1),Xd(2),...,Xd(Nd),Xc).
!
! NB!! Since R is symmetric, only the upper triangular contains the
! sorted conditional covariance. The whole matrix
! is easily obtained by copying elements of the upper triangle to
! the lower or by uncommenting some lines in the end of this subroutine

! revised pab 18.04.2000
!  new name rind60
!  New assumption of BIG for the conditional sorted variables:
!                         BIG(I,I)=sqrt(Var(X(I)|X(I+1)...X(N))=SQI
!                         BIG(1:I-1,I)=COV(X(1:I-1),X(I)|X(I+1)...X(N))/SQI
!      Otherwise
!                         BIG(I,I) = Var(X(I)|X(I+1)...X(N)
!                         BIG(1:I-1,I)=COV(X(1:I-1),X(I)|X(I+1)...X(N))
!  This also affects C1C2: SQ0=sqrt(Var(X(I)|X(I+1)...X(N)) is removed from input
!  =>  A lot of wasteful divisions are avoided



! Using SQ to temporarily store the diagonal of R
! Adding a nugget effect to ensure the the inversion is
! not corrupted by round off errors
! good choice for nugget might be 1e-8
                             !call getdiag(SQ,R)
      INFORM = 0
      ALLOCATE(SQ(1:Ntdc))

      IF (Nd+Nj+Njj+Ntscis.GT.0) THEN
         ALLOCATE(CSTD2(1:Ntd,1:Nd+Nj+Njj+Ntscis))
         CSTD2=0.d0             ! initialize CSTD
      ENDIF
      !CALL ECHO(R,Ntdc)
      DO ix = 1, Ntdc
         R(ix,ix)=R(ix,ix)+Nugget
         SQ(ix)=R(ix,ix)
         index1 (ix) = ix       ! initialize index1
      ENDDO

      Ntmj=Nt-Nj
      !NsXtmj(Njj+Nd+Nj+1)=Ntmj      ! index to last stochastic variable of Nt-Nj of Xt
      !NsXdj(Nd+Nj+1)=Ntmj+1     ! index to first stochastic variable of Xd and Nj of Xt
      !degenerate=0
      Njleft=Nj
      NstoXd=Ntmj+1;Nstoc=Ntmj


      DO ix = 1, Nc             ! Condsort Xc
         r1 = Ntdc-ix
         m=r1+2-MAXLOC(SQ(r1+1:Ntd+1:-1))
         IF (SQ(m(1)).LT.XCEPS2) THEN
            INFORM = 1
            !PRINT *,'Condsort, degenerate Xc'
            IF (SQ(m(1)).LT.-XCEPS2) THEN
               !print *, 'Condsort, Not semi-positive definit'
            ENDIF
                                !degenerate=1
            GOTO 200            ! RETURN    !degenerate case
         ENDIF
         m1=index1(m(1));
         CALL swapint(index1(m(1)),index1(Ntdc-ix+1))
         !CALL swapRe(SQ(r1+1),SQ(m(1)))
         SQ(r1+1) = SQRT(SQ(m(1)))
         R(index1(1:r1+1),m1) = R(index1(1:r1+1),m1)/SQ(r1+1)
         R(m1,index1(1:r1))   = R(index1(1:r1),m1)
                                ! sort and calculate conditional covariances
         CALL CONDSORT2(R,SQ,index1,Nstoc,NstoXd,Njleft,m1,Ntdc-ix)
      ENDDO                     ! ix

      NsXdj(Nd+Nj+1)  = NstoXd  ! index to first stochastic variable of Xd and Nj of Xt
      NsXtmj(Nd+Nj+Njj+Ntscis+1) = Nstoc ! index to last stochastic variable of Nt-Nj of Xt
      !print *, 'condsort index1', index1
      !print *, 'condsort Xd'
      !call echo(R,Ntdc)

      DO ix = 1, Nd+Nj          ! Condsort Xd +  Nj of Xt
         r1 = Ntd-ix
         IF (Njleft.GT.0) THEN
            m=r1+2-MAXLOC(SQ(r1+1:1:-1))
            Ntmp=NstoXd+Njleft-1
            IF (((NstoXd.LE.m(1)).AND.(m(1).LE.Ntmp))
     &           .OR.(m(1).LE.Nstoc)) THEN
               CALL swapint(index1(m(1)),index1(Ntmp))
               CALL swapRe(SQ(m(1)),SQ(Ntmp))
               m(1)=Ntmp
               Njleft=Njleft-1
            END IF
         ELSE
            m=r1+2-MAXLOC(SQ(r1+1:Ntmj+1:-1))
         END IF
         IF (SQ(m(1)).LT.EPS2) THEN
                                !PRINT *,'Condsort, degenerate Xd'
                                !degenerate=1
            Ntmp=Nd+Nj+1-ix
            NsXtmj(Ntscis+Njj+1:Ntmp+Ntscis+Njj+1)=Nstoc
            NsXdj(1:Ntmp+1)=NstoXd
            IF (ix.EQ.1) THEN
               DO iy=1,Ntd      !sqrt(VAR(X(I)|X(Ntd-ix+1:Ntdc))
                  r1=index1(iy)
                  CSTD2(r1,Ntscis+Njj+1:Ntmp+Ntscis+Njj)=SQRT(SQ(iy))
               ENDDO
            ELSE
               DO iy=ix,Nd+Nj
                  CSTD2(:,Nd+Nj+Ntscis+Njj+1-iy)=
     &                 CSTD2(:,Ntmp+Ntscis+Njj+1)
               ENDDO
            ENDIF
            GOTO 200            ! degenerate case
         END IF
         m1=index1(m(1));
         CALL swapint(index1(m(1)),index1(r1+1))
         !CSTD2(m1,Nd+Nj+Ntscis+Njj+1-ix)=SQRT(SQ(m(1)))
         !CALL swapRe(SQ(Ntd-ix+1),SQ(m(1)))
         SQ(r1+1) = SQRT(SQ(m(1)))
         CSTD2(m1,Nd+Nj+Ntscis+Njj+1-ix) = SQ(r1+1)

         R(index1(1:r1+1),m1) = R(index1(1:r1+1),m1)/SQ(r1+1)
         R(m1,index1(1:r1)) = R(index1(1:r1),m1)

                                ! Calculating conditional variances
         CALL CONDSORT2(R,SQ,index1,Nstoc,NstoXd,Njleft,m1,Ntd-ix)
                                ! saving indices
         NsXtmj(Nd+Nj+Njj+Ntscis+1-ix)=Nstoc
         NsXdj(Nd+Nj+1-ix)=NstoXd

                                ! Calculating standard deviations non-deterministic variables
         DO row=1,NsXtmj(Nd+Nj+Njj+Ntscis+2-ix)        !Nstoc
            r1=index1(row)
            CSTD2(r1,Nd+Nj+Njj+Ntscis+1-ix)=SQRT(SQ(row)) !sqrt(VAR(X(I)|X(Ntd-ix+1:Ntdc))
         ENDDO
         DO row=NsXdj(Nd+Nj+2-ix),Ntd-ix              !NstoXd,Ntd-ix
            r1=index1(row)
            CSTD2(r1,Nd+Nj+Ntscis+Njj+1-ix)=SQRT(SQ(row)) !sqrt(VAR(X(I)|X(Ntd-ix+1:Ntdc))
         ENDDO
      ENDDO                     ! ix


 200  IF ((SCIS.GT.0).OR. (Njj.gt.0)) THEN ! check on Njj instead
          ! Calculating conditional variances and sort for Nstoc of Xt
         CALL CONDSORT3(R,CSTD2,SQ,index1,NsXtmj,Nstoc)
         !Nst0=Nstoc
      ENDIF
      IF ((Nd+Nj+Njj+Ntscis.GT.0)) THEN
         DO row=1,Ntd           ! sorting CSTD according to index1
            r1=index1(row)
            CSTD(row,:)= CSTD2(r1,:)
         END DO
         DEALLOCATE(CSTD2)
      ELSE
         IF (Nc.EQ.0) THEN
            ix=1; Nstoc=Ntmj
            DO WHILE (ix.LE.Nstoc)
               IF (SQ(ix).LT.EPS2) THEN
                  DO WHILE ((SQ(Nstoc).LT.EPS2).AND.(ix.LT.Nstoc))
                     SQ(Nstoc)=0.d0 !max(0.d0,SQ(Nstoc))
                     Nstoc=Nstoc-1
                  END DO
                  CALL swapint(index1(ix),index1(Nstoc)) ! swap indices
                  !CALL swapRe(SQ(ix),SQ(Nstoc))
                  SQ(ix)=SQ(Nstoc);SQ(Nstoc)=0.d0
                  Nstoc=Nstoc-1
               ENDIF
               ix=ix+1
            END DO
         ENDIF
         CSTD(1:Nt,1)=SQRT(SQ(1:Nt))
         NsXtmj(1)=Nstoc
      ENDIF

      changed=0
      DO row=Ntdc,1,-1          ! sorting the upper triangular of the
         r1=index1(row)         ! covariance matrix according to index1
         xedni(r1)=row
         !PRINT *,'condsort,xedni',xedni
         !PRINT *,'condsort,r1,row',r1,row
         IF ((r1.NE.row).OR.(changed.EQ.1)) THEN
            changed=1
            R(row,row)=SQ(row)
            DO col=row+1,Ntdc
               c1=index1(col)
               IF (c1.GT.r1) THEN
                  R(row,col)=R(c1,r1)
               ELSE
                  R(row,col)=R(r1,c1)
               END IF
            END DO
         END IF
      END DO
                                ! you may sort the lower triangular according
                                ! to index1 also, but it is not needed
                                ! since R is symmetric.  Uncomment the
                                ! following if the whole matrix is needed
!      DO col=1,Ntdc
!         DO row=col+1,Ntdc
!            R(row,col)=R(col,row) ! R symmetric
!         END DO
!      END DO
!      IF (degenerate.EQ.1) THEN
!         PRINT *,'condsort,R='
!         call echo(R,Ntdc)
!         PRINT *,'condsort,SQ='
!         call echo(CSTD,Ntd)
!         PRINT *,'index=',index1
!         PRINT *,'xedni=',xedni
!      ENDIF
!      PRINT * , 'big'
!600   FORMAT(4F8.4)
!      PRINT 600, R
!      PRINT 600, SQ
      DEALLOCATE(SQ)

      RETURN
      END SUBROUTINE CONDSORT


      SUBROUTINE CONDSORT2(R,SQ,index1,Nstoc,NstoXd,Njleft,m1,N)
      USE GLOBALDATA, ONLY : Ntd,EPS2,XCEPS2
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R
      DOUBLE PRECISION, DIMENSION(:),   INTENT(inout) :: SQ
      INTEGER,          DIMENSION(:  ), INTENT(inout)   :: index1
      INTEGER, INTENT(inout)   :: Nstoc,NstoXd,Njleft
      INTEGER, INTENT(in)   :: m1,N
! local variables
      INTEGER       :: Nsold,Ndold, Ntmp
      INTEGER       :: r1,c1,row,col,iy

! save their old values
      Nsold=Nstoc;Ndold=NstoXd

                                ! Calculating conditional variances for the
                                ! Xc variables.
      DO row=Ntd+1,N
         r1 = index1(row)
         SQ(row) = R(r1,r1)-R(r1,m1)*R(m1,r1)     !/R(m1,m1)
         IF (SQ(row).LT.XCEPS2) THEN
            IF (SQ(row).LT.-XCEPS2) THEN
               !print *, 'Condsort2,Error: Covariance negative definit'
            ENDIF
            R(r1,r1) = 0.d0
            SQ(row) = 0.d0
            !PRINT *,'condsort2, degenerate xc'
            RETURN              ! degenerate case XIND should return NaN
         ELSE
            R(r1,r1)=SQ(row)
            DO col=row+1,N
               c1 = index1(col)
               R(c1,r1) = R(r1,c1)-R(r1,m1)*R(m1,c1)      !/R(m1,m1)
               R(r1,c1) = R(c1,r1)
            ENDDO
         ENDIF
      ENDDO                     ! Calculating conditional variances for the
                                ! first Nstoc variables.
                                ! variables with variance less than EPS2
                                ! will be treated as deterministic and not
                                ! stochastic variables and are therefore moved
                                ! to the end among these Nt-Nj first variables.
                                ! Nstoc is the # of variables we treat
                                ! stochastically
      iy=1
      DO WHILE (iy.LE.Nstoc)
         r1=index1(iy)
         SQ(iy)=R(r1,r1)-R(r1,m1)*R(m1,r1)               !/R(m1,m1)
         IF (SQ(iy).LT.EPS2) THEN
            IF (SQ(iy).LT.-EPS2) THEN
               !print *, 'Condsort2,Error: Covariance negative definit'
            ENDIF
            r1=index1(Nstoc)
            SQ(Nstoc)=R(r1,r1)-R(r1,m1)*R(m1,r1)         !/R(m1,m1)

            DO WHILE ((SQ(Nstoc).LT.EPS2).AND.(iy.LT.Nstoc))
               IF (SQ(Nstoc).LT.-EPS2) THEN
                !print *, 'Condsort2,Error: Covariance negative definit'
               ENDIF
               SQ(Nstoc)=0.d0 !MAX(0.d0,SQ(Nstoc))
               Nstoc=Nstoc-1
               r1=index1(Nstoc)
               SQ(Nstoc)=R(r1,r1)-R(r1,m1)*R(m1,r1)       !/R(m1,m1)
            END DO
            CALL swapint(index1(iy),index1(Nstoc)) ! swap indices
            !CALL swapre(SQ(iy),SQ(Nstoc))          ! swap values
            SQ(iy)=SQ(Nstoc);SQ(Nstoc)=0.d0
            Nstoc=Nstoc-1
         ENDIF
         iy=iy+1
      END DO

                                ! Calculating conditional variances for the
                                ! stochastic variables Xd and Njleft of Xt.
                                ! Variables with conditional variance less than
                                ! EPS2 are moved to the beginning among these
                                ! with only One exception: if it is one of the
                                ! Xt variables and Nstoc>0 then it switch place
                                ! with Xt(Nstoc)

      DO iy=Ndold,MIN(Ntd,N)
         r1=index1(iy)
         SQ(iy)=R(r1,r1)-R(r1,m1)*R(m1,r1)         !/R(m1,m1)
         IF (SQ(iy).LT.EPS2) THEN
            IF (Njleft.GT.0) THEN
               Ntmp=NstoXd+Njleft
               IF (iy.LT.Ntmp) THEN
                  IF (Nstoc.GT.0) THEN !switch place with Xt(Nstoc)
                     CALL swapint(index1(iy),index1(Nstoc))
                     !CALL swapre(SQ(iy),SQ(Nstoc))
                     SQ(iy)=SQ(Nstoc);SQ(Nstoc)=0.d0
                     Nstoc=Nstoc-1
                  ELSE
                     CALL swapint(index1(iy),index1(NstoXd))
                     !CALL swapre(SQ(iy),SQ(NstoXd))
                     SQ(iy)=SQ(NstoXd);SQ(NstoXd)=0.d0
                     Njleft=Njleft-1
                     NstoXd=NstoXd+1
                  ENDIF
               ELSE
                  CALL swapint(index1(iy),index1(Ntmp))
                  CALL swapint(index1(Ntmp),index1(NstoXd))
                  !CALL swapre(SQ(iy),SQ(Ntmp))
                  !CALL swapre(SQ(Ntmp),SQ(NstoXd))
                  SQ(iy)=SQ(Ntmp);SQ(Ntmp)=SQ(NstoXd)
                  SQ(NstoXd)=0.d0
                  NstoXd=NstoXd+1
               ENDIF
            ELSE
               CALL swapint(index1(iy),index1(NstoXd))
               !CALL swapre(SQ(iy),SQ(NstoXd)) !
               SQ(iy)=SQ(NstoXd);SQ(NstoXd)=0.d0
               NstoXd=NstoXd+1
            ENDIF
         ENDIF                  ! SQ < EPS2
      ENDDO


            ! Calculating Covariances for non-deterministic variables
      DO row=1,Nstoc
         r1=index1(row)
         R(r1,r1)=SQ(row)
         DO col=row+1,Nstoc
            c1=index1(col)
            R(c1,r1)=R(r1,c1)-R(r1,m1)*R(m1,c1)  !/R(m1,m1)
            R(r1,c1)=R(c1,r1)
         ENDDO
         DO col=NstoXd,N
            c1=index1(col)
            R(c1,r1)=R(r1,c1)-R(r1,m1)*R(m1,c1)  !/R(m1,m1)
            R(r1,c1)=R(c1,r1)
         ENDDO
      ENDDO
      DO row=NstoXd,MIN(Ntd,N)
         r1=index1(row)
         R(r1,r1)=SQ(row)

         DO col=row+1,N
            c1=index1(col)
            R(c1,r1)=R(r1,c1)-R(r1,m1)*R(m1,c1)  !/R(m1,m1)
            R(r1,c1)=R(c1,r1)
         ENDDO
      ENDDO

                                ! Set covariances for Deterministic variables to zero
                                ! in order to avoid numerical problems

      DO row=Ndold,NStoXd-1
         r1=index1(row)
         SQ(row)  = 0.d0 !MAX(SQ(row),0.d0)
         R(r1,r1) = SQ(row)
         DO col=row+1,N
            c1=index1(col)
            R(c1,r1)=0.d0
            R(r1,c1)=0.d0
         ENDDO
         DO col=1,Nsold
            c1=index1(col)
            R(c1,r1)=0.d0
            R(r1,c1)=0.d0
         ENDDO
      ENDDO

      DO row=Nstoc+1,Nsold
         r1=index1(row)
         SQ(row)  = 0.d0 !MAX(SQ(row),0.d0)
         R(r1,r1)=SQ(row)
         DO col=1,row-1
            c1=index1(col)
            R(c1,r1)=0.d0
            R(r1,c1)=0.d0
         ENDDO
         DO col=NstoXd,N
            c1=index1(col)
            R(c1,r1)=0.d0
            R(r1,c1)=0.d0
         ENDDO
      ENDDO
      RETURN
      END SUBROUTINE CONDSORT2

      SUBROUTINE CONDSORT3(R,CSTD2,SQ,index1,NsXtmj,Nstoc)
      USE GLOBALDATA, ONLY : EPS2,Njj,Ntscis
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R,CSTD2
      DOUBLE PRECISION, DIMENSION(:),   INTENT(inout) :: SQ ! diag. of R
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: index1,NsXtmj
      INTEGER,          DIMENSION(1) :: m
      INTEGER, INTENT(inout)   :: Nstoc
! local variables
      INTEGER  :: m1
      INTEGER  :: Nsold
      INTEGER  :: r1,c1,row,col,iy,ix
! This function condsort all the Xt variables for use with RINDSCIS and
! MNORMPRB

      !Nsoold=Nstoc
      ix=1

      DO WHILE ((ix.LE.Nstoc).and.(ix.LE.(Ntscis+Njj)))
         m=ix-1+MAXLOC(SQ(ix:Nstoc))
         IF (SQ(m(1)).LT.EPS2) THEN
            !PRINT *,'Condsort3, error degenerate X'
            NsXtmj(1:Njj+Ntscis)=0
            Nstoc=0       !degenerate=1
            RETURN    !degenerate case
         ENDIF
         m1=index1(m(1));
         CALL swapint(index1(m(1)),index1(ix))
         SQ(ix) = SQRT(SQ(m(1)))
         CSTD2(m1,ix) = SQ(ix)

         R(index1(ix:Nstoc),m1) = R(index1(ix:Nstoc),m1)/SQ(ix)
         R(m1,index1(ix+1:Nstoc)) = R(index1(ix+1:Nstoc),m1)
                                ! Calculating conditional variances for the
                                ! first Nstoc variables.
                                ! variables with variance less than EPS2
                                ! will be treated as deterministic and not
                                ! stochastic variables and are therefore moved
                                ! to the end among these variables.
                                ! Nstoc is the # of variables we treat
                                ! stochastically
         iy=ix+1;Nsold=Nstoc
         DO WHILE (iy.LE.Nstoc)
            r1=index1(iy)
            SQ(iy)=R(r1,r1)-R(r1,m1)*R(m1,r1)         !/R(m1,m1)
            IF (SQ(iy).LT.EPS2) THEN
               IF (SQ(iy).LT.-EPS2) THEN
                  !print *,'Cndsrt3,Error:Covariance negative definit'
               ENDIF
               r1=index1(Nstoc)
               SQ(Nstoc)=R(r1,r1)-R(r1,m1)*R(m1,r1)     !/R(m1,m1)
               DO WHILE ((SQ(Nstoc).LT.EPS2).AND.(iy.LT.Nstoc))
                  IF (SQ(Nstoc).LT.-EPS2) THEN
                     !print *,'Cndsrt3,Error:Covariance negative definit'
                  ENDIF
                  SQ(Nstoc)=0.d0 !MAX(0.d0,SQ(Nstoc))
                  Nstoc=Nstoc-1
                  r1=index1(Nstoc)
                  SQ(Nstoc)=R(r1,r1)-R(r1,m1)*R(m1,r1)     !/R(m1,m1)
               END DO
               CALL swapint(index1(iy),index1(Nstoc)) ! swap indices
               !CALL swapre(SQ(iy),SQ(Nstoc)) !
               SQ(iy)=SQ(Nstoc); SQ(Nstoc)=0.d0 ! swap values
               Nstoc=Nstoc-1
            ENDIF
            iy=iy+1
         END DO
         NsXtmj(ix)=Nstoc ! saving index to last stoch. var. after conditioning
             ! Calculating Covariances for non-deterministic variables
         DO row=ix+1,Nstoc
            r1=index1(row)
            R(r1,r1)=SQ(row)
            CSTD2(r1,ix)=SQRT(SQ(row)) ! saving stdev after conditioning on ix
            DO col=row+1,Nstoc
               c1=index1(col)
               R(c1,r1)=R(r1,c1)-R(r1,m1)*R(m1,c1)   !/R(m1,m1)
               R(r1,c1)=R(c1,r1)
            ENDDO
         ENDDO
            ! similarly for deterministic values
         DO row=Nstoc+1,Nsold
            r1=index1(row)
            SQ(row)=0.d0 !MAX(SQ(row),0.d0)
            R(r1,r1)=SQ(row)
            DO col=ix+1,Nsold   !row-1
               c1=index1(col)
               R(c1,r1)=0.d0
               R(r1,c1)=0.d0
            ENDDO
         ENDDO
         ix=ix+1
      ENDDO
      NsXtmj(Nstoc+1:Njj+Ntscis)=Nstoc
      RETURN
      END SUBROUTINE CONDSORT3

      SUBROUTINE swapRe(m,n)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(inout) :: m,n
      DOUBLE PRECISION                :: tmp
      tmp=m
      m=n
      n=tmp
      END SUBROUTINE swapRe

      SUBROUTINE swapint(m,n)
      IMPLICIT NONE
      INTEGER, INTENT(inout) :: m,n
      INTEGER                :: tmp
      tmp=m
      m=n
      n=tmp
      END SUBROUTINE swapint

      SUBROUTINE getdiag(diag,matrix)
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(out) :: diag
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in)  :: matrix
      DOUBLE PRECISION, DIMENSION(:  ), ALLOCATABLE :: vector

      ALLOCATE(vector(SIZE(matrix)))
      vector=PACK(matrix,.TRUE.)
      diag=vector(1:SIZE(matrix):SIZE(matrix,dim=1)+1)
      DEALLOCATE(vector)
      END SUBROUTINE getdiag

      END MODULE RIND71MOD








