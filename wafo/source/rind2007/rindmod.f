! Programs available in module RINDMOD :
!
!   1) setConstants
!   2) RINDD
!
! SETCONSTANTS set member variables controlling the performance of RINDD
!
! CALL setConstants(method,xcscale,abseps,releps,coveps,maxpts,minpts,nit,xcutoff,Nc1c2)
!
! METHOD  = INTEGER defining the SCIS integration method
!             1 Integrate by SADAPT for Ndim<9 and by KRBVRC otherwise
!             2 Integrate by SADAPT for Ndim<20 and by KRBVRC otherwise
!             3 Integrate by KRBVRC by Genz (1993) (Fast Ndim<101) (default)
!             4 Integrate by KROBOV by Genz (1992) (Fast Ndim<101)
!             5 Integrate by RCRUDE by Genz (1992) (Slow Ndim<1001)
!             6 Integrate by SOBNIED               (Fast Ndim<1041)
!             7 Integrate by DKBVRC by Genz (2003) (Fast Ndim<1001)
!
!   XCSCALE = REAL to scale the conditinal probability density, i.e.,
!             f_{Xc} = exp(-0.5*Xc*inv(Sxc)*Xc + XcScale) (default XcScale =0)
!   ABSEPS  = REAL absolute error tolerance.       (default 0)
!   RELEPS  = REAL relative error tolerance.       (default 1e-3)
!   COVEPS  = REAL error tolerance in Cholesky factorization (default 1e-13)
!   MAXPTS  = INTEGER, maximum number of function values allowed. This
!             parameter can be used to limit the time. A sensible
!             strategy is to start with MAXPTS = 1000*N, and then
!             increase MAXPTS if ERROR is too large.
!             (Only for METHOD~=0) (default 40000)
!   MINPTS  = INTEGER, minimum number of function values allowed.
!             (Only for METHOD~=0) (default 0)
!   NIT     = INTEGER, maximum number of Xt variables to integrate
!             This parameter can be used to limit the time.
!             If NIT is less than the rank of the covariance matrix,
!             the returned result is a upper bound for the true value
!             of the integral.  (default 1000)
!   XCUTOFF = REAL cut off value where the marginal normal
!             distribution is truncated. (Depends on requested
!             accuracy. A value between 4 and 5 is reasonable.)
!  NC1C2    = number of times to use the regression equation to restrict
!            integration area. Nc1c2 = 1,2 is recommended. (default 2)
!
!
!RIND computes  E[Jacobian*Indicator|Condition]*f_{Xc}(xc(:,ix))
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
!CALL RINDD(E,err,terr,S,m,xc,Nt,indI,Blo,Bup,INFIN);
!
!        E = expectation/density as explained above size 1 x Nx (out)
!      ERR = estimated sampling error  size 1 x Nx (out)
!     TERR = estimated truncation error size 1 x Nx (out)
!        S = Covariance matrix of X=[Xt;Xd;Xc] size N x N (N=Nt+Nd+Nc) (in)
!        m = the expectation of X=[Xt;Xd;Xc]   size N x 1              (in)
!       xc = values to condition on            size Nc x Nx            (in)
!     indI = vector of indices to the different barriers in the        (in)
!            indicator function,  length NI, where   NI = Nb+1
!            (NB! restriction  indI(1)=0, indI(NI)=Nt+Nd )
!  Blo,Bup = Lower and upper barrier coefficients used to compute the   (in)
!            integration limits A and B, respectively.
!            size  Mb x Nb. If  Mb<Nc+1 then
!            Blo(Mb+1:Nc+1,:) is assumed to be zero.
!    INFIN = INTEGER, array of integration limits flags:  size 1 x Nb   (in)
!            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!            if INFIN(I) = 0, Ith limits are (-infinity, Hup(I)];
!            if INFIN(I) = 1, Ith limits are [Hlo(I), infinity);
!            if INFIN(I) = 2, Ith limits are [Hlo(I), Hup(I)].
!
! The relation to the integration limits Hlo and Hup are as follows
!    IF INFIN(j)>=0,
!      IF INFIN(j)~=0,  A(i)=Blo(1,j)+Blo(2:Mb,j).'*xc(1:Mb-1,ix),
!      IF INFIN(j)~=1,  B(i)=Bup(1,j)+Bup(2:Mb,j).'*xc(1:Mb-1,ix),
!
!            where i=indI(j-1)+1:indI(j), j=1:NI-1, ix=1:Nx
!            Thus the integration limits may change with the conditional
!            variables.
!Example:
! The indices, indI=[0 3 5 6], and coefficients Blo=[0 0 -1],
! Bup=[0 0 5], INFIN=[0 1 2]
! means that   A = [-inf -inf -inf 0 0 -1]  B = [0 0 0 inf inf 5]
!
!
! (Recommended limitations Nx,Nt<101, Nd<7 and Nc<11)
! Also note that the size information have to be transferred to RINDD
! through the input arguments E,S,m,Nt,IndI,Blo,Bup and INFIN
!
! For further description see the modules
!
! References
! Podgorski et al. (2000)
! "Exact distributions for apparent waves in irregular seas"
! Ocean Engineering,  Vol 27, no 1, pp979-1016.                        (RINDD)
!
! R. Ambartzumian, A. Der Kiureghian, V. Ohanian and H.
! Sukiasian (1998)
! "Multinormal probabilities by sequential conditioned
!  importance sampling: theory and application"                        (MVNFUN)
! Probabilistic Engineering Mechanics, Vol. 13, No 4. pp 299-308
!
! Alan Genz (1992)
! 'Numerical Computation of Multivariate Normal Probabilites'          (MVNFUN)
! J. computational Graphical Statistics, Vol.1, pp 141--149
!
! Alan Genz and Koon-Shing Kwong (2000?)
! 'Numerical Evaluation of Singular Multivariate Normal Distributions' (MVNFUN,COVSRT)
! Computational Statistics and Data analysis
!
!
! P. A. Brodtkorb (2004),                                 (RINDD, MVNFUN, COVSRT)
! Numerical evaluation of multinormal expectations
! In Lund university report series
! and in the Dr.Ing thesis:
! The probability of Occurrence of dangerous Wave Situations at Sea.
! Dr.Ing thesis, Norwegian University of Science and Technolgy, NTNU,
! Trondheim, Norway.

! Tested on:  DIGITAL UNIX Fortran90 compiler
!             PC pentium II with Lahey Fortran90 compiler
!             Solaris with SunSoft F90 compiler Version 1.0.1.0  (21229283)
! History:
! Revised pab aug. 2009
! -renamed from rind2007 to rindmod
! Revised pab July 2007
! - separated the absolute error into ERR and TERR.
! - renamed from alanpab24 -> rind2007
! revised pab 23may2004
! RIND module totally rewritten according to the last reference.


      MODULE GLOBALCONST        ! global constants
      IMPLICIT NONE
      DOUBLE PRECISION, PARAMETER :: gSQTWPI1= 0.39894228040143D0  !=1/sqrt(2*pi)
      DOUBLE PRECISION, PARAMETER :: gSQPI1  = 0.56418958354776D0  !=1/sqrt(pi)
      DOUBLE PRECISION, PARAMETER :: gSQPI   = 1.77245385090552D0  !=sqrt(pi)
      DOUBLE PRECISION, PARAMETER :: gSQTW   = 1.41421356237310D0  !=sqrt(2)
      DOUBLE PRECISION, PARAMETER :: gSQTW1  = 0.70710678118655D0  !=1/sqrt(2)
      DOUBLE PRECISION, PARAMETER :: gPI1    = 0.31830988618379D0  !=1/pi
      DOUBLE PRECISION, PARAMETER :: gPI     = 3.14159265358979D0  !=pi
      DOUBLE PRECISION, PARAMETER :: gTWPI   = 6.28318530717958D0  !=2*pi
      DOUBLE PRECISION, PARAMETER :: gSQTWPI = 2.50662827463100D0  !=sqrt(2*pi)
      DOUBLE PRECISION, PARAMETER :: gONE    = 1.D0
      DOUBLE PRECISION, PARAMETER :: gTWO    = 2.D0
      DOUBLE PRECISION, PARAMETER :: gHALF   = 0.5D0
      DOUBLE PRECISION, PARAMETER :: gZERO   = 0.D0
      DOUBLE PRECISION, PARAMETER :: gINFINITY = 37.D0 ! SQRT(-gTWO*LOG(1.D+12*TINY(gONE)))
!     Set gINFINITY (infinity).
!     Such that EXP(-2.x^2) > 10^(12) times TINY
!     SAVE gINFINITY
      END MODULE GLOBALCONST

      MODULE RINDMOD
      USE GLOBALCONST
!      USE PRINTMOD  ! used for debugging only
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: RINDD, SetConstants
	PUBLIC :: mCovEps, mAbsEps,mRelEps, mXcutOff, mXcScale
      PUBLIC :: mNc1c2, mNIT, mMaxPts,mMinPts, mMethod, mSmall
      private :: preInit
      private :: initIntegrand
      private :: initfun,mvnfun,cvsrtxc,covsrt1,covsrt,rcscale,rcswap
      private :: cleanUp

	INTERFACE RINDD
      MODULE PROCEDURE RINDD
      END INTERFACE

	INTERFACE SetConstants
      MODULE PROCEDURE SetConstants
      END INTERFACE

! mInfinity = what is considered as infinite value in FI
! mFxcEpss  = if fxc is less, do not compute E(...|Xc)
! mXcEps2   =  if any Var(Xc(j)|Xc(1),...,Xc(j-1)) <= XCEPS2 then return NAN
      double precision, parameter :: mInfinity = 8.25d0 ! 37.0d0
      double precision, parameter :: mFxcEpss = 1.0D-20
      double precision, save      :: mXcEps2   = 2.3d-16
!     Constants defining accuracy of integration:
!     mCovEps = termination criteria for Cholesky decomposition
!     mAbsEps = requested absolute tolerance
!     mRelEps = requested relative tolerance
!     mXcutOff = truncation value to c1c2
!     mXcScale = scale factor in the exponential (in order to avoid overflow)
!     mNc1c2  = number of times to use function c1c2, i.e.,regression
!               equation to restrict integration area.
!     mNIT    = maximum number of Xt variables to integrate
!     mMethod = integration method:
!            1 Integrate all by SADAPT if NDIM<9 otherwise by KRBVRC (default)
!            2 Integrate all by SADAPT if NDIM<19 otherwise by KRBVRC
!            3 Integrate all by KRBVRC by Genz (1998) (Fast and reliable)
!            4 Integrate all by KROBOV by Genz (1992) (Fast and reliable)
!            5 Integrate all by RCRUDE by Genz (1992) (Reliable)
!            6 Integrate all by SOBNIED by Hong and Hickernell
!            7 Integrate all by DKBVRC by Genz (2003) (Fast Ndim<1001)
      double precision, save :: mCovEps  = 1.0d-10
      double precision, save :: mAbsEps  = 0.01d0
      double precision, save :: mRelEps  = 0.01d0
      double precision, save :: mXcutOff = 5.d0
      double precision, save :: mXcScale = 0.0d0
      integer, save :: mNc1c2  = 2
      integer, save :: mNIT    = 1000
      integer, save :: mMaxPts = 40000
      integer, save :: mMinPts = 0
      integer, save :: mMethod = 3


!     Integrand variables:
!     mBIG    = Cholesky Factor/Covariance matrix:
!               Upper triangular part is the cholesky factor
!               Lower triangular part contains the conditional
!               standarddeviations
!               (mBIG2 is only used if mNx>1)
!     mCDI    = Cholesky DIagonal elements
!     mA,mB   = Integration limits
!     mINFI   = integrationi limit flags
!     mCm     = conditional mean
!     mINFIXt,
!     mINFIXd = # redundant variables of Xt and Xd,
!              respectively
!     mIndex1,
!     mIndex2 = indices to the variables original place.   Size  Ntdc
!     xedni   = indices to the variables new place.        Size  Ntdc
!     mNt     = # Xt variables
!     mNd     = # Xd variables
!     mNc     = # Xc variables
!     mNtd    = mNt + mNd
!     mNtdc   = mNt + mNd + mNc
!     mNx     = # different integration limits

      double precision,allocatable, dimension(:,:) :: mBIG,mBIG2
      double precision,allocatable, dimension(:)   :: mA,mB,mCDI,mCm
      INTEGER, DIMENSION(:),ALLOCATABLE :: mInfi,mIndex1,mIndex2,mXedni
      INTEGER,SAVE :: mNt,mNd,mNc,mNtdc, mNtd, mNx ! Size information
      INTEGER,SAVE :: mInfiXt,mInfiXd
      logical,save :: mInitIntegrandCalled = .FALSE.

      DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: mCDIXd, mCmXd
      DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: mXd, mXc, mY
      double precision, save :: mSmall = 2.3d-16

!     variables set in initfun and used in mvnfun:
      INTEGER, PRIVATE :: mI0,mNdleftN0
      DOUBLE PRECISION, PRIVATE :: mE1,mD1, mVAL0

      contains
      subroutine setConstants(method,xcscale,abseps,releps,coveps,
     &     maxpts,minpts,nit,xcutoff,Nc1c2)
      double precision, optional, intent(in) :: xcscale,abseps,releps
     $     ,coveps, xcutoff
      integer, optional,intent(in) :: method,nit,maxpts,minpts,Nc1c2
      double precision, parameter :: one = 1.0d0
      mSmall = spacing(one)
      if (present(method))  mMethod = method
      if (present(xcscale)) mXcScale = xcscale
      if (present(abseps)) mAbsEps = max(abseps,mSmall)
      if (present(releps)) mRelEps = max(releps,0.0d0)
      if (present(coveps)) mCovEps = max(coveps,1d-12)
      if (present(maxpts)) mMaxPts = maxpts
      if (present(minpts)) mMinPts = minpts
      if (present(nit))    mNit    = nit
      if (present(xcutOff)) mXcutOff = xCutOff
      if (present(Nc1c2))  mNc1c2   = max(Nc1c2,1)
!      print *, 'method=', mMethod
!      print *, 'xcscale=', mXcScale
!      print *, 'abseps=', mAbsEps
!      print *, 'releps=', mRelEps
!      print *, 'coveps=', mCovEps
!      print *, 'maxpts=', mMaxPts
!      print *, 'minpts=', mMinPts
!      print *, 'nit=',    mNit
!      print *, 'xcutOff=', mXcutOff
!      print *, 'Nc1c2=',  mNc1c2
      end subroutine setConstants

      subroutine preInit(BIG,Xc,Nt,inform)
      double precision,dimension(:,:), intent(in) :: BIG
      double precision,dimension(:,:), intent(in) :: Xc
      integer, intent(in)  :: Nt
      integer, intent(out) :: inform
!     Local variables
      integer :: I,J
      inform = 0
      mInitIntegrandCalled = .FALSE.
!     Find the size information
!~~~~~~~~~~~~~~~~~~~~~~~~~~
      mNt   = Nt
      mNc   = SIZE( Xc, dim = 1 )
      mNx   = MAX( SIZE( Xc, dim = 2), 1 )
      mNtdc = SIZE( BIG, dim = 1 )
      ! make sure it does not exceed Ntdc-Nc
      IF (mNt+mNc.GT.mNtdc) mNt = mNtdc - mNc
      mNd  = mNtdc-mNt-mNc
      mNtd = mNt+mNd
      IF (mNd < 0) THEN
!         PRINT *,'RIND Nt,Nd,Nc,Ntdc=',Nt,Nd,Nc,Ntdc
         ! Size information inconsistent
         inform = 3
         return
      ENDIF

 !     PRINT *,'Nt Nd Nc Ntd Ntdc,',Nt, Nd, Nc, Ntd, Ntdc

! ALLOCATION
!~~~~~~~~~~~~
      IF (mNd>0) THEN
         ALLOCATE(mXd(mNd),mCmXd(mNd),mCDIXd(mNd))
         mCmXd(:)  = gZERO
         mCDIXd(:) = gZERO
         mxd(:)    = gZERO
      END IF
      ALLOCATE(mBIG(mNtdc,mNtdc),mCm(mNtdc),mY(mNtd))
      ALLOCATE(mIndex1(mNtdc),mA(mNtd),mB(mNtd),mINFI(mNtd),mXc(mNc))
      ALLOCATE(mCDI(mNtd),mXedni(mNtdc),mIndex2(mNtdc))

!     Initialization
!~~~~~~~~~~~~~~~~~~~~~
!     Copy upper triangular of input matrix, only.
      do i = 1,mNtdc
         mBIG(1:i,i)    = BIG(1:i,i)
      end do

      mIndex2 = (/(J,J=1,mNtdc)/)

!      CALL mexprintf('BIG Before CovsrtXc'//CHAR(10))
!      CALL ECHO(BIG)
!     sort BIG by decreasing cond. variance for Xc
      CALL CVSRTXC(mNt,mNd,mBIG,mIndex2,INFORM)
!      CALL mexprintf('BIG after CovsrtXc'//CHAR(10))
!      CALL ECHO(BIG)

      IF (INFORM.GT.0) return ! degenerate case exit VALS=0 for all
                                ! (should perhaps return NaN instead??)


      DO I=mNtdc,1,-1
         J = mIndex2(I)            ! covariance matrix according to index2
         mXedni(J) = I
      END DO

      IF (mNx>1) THEN
         ALLOCATE(mBIG2(mNtdc,mNtdc))
         do i = 1,mNtdc
            mBIG2(1:i,i) = mBIG(1:i,i) !Copy input matrix
         end do
      ENDIF
      return
      end subroutine preInit
      subroutine initIntegrand(ix,Xc,Ex,indI,Blo,Bup,INFIN,
     &     fxc,value,abserr,NDIM,inform)
      integer, intent(in) :: ix ! integrand number
      double precision, dimension(:),intent(in) :: Ex
      double precision, dimension(:,:), intent(in) :: Xc,Blo,Bup
      integer, dimension(:), intent(in) :: indI,INFIN
      double precision, intent(out) :: fxc,value,abserr
      integer, intent(out) :: NDIM, inform
!     Locals
      DOUBLE PRECISION   :: SQ0,xx,quant
      integer :: I,J
      inform = 0
      NDIM   = 0
      VALUE   = gZERO
      fxc     = gONE
      abserr  = mSmall

      IF (mInitIntegrandCalled)  then
         do i = 1,mNtdc
            mBIG(1:i,i)    = mBIG2(1:i,i) !Copy input matrix
         end do
      else
         mInitIntegrandCalled = .TRUE.
      endif

                                ! Set the original means of the variables
      mCm(:)  = Ex(mIndex2(1:mNtdc)) !   Cm(1:Ntdc)  =Ex (index1(1:Ntdc))
      IF (mNc>0) THEN
         mXc(:) = Xc(:,ix)
                                !mXc(1:Nc)    = Xc(1:Nc,ix)
         QUANT = DBLE(mNc)*LOG(gSQTWPI1)
         I = mNtdc
         DO J = 1, mNc
!     Iterative conditioning on the last Nc variables
            SQ0 = mBIG(I,I)     ! SQRT(Var(X(i)|X(i+1),X(i+2),...,X(Ntdc)))
            xx = (mXc(mIndex2(I) - mNtd) - mCm(I))/SQ0
                                !Trick to calculate
                                !fxc = fxc*SQTWPI1*EXP(-0.5*(XX**2))/SQ0
            QUANT = QUANT - gHALF*xx*xx  - LOG(SQ0)
                                ! conditional mean (expectation)
                                ! E(X(1:i-1)|X(i),X(i+1),...,X(Ntdc))
            mCm(1:I-1) = mCm(1:I-1) + xx*mBIG(1:I-1,I)
            I = I-1
         ENDDO
                                ! Calculating the
                                ! fxc probability density for i=Ntdc-J+1,
                                ! fXc=f(X(i)|X(i+1),X(i+2)...X(Ntdc))*
                                !     f(X(i+1)|X(i+2)...X(Ntdc))*..*f(X(Ntdc))
         fxc = EXP(QUANT+mXcScale)

                                ! if fxc small:  don't bother
                                ! calculating it, goto end
         IF (fxc  < mFxcEpss) then
            abserr = gONE
            inform = 1
            return
         endif
      END IF
!     Set integration limits mA,mB and mINFI
!     NOTE: mA and mB are integration limits with mCm subtracted
      CALL setIntLimits(mXc,indI,Blo,Bup,INFIN,inform)
      if (inform>0) return
      mIndex1(:) = mIndex2(:)
      CALL COVSRT(.FALSE., mNt,mNd,mBIG,mCm,mA,mB,mINFI,
     &        mINDEX1,mINFIXt,mINFIXd,NDIM,mY,mCDI)

      CALL INITFUN(VALUE,abserr,INFORM)
!     IF INFORM>0 : degenerate case:
!     Integral can be calculated excactly, ie.
!     mean of deterministic variables outside the barriers,
!     or NDIM = 1
      return
      end subroutine initIntegrand
      subroutine cleanUp
!     Deallocate all work arrays and vectors
      IF (ALLOCATED(mXc))     DEALLOCATE(mXc)
      IF (ALLOCATED(mXd))     DEALLOCATE(mXd)
      IF (ALLOCATED(mCm))     DEALLOCATE(mCm)
      IF (ALLOCATED(mBIG2))   DEALLOCATE(mBIG2)
      IF (ALLOCATED(mBIG))    DEALLOCATE(mBIG)
      IF (ALLOCATED(mIndex2)) DEALLOCATE(mIndex2)
      IF (ALLOCATED(mIndex1)) DEALLOCATE(mIndex1)
      IF (ALLOCATED(mXedni))  DEALLOCATE(mXedni)
      IF (ALLOCATED(mA))      DEALLOCATE(mA)
      IF (ALLOCATED(mB))      DEALLOCATE(mB)
      IF (ALLOCATED(mY))      DEALLOCATE(mY)
      IF (ALLOCATED(mCDI))    DEALLOCATE(mCDI)
      IF (ALLOCATED(mCDIXd))  DEALLOCATE(mCDIXd)
      IF (ALLOCATED(mCmXd))   DEALLOCATE(mCmXd)
      IF (ALLOCATED(mINFI))   DEALLOCATE(mINFI)
      end subroutine cleanUp
      function integrandBound(I0,N,Y,FINY) result (bound1)
      use FIMOD
      integer, intent(in) :: I0,N,FINY
      double precision, intent(in) :: Y
      double precision :: bound1
! locals
      integer :: I,IK,FINA, FINB
      double precision :: AI,BI,D1,E1
      double precision :: TMP
!     Computes the upper bound for the intgrand
      bound1 = gzero
      if (FINY<1) return
      FINA = 0
      FINB = 0
      IK = 2
      DO I = I0, N
             ! E(Y(I) | Y(1))/STD(Y(IK)|Y(1))
         TMP = mBIG(IK-1,I)*Y
         IF (mINFI(I) > -1) then
!     May have infinite int. Limits if Nd>0
            IF ( mINFI(I) .NE. 0 ) THEN
               IF ( FINA .EQ. 1 ) THEN
                  AI = MAX( AI, mA(I) - tmp )
               ELSE
                  AI   = mA(I) - tmp
                  FINA = 1
               END IF
            END IF
            IF ( mINFI(I) .NE. 1 ) THEN
               IF ( FINB .EQ. 1 ) THEN
                  BI = MIN( BI, mB(I) - tmp)
               ELSE
                  BI   = mB(I) - tmp
                  FINB = 1
               END IF
            END IF
         endif

         IF (I.EQ.N.OR.mBIG(IK+1,I+1)>gZERO) THEN
            CALL MVNLMS( AI, BI,2*FINA+FINB-1, D1, E1 )
            IF (D1<E1) bound1 = E1-D1
            return
         ENDIF
      ENDDO
      RETURN
      end function integrandBound
      SUBROUTINE INITFUN(VALUE,abserr,INFORM)
      USE JACOBMOD
      use SWAPMOD
      USE FIMOD
!      USE GLOBALDATA, ONLY: NIT,EPS2,EPS,xCutOff,NC1C2,ABSEPS
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(OUT) :: VALUE,abserr
      INTEGER, INTENT(out) :: INFORM
! local variables:
      INTEGER ::  N,NdleftO
      INTEGER :: I, J,  FINA, FINB
      DOUBLE PRECISION :: AI, BI, D0,E0,R12, R13, R23
      DOUBLE PRECISION :: xCut , upError,loError,maxTruncError
      LOGICAL :: useC1C2,isXd

!
!     Integrand subroutine
!
!INITFUN initialize the Multivariate Normal integrand function
! COF  - conditional sorted ChOlesky Factor of the covariance matrix (IN)
! CDI  - Cholesky DIagonal elements used to calculate the mean
! Cm   - conditional mean of Xd and Xt given Xc, E(Xd,Xt|Xc)
! xd   - variables to the jacobian variable, need no initialization size Nd
! xc   - conditional variables (IN)
! INDEX1 - if INDEX1(I)>Nt then variable no. I is one of the Xd
!          variables otherwise it is one of Xt.

      !PRINT *,'Mvnfun,ndim',Ndim
      INFORM = 0
      VALUE  = gZERO
      abserr = max(mCovEps , 6.0d0*mSmall)
      mVAL0   = gONE


      mNdleftN0 = mNd               ! Counter for number of Xd variables left

      mI0  = 0
      FINA = 0
      FINB = 0
      N = mNt + mNd - mINFIXt - mINFIXd-1
      IF (mINFIXt+mINFIXd > 0) THEN
!     CHCKLIM Check if the conditional mean Cm = E(Xt,Xd|Xc) for the
!     deterministic variables are between the barriers, i.e.,
!     A=Hlo-Cm< 0 <B=Hup-Cm
!     INFIN  INTEGER, array of integration limits flags:
!            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!            if INFIN(I) = 0, Ith limits are (-infinity, B(I)];
!            if INFIN(I) = 1, Ith limits are [A(I), infinity);
!            if INFIN(I) = 2, Ith limits are [A(I), B(I)].
         I = N+1
         DO J=1, mINFIXt + mINFIXd
            I = I + 1
            IF (mINFI(I)>-1) THEN
               IF ((mINFI(I).NE.0).AND.(mAbsEps  < mA(I))) GOTO 200
               IF ((mINFI(I).NE.1).AND.(mB(I) < -mAbsEps )) GOTO 200
            ENDIF
         ENDDO

         IF (mINFIXd>0) THEN
            ! Redundant variables of Xd: replace Xd with the mean
            I = mNt + mNd !-INFIS
            J = mNdleftN0-mINFIXd

            DO WHILE (mNdleftN0>J)
               isXd = (mNt < mIndex1(I))
               IF (isXd) THEN
                  mXd (mNdleftN0) =  mCm (I)
                  mNdleftN0 = mNdleftN0-1
               END IF
               I = I-1
            ENDDO
         ENDIF
         IF (N+1 < 1) THEN
!     Degenerate case, No relevant variables left to integrate
!     Print *,'rind ndim1',Ndim1
            IF (mNd>0) THEN
               VALUE = jacob (mXd,mXc) ! jacobian of xd,xc
            ELSE
               VALUE = gONE
            END IF
            GOTO 200
         ENDIF
      ENDIF
      IF (mNIT<=100) THEN
         xCut = mXcutOff

         J = 1
         DO I = 2, N+1
            IF (mBIG(J+1,I)>gZERO) THEN
               J = J + 1
            ELSE
               ! Add xCut std to deterministic variables to get an upper
               ! bound for integral
              mA(I) =  mA(I) - xCut * mBIG(I,J)
              mB(I) =  mB(I) + xCut * mBIG(I,J)
            ENDIF
         END DO
      ELSE
         xCut = gZERO
      ENDIF

      NdleftO = mNdleftN0
      useC1C2 = (1<=mNc1c2)
      DO I = 1, N+1
         IF (mINFI(I) > -1) then
!     May have infinite int. Limits if Nd>0
            IF ( mINFI(I) .NE. 0 ) THEN
               IF ( FINA .EQ. 1 ) THEN
                  AI = MAX( AI, mA(I) )
               ELSE
                  AI   = mA(I)
                  FINA = 1
               END IF
            END IF
            IF ( mINFI(I) .NE. 1 ) THEN
               IF ( FINB .EQ. 1 ) THEN
                  BI = MIN( BI, mB(I) )
               ELSE
                  BI   = mB(I)
                  FINB = 1
               END IF
            END IF
         endif
         isXd = (mINDEX1(I)>mNt)
         IF (isXd) THEN         ! Save the mean for Xd
            mCmXd(mNdleftN0)  = mCm(I)
            mCDIXd(mNdleftN0) = mCDI(I)
            mNdleftN0        = mNdleftN0-1
         END IF

         IF (I.EQ.N+1.OR.mBIG(2,I+1)>gZERO) THEN
            IF (useC1C2.AND.I<N) THEN
               mY(:) = gZERO


               CALL MVNLMS( AI, BI,2*FINA+FINB-1, D0, E0 )
               IF (D0>=E0) GOTO 200

               CALL C1C2(I+1,N+1,1,mA,mB,mINFI,mY,mBIG,AI,BI,FINA,FINB)
               CALL MVNLMS( AI, BI,2*FINA+FINB-1, mD1, mE1 )
               IF (mD1>=mE1) GOTO 200
               maxTruncError = FI(-ABS(mXcutOff))*dble(mNc1c2)
               upError = abs(E0-mE1)
               loError = abs(D0-mD1)
               if (upError>mSmall) then
                  upError = upError*integrandBound(I+1,N+1,BI,FINB)
               endif
               if (loError>mSmall) then
                  loError = loError*integrandBound(I+1,N+1,AI,FINA)
               endif
               abserr  = abserr + min(upError + loError,maxTruncError)
               !CALL printvar(log10(loError+upError+msmall),'lo+up-err')
            ELSE
               CALL MVNLMS( AI, BI,2*FINA+FINB-1, mD1, mE1 )
               IF (mD1>=mE1) GOTO 200
            ENDIF
            !CALL MVNLMS( AI, BI,2*FINA+FINB-1, mD1, mE1 )
            !IF (mD1>=mE1) GOTO 200
            IF ( NdleftO<=0) THEN
               IF (mNd>0) mVAL0 = JACOB(mXd,mXc)
               SELECT CASE (I-N)
               CASE (1)   !IF (I.EQ.N+1) THEN
                  VALUE  = (mE1-mD1)*mVAL0
                  abserr = abserr*mVAL0
                  GO TO 200
               CASE (0)     !ELSEIF (I.EQ.N) THEN
                                !D1=1/sqrt(1-rho^2)=1/STD(X(I+1)|X(1))
                  mD1 = SQRT( gONE + mBIG(1,I+1)*mBIG(1,I+1) )
                  mINFI(2) = mINFI(I+1)
                  mA(1) = AI
                  mB(1) = BI
                  mINFI(1) = 2*FINA+FINB-1
                  IF ( mINFI(2) .NE. 0 ) mA(2) = mA(I+1)/mD1
                  IF ( mINFI(2) .NE. 1 ) mB(2) = mB(I+1)/mD1
                  VALUE = BVNMVN( mA, mB,mINFI,mBIG(1,I+1)/mD1 )*mVAL0
                  abserr = (abserr+1.0d-14)*mVAL0
                  GO TO 200
               CASE ( -1 )  !ELSEIF (I.EQ.N-1) THEN
                  IF (.FALSE.) THEN
! TODO :this needs further checking! (it should work though)
                  !1/D1= sqrt(1-r12^2) = STD(X(I+1)|X(1))
                  !1/E1=  STD(X(I+2)|X(1)X(I+1))
                  !D1  = BIG(I+1,1)
                  !E1  = BIG(I+2,2)

                  mD1 = gONE/SQRT( gONE + mBIG(1,I+1)*mBIG(1,I+1) )
                  R12 = mBIG( 1, I+1 ) * mD1
                  if (mBIG(3,I+2)>gZERO) then
                     mE1 = gONE/SQRT( gONE + mBIG(1,I+2)*mBIG(1,I+2) +
     &                    mBIG(2,I+2)*mBIG(2,I+2) )
                     R13 = mBIG( 1, I+2 ) * mE1
                     R23 = mBIG( 2, I+2 ) * (mE1 * mD1) + R12 * R13
                  else
                     mE1  = mCDI(I+2)
                     R13 = mBIG( 1, I+2 ) * mE1
                     R23 = mE1*mD1 + R12 * R13
                     IF ((mE1  <  gZERO).AND. mINFI(I+2)>-1) THEN
                        CALL SWAP(mA(I+2),mB(I+2))
                        IF (mINFI(I+2).NE. 2) mINFI(I+2) = 1-mINFI(I+2)
                     END IF
                     !R23 = BIG( 2, I+2 ) * (E1 * D1) + R12 * R13
                  endif
                  mINFI(2) = mINFI(I+1)
                  mINFI(3) = mINFI(I+2)
                  mA(1) = AI
                  mB(1) = BI
                  mINFI(1) = 2*FINA+FINB-1
                  IF ( mINFI(2) .NE. 0 ) mA(2) = mA(I+1) * mD1
                  IF ( mINFI(2) .NE. 1 ) mB(2) = mB(I+1) * mD1
                  IF ( mINFI(3) .NE. 0 ) mA(3) = mA(I+2) * mE1
                  IF ( mINFI(3) .NE. 1 ) mB(3) = mB(I+2) * mE1
                  if(.false.) then
                     CALL PRINTVECD((/R12, R13, R23 /),'R12 = ')
                     CALL PRINTVECD((/mD1, mE1 /),'D1 = ')
                     CALL PRINTVECD(mBIG(1,1:3),'BIG(1,1:3) = ')
                     CALL PRINTVECD(mBIG(2,2:3),'BIG(2,2:3) = ')
                     CALL PRINTVECD(mBIG(1:3,1),'BIG(1:3,1) = ')
                     CALL PRINTVECD(mBIG(2:3,2),'BIG(2:3,2) = ')
                     CALL PRINTVECD(mA(1:I+2),'A = ')
                     CALL PRINTVECD(mB(1:I+2),'B = ')
                     CALL PRINTVECI(mINFI(1:I+2),'INFI = ')
                     CALL PRINTVECI(mINDEX1(1:I+2),'index1 = ')
                  endif
                  VALUE = TVNMVN( mA, mB,mINFI,
     &                 (/R12, R13, R23 /),1.0d-13) * mVAL0
                  ABSERR = (ABSERR + 1.0d-13)*mVAL0
                  GOTO 200
                  ENDIF
               END SELECT !ENDIF
            ENDIF
            ABSERR = mVAL0*ABSERR
            mVAL0 = mVAL0 * (mE1-mD1)
            mI0   = I
            RETURN
         ENDIF
      ENDDO
      RETURN
 200  INFORM = 1
      RETURN
      END SUBROUTINE INITFUN
!
!     Integrand subroutine
!
      FUNCTION MVNFUN( Ndim, W ) RESULT (VAL)
      USE JACOBMOD
      USE FIMOD
      IMPLICIT NONE
      INTEGER, INTENT (IN) :: Ndim
      DOUBLE PRECISION, DIMENSION(:), INTENT(in) :: W
      DOUBLE PRECISION :: VAL
! local variables:
      INTEGER ::  N,I, J, FINA, FINB
      INTEGER ::  NdleftN, NdleftO ,IK
      DOUBLE PRECISION :: TMP, AI, BI, DI, EI
      LOGICAL :: useC1C2, isXd
!MVNFUN Multivariate Normal integrand function
! where the integrand is transformed from an integral
! having integration limits A and B to an
! integral having  constant integration limits i.e.
!   B                                  1
! int jacob(xd,xc)*f(xd,xt)dxt dxd = int F2(W) dW
!  A                                  0
!
! W    - new transformed integration variables, valid range 0..1
!        The vector must have the length Ndim returned from Covsrt
! mBIG - conditional sorted ChOlesky Factor of the covariance matrix (IN)
! mCDI - Cholesky DIagonal elements used to calculate the mean
! mCm  - conditional mean of Xd and Xt given Xc, E(Xd,Xt|Xc)
! mXd  - variables to the jacobian variable, need no initialization size Nd
! mXc  - conditional variables (IN)
! mINDEX1 - if mINDEX1(I)>Nt then variable No. I is one of the Xd
!           variables otherwise it is one of Xt

      !PRINT *,'Mvnfun,ndim',Ndim

!      xCut = gZERO ! xCutOff

      N    = mNt+mNd-mINFIXt-mINFIXd-1
      IK   = 1                    ! Counter for Ndim
      FINA = 0
      FINB = 0

      NdleftN = mNdleftN0          ! Counter for number of Xd variables left
      VAL     = mVAL0
      NdleftO = mNd - mINFIXd
      mY(IK) = FIINV( mD1 + W(IK)*( mE1 - mD1 ) )
      useC1C2 = (IK+1.LE.mNc1c2)
      IF (useC1C2) THEN
         ! Calculate the conditional mean
         ! E(Y(I) | Y(1),...Y(I0))/STD(Y(I)|Y(1),,,,Y(I0))
         mY(mI0+1:N+1) = mBIG(IK, mI0+1:N+1)*mY(IK)
      ENDIF
      IF (NdleftO.GT.NdleftN ) THEN
         mXd(NdleftN+1:NdleftO) = mCmXd(NdleftN+1:NdleftO)+
     &        mY(IK) * mCDIXd(NdleftN+1:NdleftO)
      ENDIF
      NdleftO = NdleftN
      IK = 2                    !=IK+1


      DO I = mI0+1, N+1
         IF (useC1C2) THEN
             TMP = mY(I)
          ELSE
            TMP = 0.d0
            DO J = 1, IK-1
               ! E(Y(I) | Y(1),...Y(IK-1))/STD(Y(IK)|Y(1),,,,Y(IK-1))
               TMP = TMP + mBIG(J,I)*mY(J)
            END DO
         ENDIF
         IF (mINFI(I) < 0) GO TO 100
            ! May have infinite int. Limits if Nd>0
         IF ( mINFI(I) .NE. 0 ) THEN
            IF ( FINA .EQ. 1 ) THEN
               AI = MAX( AI, mA(I) - TMP)
            ELSE
               AI = mA(I) - TMP
               FINA = 1
            END IF
            IF (FINB.EQ.1.AND.BI<=AI) GOTO 200
         END IF
         IF ( mINFI(I) .NE. 1 ) THEN
            IF ( FINB .EQ. 1 ) THEN
               BI = MIN( BI, mB(I) - TMP)
            ELSE
               BI = mB(I) - TMP
               FINB = 1
            END IF
            IF (FINA.EQ.1.AND.BI<=AI) GOTO 200
         END IF
 100     isXd = (mNt<mINDEX1(I))
         IF (isXd) THEN
!     Save the mean of xd and Covariance diagonal element
            ! Conditional mean E(Xi|X1,..X)
            mCmXd(NdleftN)  = mCm(I) + TMP * mCDI(I)
             ! Covariance diagonal
            mCDIXd(NdleftN) = mCDI(I)
            NdleftN        = NdleftN - 1
         END IF
         IF (I == N+1 .OR. mBIG(IK+1,I+1) > gZERO ) THEN
            IF (useC1C2) THEN
!     Note: for J =I+1:N+1:  Y(J) = conditional expectation, E(Yj|Y1,...Yk)
               CALL C1C2(I+1,N+1,IK,mA,mB,mINFI,mY,mBIG,AI,BI,FINA,FINB)
            ENDIF
            CALL MVNLMS( AI, BI, 2*FINA+FINB-1, DI, EI )
            IF ( DI >= EI ) GO TO 200
            VAL = VAL * ( EI - DI )

            IF ( I <= N .OR. (NdleftN < NdleftO)) THEN
               mY(IK) = FIINV( DI + W(IK)*( EI - DI ) )
               IF (NdleftN < NdleftO ) THEN
                  mXd(NdleftN+1:NdleftO) = mCmXd(NdleftN+1:NdleftO)+
     &                 mY(IK) * mCDIXd(NdleftN+1:NdleftO)
                  NdleftO = NdleftN
               ENDIF
               useC1C2 = (IK+1<=mNc1c2)
               IF (useC1C2) THEN

                  ! E(Y(J) | Y(1),...Y(I))/STD(Y(J)|Y(1),,,,Y(I))
                  mY(I+1:N+1) = mY(I+1:N+1) + mBIG(IK, I+1:N+1)*mY(IK)
               ENDIF
            ENDIF
            IK   = IK + 1
            FINA = 0
            FINB = 0
         END IF
      END DO
      IF (mNd>0) VAL = VAL * jacob(mXd,mXc)
      RETURN
 200  VAL = gZERO
      RETURN
      END FUNCTION MVNFUN


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!******************* RINDD - the main program *********************!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      SUBROUTINE RINDD(VALS,ERR,TERR,Big,Ex,Xc,Nt,
     &               indI,Blo,Bup,INFIN)
      USE RCRUDEMOD
      USE KRBVRCMOD
      USE ADAPTMOD
      USE KROBOVMOD
      USE DKBVRCMOD
      USE SSOBOLMOD
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(out):: VALS, ERR ,TERR
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: BIG
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: Xc
      DOUBLE PRECISION, DIMENSION(:), INTENT(in) :: Ex
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: Blo, Bup
      INTEGER,          DIMENSION(:), INTENT(in) :: indI,INFIN
      INTEGER,                          INTENT(in) :: Nt
!      DOUBLE PRECISION,                 INTENT(in) :: XcScale
! local variables
      INTEGER :: ix, INFORM, NDIM, MAXPTS, MINPTS
      DOUBLE PRECISION :: VALUE,fxc,absERR,absERR2
      double precision :: LABSEPS,LRELEPS


      VALS(:) = gZERO
      ERR(:)  = gONE
      TERR(:) = gONE

      call preInit(BIG,Xc,Nt,inform)
      IF (INFORM.GT.0) GOTO 110 ! degenerate case exit VALS=0 for all
                                ! (should perhaps return NaN instead??)

!     Now the loop over all different values of
!     variables Xc (the one one is conditioning on)
!     is started. The density f_{Xc}(xc(:,ix))
!     will be computed and denoted by  fxc.
      DO  ix = 1, mNx
         call initIntegrand(ix,Xc,Ex,indI,Blo,Bup,infin,
     &        fxc,value,abserr,NDIM,inform)


         IF (INFORM.GT.0) GO TO 100

         MAXPTS  = mMAXPTS
         MINPTS  = mMINPTS
         LABSEPS = max(mABSEPS-abserr,0.2D0*mABSEPS)       !*fxc
         LRELEPS = mRELEPS
         ABSERR2 = mSmall

         SELECT CASE (mMethod)
         CASE (:1)
            IF (NDIM < 9) THEN
               CALL SADAPT(NDIM,MAXPTS,MVNFUN,LABSEPS,
     &              LRELEPS,ABSERR2,VALUE,INFORM)
               VALUE = MAX(VALUE,gZERO)
            ELSE
               CALL KRBVRC(NDIM, MINPTS, MAXPTS, MVNFUN,LABSEPS,LRELEPS,
     &              ABSERR2, VALUE, INFORM )
            ENDIF
         CASE (2)
!        Call the subregion adaptive integration subroutine
            IF ( NDIM .GT. 19.) THEN
!     print *, 'Ndim too large for SADMVN => Calling KRBVRC'
               CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN, LABSEPS,
     &              LRELEPS, ABSERR2, VALUE, INFORM )
            ELSE
               CALL SADAPT(NDIM,MAXPTS,MVNFUN,LABSEPS,
     &              LRELEPS,ABSERR2,VALUE,INFORM)
               VALUE = MAX(VALUE,gZERO)
            ENDIF
         CASE (3)               !       Call the Lattice rule integration procedure
            CALL KRBVRC( NDIM, MINPTS, MAXPTS, MVNFUN, LABSEPS,
     &           LRELEPS, ABSERR2, VALUE, INFORM )
         CASE (4)               !       Call the Lattice rule
                                !       integration procedure
            CALL KROBOV( NDIM, MINPTS, MAXPTS, MVNFUN, LABSEPS,
     &           LRELEPS,ABSERR2, VALUE, INFORM )
         CASE (5)    ! Call Crude Monte Carlo integration procedure
            CALL RANMC( NDIM, MAXPTS, MVNFUN, LABSEPS,
     &           LRELEPS, ABSERR2, VALUE, INFORM )
         CASE (6)       !       Call the scrambled Sobol sequence rule integration procedure
           CALL SOBNIED( NDIM, MINPTS, MAXPTS, MVNFUN, LABSEPS, LRELEPS,
     &           ABSERR2, VALUE, INFORM )
         CASE (7:)
           CALL DKBVRC( NDIM, MINPTS, MAXPTS, MVNFUN, LABSEPS, LRELEPS,
     &           ABSERR2, VALUE, INFORM )
         END SELECT

!     IF (INFORM.gt.0) print *,'RIND, INFORM,error =',inform,error
 100     VALS(ix) = VALUE*fxc
         IF (SIZE(ERR, DIM = 1).EQ.mNx) ERR(ix)   = abserr2*fxc
         IF (SIZE(TERR, DIM = 1).EQ.mNx) TERR(ix) = abserr*fxc
      ENDDO                     !ix

 110  CONTINUE
      call cleanUp
      RETURN
      END SUBROUTINE RINDD

      SUBROUTINE setIntLimits(xc,indI,Blo,Bup,INFIN,inform)
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(in)  :: xc
      INTEGER,          DIMENSION(:  ), INTENT(in)  :: indI,INFIN
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in)  :: Blo,Bup
      integer, intent(out) :: inform
!Local variables
      INTEGER   :: I, J, K, L,Mb1,Nb,NI,Nc
      DOUBLE PRECISION :: xCut, SQ0
!this procedure set mA,mB and mInfi according to Blo/Bup and INFIN
!
!     INFIN  INTEGER, array of integration limits flags:
!            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!            if INFIN(I) = 0, Ith limits are (-infinity, mB(I)];
!            if INFIN(I) = 1, Ith limits are [mA(I), infinity);
!            if INFIN(I) = 2, Ith limits are [mA(I), mB(I)].
! Note on member variables:
! mXedni     = indices to the variables new place after cvsrtXc.  Size Ntdc
! mCm        = E(Xt,Xd|Xc), i.e., conditional mean given Xc
! mBIG(:,1:Ntd) = Cov(Xt,Xd|Xc)

      xCut = ABS(mInfinity)
      Mb1 = size(Blo,DIM=1)-1
      Nb = size(Blo,DIM=2)
      NI = size(indI,DIM=1)
      Nc = size(xc,DIM=1)
      if (Mb1>Nc .or. Nb.NE.NI-1) then
!     size of variables inconsistent
         inform = 4
         return
      endif

!      IF (Mb.GT.Nc+1) print *,'barrier: Mb,Nc =',Mb,Nc
!      IF (Nb.NE.NI-1) print *,'barrier: Nb,NI =',Nb,NI
      DO J = 2, NI
         DO I = indI (J - 1) + 1 , indI (J)
            L        = mXedni(I)
            mINFI(L) = INFIN(J-1)
            SQ0      = SQRT(mBIG(L,L))
            mA(L)    = -xCut*SQ0
            mB(L)    =  xCut*SQ0
            IF (mINFI(L).GE.0) THEN
               IF  (mINFI(L).NE.0) THEN
                  mA(L) = Blo (1, J - 1)-mCm(L)
                  DO K = 1, Mb1
                     mA(L) = mA(L)+Blo(K+1,J-1)*xc(K)
                  ENDDO         ! K
                  ! This can only be done if
                  if (mA(L)< -xCut*SQ0) mINFI(L) = mINFI(L)-2
               ENDIF
               IF  (mINFI(L).NE.1) THEN
                  mB(L) = Bup (1, J - 1)-mCm(L)
                  DO K = 1, Mb1
                     mB(L) = mB(L)+Bup(K+1,J-1)*xc(K)
                  ENDDO
                  if (xCut*SQ0<mB(L)) mINFI(L) = mINFI(L)-1
               ENDIF            !
            ENDIF
         ENDDO                  ! I
      ENDDO                     ! J
!     print * ,'barrier hup:',size(Hup),Hup(xedni(1:indI(NI)))
!     print * ,'barrier hlo:',size(Hlo),Hlo(xedni(1:indI(NI)))
      RETURN
      END SUBROUTINE setIntLimits


      FUNCTION GETTMEAN(A,B,INFJ,PRB) RESULT (MEAN1)
      USE GLOBALCONST
      IMPLICIT NONE
!     GETTMEAN Returns the expected mean, E(I(a<x<b)*X)
      DOUBLE PRECISION, INTENT(IN) :: A,B,PRB
      INTEGER , INTENT(IN) :: INFJ
      DOUBLE PRECISION :: MEAN1
      DOUBLE PRECISION :: YL,YU
!     DOUBLE PRECISION, PARAMETER:: ZERO = 0.0D0, HALF = 0.5D0

      IF ( PRB .GT. gZERO) THEN
         YL = gZERO
         YU = gZERO
         IF (INFJ.GE.0) THEN
            IF (INFJ .NE. 0) YL =-EXP(-gHALF*(A*A))*gSQTWPI1
            IF (INFJ .NE. 1) YU =-EXP(-gHALF*(B*B))*gSQTWPI1
         ENDIF
         MEAN1 = ( YU - YL )/PRB
      ELSE
         SELECT CASE (INFJ)
         CASE (:-1)
            MEAN1 = gZERO
         CASE (0)
            MEAN1 = B
         CASE (1)
            MEAN1 = A
         CASE (2:)
            MEAN1 = ( A + B ) * gHALF
         END SELECT
      END IF
      RETURN
      END FUNCTION
      SUBROUTINE ADJLIMITS(A,B, infi)
!      USE GLOBALDATA, ONLY : xCutOff
      IMPLICIT NONE
!     Adjust INFI when integration limits A and/or B is too far out in the tail
      DOUBLE PRECISION, INTENT(IN)     :: A,B
      INTEGER,          INTENT(IN OUT) :: infi
!      DOUBLE PRECISION, PARAMETER :: xCutOff = 8.D0
      IF (infi>-1) THEN
         IF (infi.NE.0)THEN
            IF (A  <  -mXcutOff) THEN
               infi = infi-2
!               CALL mexprintf('ADJ A')
            ENDIF
         ENDIF
         IF (infi.NE.1) THEN
            IF (mXCutOff  <  B) THEN
               infi = infi-1
!               CALL mexprintf('ADJ B')
            ENDIF
         END IF
      END IF
      RETURN
      END SUBROUTINE ADJLIMITS
      SUBROUTINE C1C2(I0,I1,IK,A,B,INFIN, Cm, BIG, AJ, BJ, FINA,FINB)
! The regression equation for the conditional distr. of Y given X=x
! is equal  to the conditional expectation of Y given X=x, i.e.,
!
!       E(Y|X=x) = E(Y) + Cov(Y,X)/Var(X)[x-E(X)]
!
!  Let
!     x1 = (x-E(X))/SQRT(Var(X)) be zero mean,
!     C1< x1 <C2,
!     B1(I) = COV(Y(I),X)/SQRT(Var(X)).
!  Then the process  Y(I) with mean Cm(I) can be written as
!
!       y(I) = Cm(I) + x1*B1(I) + Delta(I) for  I=I0,...,I1.
!
!  where SQ(I) = sqrt(Var(Y|X)) is the standard deviation of Delta(I).
!
!  Since we are truncating all Gaussian  variables to
!  the interval [-C,C], then if for any I
!
!  a) Cm(I)+x1*B1(I)-C*SQ(I)>B(I)  or
!
!  b) Cm(I)+x1*B1(I)+C*SQ(I)<A(I)  then
!
!  the (XIND|Xn=xn) = 0 !!!!!!!!!
!
!  Consequently, for increasing the accuracy (by excluding possible
!  discontinuouities) we shall exclude such values for which (XIND|X1=x1) = 0.
!  Hence we assume that if Aj<x<Bj any of the previous conditions are
!  satisfied
!
!  OBSERVE!!, Aj, Bj has to be set to (the normalized) lower and upper bounds
!  of possible values for x1,respectively, i.e.,
!           Aj=max((A-E(X))/SQRT(Var(X)),-C), Bj=min((B-E(X))/SQRT(Var(X)),C)
!  before calling C1C2 subroutine.
!
      USE GLOBALCONST
!      USE GLOBALDATA, ONLY : EPS2,EPS ,xCutOff
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:), INTENT(in) :: Cm, A,B !, B1, SQ
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(in) :: BIG
      INTEGER,          DIMENSION(:), INTENT(in) :: INFIN
      DOUBLE PRECISION,            INTENT(inout) :: AJ,BJ
      INTEGER,                     INTENT(inout) :: FINA, FINB
      INTEGER, INTENT(IN) :: I0, I1, IK

!     Local variables
      DOUBLE PRECISION   :: xCut
!      DOUBLE PRECISION, PARAMETER :: TOL = 1.0D-16
      DOUBLE PRECISION :: CSQ, BdSQ0, LTOL
      INTEGER :: INFI, I

      xCut = MIN(ABS(mXcutOff),mInfinity)
      LTOL = mSmall ! EPS2
!      AJ = MAX(AJ,-xCut)
!      BJ = MIN(BJ,xCut)
!      IF (AJ.GE.BJ) GO TO 112
!      CALL PRINTVAR(AJ,TXT='BC1C2: AJ')
!      CALL PRINTVAR(BJ,TXT='BC1C2: BJ')

      IF (I1 < I0)  RETURN       !Not able to change integration limits
      DO I = I0,I1
!     C = xCutOff
         INFI = INFIN(I)
         IF (INFI>-1) THEN
            !BdSQ0 = B1(I)
            !CSQ   = xCut * SQ(I)
            BdSQ0  = BIG(IK,I)
            CSQ    = xCut * BIG(I,IK)
            IF (BdSQ0 > LTOL) THEN
               IF ( INFI .NE. 0 ) THEN
                  IF (FINA.EQ.1) THEN
                     AJ = MAX(AJ,(A(I) - Cm(I) - CSQ)/BdSQ0)
                  ELSE
                     AJ = (A(I) - Cm(I) - CSQ)/BdSQ0
                     FINA = 1
                  ENDIF
                  IF (FINB.GT.0) AJ = MIN(AJ,BJ)
               END IF
               IF ( INFI .NE. 1 ) THEN
                  IF (FINB.EQ.1) THEN
                     BJ = MIN(BJ,(B(I) - Cm(I) + CSQ)/BdSQ0)
                  ELSE
                     BJ = (B(I) - Cm(I) + CSQ)/BdSQ0
                     FINB = 1
                  ENDIF
                  IF (FINA.GT.0) BJ = MAX(AJ,BJ)
               END IF
             ELSEIF (BdSQ0  <  -LTOL) THEN
                IF ( INFI .NE. 0 ) THEN
                  IF (FINB.EQ.1) THEN
                     BJ = MIN(BJ,(A(I) - Cm(I) - CSQ)/BdSQ0)
                  ELSE
                     BJ = (A(I) - Cm(I) - CSQ)/BdSQ0
                     FINB = 1
                  ENDIF
                  IF (FINA.GT.0) BJ = MAX(AJ,BJ)
               END IF
               IF ( INFI .NE. 1 ) THEN
                  IF (FINA.EQ.1) THEN
                     AJ = MAX(AJ,(B(I) - Cm(I) + CSQ)/BdSQ0)
                  ELSE
                     AJ = (B(I) - Cm(I) + CSQ)/BdSQ0
                     FINA = 1
                  ENDIF
                  IF (FINB.GT.0) AJ = MIN(AJ,BJ)
               END IF
            END IF
         ENDIF
      END DO
!      IF (FINA>0 .AND. FINB>0) THEN
!         IF (AJ<BJ) THEN
!            IF (AJ   <= -xCut) FINA = 0
!            IF (xCut <= BJ   ) FINB = 0
!         ENDIF
!      ENDIF
!      CALL PRINTVAR(AJ,TXT='AC1C2: AJ')
!      CALL PRINTVAR(BJ,TXT='AC1C2: BJ')
      RETURN
      END SUBROUTINE C1C2
      SUBROUTINE CVSRTXC (Nt,Nd,R,index1,INFORM)
!      USE GLOBALDATA, ONLY :  XCEPS2
!      USE GLOBALCONST
      IMPLICIT NONE
      INTEGER, INTENT(in) :: Nt,Nd
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: index1
      INTEGER, INTENT(out) :: INFORM
! local variables
      DOUBLE PRECISION, DIMENSION(:), allocatable   :: SQ
      INTEGER,          DIMENSION(1)                :: m
      INTEGER :: M1,K,I,J,Ntdc,Ntd,Nc, LO
      DOUBLE PRECISION :: LTOL, maxSQ
!     if any Var(Xc(j)|Xc(1),...,Xc(j-1)) <= XCEPS2 then return NAN
      double precision :: XCEPS2
!CVSRTXC calculate the conditional covariance matrix of Xt and Xd given Xc
! as well as the cholesky factorization for the Xc variable(s)
! The Xc variables are sorted by the largest conditional covariance
!
! R         = In : Cov(X) where X=[Xt Xd Xc] is stochastic vector
!             Out: sorted Conditional Covar. matrix, i.e.,
!                    [ Cov([Xt,Xd] | Xc)    Shape N X N  (N=Ntdc=Nt+Nd+Nc)
! index1    = In/Out : permutation vector giving the indices to the variables
!             original place.   Size  Ntdc
! INFORM    = Out, Returns
!             0 If Normal termination.
!             1 If R is degenerate, i.e., Cov(Xc) is singular.
!
! R=Cov([Xt,Xd,Xc]) is a covariance matrix of the stochastic
! vector X=[Xt Xd Xc] where the variables Xt, Xd and Xc have the size
! Nt, Nd and Nc, respectively.
! Xc are the conditional variables.
! Xd and Xt are the variables to integrate.
!(Xd,Xt = variables in the jacobian and indicator respectively)
!
! Note: CVSRTXC only works on the upper triangular part of R

      INFORM = 0
      Ntdc = size(R,DIM=1)
      Ntd  = Nt   + Nd
      Nc   = Ntdc - Ntd

      IF (Nc < 1) RETURN



      ALLOCATE(SQ(1:Ntdc))
      maxSQ = gZERO
      DO I = 1, Ntdc
         SQ(I)  = R(I,I)
         if (SQ(I)>maxSQ) maxSQ = SQ(I)
      ENDDO

      XCEPS2 = Ntdc*mSmall*maxSQ
      mXcEps2 = XCEPS2
      LTOL   = mSmall
      LO = 1
      K = Ntdc
      DO I = 1, Nc             ! Condsort Xc
         m  = K+1-MAXLOC(SQ(K:Ntd+1:-1))
         M1 = m(1)
         IF (SQ(m1)<=XCEPS2) THEN
!     PRINT *,'CVSRTXC: Degenerate case of Xc(Nc-J+1) for J=',ix
            !CALL mexprintf('CVSRTXC: Degenerate case of Xc(Nc-J+1)')
            INFORM = 1
            GOTO 200   ! RETURN    !degenerate case
         ENDIF
         IF (M1.NE.K) THEN
            ! Symmetric row column permuations
            ! Swap row and columns, but only upper triangular part
            CALL RCSWAP( M1, K, Ntdc,Ntd, R,INDEX1,SQ)
         END IF
         R(K,K) = SQRT(SQ(K))
         IF (K .EQ. LO) GOTO 200
         R(LO:K-1,K) = R(LO:K-1,K)/R(K,K)
! Cov(Xi,Xj|Xk,Xk+1,..,Xn)  = ....
!         Cov(Xi,Xj|Xk+1,..,Xn) - Cov(Xi,Xk|Xk+1,..Xn)*Cov(Xj,Xk|Xk+1,..Xn)
         DO J = LO,K-1
                                ! Var(Xj | Xk,Xk+1,...,Xn)
            SQ(J)  =  R(J,J) - R(J,K)*R(J,K)
            IF (SQ(J)<=LTOL.AND.J<=Ntd) THEN
               IF (LO < J) THEN
                  CALL RCSWAP(LO, J, Ntdc,Ntd, R,INDEX1,SQ)
               ENDIF
               R(LO,LO:K-1) = gZERO
               IF (SQ(LO) < -10.0D0*SQRT(LTOL)) THEN
                  ! inform = 2
                  !R(LO,K) = gZERO
                 ! CALL mexprintf('Negative definit BIG!'//CHAR(10))
               ENDIF
               SQ(LO) = gZERO
               LO = LO + 1
            ELSE
               R(J,J) = SQ(J)
               R(LO:J-1,J) = R(LO:J-1,J) - R(LO:J-1,K)*R(J,K)
            ENDIF
         END DO
         K = K - 1
      ENDDO
 200  DEALLOCATE(SQ)
      RETURN
      END SUBROUTINE CVSRTXC

      SUBROUTINE  RCSCALE(chkLim,K,K0,N1,N,K1,INFIS,CDI,Cm,
     &     R,A,B,INFI,INDEX1,Y)
      USE GLOBALCONST
      USE SWAPMOD
      IMPLICIT NONE
!RCSCALE:  Scale  covariance matrix and limits
!
!   CALL  RCSCALE( k, k0, N1, N,K1, CDI,Cm,R,A, B, INFIN,index1,Y)
!
!  chkLim  = TRUE  if check if variable K is redundant
!            FALSE
!    K     = index to variable which is deterministic,i.e.,
!            STD(Xk|X1,...Xr) = 0
!    N1    = Number of significant variables of [Xt,Xd]
!    N     = length(Xt)+length(Xd)
!    K1    = index to current variable we are conditioning on.
!   CDI    = Cholesky diagonal elements which contains either
!               CDI(J) = STD(Xj | X1,...,Xj-1,Xc) if Xj is stochastic given
!                        X1,...Xj, Xc
!              or
!               CDI(J) = COV(Xj,Xk | X1,..,Xk-1,Xc  )/STD(Xk | X1,..,Xk-1,Xc)
!               if Xj is determinstically determined given X1,..,Xk,Xc
!               for some k<j.
!   Cm     = conditional mean
!    R     = Matrix containing cholesky factor for
!            X = [Xt,Xd,Xc] in upper triangular part. Lower triangular
!            part contains conditional stdevs. size Ntdc X Ntdc
! INDEX1   = permutation index vector. (index to variables original place).
!    A,B   = lower and upper integration limit, respectively.
!    INFIN = if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!            if INFIN(I) = 0, Ith limits are (-infinity, B(I)];
!            if INFIN(I) = 1, Ith limits are [A(I), infinity);
!            if INFIN(I) = 2, Ith limits are [A(I), B(I)].
!    Y     = work vector
!
! NOTE: RCSWAP works only on the upper triangular part of C
! + check if variable k is redundant
!  If the conditional covariance matrix diagonal entry is zero,
!  permute limits and/or rows, if necessary.
      LOGICAL, INTENT(IN) :: chkLim
      INTEGER, INTENT(IN) :: K, K0,N
      INTEGER, INTENT(INOUT) :: N1,K1,INFIS
      DOUBLE PRECISION, DIMENSION(:),  INTENT(INOUT) :: CDI,A,B,Cm
      DOUBLE PRECISION, DIMENSION(:,:),INTENT(INOUT) :: R
      INTEGER,          DIMENSION(:),  INTENT(INOUT) :: INFI,INDEX1
      DOUBLE PRECISION, DIMENSION(:),OPTIONAL,INTENT(INOUT) :: Y
!Local variables
      DOUBLE PRECISION, PARAMETER :: LTOL = 1.0D-16
      double precision :: xCut
      DOUBLE PRECISION :: D,AK,BK,CVDIAG
      INTEGER :: KK,K00, KKold, I, J, Ntdc, INFK
      K00 = K0
      DO WHILE( (0 < K00).AND. (ABS(R(K00,K)).LE.LTOL) )
         R(K00,K) = gZERO
         K00      = K00 - 1
      ENDDO
      IF (K00.GT.0) THEN
      !  CDI(K) = COV(Xk Xj| X1,..,Xj-1,Xc  )/STD(Xj | X1,..,Xj-1,Xc)
         CDI(K) = R(K00,K)
         A(K)   = A(K)/CDI(K)
         B(K)   = B(K)/CDI(K)

         IF ((CDI(K)  <  gZERO).AND. INFI(K).GE. 0) THEN
            CALL SWAP(A(K),B(K))
            IF (INFI(K).NE. 2) INFI(K) = 1-INFI(K)
         END IF


                                !Scale conditional covariances
         R(1:K00,K) = R(1:K00,K)/CDI(K)
                                !Scale conditional standard dev.s used in regression eq.
         R(K,1:K00) = R(K,1:K00)/ABS(CDI(K))


         R(K00+1:K,K)   = gZERO
         !R(K,K00+1:K-1) = gZERO ! original
         R(K,K00:K-1) = gZERO    ! check this

         !
         if (chkLim.AND.K00>1) then
            ! Check if variable is redundant
            ! TODO:  this chklim-block does not work correctly yet
            xCut = mInfinity
            I = 1
            Ak = R(I,K)*xCut
            Bk = - (R(I,K))*xCut
            if (INFI(I)>=0) then
               if (INFI(I).ne.0) then
                  Ak = -(R(I,K))*MAX(A(I),-xCut)
               endif
               if (INFI(I).ne.1) then
                  Bk = - (R(I,K))*MIN(B(I),xCut)
               endif
            endif

            if (R(I,K)<gZERO) THEN
               CALL SWAP(Ak,Bk)
            endif
            !call printvar(infi(k),'infi(k)')
            !call printvar(A(k),'AK')
            !call printvar(B(k),'BK')
            !call printvar(Ak,'AK')
            !call printvar(Bk,'BK')
            INFK = INFI(K)
            Ak   = A(K)+Ak
            Bk   = B(K)+Bk
            D = gZERO
            DO I = 2, K00-1
               D = D + ABS(R(I,K))
            END DO
            CVDIAG = abs(R(k,k00))
            !call printvar(cvdiag,'cvdiag')
            Ak = (Ak + (D+cvdiag)*xCut)
            Bk = (Bk - (D+cvdiag)*xCut)
            !call printvar(Ak,'AK')
            !call printvar(Bk,'BK')
            ! If Ak<-xCut and xCut<Bk then variable Xk is redundant
            CALL ADJLIMITS(Ak,Bk,INFK)

! Should change this to check against A(k00) and B(k00)
            IF (INFK < 0) THEN
               !variable is redundnant
               !                     CALL mexPrintf('AdjLim'//CHAR(10))
               IF ( K < N1 ) THEN
                  CALL RCSWAP( K, N1, N1,N, R,INDEX1,Cm, A, B, INFI)

                  ! move conditional standarddeviations
                  R(K,1:K0) = R(N1,1:K0)
                  CDI(K)    = CDI(N1)

                  IF (PRESENT(Y)) THEN
                     Y(K) = Y(N1)
                  ENDIF
               ENDIF
               CDI(N1)    = gZERO
               R(1:N1,N1) = gZERO
               R(N1,1:N1) = gZERO

               INFIS = INFIS+1
               N1    = N1-1
              ! CALL printvar(index1(N1),'index1(n1)')
              ! CALL mexPrintf('RCSCALE: updated N1')
              ! CALL printvar(INFIS,'INFIS ')
               return
            END IF
         endif
         KKold = K
         KK = K-1
         DO I = K0, K00+1, -1
            DO WHILE ((I.LE.KK) .AND. ABS(R(I,KK)).GT.LTOL)
               DO J = 1,I       !K0
                  ! SWAP Covariance matrix
                  CALL SWAP(R(J,KK),R(J,KKold))
                  ! SWAP conditional standarddeviations
                  CALL SWAP(R(KK,J),R(KKold,J))
               END DO
               CALL SWAP(CDI(KK),CDI(KKold))
               CALL SWAP(Cm(KK),Cm(KKold))
               CALL SWAP(INDEX1(KK),INDEX1(KKold))
               CALL SWAP(A(KK),A(KKold))
               CALL SWAP(B(KK),B(KKold))
               CALL SWAP(INFI(KK),INFI(KKold))
               IF (PRESENT(Y)) THEN
                   CALL SWAP(Y(KK),Y(KKold))
               ENDIF
               Ntdc = SIZE(R,DIM=1)
               IF (N < Ntdc) THEN
                  ! SWAP Xc entries, i.e, Cov(Xt,Xc) and Cov(Xd,Xc)
                  DO J = N+1, Ntdc
                     CALL SWAP( R(KK,J), R(KKold,J) )
                  END DO
               ENDIF
               KKold = KK
               KK    = KK - 1
            ENDDO
         END DO
         IF (KK < K1) THEN
            K1 = K1 + 1
!            CALL mexPrintf('RCSCALE: updated K1'//CHAR(10))
         END IF
!         CALL PRINTVAR(K,TXT='K')
!         CALL PRINTVAR(KK,TXT='KK')
!         CALL PRINTVAR(K1,TXT='K1')
!         CALL PRINTVAR(K00,TXT='K00')
!         CALL PRINTVAR(K0,TXT='K0')
!         CALL PRINTCOF(N,A,B,INFI,R,INDEX1)
      ELSE
!     Remove variable if it is conditional independent of all other variables
!         CALL mexPrintf('RCSCALE ERROR*********************'//char(10))
!         call PRINTCOF(N,A,B,INFI,R,INDEX1)
!         CALL mexPrintf('RCSCALE ERROR*********************'//char(10))
      ENDIF
!      if (chkLim) then
!         call PRINTCOF(N,A,B,INFI,R,INDEX1)
!      endif
      END SUBROUTINE RCSCALE

      SUBROUTINE COVSRT(BCVSRT, Nt,Nd,R,Cm,A,B,INFI,INDEX1,
     &     INFIS,INFISD, NDIM, Y, CDI )
      USE FIMOD
      USE SWAPMOD
      USE GLOBALCONST
!      USE GLOBALDATA, ONLY : EPS2,NIT,xCutOff
      IMPLICIT NONE
!COVSRT  sort integration limits and determine Cholesky factor.
!
!     Nt, Nd = size info about Xt and Xd variables.
!     R      = Covariance/Cholesky factored matrix for [Xt,Xd,Xc] (in)
!              On input:
!              Note: Only upper triangular part is needed/used.
!               1a) the first upper triangular the Nt + Nd times Nt + Nd
!                   block contains COV([Xt,Xd]|Xc)
!                 (conditional covariance matrix for Xt and Xd given Xc)
!               2a) The upper triangular part of the Nt+Nd+Nc times Nc
!                   last block contains the cholesky matrix for Xc,
!                   i.e.,
!
!              On output:
!               1b) part 2a) mentioned above is unchanged, only necessary
!                 permutations according to INDEX1 is done.
!               2b) part 1a) mentioned above is changed to a special
!                  form of cholesky matrix: (N = Nt+Nd-INFIS-INFISD)
!   C = COVARIANCE
!   R(1,1) = 1
!   R(1,2:N) = [C(X1,X2)/STD(X1)/STD(X2|X1),..,C(X1,XN)/STD(X1)/STD(XN|XN-1,..,X1)]
!   R(2,2) = 1
!   R(2,3:N) =[C(X2,X3|X1)/STD(X2|X1)/STD(X3|X2,X1),..,C(X2,XN|X1)/STD(X2|X1)/STD(XN|XN-1,..,X1)]
!                  ....
!                  etc.
!               3b) The lower triangular part of R contains the
!               normalized conditional standard deviations (which is
!               used in the reqression approximation C1C2), i.e.,
!               R(2:N,1)  = [STD(X2|X1) STD(X3|X1),....,STD(XN|X1) ]/STD(X1)
!               R(3:N,2)  = [STD(X3|X1,X2),....,STD(XN|X1,X2) ]/STD(X2|X1)
!               .....
!               etc.
!     Cm     = Conditional mean given Xc
!     A,B    = lower and upper integration limits length Nt+Nd
!     INFIN  = INTEGER, array of integration limits flags:  length Nt+Nd   (in)
!             if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!             if INFIN(I) = 0, Ith limits are (-infinity, B(I)];
!             if INFIN(I) = 1, Ith limits are [A(I), infinity);
!             if INFIN(I) = 2, Ith limits are [A(I), B(I)].
!    INDEX1  = permutation index vector, i.e., giving the indices to the
!              variables original place.
!    INFIS   = Number of redundant variables of Xt
!    INFISD  = Number of redundant variables of Xd
!    NDIM    = Number of relevant dimensions to integrate. This is the
!             same as the rank of the submatrix of Cov([Xt,Xd]) minus
!             the INFIS variables of Xt and INFISD variables of Xd.
!    Y       = working array
!    CDI     = Cholesky diagonal elements which contains either
!               CDI(J) = STD(Xj | X1,...,Xj-1,Xc) if Xj is stochastic given
!                        X1,...Xj, Xc
!              or
!               CDI(J) = COV(Xj,Xk | X1,..,Xk-1,Xc  )/STD(Xk | X1,..,Xk-1,Xc)
!               if Xj is determinstically determined given X1,..,Xk,Xc
!               for some k<j.
!
!     Subroutine to sort integration limits and determine Cholesky
!     factor.
!
!     Note: COVSRT works only on the upper triangular part of R
      LOGICAL,                          INTENT(in)    :: BCVSRT
      INTEGER,                          INTENT(in)    :: Nt,Nd
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(inout) :: Cm,A,B
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: INFI
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: INDEX1
      INTEGER,                          INTENT(out) :: INFIS,INFISD,NDIM
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(out)   :: Y, CDI
      double precision :: covErr
!     Local variables
      INTEGER :: N,N1,I, J, K, L, JMIN, Ndleft
      INTEGER ::  K1, K0, Nullity,INFJ,FINA,FINB
      DOUBLE PRECISION :: SUMSQ, AJ, BJ, TMP,D, E,EPSL
      DOUBLE PRECISION :: AA, Ca, Pa, APJ, PRBJ,RMAX
      DOUBLE PRECISION :: CVDIAG, AMIN, BMIN, PRBMIN
      DOUBLE PRECISION :: LTOL,TOL, ZERO,xCut
      LOGICAL          :: isOK = .TRUE.
      LOGICAL          :: isXd = .FALSE.
      LOGICAL          :: isXt = .FALSE.
      LOGICAL          :: chkLim
!      PARAMETER ( SQTWPI = 2.506628274631001D0, )
      PARAMETER (ZERO = 0.D0, TOL = 1D-16)

      xCut = mInfinity
!     xCut = MIN(ABS(xCutOff),8.0D0)
      INFIS  = 0
      INFISD = 0
      Ndim   = 0
      Ndleft = Nd
      N      = Nt + Nd

      LTOL = mSmall
      TMP  = ZERO
      DO I = 1, N
         IF (R(I,I).GT.TMP) TMP = R(I,I)
      ENDDO
      EPSL = tmp*MAX(mCovEps,N*mSmall) !tmp
      !IF (N < 10) EPSL = MIN(1D-10,EPSL)
      !LTOL = EPSL
!      EPSL = MAX(EPS2,LTOL)
      IF (TMP.GT.EPSL) THEN
         DO I = 1, N
            IF ((INFI(I)  <  0).OR.(R(I,I)<=LTOL)) THEN
               IF (INDEX1(I)<=Nt)  THEN
                  INFIS = INFIS+1
               ELSEIF (R(I,I)<=LTOL) THEN
                  INFISD = INFISD+1
               ENDIF
            ENDIF
         END DO
      ELSE
         LTOL       = EPSL
         INFIS      = Nt
         INFISD     = Nd
         R(1:N,1:N) = ZERO
      ENDIF

      covErr = 20.d0*LTOL

      N1  = N-INFIS-INFISD
      CDI(N1+1:N) = gZERO
      !PRINT *,'COVSRT'
      !CALL PRINTCOF(N,A,B,INFI,R,INDEX1)

!     Move any redundant variables of Xd to innermost positions.
      LP3: DO I = N, N-INFISD+1, -1
         isXt = (INDEX1(I)<=Nt)
         IF ( (R(I,I) > LTOL) .OR. (isXt)) THEN
            DO J = 1,I-1
               isXd = (INDEX1(J)>Nt)
               IF ( (R(J,J) <= LTOL) .AND.isXd) THEN
                  CALL RCSWAP(J, I, N, N, R,INDEX1,Cm, A, B, INFI)
                  !GO TO 10
                  CYCLE LP3
               ENDIF
            END DO
         ENDIF
! 10
      END DO LP3
!
!     Move any doubly infinite limits or any redundant of Xt to the next
!     innermost positions.
!
      LP4: DO I = N-INFISD, N1+1, -1
         isXd = (INDEX1(I)>Nt)
         IF ( ((INFI(I) > -1).AND.(R(I,I) > LTOL))
     &        .OR. isXd) THEN
            DO J = 1,I-1
               isXt = (INDEX1(J)<=Nt)
               IF ( (INFI(J)  <  0 .OR. (R(J,J)<= LTOL))
     &              .AND. (isXt)) THEN
                  CALL RCSWAP( J, I, N,N, R,INDEX1,Cm, A, B, INFI)
                  !GO TO 15
                  CYCLE LP4
               ENDIF
            END DO
         ENDIF
!15
      END DO LP4

!      CALL mexprintf('Before sorting')
!      CALL PRINTCOF(N,A,B,INFI,R,INDEX1)
!      CALL PRINTVEC(CDI,'CDI')
!      CALL PRINTVEC(Cm,'Cm')

      IF ( N1 <= 0 ) GOTO 200
!
!     Sort remaining limits and determine Cholesky factor.
!
      Y(1:N1) = gZERO
      K       = 1
      Ndleft  = Nd - INFISD
      Nullity = 0
      DO  WHILE (K .LE. N1)

!     IF (Ndim.EQ.3) EPSL = MAX(EPS2,1D-10)
!     Determine the integration limits for variable with minimum
!     expected probability and interchange that variable with Kth.

         K0     = K - Nullity
         PRBMIN = gTWO
         JMIN   = K
         CVDIAG = ZERO
         RMAX   = ZERO
         IF ((Ndleft>0) .OR. (NDIM < Nd+mNIT)) THEN
            DO J = K, N1
               isXd = (INDEX1(J)>Nt)
               isOK = ((NDIM <= Nd+mNIT).OR.isXd)
               IF ( R(J,J) <= K0*K0*EPSL .OR. (.NOT. isOK)) THEN
                  RMAX = max(RMAX,ABS(R(J,J)))
               ELSE
                  TMP = ZERO    ! =  conditional mean of Y(I) given Y(1:I-1)
                  DO I = 1, K0 - 1
                     TMP = TMP + R(I,J)*Y(I)
                  END DO
                  SUMSQ = SQRT( R(J,J))

                  IF (INFI(J)>-1) THEN
                                ! May have infinite int. limits if Nd>0
                     IF (INFI(J).NE.0) THEN
                        AJ = ( A(J) - TMP )/SUMSQ
                     ENDIF
                     IF (INFI(J).NE.1) THEN
                        BJ = ( B(J) - TMP )/SUMSQ
                     ENDIF
                  ENDIF
                  IF (isXd) THEN
                     AA = (Cm(J)+TMP)/SUMSQ ! inflection point
                     CALL EXLMS(AA,AJ,BJ,INFI(J),D,E,Ca,Pa)
                     PRBJ = E - D
                  ELSE
                                !CALL MVNLMS( AJ, BJ, INFI(J), D, E )
                     CALL MVNLIMITS(AJ,BJ,INFI(J),APJ,PRBJ)
                  ENDIF
                                !IF ( EMIN + D .GE. E + DMIN ) THEN
                  IF ( PRBJ  <  PRBMIN ) THEN
                     JMIN = J
                     AMIN = AJ
                     BMIN = BJ
                     PRBMIN = MAX(PRBJ,ZERO)
                     CVDIAG = SUMSQ
                  ENDIF
               ENDIF
            END DO
         END IF
!
!     Compute Ith column of Cholesky factor.
!     Compute expected value for Ith integration variable (without
!     considering the jacobian) and
!     scale Ith covariance matrix row and limits.
!
! 40
         IF ( CVDIAG.GT.TOL) THEN
            isXd = (INDEX1(JMIN)>Nt)
            IF (isXd) THEN
               Ndleft = Ndleft - 1
            ELSEIF (BCVSRT.EQV..FALSE..AND.(PRBMIN+LTOL>=gONE)) THEN
!BCVSRT.EQ.
               J = 1
               AJ = R(J,JMIN)*xCut
               BJ = - (R(J,JMIN))*xCut
               if (INFI(J)>=0) then
                  if (INFI(J).ne.0) then
                     AJ = -(R(J,JMIN))*MAX(A(J),-xCut)
                  endif
                  if (INFI(J).ne.1) then
                     BJ = - (R(J,JMIN))*MIN(B(J),xCut)
                  endif
               endif
               if (R(J,JMIN)<gZERO) THEN
                  CALL SWAP(AJ,BJ)
               endif
               INFJ = INFI(JMIN)
               AJ   = A(JMIN)+AJ
               BJ   = B(JMIN)+BJ

               D = gZERO
               DO J = 2, K0-1
                  D = D + ABS(R(J,JMIN))
               END DO

               AJ = (AJ + D*xCut)/CVDIAG
               BJ = (BJ - D*xCut)/CVDIAG
               CALL ADJLIMITS(AJ,BJ,INFJ)
               IF (INFJ < 0) THEN
                  !variable is redundnant
                  !                     CALL mexPrintf('AdjLim'//CHAR(10))
                  IF ( JMIN < N1 ) THEN
                   CALL RCSWAP( JMIN,N1,N1,N,R,INDEX1,Cm,A,B,INFI)
                     ! move conditional standarddeviations
                     R(JMIN,1:K0-1) = R(N1,1:K0-1)

                     Y(JMIN) = Y(N1)
                  ENDIF
                  R(1:N1,N1)     = gZERO
                  R(N1,1:N1)     = gZERO
                  Y(N1)   = gZERO
                  INFIS = INFIS+1
                  N1    = N1-1
                  GOTO 100
               END IF
            ENDIF
            NDIM = NDIM + 1     !Number of relevant dimensions to integrate

            IF ( K < JMIN ) THEN
               CALL RCSWAP( K, JMIN, N1,N, R,INDEX1,Cm, A, B, INFI)
               ! SWAP conditional standarddeviations
               DO J = 1,K0-1  !MIN(K0, K-1)
                  CALL SWAP(R(K,J),R(JMIN,J))
               END DO
            END IF

            R(K0,K) = CVDIAG
            CDI(K)  = CVDIAG     ! Store the diagonal element
            DO I = K0+1,K
               R(I,K) = gZERO;
               R(K,I) = gZERO
            END DO

            K1 = K
            I  = K1 + 1
            DO WHILE (I <= N1)
               TMP = ZERO
               DO J = 1, K0 - 1
                  !tmp = tmp + L(i,j).*L(k1,j)
                  TMP = TMP + R(J,I)*R(J,K1)
               END DO
                  ! Cov(Xk,Xi|X1,X2,...Xk-1)/STD(Xk|X1,X2,...Xk-1)
               R(K0,I)  = (R(K1,I) - TMP) /CVDIAG
                  ! Var(Xi|X1,X2,...Xk)
               R(I,I) = R(I,I) - R(K0,I) * R(K0,I)

               IF (R(I,I).GT.LTOL) THEN
                  R(I,K0) = SQRT(R(I,I)) ! STD(Xi|X1,X2,...Xk)
               ELSE   !!IF (R(I,I) .LE. LTOL) THEN !TOL
                                !CALL mexprintf('Singular')
                  isXd = (index1(I)>Nt)
                  if (isXd) then
                     Ndleft = Ndleft - 1
                  ELSEIF (BCVSRT.EQV..FALSE.) THEN
!     BCVSRT.EQ.
                     J = 1
                     AJ = R(J,I)*xCut
                     BJ = - (R(J,I))*xCut
                     if (INFI(J)>=0) then
                        if (INFI(J).ne.0) then
                           AJ = -(R(J,I))*MAX(A(J),-xCut)
                        endif
                        if (INFI(J).ne.1) then
                           BJ = - (R(J,I))*MIN(B(J),xCut)
                        endif
                     endif
                     if (R(J,I)<gZERO) THEN
                        CALL SWAP(AJ,BJ)
                     endif
                     INFJ = INFI(I)
                     AJ   = A(I)+AJ
                     BJ   = B(I)+BJ

                     D = gZERO
                     DO J = 2, K0
                        D = D + ABS(R(J,I))
                     END DO

                     AJ = (AJ + D*xCut)-mXcutOff
                     BJ = (BJ - D*xCut)+mXcutOff
                     !call printvar(Aj,'Aj')
                     !call printvar(Bj,'Bj')
                     CALL ADJLIMITS(AJ,BJ,INFJ)
                     IF (INFJ < 0) THEN
                                !variable is redundnant
                        !CALL mexPrintf('AdjLim'//CHAR(10))
                        IF ( I < N1 ) THEN
                           CALL RCSWAP( I,N1,N1,N,R,INDEX1,Cm,A,B,INFI)
                                ! move conditional standarddeviations
                           R(I,1:K0-1) = R(N1,1:K0-1)

                           Y(I) = Y(N1)
                        ENDIF
                        R(1:N1,N1)     = gZERO
                        R(N1,1:N1)     = gZERO
                        Y(N1)   = gZERO
                        INFIS = INFIS+1
                        N1    = N1-1

                        !CALL mexprintf('covsrt updated N1')
                        !call printvar(INFIS,' Infis')
                        GOTO 75
                     END IF
                  END IF
                  IF (mNIT>100) THEN
                     R(I,K0) = gZERO
                  ELSE
                     R(I,K0) = MAX(SQRT(MAX(R(I,I), gZERO)),LTOL)
                  ENDIF
                  Nullity = Nullity + 1
                  K  = K + 1
                  IF (K  <  I) THEN
                     CALL RCSWAP( K, I, N1,N,R,INDEX1,Cm, A, B, INFI)
                     ! SWAP conditional standarddeviations
                     DO J = 1, K0
                        CALL SWAP(R(K,J),R(I,J))
                     END DO
                  ENDIF
                  chkLim = .FALSE. !((.not.isXd).AND.(BCVSRT.EQ..FALSE.))
                  L = INFIS
                  CALL  RCSCALE(chkLim,K,K0,N1,N,K1,INFIS,CDI,Cm,
     &                 R,A,B,INFI,INDEX1)
                  if (L.ne.INFIS) THEN
                     K = K - 1
                     I = I - 1
                  ENDIF
               END IF
               I = I + 1
 75            CONTINUE
            END DO
            INFJ = INFI(K1)

            IF (K1 .EQ.1) THEN
               FINA = 0
               FINB = 0
               IF (INFJ.GE.0) THEN
                  IF  (INFJ.NE.0) FINA = 1
                  IF  (INFJ.NE.1) FINB = 1
               ENDIF
               CALL C1C2(K1+1,N1,K0,A,B,INFI, Y, R,
     &              AMIN, BMIN, FINA,FINB)
               INFJ = 2*FINA+FINB-1
               CALL MVNLIMITS(AMIN,BMIN,INFJ,APJ,PRBMIN)
            ENDIF

            Y(K0) = gettmean(AMIN,BMIN,INFJ,PRBMIN)


            R( K0, K1 ) = R( K0, K1 ) / CVDIAG
            DO J = 1, K0 - 1
               ! conditional covariances
               R( J, K1 ) = R( J, K1 ) / CVDIAG
               ! conditional standard dev.s used in regression eq.
               R( K1, J ) = R( K1, J ) / CVDIAG
            END DO

            A( K1 ) = A( K1 )/CVDIAG
            B( K1 ) = B( K1 )/CVDIAG

            K  = K  + 1
100         CONTINUE
         ELSE
            covErr = RMAX
            R(K:N1,K:N1) = gZERO
            I = K
            DO WHILE (I <= N1)
!  Scale  covariance matrix rows and limits
!  If the conditional covariance matrix diagonal entry is zero,
!  permute limits and/or rows, if necessary.
               chkLim = ((index1(I)<=Nt).AND.(BCVSRT.EQV..FALSE.))
               L = INFIS
               CALL RCSCALE(chkLim,I,K0-1,N1,N,K1,INFIS,CDI,Cm,
     &              R,A,B,INFI,INDEX1)
               if (L.EQ.INFIS) I = I + 1
            END DO
            Nullity = N1 - K0 + 1
            GOTO 200  !RETURN
         END IF
      END DO
 200  CONTINUE
      IF (Ndim .GT. 0) THEN  ! N1<K
         ! K1 = index to the last stochastic varible to integrate
         ! If last stoch. variable is Xt: reduce dimension of integral by 1
         IF (ALL(INDEX1(K1:N1).LE.Nt)) Ndim = Ndim-1
      ENDIF
!      CALL mexprintf('After sorting')

!      CALL PRINTCOF(N,A,B,INFI,R,INDEX1)
!      CALL printvar(A(1),'A1')
!      CALL printvar(B(1),'B1')
!      CALL printvar(INFIS,'INFIS')
!      CALL PRINTVEC(CDI,'CDI')
!      CALL PRINTVEC(Y,'Y')
!      CALL PRINTVEC(AA1,'AA1')
!      CALL PRINTVEC(BB1,'BB1')
!      CALL PRINTVAR(NDIM,TXT='NDIM')
!      CALL PRINTVAR(NIT,TXT='NIT')
!      DEALLOCATE(AA1)
!      DEALLOCATE(BB1)
      RETURN
      END SUBROUTINE COVSRT

      SUBROUTINE COVSRT1(BCVSRT, Nt,Nd,R,Cm,A,B,INFI,INDEX1,
     &     INFIS,INFISD, NDIM, Y, CDI )
      USE FIMOD
      USE SWAPMOD
!      USE GLOBALCONST
!      USE GLOBALDATA, ONLY : EPS2,NIT,xCutOff,Nc1c2
      IMPLICIT NONE
!COVSRT1  sort integration limits and determine Cholesky factor.
!
!     Nt, Nd = size info about Xt and Xd variables.
!     R      = Covariance/Cholesky factored matrix for [Xt,Xd,Xc] (in)
!              On input:
!              Note: Only upper triangular part is needed/used.
!               1a) the first upper triangular the Nt + Nd times Nt + Nd
!                   block contains COV([Xt,Xd]|Xc)
!                   (conditional covariance matrix for Xt and Xd given Xc)
!               2a) The upper triangular part of the Nt+Nd+Nc times Nc
!                   last block contains the cholesky matrix for Xc, i.e.,
!
!              On output:
!               1b) part 2a) mentioned above is unchanged, only necessary
!                   permutations according to INDEX1 is done.
!               2b) part 1a) mentioned above is changed to a special
!                   form of cholesky matrix: (N = Nt+Nd-INFIS-INFISD)
!                  R(1,1) = 1
!                  R(1,2:N) = [COV(X1,X2)/STD(X1),....COV(X1,XN)/STD(X1)]
!                  R(2,2) = 1
!                  R(2,3:N) = [COV(X2,X3)/STD(X2|X1),....COV(X2,XN)/STD(X2|X1)]
!                  ....
!                  etc.
!               3b) The lower triangular part of R contains the
!               conditional standard deviations, i.e.,
!               R(2:N,1)  = [STD(X2|X1) STD(X3|X1),....,STD(XN|X1) ]
!               R(3:N,2)  = [STD(X3|X1,X2),....,STD(XN|X1,X2) ]
!               .....
!               etc.
!
!     Cm     = Conditional mean given Xc
!     A,B    = lower and upper integration limits length Nt+Nd
!     INFIN  = INTEGER, array of integration limits flags:  length Nt+Nd   (in)
!             if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!             if INFIN(I) = 0, Ith limits are (-infinity, B(I)];
!             if INFIN(I) = 1, Ith limits are [A(I), infinity);
!             if INFIN(I) = 2, Ith limits are [A(I), B(I)].
!    INDEX1  = permutation index vector
!    INFIS   = Number of redundant variables of Xt
!    INFISD  = Number of redundant variables of Xd
!    NDIM    = Number of relevant dimensions to integrate. This is the
!             same as the rank of the submatrix of Cov([Xt,Xd]) minus
!             the INFIS variables of Xt and INFISD variables of Xd.
!    Y       = working array
!    CDI     = Cholesky diagonal elements which contains either
!               CDI(J) = STD(Xj| X1,,,Xj-1,Xc) if Xj is stochastic given
!                X1,...Xj, Xc
!              or
!               CDI(J) = COV(Xj,Xk|X1,..,Xk-1,Xc  )/STD(Xk| X1,,,Xk-1,Xc)
!               if Xj is determinstically determined given X1,..,Xk,Xc
!               for some k<j.
!
!     Subroutine to sort integration limits and determine Cholesky
!     factor.
!
!     Note: COVSRT1 works only on the upper triangular part of R
      LOGICAL,                          INTENT(in)    :: BCVSRT
      INTEGER,                          INTENT(in)    :: Nt,Nd
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout) :: R
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(inout) :: Cm,A,B
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: INFI
      INTEGER,          DIMENSION(:  ), INTENT(inout) :: INDEX1
      INTEGER,                          INTENT(out) :: INFIS,INFISD,NDIM
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(out)   :: Y, CDI
!     Local variables
      INTEGER :: N,N1,I, J, K, L, JMIN,Ndleft
      INTEGER ::  K1, K0, Nullity, INFJ, FINA, FINB
      DOUBLE PRECISION :: SUMSQ, AJ, BJ, TMP,D, E,EPSL
      DOUBLE PRECISION :: AA, Ca, Pa, APJ, PRBJ, RMAX, xCut
      DOUBLE PRECISION :: CVDIAG, AMIN, BMIN, PRBMIN
      DOUBLE PRECISION :: LTOL,TOL, ZERO,EIGHT
!      INTEGER, PARAMETER :: NMAX = 1500
!      DOUBLE PRECISION, DIMENSION(NMAX) :: AP,BP
!      INTEGER, DIMENSION(NMAX) :: INFP
!      INTEGER :: Nabp
      LOGICAL          :: isOK = .TRUE.
      LOGICAL          :: isXd = .FALSE.
      LOGICAL          :: isXt = .FALSE.
      LOGICAL          :: chkLim
!      PARAMETER ( SQTWPI = 2.506628274631001D0)
      PARAMETER (ZERO = 0.D0,EIGHT = 8.D0, TOL=1.0D-16)

      xCut = MIN(ABS(mXcutOff),EIGHT)
      INFIS  = 0
      INFISD = 0
      Ndim   = 0
      Ndleft = Nd
      N      = Nt+Nd

!     IF (N < 10) EPSL = MIN(1D-10,EPS2)
      LTOL = TOL
      TMP  = ZERO
      DO I = 1, N
         IF (R(I,I).GT.TMP) TMP = R(I,I)
      ENDDO

      EPSL = MAX(mCovEps,N*TMP*mSmall)
      IF (TMP.GT.EPSL) THEN
         DO I = 1, N
            IF ((INFI(I)  <  0).OR.(R(I,I).LE.LTOL)) THEN
               IF (INDEX1(I).LE.Nt)  THEN
                  INFIS = INFIS+1
               ELSEIF (R(I,I).LE.LTOL) THEN
                  INFISD = INFISD+1
               ENDIF
            ENDIF
         END DO
      ELSE
         !CALL PRINTCOF(N,A,B,INFI,R,INDEX1)
         !CALL PRINTVEC(CDI)
         !CALL PRINTVEC(Cm)
         LTOL   = EPSL
         INFIS  = Nt
         INFISD = Nd
         R(1:N,1:N) = ZERO
      ENDIF
      N1  = N-INFIS-INFISD
      CDI(N1+1:N) = ZERO
!     PRINT *,'COVSRT'
!     CALL PRINTCOF(N,A,B,INFI,R,INDEX1)

!     Move any redundant variables of Xd to innermost positions.
      LP1:  DO I = N, N-INFISD+1, -1
         isXt = (INDEX1(I).LE.Nt)
         IF ( R(I,I) .GT. LTOL .OR. isXt) THEN
            DO J = 1,I-1
               isXd = (INDEX1(J).GT.Nt)
               IF ( R(J,J) .LE. LTOL .AND. isXd) THEN
                  CALL RCSWAP( J, I, N,N, R,INDEX1,Cm, A, B, INFI)
                  !GO TO 10
                  CYCLE LP1
               ENDIF
            END DO
         ENDIF
! 10
      END DO LP1
!
!     Move any doubly infinite limits or any redundant of Xt to the next
!     innermost positions.
!
      LP2: DO I = N-INFISD, N1+1, -1
         isXd = (INDEX1(I).GT.Nt)
         IF ( ((INFI(I) .GE. 0).AND. (R(I,I).GT. LTOL) )
     &        .OR. isXd) THEN
            DO J = 1,I-1
               isXt = (INDEX1(J).LE.Nt)
               IF ( (INFI(J)  <  0 .OR. (R(J,J).LE. LTOL))
     &              .AND. isXt) THEN
                  CALL RCSWAP( J, I, N,N, R,INDEX1,Cm, A, B, INFI)
                  !GO TO 15
                  CYCLE LP2
               ENDIF
            END DO
         ENDIF
 !15
      END DO LP2

      IF ( N1 .LE. 0 ) RETURN
!      CALL mexprintf('Before sorting')
!      CALL PRINTCOF(N,A,B,INFI,R,INDEX1)


!     Sort remaining limits and determine Cholesky factor.
      Y(1:N1) = ZERO
      K  = 1
!      N1  = N-INFIS-INFISD
      Ndleft = Nd - INFISD
      Nullity = 0

!      Nabp  = 0
!      AP(1:N1) = ZERO
!      BP(1:N1) = zero
      DO  WHILE (K .LE. N1)

!     Determine the integration limits for variable with minimum
!     expected probability and interchange that variable with Kth.

         K0     = K-Nullity
         PRBMIN = 2.d0
         JMIN   = K
         CVDIAG = ZERO
         RMAX   = ZERO
         IF (Ndleft.GT.0 .OR. NDIM < Nd+mNIT) THEN
            DO J = K,N1
               isXd = (INDEX1(J).GT.Nt)
               isOK = ((NDIM <= Nd+mNIT).OR.isXd)
               IF ( R(J,J) .LE. K0*K0*EPSL.OR. (.NOT. isOK)) THEN
                  RMAX = max(RMAX,R(J,J))
               ELSE
                  TMP = Y(J) ! =  the conditional mean of Y(J) given Y(1:J-1)
                  SUMSQ = SQRT( R(J,J))

                  IF (INFI(J) < 0) GO TO 30 ! May have infinite int. limits if Nd>0
                  IF (INFI(J).NE.0) THEN
                     AJ = ( A(J) - TMP )/SUMSQ
                  ENDIF
                  IF (INFI(J).NE.1) THEN
                     BJ = ( B(J) - TMP )/SUMSQ
                  ENDIF
 30               IF (INDEX1(J).GT.Nt) THEN
                     AA = (Cm(J)+TMP)/SUMSQ ! inflection point
                     CALL EXLMS(AA,AJ,BJ,INFI(J),D,E,Ca,Pa)
                     PRBJ = E-D
                  ELSE
                                !CALL MVNLMS( AJ, BJ, INFI(J), D, E )
                     CALL MVNLIMITS(AJ,BJ,INFI(J),APJ,PRBJ)
                  ENDIF
                  IF ( PRBJ  <  PRBMIN ) THEN
                     JMIN = J
                     AMIN = AJ
                     BMIN = BJ
                     PRBMIN = MAX(PRBJ,ZERO)
                     CVDIAG = SUMSQ
                  ENDIF
               ENDIF
            END DO
         END IF
!
!     Compute Ith column of Cholesky factor.
!     Compute expected value for Ith integration variable (without
!     considering the jacobian) and
!     scale Ith covariance matrix row and limits.
!
!40
         IF ( CVDIAG.GT.TOL) THEN
            IF (INDEX1(JMIN).GT.Nt) THEN
               Ndleft = Ndleft-1
            ELSE
               IF (BCVSRT.EQV..FALSE..AND.(PRBMIN+LTOL.GE.gONE)) THEN
!BCVSRT.EQ.
                  I = 1
                  AJ = R(I,JMIN)*xCut
                  BJ = - (R(I,JMIN))*xCut
                  if (INFI(1)>=0) then
                     if (INFI(1).ne.0) then
                        AJ = -(R(I,JMIN))*MAX(A(I),-xCut)
                     endif
                     if (INFI(1).ne.1) then
                        BJ = - (R(I,JMIN))*MIN(B(I),xCut)
                     endif
                  endif
                  if (R(I,JMIN)<gZERO) THEN
                     CALL SWAP(AJ,BJ)
                  endif
                  INFJ = INFI(JMIN)
                  AJ   = A(JMIN)+AJ
                  BJ   = B(JMIN)+BJ

                  D = gZERO
                  DO I = 2, K0-1
                     D = D + ABS(R(I,JMIN))
                  END DO



                  AJ = (AJ + D*xCut)/CVDIAG
                  BJ = (BJ - D*xCut)/CVDIAG
                  CALL ADJLIMITS(AJ,BJ,INFJ)
                  IF (INFJ < 0) THEN
                     !variable is redundnant
                     !CALL mexPrintf('AdjLim'//CHAR(10))
                   IF ( JMIN < N1 ) THEN
                   CALL RCSWAP( JMIN, N1, N1,N, R,INDEX1,Cm, A, B, INFI)
                                ! SWAP conditional standarddeviations
                      DO I = 1,K0-1
                         CALL SWAP(R(JMIN,I),R(N1,I))
                      END DO
                      CALL SWAP(Y(N1),Y(JMIN))
                   ENDIF
                   INFIS = INFIS+1
                   N1    = N1-1
                   GOTO 100
                  END IF
                ENDIF
            ENDIF
            NDIM = NDIM + 1     !Number of relevant dimensions to integrate

            IF ( K < JMIN ) THEN

               CALL RCSWAP( K, JMIN, N1,N, R,INDEX1,Cm, A, B, INFI)
               ! SWAP conditional standarddeviations
               DO J=1,K0-1
                  CALL SWAP(R(K,J),R(JMIN,J))
               END DO
               CALL SWAP(Y(K),Y(JMIN))
            END IF


            R(K0,K:N1) = R(K0,K:N1)/CVDIAG
            R(K0,K) = CVDIAG
            CDI(K)  = CVDIAG     ! Store the diagonal element
            DO I = K0+1,K
               R(I,K) = ZERO
               R(K,I) = ZERO
            END DO

            K1  = K
            !IF (K .EQ. N1) GOTO 200

!  Cov(Xi,Xj|Xk,Xk+1,..,Xn)=
!             Cov(Xi,Xj|Xk+1,..,Xn) -
 !             Cov(Xi,Xk|Xk+1,..Xn)*Cov(Xj,Xk|Xk+1,..Xn)
            I = K1 +1
            DO WHILE (I <= N1)
                ! Var(Xj | Xk,Xk+1,...,Xn)
               R(I,I)  =  R(I,I) - R(K0,I)*R(K0,I)
               IF (R(I,I).GT.LTOL) THEN
                  R(I,K0)     = SQRT(R(I,I)) ! STD(Xi|X1,X2,...Xk)
                  R(I,I+1:N1) = R(I,I+1:N1) - R(K0,I+1:N1)*R(K0,I)
               ELSE
                  R(I,K0) = MAX(SQRT(MAX(R(I,I), gZERO)),LTOL)
                  Nullity = Nullity + 1
                  K  = K + 1
                  IF (K  <  I) THEN
                     CALL RCSWAP( K, I, N1,N,R,INDEX1,Cm, A, B, INFI)
                     ! SWAP conditional standarddeviations
                     DO J=1,K0
                        CALL SWAP(R(K,J),R(I,J))
                     END DO
                     CALL SWAP(Y(K),Y(I))
                  ENDIF
                  isXd = (INDEX1(K).GT.Nt)
                  IF (isXd) Ndleft = Ndleft-1
                  chkLim = ((.not.isXd).AND.(BCVSRT.EQV..FALSE.))
                  L = INFIS
                  CALL  RCSCALE(chkLim,K,K0,N1,N,K1,INFIS,CDI,Cm,
     &                 R,A,B,INFI,INDEX1,Y)
                  IF (L.NE.INFIS) I = I - 1
               END IF
               I = I +1
            END DO
            INFJ = INFI(K1)
            IF (K0 == 1) THEN
               FINA = 0
               FINB = 0
               IF (INFJ.GE.0) THEN
                  IF  (INFJ.NE.0) FINA = 1
                  IF  (INFJ.NE.1) FINB = 1
               ENDIF
               CALL C1C2(K1+1,N1,K0,A,B,INFI, Y, R,
     &              AMIN, BMIN, FINA,FINB)
               INFJ = 2*FINA+FINB-1
               CALL MVNLIMITS(AMIN,BMIN,INFJ,APJ,PRBMIN)
            ENDIF
            Y(K0) = GETTMEAN(AMIN,BMIN,INFJ,PRBMIN)

                     ! conditional mean (expectation)
                                ! E(Y(K+1:N)|Y(1),Y(2),...,Y(K))
            Y(K+1:N1) = Y(K+1:N1)+Y(K0)*R(K0,K+1:N1)
            R(K0,K1)  = R(K0,K1)/CVDIAG ! conditional covariances
            DO J = 1, K0 - 1
               R(J,K1) = R(J,K1)/CVDIAG ! conditional covariances
               R(K1,J) = R(K1,J)/CVDIAG ! conditional standard dev.s used in regression eq.
            END DO
            A(K1) = A(K1)/CVDIAG
            B(K1) = B(K1)/CVDIAG

            K  = K  + 1
 100        CONTINUE
         ELSE
            R(K:N1,K:N1) = gZERO
!            CALL PRINTCOF(N,A,B,INFI,R,INDEX1)
            I = K
            DO WHILE (I <= N1)
!  Scale  covariance matrix rows and limits
!  If the conditional covariance matrix diagonal entry is zero,
!  permute limits and/or rows, if necessary.
               chkLim = ((index1(I)<=Nt).AND.(BCVSRT.EQV..FALSE.))
               L = INFIS
               CALL RCSCALE(chkLim,I,K0-1,N1,N,K1,INFIS,CDI,Cm,
     &              R,A,B,INFI,INDEX1)
               if (L.EQ.INFIS) I = I + 1
            END DO
            Nullity = N1 - K0 + 1
            GOTO 200  !RETURN
         END IF
      END DO

 200  CONTINUE
      IF (Ndim .GT. 0) THEN     ! N1<K
         ! K1 = index to the last stochastic varible to integrate
         IF (ALL(INDEX1(K1:N1).LE.Nt)) Ndim = Ndim - 1
      ENDIF
!      CALL mexprintf('After sorting')
!      CALL PRINTCOF(N,A,B,INFI,R,INDEX1)
!      CALL PRINTVEC(CDI)
!      CALL PRINTVAR(NDIM,TXT='NDIM')
      RETURN
      END SUBROUTINE COVSRT1

      SUBROUTINE RCSWAP( P, Q, N,Ntd, C,IND,Cm, A, B, INFIN )
      USE SWAPMOD
      IMPLICIT NONE
! RCSWAP  Swaps rows and columns P and Q in situ, with P <= Q.
!
!
!   CALL  RCSWAP( P, Q, N, Ntd, C,IND A, B, INFIN, Cm)
!
!    P, Q  = row/column number to swap P<=Q<=N
!    N     = Number of significant variables of [Xt,Xd]
!    Ntd   = length(Xt)+length(Xd)
!    C     = upper triangular cholesky factor.Cov([Xt,Xd,Xc]) size Ntdc X Ntdc
!    IND   = permutation index vector. (index to variables original place).
!    Cm    = conditional mean
!    A,B   = lower and upper integration limit, respectively.
!    INFIN = if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!            if INFIN(I) = 0, Ith limits are (-infinity, B(I)];
!            if INFIN(I) = 1, Ith limits are [A(I), infinity);
!            if INFIN(I) = 2, Ith limits are [A(I), B(I)].
!
! NOTE: RCSWAP works only on the upper triangular part of C
      DOUBLE PRECISION, DIMENSION(:,:),INTENT(inout) :: C
      INTEGER, DIMENSION(:),INTENT(inout) :: IND
      INTEGER, DIMENSION(:),          OPTIONAL,INTENT(inout) :: INFIN
      DOUBLE PRECISION, DIMENSION(:), OPTIONAL,INTENT(inout) :: A,B,Cm
      INTEGER,INTENT(in) :: P, Q, N, Ntd
! local variable
      INTEGER :: J, Ntdc
      LOGICAL :: isXc
      IF (PRESENT(Cm))    CALL SWAP( Cm(P), Cm(Q) )
      IF (PRESENT(A))     CALL SWAP( A(P), A(Q) )
      IF (PRESENT(B))     CALL SWAP( B(P), B(Q) )
      IF (PRESENT(INFIN)) CALL SWAP(INFIN(P),INFIN(Q))

      CALL SWAP(IND(P),IND(Q))

      CALL SWAP( C(P,P), C(Q,Q) )
      DO J = 1, P-1
         CALL SWAP( C(J,P), C(J,Q) )
      END DO
      DO J = P+1, Q-1
         CALL SWAP( C(P,J), C(J,Q) )
      END DO
      DO J = Q+1, N
         CALL SWAP( C(P,J), C(Q,J) )
      END DO
      Ntdc = SIZE(C,DIM=1)
      isXc = (N < Ntdc)
      IF (isXc) THEN
         !Swap row P and Q of Xc variables
         DO J = Ntd+1, Ntdc
            CALL SWAP( C(P,J), C(Q,J) )
         END DO
      ENDIF
      RETURN
      END SUBROUTINE RCSWAP
      end module RINDMOD
