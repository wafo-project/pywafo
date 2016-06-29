! This is a interface-file for Python
! This file contains a interface to RIND a subroutine
! for computing multivariate normal expectations.
! The file is self contained and should compile without errors on (Fortran90)
! standard Fortran compilers.
!
! The interface was written by
!     Per Andreas Brodtkorb
!     Norwegian Defence Research Establishment
!     P.O. Box 115
!     N-3191 Horten
!     Norway
!     Email: Per.Brodtkorb@ffi.no
!
!
! RIND Computes multivariate normal expectations
!
!  E[Jacobian*Indicator|Condition ]*f_{Xc}(xc(:,ix))
!  where
!      "Indicator" = I{ H_lo(i) < X(i) < H_up(i), i=1:N_t+N_d }
!      "Jacobian"  = J(X(Nt+1),...,X(Nt+Nd+Nc)), special case is
!      "Jacobian"  = |X(Nt+1)*...*X(Nt+Nd)|=|Xd(1)*Xd(2)..Xd(Nd)|
!      "condition" = Xc=xc(:,ix),  ix=1,...,Nx.
!      X = [Xt; Xd; Xc], a stochastic vector of Multivariate Gaussian
!          variables where Xt,Xd and Xc have the length Nt, Nd and Nc,
!          respectively. (Recommended limitations Nx,Nt<=100, Nd<=6 and Nc<=10)
!
!  CALL: [value,error,terror,inform]=rind(S,m,indI,Blo,Bup,INFIN,xc,
!         Nt,SCIS,XcScale,ABSEPS,RELEPS,COVEPS,MAXPTS,MINPTS,seed,NIT,xCutOff,Nc1c2);
!
!
!    VALUE  = estimated value for the expectation as explained above size 1 x Nx
!    ERROR  = estimated sampling error, with 99% confidence level.   size 1 x Nx
!   TERROR  = estimated truncation error
!   INFORM  = INTEGER, termination status parameter: (not implemented yet)
!            if INFORM = 0, normal completion with ERROR < EPS;
!            if INFORM = 1, completion with ERROR > EPS and MAXPTS
!                           function vaules used; increase MAXPTS to
!                           decrease ERROR;
!            if INFORM = 2, N > 100 or N < 1.
!
!         S = Covariance matrix of X=[Xt;Xd;Xc] size Ntdc x Ntdc (Ntdc=Nt+Nd+Nc)
!         m = the expectation of X=[Xt;Xd;Xc]   size N x 1
!      indI = vector of indices to the different barriers in the
!            indicator function,  length NI, where   NI = Nb+1
!             (NB! restriction  indI(1)=0, indI(NI)=Nt+Nd )
! B_lo,B_up = Lower and upper barriers used to compute the integration
!             limits, Hlo and Hup, respectively. size  Mb x Nb
!    INFIN  = INTEGER, array of integration limits flags:  size 1 x Nb   (in)
!             if INFIN(I) < 0, Ith limits are (-infinity, infinity);
!             if INFIN(I) = 0, Ith limits are (-infinity, Hup(I)];
!             if INFIN(I) = 1, Ith limits are [Hlo(I), infinity);
!             if INFIN(I) = 2, Ith limits are [Hlo(I), Hup(I)].
!        xc = values to condition on            size Nc x Nx
!        Nt = size of Xt
!      SCIS = Integer defining integration method
!             1 Integrate all by SADAPT for Ndim<9 and by KRBVRC otherwise
!             2 Integrate all by SADAPT by Genz (1992) (Fast)
!             3 Integrate all by KRBVRC by Genz (1993) (Fast)
!             4 Integrate all by KROBOV by Genz (1992) (Fast)
!             5 Integrate all by RCRUDE by Genz (1992)
!   XcScale = REAL to scale the conditinal probability density, i.e.,
!              f_{Xc} = exp(-0.5*Xc*inv(Sxc)*Xc + XcScale)
!    ABSEPS = REAL absolute error tolerance.
!    RELEPS = REAL relative error tolerance.
!    COVEPS = REAL error in cholesky factorization
!    MAXPTS = INTEGER, maximum number of function values allowed. This
!             parameter can be used to limit the time. A sensible
!             strategy is to start with MAXPTS = 1000*N, and then
!             increase MAXPTS if ERROR is too large.
!    MINPTS = INTEGER, minimum number of function values allowed
!    SEED   = INTEGER, seed to the random generator used in the integrations
!    NIT    = INTEGER, maximum number of Xt variables to integrate
!   xCutOff = REAL upper/lower truncation limit of the marginal normal CDF
!    Nc1c2  = INTEGER number of times to use the regression equation to restrict
!             integration area. Nc1c2 = 1,2 is recommended.
!
!
!   If  Mb<Nc+1 then B_lo(Mb+1:Nc+1,:) is assumed to be zero.
!   The relation to the integration limits Hlo and Hup are as follows
!    IF INFIN(j)>=0,
!      IF INFIN(j)~=0,  Hlo(i)=Blo(1,j)+Blo(2:Mb,j).'*xc(1:Mb-1,ix),
!      IF INFIN(j)~=1,  Hup(i)=Bup(1,j)+Bup(2:Mb,j).'*xc(1:Mb-1,ix),
!
!   where i=indI(j-1)+1:indI(j), j=2:NI, ix=1:Nx
!
! This file was successfully compiled for matlab 5.3
! using Compaq Visual Fortran 6.1, and Windows 2000 and windows XP.
! The example here uses Fortran90 source.
! First, you will need to modify your mexopts.bat file.
! To find it, issue the command prefdir(1) from the Matlab command line,
! the directory it answers with will contain your mexopts.bat file.
! Open it for editing. The first section will look like:
!
!rem ********************************************************************
!rem General parameters
!rem ********************************************************************
!set MATLAB=%MATLAB%
!set DF_ROOT=C:\Program Files\Microsoft Visual Studio
!set VCDir=%DF_ROOT%\VC98
!set MSDevDir=%DF_ROOT%\Common\msdev98
!set DFDir=%DF_ROOT%\DF98
!set PATH=%MSDevDir%\bin;%DFDir%\BIN;%VCDir%\BIN;%PATH%
!set INCLUDE=%DFDir%\INCLUDE;%DFDir%\IMSL\INCLUDE;%INCLUDE%
!set LIB=%DFDir%\LIB;%VCDir%\LIB
!
! then you are ready to compile this file at the matlab prompt using the following command:
!
!   mex -O -output mexrind2007 intmodule.f  jacobmod.f rind2007.f mexrind2007.f
!


      subroutine set_constants(method,xcscale,abseps,releps,coveps,
     & maxpts,minpts,nit,xcutoff,Nc1c2, NINT1, xsplit)
      use rindmod, only : setconstants
      use rind71mod, only : setdata
      double precision :: xcscale,abseps,releps,coveps,xcutoff,xsplit
      integer method, maxpts, minpts, nit, Nc1c2, NINT1
Cf2py double precision, optional :: xcscale = 0.0e0
Cf2py double precision, optional :: abseps = 0.01e0
Cf2py double precision, optional :: releps = 0.01e0
Cf2py double precision, optional :: coveps = 1.0e-10
Cf2py double precision, optional :: xcutoff = 5.0e0
Cf2py double precision, optional :: xsplit = 5.0e0

Cf2py integer, optional :: method = 3
Cf2py integer, optional :: minpts = 0
Cf2py integer, optional :: maxpts = 40000
Cf2py integer, optional :: nit = 1000
Cf2py integer, optional :: Nc1c2 = 2
Cf2py integer, optional :: nint1 = 2

! Method>0
      call setconstants(method,xcscale,abseps,releps,coveps,
     &     maxpts,minpts,nit,xcutoff,Nc1c2)
! method==0
      call SETDATA(method,xcscale,abseps,releps,coveps,
     &     nit, xCutOff,NINT1,xsplit)
      return
      end subroutine set_constants
      SUBROUTINE show_constants()
      use rindmod
      print *, 'method=', mMethod
      print *, 'xcscale=', mXcScale
      print *, 'abseps=', mAbsEps
      print *, 'releps=', mRelEps
      print *, 'coveps=', mCovEps
      print *, 'maxpts=', mMaxPts
      print *, 'minpts=', mMinPts
      print *, 'nit=',    mNit
      print *, 'xcutOff=', mXcutOff
      print *, 'Nc1c2=',  mNc1c2
      end subroutine show_constants

      SUBROUTINE rind(VALS,ERR,TERR,Big,Ex,Xc,Nt,INDI,Blo,Bup,
     & INFIN,seed1,Ntdc,Nc,Nx,Ni,Mb,Nb,Nx1)
      USE rindmod
      USE rind71mod, only : rind71
      IMPLICIT NONE
      INTEGER :: Ntd
C      INTEGER :: Nj,K, I
      INTEGER :: seed1
      integer :: Nx,Nx1,Nt, Nc,Ntdc,Ni,Nb,Mb
      DOUBLE PRECISION, dimension(Ntdc,Ntdc) :: BIG
      DOUBLE PRECISION, dimension(Ntdc) :: Ex
      DOUBLE PRECISION, dimension(Nc,Nx1) :: Xc
      DOUBLE PRECISION, dimension(Mb,Nb) :: Blo,Bup
      DOUBLE PRECISION, dimension(Nx) :: VALS, ERR,TERR
      INTEGER, dimension(Ni)  :: IndI
      INTEGER, DIMENSION(Nb) :: INFIN
      INTEGER, ALLOCATABLE  :: seed(:)
      INTEGER               :: seed_size
Cf2py integer, intent(hide), depend(Ex) :: Ntdc = len(Ex)
Cf2py integer, intent(hide), depend(Xc) :: Nc = shape(Xc,0)
Cf2py integer, intent(hide), depend(Xc) :: Nx1 = shape(Xc,1)
Cf2py integer, intent(hide), depend(Xc) :: Nx = max(shape(Xc,1),1)
Cf2py integer, intent(hide), depend(Blo) :: Mb = shape(Blo,0), Nb = shape(Blo,1),
Cf2py integer, intent(hide), depend(Indi) :: Ni = len(Indi)
Cf2py depend(Ntdc)  Big
Cf2py depend(Nb)  INFIN
Cf2py depend(Mb,Nb)  Bup
Cf2py double precision, intent(out), depend(Nx) ::  VALS
Cf2py double precision, intent(out), depend(Nx) ::  ERR
Cf2py double precision, intent(out), depend(Nx) ::  TERR

C     print *, 'Ntdc=', Ntdc,' Nt=',Nt,' Nc=',Nc
C     print *, 'Nx=', Nx, 'Mb=', Mb, ' Nb=', Nb, ' Ni=',Ni
C     Ni = Nb+1
C     Nx = max(Nx1,1)
      if (Ni.EQ.Nb+1) then
      else
         print *, '(ni==nb+1) failed: rind:ni=', Ni, ', nb=',Nb
         return
      endif

      Ntd = Ntdc - Nc;
!    Nd  = Ntd - Nt

      IF (Ntd.EQ.INDI(Ni)) THEN
!     Call the computational subroutine.
        IF (mMethod.gt.0) THEN
          CALL random_seed(SIZE=seed_size)
          ALLOCATE(seed(seed_size))
                               !print *,'rindinterface seed', seed1
          CALL random_seed(GET=seed(1:seed_size)) ! get current state
          seed(1:seed_size)=seed1 ! change seed
          CALL random_seed(PUT=seed(1:seed_size))
          CALL random_seed(GET=seed(1:seed_size)) ! get current state
                                !print *,'rindinterface seed', seed
          DEALLOCATE(seed)
          CALL RINDD(VALS,ERR,TERR,Big,Ex,Xc,Nt,INDI,Blo,Bup,INFIN)
        ELSE
          CALL RIND71(VALS,Big,Ex,Xc,Nt,INDI,Blo,Bup)
          ERR(:) = -1
          TERR(:) = -1
        ENDIF
      ELSE
         print *,'INDI(Ni) must equal Nt+Nd!'
      ENDIF

      RETURN
      END SUBROUTINE rind

