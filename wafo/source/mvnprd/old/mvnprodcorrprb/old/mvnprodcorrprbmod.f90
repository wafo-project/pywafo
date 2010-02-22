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
!            call Romberg(integrand,zlo,zup,Npts
!     $           ,breakPoints,LTol,errFlg,abserr, val)
!            call AdaptiveIntWithBreaks(integrand,zlo,zup,Npts
!     $           ,breakPoints,LTol,errFlg,abserr, val)
         else
! CALL IMSL
!            k = 1 ! integration rule
!            CALL DQDAG(integrand,zlo, zup, LTol, zero, k,
!     &           val,abserr)           
!            CALL DQDAGP (integrand, zlo, zup, Npts, breakPoints, 
!     &           LTol, zero, val,abserr)
!            call AdaptiveIntWithBreaks(integrand,zlo,zup,Npts
!     $           ,breakPoints,LTol,errFlg,abserr, val)
            limit = 100
            call dqagp(integrand,zlo,zup,Npts,breakPoints,LTol,zero,
     &           limit,val,abserr,neval,errFlg)
!            call AdaptiveTrapz(integrand,zlo,zup,Npts,breakPoints,LTol
!     &                        ,errFlg,abserr, val)
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