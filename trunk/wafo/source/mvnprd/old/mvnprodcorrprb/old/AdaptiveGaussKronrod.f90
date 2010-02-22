! f2py -m adaptivegausskronrod -h adaptivegausskronrod.pyf AdaptiveGaussKronrod.f
! f2py adaptivegausskronrod.pyf AdaptiveGaussKronrod.f -c --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71
! f2py --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 -m adaptivegausskronrod  -c AdaptiveGaussKronrod.f
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
      subroutine dqagp(f,a,b,npts,points,epsabs,epsrel,limit,result1,
     *   abserr,neval,ier)
!      use functionInterface
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
      subroutine dqagpe(f,a,b,npts,points,epsabs,epsrel,limit,result1,
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
      double precision, intent(out) :: result1,abserr
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
      result1  = 0.0d+00
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
        result1 = result1 + area1
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
      dres   = dabs(result1)
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
   80 if(ier.ne.0.or.abserr.le.errbnd) go to 210

!
!           initialization
!           --------------
!
      rlist2(1) = result1
      maxerr    = iord(1)
      errmax    = elist(maxerr)
      area      = result1
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
        result1 = reseps
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
      if(result1.ne.0.0d+00.and.area.ne.0.0d+00)go to 175
      if(abserr.gt.errsum)go to 190
      if(area.eq.0.0d+00) go to 210
      go to 180
  175 if(abserr/dabs(result1).gt.errsum/dabs(area))go to 190
!
!           test on divergence.
!
  180 if(ksgn.eq.(-1).and.dmax1(dabs(result1),dabs(area)).le.
     *  resabs*0.1d-01) go to 210
      if(0.1d-01.gt.(result1/area).or.(result1/area).gt.0.1d+03.or.
     *  errsum.gt.dabs(area)) ier = 6
      go to 210
!
!           compute global integral sum.
!
  190 result1 = 0.0d+00
      do 200 k = 1,last
        result1 = result1+rlist(k)
  200 continue
      abserr = errsum
  210 if(ier.gt.2) ier = ier-1
      result1 = result1*sign
  999 return
      end subroutine dqagpe
      subroutine dqk21(f,a,b,result1,abserr,resabs,resasc)
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
      double precision,intent(in) :: a,b
	  double precision, intent(out) :: abserr, result1,resabs,resasc
	  double precision :: f,absc,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,
     *  resg,resk,reskh,uflow,wg,wgk,xgk
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
      result1 = resk*hlgth
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
      subroutine dqk15(f,a,b,result1,abserr,resabs,resasc)
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
	  double precision,intent(in) :: a,b
	  double precision, intent(out) :: abserr, result1,resabs,resasc
      double precision :: f, absc,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,
     *  resg,resk,reskh,uflow,wg,wgk,xgk
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
      result1 = resk*hlgth
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
      subroutine dqk9(f,a,b,result1,abserr,resabs,resasc)
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
	  double precision,intent(in) :: a,b
	  double precision, intent(out) :: abserr, result1,resabs,resasc
      double precision :: f,absc,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,
     *  resg,resk0,resk,reskh,uflow,wg,wgk0,wgk,xgk
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
      result1 = resk
      call dea3(resg,resk0,resk,abserr,result1)
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
      subroutine dqkl9(f,a,b,result1,abserr,resabs,resasc)
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
	  double precision,intent(in) :: a,b
	  double precision, intent(out) :: abserr, result1,resabs,resasc
      double precision :: f,absc,centr,dhlgth,
     *  epmach,fc,fsum,fval1,fval2,fv1,fv2,hlgth,
     *  resg,resk0,resk,reskh,uflow,wg,wgk0,wgk,xgk
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
      result1 = resk
      call dea3(resg,resk0,resk,abserr,result1)
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
      subroutine dqelg(n,epstab,result1,abserr,res3la,nres)
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
     *  oflow,res,result1,res3la,ss,tol1,tol2,tol3
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
      result1 = epstab(n)
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
c           result1 = e2
c           abserr = abs(e1-e0)+abs(e2-e1)
c
        result1 = res
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
        result1 = res
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
      res3la(nres) = result1
      abserr = oflow
      go to 100
c
c           compute error estimate
c
   90 abserr = dabs(result1-res3la(3))+dabs(result1-res3la(2))
     *  +dabs(result1-res3la(1))
      res3la(1) = res3la(2)
      res3la(2) = res3la(3)
      res3la(3) = result1
  100 abserr = dmax1(abserr,0.5d+01*epmach*dabs(result1))
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