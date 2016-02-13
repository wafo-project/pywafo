
      subroutine prbnormtndpc(rho,a,b,NDF,N,abseps,IERC,HNC,PRB,BOUND,
     *  IFAULT)
      double precision A(N),B(N),rho(N),D(N)
      integer INFIN(N)
      integer NDF,N,IERC
      integer IFAULT
      double precision HNC
C      double precision EPS
      double precision PRB, BOUND
      double precision, parameter :: infinity = 37.0d0
Cf2py integer, intent(hide), depend(rho) :: N = len(rho)
Cf2py depend(N)  a
Cf2py depend(N)  b
Cf2py integer, optional :: NDF = 0
Cf2py double precision, optional :: abseps = 0.001
Cf2py double precision, optional :: HNC = 0.24
Cf2py integer, optional :: IERC =0
Cf2py double precision, intent(out) :: PRB
Cf2py double precision, intent(out) :: BOUND
Cf2py integer, intent(out) :: IFAULT

CCf2py intent(in) N,IERC
CCf2py intent(in) HINC,EPS
CCf2py intent(in) INF
CCf2py intent(in) A,B,rho



* Set INFIN  INTEGER, array of integration limits flags:
*            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
*            if INFIN(I) = 0, Ith limits are [LOWER(I), infinity);
*            if INFIN(I) = 1, Ith limits are (-infinity, UPPER(I)];
*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
      Ndim = 0
      DO K = 1,N
           Ndim = Ndim + 1
           INFIN(Ndim) = 2
           D(k) = 0.0
           if (A(K)-D(K).LE.-INFINITY) THEN
              if (B(K)-D(K) .GE. INFINITY) THEN
                 Ndim = Ndim - 1
                 !INFIN(K) = -1
              else
                 INFIN(Ndim) = 1
              endif
           else if (B(K)-D(K).GE.INFINITY) THEN
              INFIN(Ndim) = 0
           endif
           if (ndim<k) then
              RHO(Ndim) = RHO(k)
              A(Ndim) = A(K)
              B(Ndim) = B(K)
C              D(Ndim) = D(K)
           endif
      ENDDO
      CALL MVSTUD(NDF,B,A,RHO,ABSEPS,Ndim,INFIN,D,IERC,HNC,
     & PRB,BOUND,IFAULT)

C CALL MVNPRD(A, B, BPD, EPS, N, INF, IERC, HINC, PROB, BOUND,
C     *  IFAULT)
      return
      end subroutine prbnormtndpc

      subroutine prbnormndpc(prb,abserr,IFT,rho,a,b,N,abseps,releps,
     &  useBreakPoints, useSimpson)
      use mvnProdCorrPrbMod, ONLY : mvnprodcorrprb
      integer :: N
      double precision,dimension(N),intent(in) :: rho,a,b
      double precision,intent(in) :: abseps
      double precision,intent(in) :: releps
      logical,         intent(in) :: useBreakPoints
      logical,         intent(in) :: useSimpson
      double precision,intent(out) :: abserr,prb
      integer, intent(out) :: IFT

Cf2py integer, intent(hide), depend(rho) :: N = len(rho)
Cf2py depend(N)  a
Cf2py depend(N)  b
Cf2py double precision, optional :: abseps = 0.001
Cf2py double precision, optional :: releps = 0.001
Cf2py logical, optional :: useBreakPoints =1
Cf2py logical, optional :: useSimpson = 1

      CALL mvnprodcorrprb(rho,a,b,abseps,releps,useBreakPoints,
     &  useSimpson,abserr,IFT,prb)

      end subroutine prbnormndpc
