
C gfortran -fPIC -c mvnprodcorrprb.f
C f2py -m mvnprdmod  -c mvnprodcorrprb.o mvnprodcorrprb_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71

C      module mvnprdmod
C      contains
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

Cf2py	integer, intent(hide), depend(rho) :: N = len(rho)
Cf2py	depend(N)  a
Cf2py	depend(N)  b
Cf2py double precision, optional :: abseps = 0.001
Cf2py double precision, optional :: releps = 0.001
Cf2py logical, optional :: useBreakPoints =1
Cf2py logical, optional :: useSimpson = 1



      CALL mvnprodcorrprb(rho,a,b,abseps,releps,useBreakPoints,
     &  useSimpson,abserr,IFT,prb)

      end subroutine prbnormndpc
C      end module mvnprdmod