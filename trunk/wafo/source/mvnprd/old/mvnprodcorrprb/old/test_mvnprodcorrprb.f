	program mvn
C gfortran -fPIC -c mvnprodcorrprb.f
C f2py -m mvnprdmod  -c mvnprodcorrprb.o mvnprodcorrprb_interface.f --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71

C      module mvnprdmod
C      contains

	use mvnProdCorrPrbMod, ONLY : mvnprodcorrprb
	integer, parameter :: N = 2
      double precision,dimension(N) :: rho,a,b
      double precision :: abseps,releps
      logical :: useBreakPoints,useSimpson
      double precision :: abserr,prb
      integer :: IFT
	
Cf2py	depend(rho)  N
Cf2py	intent(hide) :: N = len(rho) 
Cf2py	depend(N)  a 
Cf2py	depend(N)  b 
	abseps = 1.0e-3
	releps = 1.0e-3
	useBreakPoints = 1
	useSimpson = 1
	rho(:)=1.0/100000000
	a(:) = 0.0
	b(:) = 5.0

      CALL mvnprodcorrprb(rho,a,b,abseps,releps,useBreakPoints,
     &  useSimpson,abserr,IFT,prb) 
      
	print *, 'prb =', prb
	print *, 'rho =', rho
	print *, 'a =', a
	print *, 'b =', b

	print *, 'abseps =', abseps
	print *, 'releps =', releps
	print *, 'abserr =', abserr
	end program