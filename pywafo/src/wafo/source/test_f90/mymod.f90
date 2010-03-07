! f2py --fcompiler=gnu95 --compiler=mingw32 -lmsvcr71 -m mymod  -c mymod.f90

!module functionInterface
!    INTERFACE
!        FUNCTION F(Z) result (VAL)
!        DOUBLE PRECISION, INTENT(IN) :: Z
!        DOUBLE PRECISION :: VAL
!        END FUNCTION F
!    END INTERFACE
!end module functionInterface

module mod1
  integer :: i = 5
  integer :: x(4)
  double precision :: y = 5.
  double precision, dimension(2,3) :: a
  double precision, allocatable, dimension(:,:) :: b 
end module mod1
module mod
use mod1
contains
!subroutine fun(f,x1)
!real(4) :: f
!double precision, intent(in) :: x1 
!external f
!print *, "x=[",x1,"]"
!print *, "f(x)=[",dexp(x1),"]"
!end subroutine fun

subroutine foo(F)
!use functionInterface
integer k
real(4) :: f
!external f
    print *, "i=",i
    print *, "x=[",x,"]"
    print *, "a=["
    print *, "[",a(1,1),",",a(1,2),",",a(1,3),"]"
    print *, "[",a(2,1),",",a(2,2),",",a(2,3),"]"
    print *, "]"
    print *, "Setting a(1,2)=a(1,2)+3"
    a(1,2) = a(1,2)+3
    i = i +1
!    print *, F(y), y
!    call fun(2.0,y)
end subroutine foo
end module mod