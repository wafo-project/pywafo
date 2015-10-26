module hello
use types
contains
	subroutine foo(a)
    integer(intdim) :: a
              print*, "Hello from Fortran!"
              print*, "a=",a
    end subroutine foo
end module hello