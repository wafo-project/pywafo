module bindings
use types
use hello
contains
    subroutine pyfoo(a)
        integer(kind=8) :: a
        call foo(to_intdim(a))
	end subroutine pyfoo
end module bindings