

module types
	integer, parameter :: WP=4, intdim=selected_int_kind(8)
	contains
	function to_intdim(int_value)
        integer(kind=8) :: int_value
        integer(intdim) :: to_intdim
        to_intdim = int_value
    end function to_intdim
end module types
