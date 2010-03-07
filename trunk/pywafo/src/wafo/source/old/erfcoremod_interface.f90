
      module bindings
      use erfcoremod
      contains
      function pyderf(x) result (value) ! in :erfcore:erfcore.f
      double precision intent(in) :: x
      double precision :: value
      value = derf(x)
      return
      end function pyderf
      end module bindings