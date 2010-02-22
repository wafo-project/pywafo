      MODULE JACOBMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: JACOB 
      INTERFACE JACOB
      MODULE PROCEDURE JACOB
      END INTERFACE 
      CONTAINS
      FUNCTION JACOB ( xd,xc) RESULT (value1)
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:),INTENT(in) :: xd ,xc
      DOUBLE PRECISION :: value1
                        ! default
      value1 = ABS(PRODUCT(xd))  
      ! Other possibilities given below:
      !         value1 = 1.d0
      !         value1 = ABS(PRODUCT(xd)*PRODUCT(xc))  
      RETURN
      END FUNCTION JACOB
      END MODULE JACOBMOD