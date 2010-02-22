! INTMODULE contains the modules:
!        - ADAPTMOD
!        - RCRUDEMOD
!        - KROBOVMOD
!        - KRBVRCMOD
!        - DKBVRCMOD
!
! which contains several different Multidimensional Integration Subroutines
!
! See descriptions below
!
*  ADAPTMOD is a module  containing a:
* 
*   Adaptive Multidimensional Integration Subroutine
*
*   Author: Alan Genz
*           Department of Mathematics
*           Washington State University
*           Pullman, WA 99164-3113 USA
*
* Revised pab 21.11.2000
*  A bug found by Igor in dksmrc: VK was not correctly randomized
*  is now fixed
* Revised pab 07.10.2000, 
*    1) Removed LENWRK and WORK from input in ADAPT. 
*    2) Defined LENWRK internally and Put a save statement before WORK instead 
*    3) Bug fix in ADBASE: DIVAXN was undetermined when MINCLS<0. Solution:
*         put a save statement on DIVAXN in order to save/keep its last value.
*    4) MAXDIM is now a global variable defining the maximum number of dimensions
*       it is possible to integrate.
*
* revised pab 07.09.2000
*  - solaris compiler complained on the DATA statements
*    for the P and C matrices in the krbvrc and krobov routine
*    => made separate DATA statements for P and C and moved them
*      to right after the variable definitions.      
* revised pab 10.03.2000
*   - updated to f90 (i.e. changed to assumed shape arrays + changing integers to DBLE)
*   - put it into a module
*
*  This subroutine computes an approximation to the integral
*
*      1 1     1
*     I I ... I       FUNCTN(NDIM,X)  dx(NDIM)...dx(2)dx(1)
*      0 0     0  
*
***************  Parameters for SADAPT  ********************************
*
********Input  Parameters
*
*     N      INTEGER, the number of variables.
*     MAXPTS INTEGER, maximum number of function values allowed. This 
*            parameter can be used to limit the time taken. A 
*            sensible strategy is to start with MAXPTS = 1000*N, and then
*            increase MAXPTS if ERROR is too large.
*    FUNCTN  Externally declared real user defined integrand. Its 
*            parameters must be (N, Z), where Z is a real array of
*            length N.
*     ABSEPS REAL absolute error tolerance.
*     RELEPS REAL relative error tolerance.
*
*******Output  Parameters
*
*     ERROR  REAL estimated absolute error, with 99% confidence level.
*     VALUE  REAL estimated value for the integral
*     INFORM INTEGER, termination status parameter:
*            if INFORM = 0, normal completion with ERROR < EPS;
*            if INFORM = 1, completion with ERROR > EPS and MAXPTS 
*                           function vaules used; increase MAXPTS to 
*                           decrease ERROR;
*            if INFORM = 2, N > 20 or N < 1.
*
***************  Parameters for ADAPT  ********************************
*
****** Input Parameters
*
*  NDIM    Integer number of integration variables.
*  MINCLS  Integer minimum number of FUNCTN calls to be allowed; MINCLS
*          must not exceed MAXCLS. If MINCLS < 0, then ADAPT assumes
*          that a previous call of ADAPT has been made with the same
*          integrand and continues that calculation.
*  MAXCLS  Integer maximum number of FUNCTN calls to be used; MAXCLS
*          must be >= RULCLS, the number of function calls required for
*          one application of the basic integration rule.
*           IF ( NDIM .EQ. 1 ) THEN
*              RULCLS = 11
*           ELSE IF ( NDIM .LT. 15 ) THEN
*              RULCLS = 2**NDIM + 2*NDIM*(NDIM+3) + 1
*           ELSE
*              RULCLS = 1 + NDIM*(24-NDIM*(6-NDIM*4))/3
*           ENDIF
*  FUNCTN  Externally declared real user defined integrand. Its 
*          parameters must be (NDIM, Z), where Z is a real array of
*          length NDIM.
*  ABSREQ  Real required absolute accuracy.
*  RELREQ  Real required relative accuracy.
*
****** Output Parameters
*
*  MINCLS  Actual number of FUNCTN calls used by ADAPT.
*  ABSEST  Real estimated absolute accuracy.
*  FINEST  Real estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when ABSEST <= ABSREQ or
*                     ABSEST <= |FINEST|*RELREQ with MINCLS <= MAXCLS.
*          INFORM = 1 if MAXCLS was too small for ADAPT to obtain the
*                     result FINEST to within the requested accuracy.
*          INFORM = 2 if MINCLS > MAXCLS, LENWRK < 16*NDIM + 27 or 
*                     RULCLS > MAXCLS.
*
*
*
* ADAPT revised by pab 07.10.2000, 
*    1) Removed LENWRK and WORK from input. 
*    2) Defined LENWRK internally and Put a save statement before WORK instead 
*
*  WORK    Real array (length LENWRK) of working storage. This contains
*          information that is needed for additional calls of ADAPT
*          using the same integrand (input MINCLS < 0).
*  LENWRK  Integer length of real array WORK (working storage); ADAPT
*          needs LENWRK >= 16*NDIM + 27. For maximum efficiency LENWRK
*          should be about 2*NDIM*MAXCLS/RULCLS if MAXCLS FUNCTN
*          calls are needed. If LENWRK is significantly less than this,
*          ADAPT may be less efficient.
      MODULE ADAPTMOD
      IMPLICIT NONE
      INTEGER,PRIVATE, PARAMETER :: MAXDIM=20
      PRIVATE
      PUBLIC :: ADAPT, SADAPT

      INTERFACE SADAPT
      MODULE PROCEDURE SADAPT
      END INTERFACE
 
      INTERFACE ADAPT
      MODULE PROCEDURE ADAPT
      END INTERFACE
      
      INTERFACE ADBASE
      MODULE PROCEDURE ADBASE
      END INTERFACE
       
      INTERFACE  BSINIT
      MODULE PROCEDURE BSINIT
      END INTERFACE

      INTERFACE  RULNRM
      MODULE PROCEDURE RULNRM
      END INTERFACE

      INTERFACE  DIFFER
      MODULE PROCEDURE DIFFER
      END INTERFACE

      INTERFACE  BASRUL
      MODULE PROCEDURE BASRUL
      END INTERFACE

      INTERFACE FULSUM
      MODULE  PROCEDURE FULSUM
      END INTERFACE
      
      INTERFACE  TRESTR
      MODULE PROCEDURE TRESTR
      END INTERFACE
                                !--------------------------------
      CONTAINS   

!***********************************************************
!    MAIN INTEGRATION ROUTINE SADAPT
!***********************************************************  

      SUBROUTINE SADAPT(N,MAXPTS,FUNCTN,ABSEPS,
     &     RELEPS,ERROR,VALUE,INFORM)
      IMPLICIT NONE
*
*     A subroutine for computing multivariate integrals 
*     This subroutine uses an algorithm given in the paper
*     "Numerical Computation of Multivariate Normal Probabilities", in
*     J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
*          Alan Genz 
*          Department of Mathematics
*          Washington State University 
*          Pullman, WA 99164-3113
*          Email : alangenz@wsu.edu
*
* revised pab 15.03.2000
*   - changed name from SADMVN to SADAPT
*   - Made it general for any integral not just the multivariate normal integral
*
********Input  Parameters
*
*     N      INTEGER, the number of variables.
*     MAXPTS INTEGER, maximum number of function values allowed. This 
*            parameter can be used to limit the time taken. A 
*            sensible strategy is to start with MAXPTS = 1000*N, and then
*            increase MAXPTS if ERROR is too large.
*    FUNCTN  Externally declared real user defined integrand. Its 
*            parameters must be (N, Z), where Z is a real array of
*            length N.
*     ABSEPS REAL absolute error tolerance.
*     RELEPS REAL relative error tolerance.
*
*******Output  Parameters
*
*     ERROR  REAL estimated absolute error, with 99% confidence level.
*     VALUE  REAL estimated value for the integral
*     INFORM INTEGER, termination status parameter:
*            if INFORM = 0, normal completion with ERROR < EPS;
*            if INFORM = 1, completion with ERROR > EPS and MAXPTS 
*                           function vaules used; increase MAXPTS to 
*                           decrease ERROR;
*            if INFORM = 2, N > 20 or N < 1.
*
      INTEGER, INTENT(IN)  :: N,  MAXPTS
      INTEGER, INTENT(OUT) :: INFORM
      !INTEGER ::  NL, LENWRK, 
      INTEGER :: RULCLS, TOTCLS, NEWCLS, MAXCLS
      DOUBLE PRECISION, INTENT(IN)  :: ABSEPS, RELEPS
      DOUBLE PRECISION, INTENT(OUT) :: ERROR, VALUE
      DOUBLE PRECISION :: OLDVAL
      !PARAMETER ( NL = 20 )
      !PARAMETER ( LENWRK = 20*NL**2 )
      !DOUBLE PRECISION, DIMENSION(LENWRK) :: WORK
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      IF ( N .GT. MAXDIM .OR. N .LT. 1 ) THEN
         INFORM = 2
         VALUE = 0.d0
         ERROR = 1.d0
         RETURN
      ENDIF 
      INFORM = 1
*
*     Call the subregion adaptive integration subroutine
*     
      RULCLS = 1
      CALL ADAPT( N, RULCLS, 0, FUNCTN, ABSEPS, RELEPS, 
     &      ERROR, VALUE, INFORM )
      MAXCLS = MIN( 10*RULCLS, MAXPTS )
      TOTCLS = 0
      CALL ADAPT(N, TOTCLS, MAXCLS, FUNCTN, ABSEPS, RELEPS, 
     &     ERROR, VALUE, INFORM)
      IF ( ERROR .GT. MAX( ABSEPS, RELEPS*ABS(VALUE) ) ) THEN
 10      OLDVAL = VALUE
         MAXCLS = MAX( 2*RULCLS,MIN(INT(3*MAXCLS/2),MAXPTS-TOTCLS))
         NEWCLS = -1
         CALL ADAPT(N, NEWCLS, MAXCLS, FUNCTN, ABSEPS, RELEPS, 
     &        ERROR, VALUE, INFORM)
         TOTCLS = TOTCLS + NEWCLS
         ERROR = ABS(VALUE-OLDVAL) + 
     &        SQRT(DBLE(RULCLS)*ERROR**2/DBLE(TOTCLS))
         IF ( ERROR .GT. MAX( ABSEPS, RELEPS*ABS(VALUE) ) ) THEN
            IF ( MAXPTS - TOTCLS .GT. 2*RULCLS ) GO TO 10
         ELSE 
            INFORM = 0
         END IF
      ENDIF
      
      END SUBROUTINE SADAPT



!***********************************************************
!    MAIN INTEGRATION ROUTINE ADAPT
!***********************************************************  


      SUBROUTINE ADAPT(NDIM, MINCLS, MAXCLS, FUNCTN,
     &     ABSREQ, RELREQ, ABSEST, FINEST, INFORM)
      IMPLICIT NONE      
*
*   Adaptive Multidimensional Integration Subroutine
*
*   Author: Alan Genz
*           Department of Mathematics
*           Washington State University
*           Pullman, WA 99164-3113 USA
*
*  This subroutine computes an approximation to the integral
*
*      1 1     1
*     I I ... I       FUNCTN(NDIM,X)  dx(NDIM)...dx(2)dx(1)
*      0 0     0  
*
***************  Parameters for ADAPT  ********************************
*
****** Input Parameters
*
*  NDIM    Integer number of integration variables.
*  MINCLS  Integer minimum number of FUNCTN calls to be allowed; MINCLS
*          must not exceed MAXCLS. If MINCLS < 0, then ADAPT assumes
*          that a previous call of ADAPT has been made with the same
*          integrand and continues that calculation.
*  MAXCLS  Integer maximum number of FUNCTN calls to be used; MAXCLS
*          must be >= RULCLS, the number of function calls required for
*          one application of the basic integration rule.
*           IF ( NDIM .EQ. 1 ) THEN
*              RULCLS = 11
*           ELSE IF ( NDIM .LT. 15 ) THEN
*              RULCLS = 2**NDIM + 2*NDIM*(NDIM+3) + 1
*           ELSE
*              RULCLS = 1 + NDIM*(24-NDIM*(6-NDIM*4))/3
*           ENDIF
*  FUNCTN  Externally declared real user defined integrand. Its 
*          parameters must be (NDIM, Z), where Z is a real array of
*          length NDIM.
*  ABSREQ  Real required absolute accuracy.
*  RELREQ  Real required relative accuracy.
*
****** Output Parameters
*
*  MINCLS  Actual number of FUNCTN calls used by ADAPT.
*  ABSEST  Real estimated absolute accuracy.
*  FINEST  Real estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when ABSEST <= ABSREQ or
*                     ABSEST <= |FINEST|*RELREQ with MINCLS <= MAXCLS.
*          INFORM = 1 if MAXCLS was too small for ADAPT to obtain the
*                     result FINEST to within the requested accuracy.
*          INFORM = 2 if MINCLS > MAXCLS, LENWRK < 16*NDIM + 27 or 
*                     RULCLS > MAXCLS.
*
************************************************************************
*
*     Begin driver routine. This routine partitions the working storage 
*      array and then calls the main subroutine ADBASE.
*
* Revised pab 07.10.2000, 
*    1) Removed LENWRK and WORK from input. 
*    2) Defined LENWRK internally and Put a save statement before WORK instead 
*                          
*  LENWRK  Integer length of real array WORK (working storage); ADAPT
*          needs LENWRK >= 16*NDIM + 27. For maximum efficiency LENWRK
*          should be about 2*NDIM*MAXCLS/RULCLS if MAXCLS FUNCTN
*          calls are needed. If LENWRK is significantly less than this,
*          ADAPT may be less efficient.
*
*  WORK    Real array (length LENWRK) of working storage. This contains
*          information that is needed for additional calls of ADAPT
*          using the same integrand (input MINCLS < 0).
*
      INTEGER, INTENT(IN)    :: NDIM,  MAXCLS 
      INTEGER, INTENT(INOUT) :: MINCLS
      INTEGER, INTENT(OUT)   :: INFORM
      DOUBLE PRECISION, INTENT(IN)  :: ABSREQ, RELREQ 
      DOUBLE PRECISION, INTENT(OUT) :: ABSEST, FINEST
*     Local variables	 
      INTEGER, PARAMETER :: LENWRK=20*MAXDIM*MAXDIM 
      DOUBLE PRECISION, DIMENSION(LENWRK) :: WORK        ! length lenwrk
      DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: POINTS,WEGHTS,LUM
      INTEGER :: SBRGNS, MXRGNS, RULCLS, LENRUL, 
     & INERRS, INVALS, INPTRS, INLWRS, INUPRS, INMSHS, INPNTS, INWGTS, 
     & INLOWR, INUPPR, INWDTH, INMESH, INWORK
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      SAVE WORK
!      print *,'adapt, ndim', ndim 
      IF ( NDIM .EQ. 1 ) THEN
         LENRUL = 5
         RULCLS = 9
      ELSE IF ( NDIM .LT. 12 ) THEN
         LENRUL = 6
         RULCLS = 2**NDIM + 2*NDIM*(NDIM+2) + 1
      ELSE
         LENRUL = 6
!         RULCLS = 1 + 2*NDIM*(1+2*NDIM)  ! old call pab 15.03.2003
         RULCLS = 1851 + 2*NDIM*(1+2*NDIM)
      ENDIF
      IF ( LENWRK .GE. LENRUL*(NDIM+4) + 10*NDIM + 3 .AND.
     &     RULCLS. LE. MAXCLS .AND. MINCLS .LE. MAXCLS ) THEN
         MXRGNS = ( LENWRK - LENRUL*(NDIM+4) - 7*NDIM )/( 3*NDIM + 3 )
         INERRS = 1
         INVALS = INERRS + MXRGNS
         INPTRS = INVALS + MXRGNS
         INLWRS = INPTRS + MXRGNS
         INUPRS = INLWRS + MXRGNS*NDIM
         INMSHS = INUPRS + MXRGNS*NDIM
         INWGTS = INMSHS + MXRGNS*NDIM
         INPNTS = INWGTS + LENRUL*4
         INLOWR = INPNTS + LENRUL*NDIM
         INUPPR = INLOWR + NDIM
         INWDTH = INUPPR + NDIM
         INMESH = INWDTH + NDIM
         INWORK = INMESH + NDIM
          
         ALLOCATE(POINTS(NDIM,LENRUL))
         ALLOCATE(WEGHTS(LENRUL,4))
         ALLOCATE(LUM(NDIM,MXRGNS*3))

         IF (MINCLS .LT. 0 ) THEN
            SBRGNS = WORK(LENWRK)
            LUM    = reshape(WORK(INLWRS:INWGTS-1),(/ NDIM,MXRGNS*3/))
            WEGHTS = reshape(WORK(INWGTS:INPNTS-1),(/ LENRUL , 4 /))
            POINTS = reshape(WORK(INPNTS:INLOWR-1),(/ NDIM, LENRUL/))
         !ELSE
         !   WORK=0.D0;LUM=0.D0;WEGHTS=0.D0;POINTS=0.D0
         ENDIF
         CALL ADBASE(NDIM, MINCLS, MAXCLS, FUNCTN, ABSREQ, RELREQ, 
     &        ABSEST, FINEST, SBRGNS, MXRGNS, RULCLS, LENRUL, 
     &        WORK(INERRS:INVALS-1), WORK(INVALS:INPTRS-1), 
     &        WORK(INPTRS:INLWRS-1), LUM(:,1:MXRGNS), 
     &        LUM(:,MXRGNS+1:2*MXRGNS),LUM(:,2*MXRGNS+1:3*MXRGNS), 
     &        WEGHTS,POINTS,WORK(INLOWR:INUPPR-1),WORK(INUPPR:INWDTH-1), 
     &        WORK(INWDTH:INMESH-1), WORK(INMESH:INWORK-1), 
     &        WORK(INWORK:INWORK+2*NDIM-1), INFORM)
         WORK(LENWRK) = SBRGNS
! LUM = LOWERS UPPERS MESHES
         WORK(INLWRS:INWGTS-1) = reshape(LUM   ,(/ NDIM*MXRGNS*3/))  
         WORK(INWGTS:INPNTS-1) = reshape(WEGHTS,(/ LENRUL*4 /))
         WORK(INPNTS:INLOWR-1) = reshape(POINTS,(/ NDIM*LENRUL/))
         DEALLOCATE(POINTS)
         DEALLOCATE(WEGHTS)
         DEALLOCATE(LUM)
      ELSE
         INFORM = 2
         MINCLS = RULCLS
      ENDIF
      RETURN
      END SUBROUTINE ADAPT
      SUBROUTINE BSINIT(NDIM, W, LENRUL, G)
      IMPLICIT NONE
*
*     For initializing basic rule weights and symmetric sum parameters.
*
      INTEGER, INTENT(IN) :: NDIM, LENRUL
      DOUBLE PRECISION , DIMENSION(:,:), INTENT(OUT) :: W, G
*      DOUBLE PRECISION W(LENRUL,4), G(NDIM,LENRUL) 
*    Local variables
      INTEGER :: I, J
      INTEGER, PARAMETER :: NUMNUL=4, SDIM=12
      INTEGER, DIMENSION(6) ::  RULPTS
      DOUBLE PRECISION LAM1, LAM2, LAM3, LAM4, LAMP, RULCON
*
*     The following code determines rule parameters and weights for a
*      degree 7 rule (W(1,1),...,W(5,1)), two degree 5 comparison rules
*      (W(1,2),...,W(5,2) and W(1,3),...,W(5,3)) and a degree 3 
*      comparison rule (W(1,4),...W(5,4)).
*
*       If NDIM = 1, then LENRUL = 5 and total points = 9.
*       If NDIM < SDIM, then LENRUL = 6 and
*                      total points = 1+2*NDIM*(NDIM+2)+2**NDIM.
*       If NDIM > = SDIM, then LENRUL = 6 and
*                      total points = 1+2*NDIM*(1+2*NDIM).
*
!      print *,'BSINIT, ndim', ndim 
!      DO I = 1,LENRUL
!         DO J = 1,NDIM
!            G(J,I) = 0.d0
!         END DO
!         DO J = 1,NUMNUL
!            W(I,J) = 0.d0
!         END DO
!      END DO
      G = 0.D0
      W = 0.D0
      I = 2*NDIM
      RULPTS(5) = I*(NDIM-1)
      RULPTS(4) = I 
      RULPTS(3) = I
      RULPTS(2) = I
      RULPTS(1) = 1
      LAMP = 0.85d0
      LAM3 = 0.4707d0
      LAM2 = 4d0/(15.d0 - 5.d0/LAM3)
      LAM4 = 1.D0/(27.D0*LAM3*LAM3*LAM3)
      W(5,1) = ( 3.d0 - 5.d0*LAM3 )/( 180.d0*(LAM2-LAM3)*LAM2*LAM2)
      IF ( NDIM .LT. SDIM ) THEN 
         RULPTS(LENRUL) = 2**NDIM
         LAM1 = 8.d0*LAM3*(31.d0*LAM3-15.d0)/
     &        ( (3.d0*LAM3-1.d0)*(5.d0*LAM3-3.d0)*35.d0 )
         W(LENRUL,1) = LAM4/DBLE(RULPTS(LENRUL))
      ELSE
         LAM1 = ( LAM3*(15.d0 - 21.d0*LAM2) + 
     &        35.d0*DBLE(NDIM-1)*(LAM2-LAM3)/9.d0 )
     &       /  ( LAM3*(21.d0 - 35.d0*LAM2) + 
     &        35.d0*DBLE(NDIM-1)*(LAM2/LAM3-1.d0)/9.d0 )
         W(6,1) = LAM4*0.25D0  
         RULPTS(6) = 2*NDIM*(NDIM-1)
      ENDIF
      W(3,1) = ( 15.d0 - 21.d0*(LAM3+LAM1) + 35.d0*LAM3*LAM1 )
     &   /(210.d0*LAM2*(LAM2-LAM3)*(LAM2-LAM1))-DBLE(2*(NDIM-1))*W(5,1)
      W(2,1) = ( 15.d0 - 21.d0*(LAM3+LAM2) + 35.d0*LAM3*LAM2 )
     &     /( 210.d0*LAM1*(LAM1-LAM3)*(LAM1-LAM2) )
      LAM3 = SQRT(LAM3)
      IF ( NDIM .LT. SDIM ) THEN             
          G(1:NDIM,LENRUL) = LAM3
      ELSE
          G(1,6) = LAM3
          G(2,6) = LAM3
      ENDIF
      IF ( NDIM .GT. 1 ) THEN
         W(5,2) = 1.d0/(6.d0*LAM2)**2 
         W(5,3) = W(5,2) 
      ENDIF
      W(3,2) = ( 3.d0 - 5.d0*LAM1 )/( 30.d0*LAM2*(LAM2-LAM1) ) 
     &     - DBLE(2*(NDIM-1))*W(5,2) 
      W(2,2) = ( 3.d0 - 5.d0*LAM2 )/( 30.d0*LAM1*(LAM1-LAM2) )
      W(4,3) = ( 3.d0 - 5.d0*LAM2 )/( 30.d0*LAMP*(LAMP-LAM2) )
      W(3,3) = ( 3.d0 - 5.d0*LAMP )/( 30.d0*LAM2*(LAM2-LAMP) ) 
     &     - DBLE(2*(NDIM-1))*W(5,3)
      W(2,4) = 1.d0/(6.d0*LAM1)
      LAMP = SQRT(LAMP)
      LAM2 = SQRT(LAM2)
      LAM1 = SQRT(LAM1)
      G(1,2) = LAM1
      G(1,3) = LAM2
      G(1,4) = LAMP
      IF ( NDIM .GT. 1 ) THEN
         G(1,5) = LAM2
         G(2,5) = LAM2
      ENDIF
      DO J = 1, NUMNUL
         W(1,J) = 1.d0
         DO I = 2,LENRUL
            W(1,J) = W(1,J) - DBLE(RULPTS(I))*W(I,J)
         END DO
      END DO
      RULCON = 0.5d0
      CALL RULNRM( LENRUL, NUMNUL, RULPTS, W, RULCON )
      END SUBROUTINE BSINIT
!
!
      SUBROUTINE RULNRM( LENRUL, NUMNUL, RULPTS, W, RULCON )
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: LENRUL, NUMNUL
      INTEGER, DIMENSION(:), INTENT(IN) :: RULPTS
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(INOUT) :: W       !(LENRUL, *),
      DOUBLE PRECISION, INTENT(IN) ::  RULCON
*     Local variables
      INTEGER          :: I, J, K
      DOUBLE PRECISION :: ALPHA, NORMCF, NORMNL
      
*
*     Compute orthonormalized null rules.
*
!      print *,'RULNRM, lenrul, numnul', lenrul,NUMNUL 
      NORMCF = 0.d0
      DO I = 1,LENRUL
         NORMCF = NORMCF + DBLE(RULPTS(I))*W(I,1)*W(I,1)
      END DO
      DO K = 2,NUMNUL
         DO I = 1,LENRUL
            W(I,K) = W(I,K) - W(I,1)
         END DO
         DO J = 2,K-1
            ALPHA = 0.d0
            DO I = 1,LENRUL
               ALPHA = ALPHA + DBLE(RULPTS(I))*W(I,J)*W(I,K)
            END DO
            ALPHA = -ALPHA/NORMCF
            DO I = 1,LENRUL
               W(I,K) = W(I,K) + ALPHA*W(I,J)
            END DO
         END DO
         NORMNL = 0.d0
         DO I = 1,LENRUL
            NORMNL = NORMNL + DBLE(RULPTS(I))*W(I,K)*W(I,K)
         END DO
         ALPHA = SQRT(NORMCF/NORMNL)
         DO I = 1,LENRUL
            W(I,K) = ALPHA*W(I,K)
         END DO
      END DO
      DO J = 2, NUMNUL
         DO I = 1,LENRUL
            W(I,J) = W(I,J)*RULCON
         END DO
      END DO
      RETURN
      END SUBROUTINE RULNRM
!
!
      SUBROUTINE ADBASE(NDIM, MINCLS, MAXCLS, FUNCTN, ABSREQ, RELREQ,
     &     ABSEST, FINEST, SBRGNS, MXRGNS, RULCLS, LENRUL,
     &     ERRORS, VALUES, PONTRS, LOWERS, 
     &     UPPERS, MESHES, WEGHTS, POINTS, 
     &     LOWER, UPPER, WIDTH, MESH, WORK, INFORM)
      IMPLICIT NONE
*
*        Main adaptive integration subroutine
*
      INTEGER,INTENT(IN) :: NDIM,  MAXCLS, MXRGNS,LENRUL, RULCLS
      INTEGER, INTENT(INOUT) :: MINCLS, SBRGNS  
      INTEGER, INTENT(OUT) :: INFORM 
      DOUBLE PRECISION, INTENT(IN) :: ABSREQ, RELREQ
      DOUBLE PRECISION, INTENT(OUT) :: ABSEST, FINEST  
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) ::  ERRORS, VALUES, 
     &   PONTRS, LOWER, UPPER, WIDTH, MESH, WORK
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(INOUT) :: WEGHTS, POINTS
      ! shape (LENRUL,4) and (NDIM,LENRUL)
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(INOUT) :: LOWERS, UPPERS,
     &   MESHES     !SHAPE  (NDIM,MXRGNS),   
      INTEGER :: I, J,NWRGNS,  DIVAXN, TOP, RGNCLS, FUNCLS, DIFCLS
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
*
*     Initialization of subroutine
*
!      print *,'ADBASE, ndim', ndim, shape(POINTS) 
      SAVE DIVAXN               ! added pab 07.11.2000 (divaxn may have negative values otherwise)
      INFORM = 2
      FUNCLS = 0
      CALL BSINIT(NDIM, WEGHTS, LENRUL, POINTS)
      IF ( MINCLS .GE. 0) THEN
*
*       When MINCLS >= 0 determine initial subdivision of the
*       integration region and apply basic rule to each subregion.
*
         SBRGNS = 0
         DO I = 1,NDIM
            LOWER(I) = 0.d0
            MESH(I) = 1.d0
            WIDTH(I) = 1.d0/(2.d0*MESH(I))
            UPPER(I) = 1.d0
         END DO
         DIVAXN = 0
         RGNCLS = RULCLS
         NWRGNS = 1
 10      CONTINUE
         !IF (abs(DIVAXN).GT.NDIM)   PRINT *,'adbase DIVAXN1',DIVAXN
         CALL DIFFER(NDIM, LOWER, UPPER, WIDTH, WORK(1:NDIM),  
     &        WORK(NDIM+1:2*NDIM), FUNCTN, DIVAXN, DIFCLS)
         FUNCLS = FUNCLS + DIFCLS
         IF (DBLE(RGNCLS)*(MESH(DIVAXN)+1.d0)/MESH(DIVAXN)
     &        .LE. DBLE(MINCLS-FUNCLS) ) THEN
            RGNCLS = NINT(DBLE(RGNCLS)*(MESH(DIVAXN)+1.d0)/MESH(DIVAXN))
            NWRGNS = NINT(DBLE(NWRGNS)*(MESH(DIVAXN)+1.d0)/MESH(DIVAXN))
            MESH(DIVAXN) = MESH(DIVAXN) + 1.d0
            WIDTH(DIVAXN) = 1.d0/( 2.d0*MESH(DIVAXN) )
            GO TO 10
         ENDIF
         IF ( NWRGNS .LE. MXRGNS ) THEN
            DO I = 1,NDIM
               UPPER(I) = LOWER(I) + 2.d0*WIDTH(I)
               MESH(I) = 1.d0
            END DO
         ENDIF
*     
*     Apply basic rule to subregions and store results in heap.
*     
 20      SBRGNS = SBRGNS + 1
         CALL BASRUL(NDIM, LOWER, UPPER, WIDTH, FUNCTN, 
     &       WEGHTS, LENRUL, POINTS, WORK(1:NDIM), WORK(NDIM+1:2*NDIM),
     &       ERRORS(SBRGNS),VALUES(SBRGNS))
         CALL TRESTR(SBRGNS, SBRGNS, PONTRS, ERRORS)
         DO I = 1,NDIM
            LOWERS(I,SBRGNS) = LOWER(I)
            UPPERS(I,SBRGNS) = UPPER(I)
            MESHES(I,SBRGNS) = MESH(I)
         END DO
         DO I = 1,NDIM
            LOWER(I) = UPPER(I)
            UPPER(I) = LOWER(I) + 2.d0*WIDTH(I)
            IF (LOWER(I)+WIDTH(I) .LT. 1.D0)  GO TO 20
            LOWER(I) = 0.d0
            UPPER(I) = LOWER(I) + 2.d0*WIDTH(I)
         END DO
         FUNCLS = FUNCLS + SBRGNS*RULCLS
      ENDIF
*     
*     Check for termination
*
 30   FINEST = 0.d0
      ABSEST = 0.d0
      DO I = 1, SBRGNS
         FINEST = FINEST + VALUES(I)
         ABSEST = ABSEST + ERRORS(I)
      END DO
      IF ( ABSEST .GT. MAX( ABSREQ, RELREQ*ABS(FINEST) )
     &     .OR. FUNCLS .LT. MINCLS ) THEN  
*     
*     Prepare to apply basic rule in (parts of) subregion with
*     largest error.
*     
         TOP = PONTRS(1)
         RGNCLS = RULCLS
         DO I = 1,NDIM
            LOWER(I) = LOWERS(I,TOP)
            UPPER(I) = UPPERS(I,TOP)
            MESH(I) = MESHES(I,TOP)
            WIDTH(I) = (UPPER(I)-LOWER(I))/(2.D0*MESH(I))
            RGNCLS = NINT(DBLE(RGNCLS)*MESH(I))
         END DO
        !IF (abs(DIVAXN).GT.NDIM)   PRINT *,'adbase DIVAXN2',DIVAXN
         CALL DIFFER(NDIM, LOWER, UPPER, WIDTH, WORK(1:NDIM),   
     &       WORK(NDIM+1:2*NDIM), FUNCTN, DIVAXN, DIFCLS)
         FUNCLS = FUNCLS + DIFCLS
         RGNCLS = NINT(DBLE(RGNCLS)*(MESH(DIVAXN)+1.D0))/MESH(DIVAXN)
         IF ( FUNCLS + RGNCLS .LE. MAXCLS ) THEN
            IF ( SBRGNS + 1 .LE. MXRGNS ) THEN
*     
*     Prepare to subdivide into two pieces.
*    
               NWRGNS = 1
               WIDTH(DIVAXN) = 0.5d0*WIDTH(DIVAXN)
            ELSE
               NWRGNS = 0
               WIDTH(DIVAXN) = WIDTH(DIVAXN)
     &                        *MESH(DIVAXN)/( MESH(DIVAXN) + 1.d0 )
               MESHES(DIVAXN,TOP) = MESH(DIVAXN) + 1.d0 
            ENDIF
            IF ( NWRGNS .GT. 0 ) THEN
*     
*     Only allow local subdivision when space is available.
*
               DO J = SBRGNS+1,SBRGNS+NWRGNS
                  DO I = 1,NDIM
                     LOWERS(I,J) = LOWER(I)
                     UPPERS(I,J) = UPPER(I)
                     MESHES(I,J) = MESH(I)
                  END DO
               END DO
               UPPERS(DIVAXN,TOP) = LOWER(DIVAXN) + 2.d0*WIDTH(DIVAXN)
               LOWERS(DIVAXN,SBRGNS+1) = UPPERS(DIVAXN,TOP)
            ENDIF
            FUNCLS = FUNCLS + RGNCLS
            CALL BASRUL(NDIM, LOWERS(:,TOP), UPPERS(:,TOP), WIDTH, 
     &           FUNCTN, WEGHTS, LENRUL, POINTS, WORK(1:NDIM),  
     &           WORK(NDIM+1:2*NDIM),ERRORS(TOP), VALUES(TOP))
            CALL TRESTR(TOP, SBRGNS, PONTRS, ERRORS)
            DO I = SBRGNS+1, SBRGNS+NWRGNS
*     
*     Apply basic rule and store results in heap.
*     
               CALL BASRUL(NDIM, LOWERS(:,I), UPPERS(:,I), WIDTH,
     &              FUNCTN, WEGHTS, LENRUL, POINTS, WORK(1:NDIM),   
     &              WORK(NDIM+1:2*NDIM),ERRORS(I), VALUES(I))
               CALL TRESTR(I, I, PONTRS, ERRORS)
            END DO
            SBRGNS = SBRGNS + NWRGNS
            GO TO 30
         ELSE
            INFORM = 1
         ENDIF
      ELSE
         INFORM = 0
      ENDIF
      MINCLS = FUNCLS
      RETURN 
      END SUBROUTINE ADBASE
      SUBROUTINE BASRUL( NDIM, A, B, WIDTH, FUNCTN, W, LENRUL, G,
     &     CENTER, Z, RGNERT, BASEST )
      IMPLICIT NONE
*
*     For application of basic integration rule
*
      INTEGER, INTENT(IN) :: LENRUL, NDIM
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(IN) :: A, B, WIDTH     !(NDIM)
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(IN) :: W               !(LENRUL,4), 
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(INOUT) :: G            !(NDIM,LENRUL), 
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(INOUT) :: CENTER, Z    !(NDIM)
      DOUBLE PRECISION, INTENT(OUT) :: RGNERT, BASEST
      INTEGER :: I
      DOUBLE PRECISION :: FSYMSM, RGNCMP, RGNVAL,
     &     RGNVOL, RGNCPT, RGNERR
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
*
*     Compute Volume and Center of Subregion
*
!      print *,'BASRULE, ndim', ndim 
      RGNVOL = 1.d0
      DO I = 1,NDIM
         RGNVOL = 2.d0*RGNVOL*WIDTH(I)
         CENTER(I) = A(I) + WIDTH(I)
      END DO
      BASEST = 0.d0
      RGNERT = 0.d0
*
*     Compute basic rule and error
*
 10   RGNVAL = 0.d0
      RGNERR = 0.d0
      RGNCMP = 0.d0
      RGNCPT = 0.d0
      DO I = 1,LENRUL
         FSYMSM = FULSUM(NDIM, CENTER, WIDTH, Z, G(:,I), FUNCTN)
*     Basic Rule
         RGNVAL = RGNVAL + W(I,1)*FSYMSM
*     First comparison rule
         RGNERR = RGNERR + W(I,2)*FSYMSM
*     Second comparison rule
         RGNCMP = RGNCMP + W(I,3)*FSYMSM
*     Third Comparison rule
         RGNCPT = RGNCPT + W(I,4)*FSYMSM
      END DO
*
*     Error estimation
*
      RGNERR = SQRT(RGNCMP*RGNCMP + RGNERR*RGNERR)
      RGNCMP = SQRT(RGNCPT*RGNCPT + RGNCMP*RGNCMP)
      IF ( 4.d0*RGNERR .LT. RGNCMP ) RGNERR = 0.5d0*RGNERR
      IF ( 2.d0*RGNERR .GT. RGNCMP ) RGNERR = MAX( RGNERR, RGNCMP )
      RGNERT = RGNERT +  RGNVOL*RGNERR
      BASEST = BASEST +  RGNVOL*RGNVAL
*
*     When subregion has more than one piece, determine next piece and
*      loop back to apply basic rule.
*
      DO I = 1,NDIM
         CENTER(I) = CENTER(I) + 2.d0*WIDTH(I)
         IF ( CENTER(I) .LT. B(I) ) GO TO 10
         CENTER(I) = A(I) + WIDTH(I)
      END DO
      RETURN
      END SUBROUTINE BASRUL
      DOUBLE PRECISION FUNCTION FULSUM(S, CENTER, HWIDTH, X, G, F)
      IMPLICIT NONE
*
****  To compute fully symmetric basic rule sum
* 
      INTEGER, INTENT(IN) :: S
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: CENTER, HWIDTH
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) :: X, G  ! shape S
      INTEGER          :: IXCHNG, LXCHNG, I, L
      DOUBLE PRECISION :: INTSUM, GL, GI
      INTERFACE
         DOUBLE PRECISION FUNCTION F(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION F
      END INTERFACE
!      print *,'FULSUM, S', S, shape(X) 
      FULSUM = 0.d0
*
*     Compute centrally symmetric sum for permutation of G
*
 10   INTSUM = 0.d0
      !DO I = 1,S
      !   X(I) = CENTER(I) + G(I)*HWIDTH(I)
      !END DO
      X = CENTER + G*HWIDTH
 20   INTSUM = INTSUM + F(S,X)
      DO I = 1,S
         G(I) = -G(I)
         X(I) = CENTER(I) + G(I)*HWIDTH(I)
         IF ( G(I) .LT. 0.d0 ) GO TO 20
      END DO
      FULSUM = FULSUM + INTSUM
*     
*     Find next distinct permuation of G and loop back for next sum
*     
      DO I = 2,S
         IF ( G(I-1) .GT. G(I) ) THEN
            GI = G(I)
            IXCHNG = I - 1
            DO L = 1,(I-1)/2
               GL = G(L)
               G(L) = G(I-L)
               G(I-L) = GL
               IF (  GL  .LE. GI ) IXCHNG = IXCHNG - 1
               IF ( G(L) .GT. GI ) LXCHNG = L
            END DO
            IF ( G(IXCHNG) .LE. GI ) IXCHNG = LXCHNG
            G(I) = G(IXCHNG)
            G(IXCHNG) = GI
            GO TO 10
         ENDIF
      END DO
*     
*     End loop for permutations of G and associated sums
*     
*     Restore original order to G's
*     
      DO I = 1,S/2
         GI = G(I)
         G(I) = G(S+1-I)
         G(S+1-I) = GI 
      END DO
      RETURN
      END FUNCTION FULSUM
      SUBROUTINE DIFFER(NDIM, A, B, WIDTH, Z, DIF, FUNCTN, 
     &     DIVAXN, DIFCLS)
      IMPLICIT NONE
*
*     Compute fourth differences and subdivision axes
*
      INTEGER, INTENT(IN)    :: NDIM
      INTEGER, INTENT(INOUT) :: DIVAXN
      INTEGER, INTENT(OUT)   :: DIFCLS
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: A, B, WIDTH   ! (NDIM)
      DOUBLE PRECISION, DIMENSION(:),INTENT(OUT) :: Z, DIF        ! (NDIM)
      DOUBLE PRECISION :: FRTHDF, FUNCEN, WIDTHI
      INTEGER :: I
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
!      print *,'DIFFER, ndim', ndim, shape(Z) 
      DIFCLS = 0
!      IF (abs(DIVAXN).GT.NDIM)   PRINT *,'DIFFER DIVAXN1',DIVAXN

      DIVAXN = MOD(DIVAXN, NDIM ) + 1
      !print *,'DIFFER, divaxn2', divaxn
      IF ( NDIM .GT. 1 ) THEN
         !DO I = 1,NDIM 
         !  DIF(I) = 0.d0
         !   Z(I) = A(I) + WIDTH(I)
         !END DO
         DIF = 0.D0
         Z(1:NDIM) = A(1:NDIM) + WIDTH(1:NDIM)
!         print *,'Z', Z
 10      FUNCEN = FUNCTN(NDIM, Z)
         DO I = 1,NDIM
            WIDTHI = 0.2d0*WIDTH(I)
            FRTHDF = 6.d0*FUNCEN
            Z(I) = Z(I) - 4.d0*WIDTHI
            FRTHDF = FRTHDF + FUNCTN(NDIM,Z)
            Z(I) = Z(I) + 2.d0*WIDTHI
            FRTHDF = FRTHDF - 4.d0*FUNCTN(NDIM,Z)
            Z(I) = Z(I) + 4.d0*WIDTHI
            FRTHDF = FRTHDF - 4.d0*FUNCTN(NDIM,Z)
            Z(I) = Z(I) + 2.d0*WIDTHI
            FRTHDF = FRTHDF + FUNCTN(NDIM,Z)
*     Do not include differences below roundoff
!            IF ( FUNCEN + FRTHDF/8.d0 .NE. FUNCEN ) 
             IF ( FUNCEN + FRTHDF*0.125D0 .NE. FUNCEN ) 
     &           DIF(I) = DIF(I) + ABS(FRTHDF)*WIDTH(I)
            Z(I) = Z(I) - 4.d0*WIDTHI
         END DO
         DIFCLS = DIFCLS + 4*NDIM + 1
         DO I = 1,NDIM
            Z(I) = Z(I) + 2.D0*WIDTH(I)
            IF ( Z(I) .LT. B(I) ) GO TO 10
            Z(I) = A(I) + WIDTH(I)
         END DO
        !IF (abs(DIVAXN).GT.NDIM)   PRINT *,'DIFFER DIVAXN',DIVAXN,shape(dif),ndim
         DO I = 1,NDIM
            IF ( DIF(DIVAXN) .LT. DIF(I) ) DIVAXN = I
         END DO
      ENDIF
      RETURN
      END SUBROUTINE DIFFER
      SUBROUTINE TRESTR(POINTR, SBRGNS, PONTRS, RGNERS)
      IMPLICIT NONE
****BEGIN PROLOGUE TRESTR
****PURPOSE TRESTR maintains a heap for subregions.
****DESCRIPTION TRESTR maintains a heap for subregions.
*            The subregions are ordered according to the size of the
*            greatest error estimates of each subregion (RGNERS).
*
*   PARAMETERS
*
*     POINTR Integer.
*            The index for the subregion to be inserted in the heap.
*     SBRGNS Integer.
*            Number of subregions in the heap.
*     PONTRS Real array of dimension SBRGNS.
*            Used to store the indices for the greatest estimated errors
*            for each subregion.
*     RGNERS Real array of dimension SBRGNS.
*            Used to store the greatest estimated errors for each 
*            subregion.
*
****ROUTINES CALLED NONE
****END PROLOGUE TRESTR
*
*   Global variables.
*
      INTEGER, INTENT(IN) ::POINTR, SBRGNS
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) :: PONTRS
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN)    :: RGNERS
*
*   Local variables.
*
*   RGNERR Intermediate storage for the greatest error of a subregion.
*   SUBRGN Position of child/parent subregion in the heap.
*   SUBTMP Position of parent/child subregion in the heap.
*
      INTEGER SUBRGN, SUBTMP
      DOUBLE PRECISION RGNERR
*
****FIRST PROCESSING STATEMENT TRESTR
*     
!      print *,'TRESTR' 
      RGNERR = RGNERS(POINTR)
      IF ( POINTR.EQ.NINT(PONTRS(1))) THEN
*
*        Move the new subregion inserted at the top of the heap 
*        to its correct position in the heap.
*
         SUBRGN = 1
 10      SUBTMP = 2*SUBRGN
         IF ( SUBTMP .LE. SBRGNS ) THEN
            IF ( SUBTMP .NE. SBRGNS ) THEN
*     
*              Find maximum of left and right child.
*
               IF ( RGNERS(NINT(PONTRS(SUBTMP))) .LT. 
     &              RGNERS(NINT(PONTRS(SUBTMP+1))) ) SUBTMP = SUBTMP + 1
            ENDIF
*
*           Compare maximum child with parent.
*           If parent is maximum, then done.
*
            IF ( RGNERR .LT. RGNERS(NINT(PONTRS(SUBTMP))) ) THEN
*     
*              Move the pointer at position subtmp up the heap.
*     
               PONTRS(SUBRGN) = PONTRS(SUBTMP)
               SUBRGN = SUBTMP
               GO TO 10
            ENDIF
         ENDIF
      ELSE
*
*        Insert new subregion in the heap.
*
         SUBRGN = SBRGNS
 20      SUBTMP = SUBRGN/2
         IF ( SUBTMP .GE. 1 ) THEN
*
*           Compare child with parent. If parent is maximum, then done.
*     
            IF ( RGNERR .GT. RGNERS(NINT(PONTRS(SUBTMP))) ) THEN
*     
*              Move the pointer at position subtmp down the heap.
*
               PONTRS(SUBRGN) = PONTRS(SUBTMP)
               SUBRGN = SUBTMP
               GO TO 20
            ENDIF
         ENDIF
      ENDIF
      PONTRS(SUBRGN) = DBLE(POINTR)
*
****END TRESTR
*
      RETURN
      END SUBROUTINE TRESTR
      END MODULE ADAPTMOD       



*  RCRUDEMOD is a module  containing two:
*
*  Automatic Multidimensional Integration Subroutines
*               
*         AUTHOR: Alan Genz
*                 Department of Mathematics
*                 Washington State University
*                 Pulman, WA 99164-3113
*                 Email: AlanGenz@wsu.edu
*
*         Last Change: 5/15/98
* revised pab 10.03.2000
*   - updated to f90 (i.e. changed to assumed shape arrays + changing integers to DBLE)
*   - put it into a module
*   - added ranlhmc
*
*  RCRUDEMOD computes an approximation to the integral
*
*      1  1     1
*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1)
*     0  0     0
! References:
! Alan Genz (1992)
! 'Numerical Computation of Multivariate Normal Probabilites'
! J. computational Graphical Statistics, Vol.1, pp 141--149              (RANMC)
!
! William H. Press, Saul Teukolsky, 
! William T. Wetterling and Brian P. Flannery (1997)
! "Numerical recipes in Fortran 77", Vol. 1, pp 55-63            (SVDCMP,PYTHAG)
!
! Donald E. Knuth (1973) "The art of computer programming,",
! Vol. 3, pp 84-  (sorting and searching)                               (SORTRE)


! You may  initialize the random generator before you 
!  call RANLHMC or RANMC by the following lines:
!
!      call random_seed(SIZE=seed_size) 
!      allocate(seed(seed_size)) 
!      call random_seed(GET=seed(1:seed_size))  ! get current seed
!      seed(1)=seed1                            ! change seed
!      call random_seed(PUT=seed(1:seed_size)) 
!      deallocate(seed)



      MODULE RCRUDEMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: RANMC
      INTEGER ::  NDIMMAX       

      INTERFACE RANMC
      MODULE PROCEDURE RANMC
      END INTERFACE
      
      INTERFACE RCRUDE
      MODULE PROCEDURE RCRUDE
      END INTERFACE

      INTERFACE SVDCMP
      MODULE PROCEDURE SVDCMP
      END INTERFACE

      INTERFACE PYTHAG
      MODULE PROCEDURE PYTHAG
      END INTERFACE

      INTERFACE SPEARCORR
      MODULE PROCEDURE SPEARCORR
      END INTERFACE
       
      INTERFACE SORTRE
      MODULE PROCEDURE SORTRE
      END INTERFACE
      
      INTERFACE BINSORT
      MODULE PROCEDURE BINSORT
      END INTERFACE

      INTERFACE SWAPRE
      MODULE PROCEDURE SWAPRE
      END INTERFACE

      INTERFACE SWAPINT
      MODULE PROCEDURE SWAPINT
      END INTERFACE

      PARAMETER (NDIMMAX=1000)
                                    !--------------------------------
      CONTAINS   
      SUBROUTINE RANMC( N, MAXPTS, FUNCTN, ABSEPS, 
     &     RELEPS, ERROR, VALUE, INFORM )
      IMPLICIT NONE
*
*     A subroutine for computing multivariate integrals.
*     This subroutine uses the Monte-Carlo algorithm given in the paper
*     "Numerical Computation of Multivariate Normal Probabilities", in
*     J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
*          Alan Genz
*          Department of Mathematics
*          Washington State University
*          Pullman, WA 99164-3113
*          Email : alangenz@wsu.edu
*
*  This subroutine computes an approximation to the integral
*
*      1 1     1
*     I I ... I       FUNCTN(NDIM,X)  dx(NDIM)...dx(2)dx(1)
*      0 0     0  
*
***************  Parameters for RANMC  ********************************
*
****** Input Parameters
*
*     N      INTEGER, the number of variables.
*     MAXPTS INTEGER, maximum number of function values allowed. This 
*            parameter can be used to limit the time taken. A 
*            sensible strategy is to start with MAXPTS = 1000*N, and then
*            increase MAXPTS if ERROR is too large.
*     ABSEPS REAL absolute error tolerance.
*     RELEPS REAL relative error tolerance.
*
****** Output Parameters
*
*     ERROR  REAL estimated absolute error, with 99% confidence level.
*     VALUE  REAL estimated value for the integral
*     INFORM INTEGER, termination status parameter:
*            if INFORM = 0, normal completion with ERROR < EPS;
*            if INFORM = 1, completion with ERROR > EPS and MAXPTS 
*                           function vaules used; increase MAXPTS to 
*                           decrease ERROR;
*            if INFORM = 2, N > 100 or N < 1.
*
      INTEGER :: N, MAXPTS, MPT, INFORM, IVLS
      DOUBLE PRECISION :: ABSEPS, RELEPS, ERROR, VALUE, EPS
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      INFORM=0
      IF ( N .GT. NDIMMAX .OR. N .LT. 1 ) THEN
         INFORM = 2
         VALUE = 0.d0
         ERROR = 1.d0
         RETURN
      ENDIF
*
*        Call then Monte-Carlo integration subroutine
*
      MPT = 25 + 10*N
      CALL RCRUDE(N, MPT, FUNCTN, ERROR, VALUE, 0)
      IVLS = MPT
 10   EPS = MAX( ABSEPS, RELEPS*ABS(VALUE) )
      IF ( ERROR .GT. EPS .AND. IVLS .LT. MAXPTS ) THEN 
         MPT = MAX( MIN( INT(MPT*(ERROR/(EPS))**2), 
     &        MAXPTS-IVLS ), 10 )
         CALL RCRUDE(N, MPT, FUNCTN, ERROR, VALUE, 1)
         IVLS = IVLS + MPT
         GO TO 10
      ENDIF
      IF ( ERROR. GT. EPS .AND. IVLS .GE. MAXPTS ) INFORM = 1
      !IF (INFORM.EQ.1) print *,'ranmc eps',EPS 
      END SUBROUTINE RANMC
      SUBROUTINE RCRUDE(NDIM, MAXPTS, FUNCTN, ABSEST, FINEST, IR)
      IMPLICIT NONE
*
*     Crude Monte-Carlo Algorithm with simple antithetic variates
*      and weighted results on restart
*
      INTEGER :: NDIM, MAXPTS, M,  IR, NPTS
      DOUBLE PRECISION :: FINEST, ABSEST, FUN, 
     &     VARSQR, VAREST, VARPRD, FINDIF, FINVAL
      DOUBLE PRECISION, DIMENSION(NDIMMAX) :: X
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      SAVE VAREST
      IF ( IR .LE. 0 ) THEN
         VAREST = 0.d0
         FINEST = 0.d0
      ENDIF
      FINVAL = 0.d0
      VARSQR = 0.d0
      NPTS = INT(MAXPTS/2)
      DO M = 1,NPTS
         CALL random_number(X(1:NDIM))
         FUN = FUNCTN(NDIM, X(1:NDIM))
         X(1:NDIM) = 1.d0 - X(1:NDIM)
         FUN = (FUNCTN(NDIM, X(1:NDIM)) + FUN )*0.5d0
         FINDIF = ( FUN - FINVAL )/DBLE(M)
         VARSQR = DBLE( M - 2 )*VARSQR/DBLE(M) + FINDIF*FINDIF 
         FINVAL = FINVAL + FINDIF
      END DO
      VARPRD = VAREST*VARSQR
      FINEST = FINEST + ( FINVAL - FINEST )/(1.d0 + VARPRD)
      IF ( VARSQR .GT. 0 ) VAREST = (1.d0 + VARPRD)/VARSQR
      ABSEST = 3.d0*SQRT( VARSQR/( 1.d0 + VARPRD ) )
      MAXPTS=2*NPTS
      END SUBROUTINE RCRUDE

      SUBROUTINE BINSORT(indices,rarray)
      IMPLICIT NONE
      TYPE ENTRY
         DOUBLE PRECISION, POINTER :: VAL
         INTEGER :: IX
         TYPE( ENTRY), POINTER :: NEXT
      END TYPE ENTRY 
      DOUBLE PRECISION, DIMENSION(:), INTENT(in) :: rarray
      INTEGER,          DIMENSION(:), INTENT(inout)  :: indices
      DOUBLE PRECISION, DIMENSION(SIZE(rarray)),TARGET  :: A
      TYPE(ENTRY), DIMENSION(:), ALLOCATABLE,TARGET  :: B      
      TYPE(ENTRY), POINTER   :: FIRST,CURRENT

! local variables
      INTEGER  :: i,im,n
      DOUBLE PRECISION :: mx, mn
! Bucket sort: 
! This subroutine sorts the indices according to rarray. The Assumption is that rarray consists of 
! uniformly distributed numbers. If the assumption holds it runs in O(n) time
      n=size(indices)
      IF (n.EQ.1) RETURN
      !indices=(/(i,i=1,n)/)
      mx = MAXVAL(rarray)
      mn = MINVAL(rarray)
      A=(rarray-mn)/(mx-mn)  ! make sure the numbers are between 0 and 1
     
      !print *,'binsort ind=',indices
      !print *,'binsort rar=',rarray
      !print *,'binsort rar=',A
      ALLOCATE(B(0:n-1))
      !IF (ASSOCIATED(B(0)%VAL)) print *,'binsort B(0)=',B(0)%VAL
      DO I=0,n-1
         NULLIFY(B(I)%VAL)
         NULLIFY(B(I)%NEXT)
      ENDDO
      
      DO I=1,n
         IM=min(ABS(FLOOR(n*A(I))),N-1)
         IF (ASSOCIATED(B(IM)%VAL)) THEN  ! insert the new item by insertion sorting
            ALLOCATE(CURRENT)
            IF (A(I).LT.B(IM)%VAL) THEN
              CURRENT = B(IM) 
              B(IM)   = ENTRY(A(I),indices(I),CURRENT)
            ELSE
               FIRST => B(IM)
               DO WHILE(ASSOCIATED(FIRST%NEXT).AND.
     &              FIRST%NEXT%VAL.LT.A(I))
                  FIRST=FIRST%NEXT
               END DO
            
               CURRENT = ENTRY(A(I),indices(I),FIRST%NEXT)
               FIRST%NEXT => CURRENT  
            ENDIF   
         ELSE
            B(IM)%VAL => A(I)
            B(IM)%IX  = indices(I)
         ENDIF
      END DO
      IM=0
      I=0 
      DO WHILE (IM.LT.N .AND. I.LT.N)
         IF (ASSOCIATED(B(I)%VAL)) THEN
            IM=IM+1
            indices(IM)=B(I)%IX
            DO WHILE (ASSOCIATED(B(I)%NEXT)) 
               CURRENT => B(I)%NEXT
               B(I)%NEXT => B(I)%NEXT%NEXT
               IM=IM+1
               indices(IM)=CURRENT%IX
               DEALLOCATE(CURRENT)
            END DO
         ENDIF
         I=I+1
      END DO
      DEALLOCATE(B)
      !print *,'binsort ind=',indices
      RETURN
      END SUBROUTINE BINSORT

      SUBROUTINE SORTRE(indices,rarray)
      IMPLICIT NONE
      DOUBLE PRECISION, DIMENSION(:), INTENT(inout) :: rarray
      INTEGER,          DIMENSION(:), INTENT(inout) :: indices
! local variables
       INTEGER  :: i,im,j,k,m,n
   
! diminishing increment sort as described by
! Donald E. Knuth (1973) "The art of computer programming,",
! Vol. 3, pp 84-  (sorting and searching)
      n=size(indices)
      ! if the below is commented out then assume indices are already initialized
      !indices=(/(i,i=1,n)/)
!100   continue
      if (n.le.1) goto 800
      m=1
200   continue
      m=m+m
      if (m.lt.n) goto 200
      m=m-1
300   continue
      m=m/2
      if (m.eq.0) goto 800
      k=n-m
      j=1
400   continue
      i=j
500   continue
      im=i+m
      if (rarray(i).gt.rarray(im)) goto 700          
600   continue
      j=j+1
      if (j.gt.k) goto 300
      goto 400
700   continue
      CALL swapre(rarray(i),rarray(im))
      CALL swapint(indices(i),indices(im))
      i=i-m
      if (i.lt.1) goto 600
      goto 500
800   continue
      RETURN   
      END SUBROUTINE SORTRE

      SUBROUTINE swapRe(m,n)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(inout) :: m,n
      DOUBLE PRECISION                :: tmp      
      tmp=m
      m=n
      n=tmp
      END SUBROUTINE swapRe
  
      SUBROUTINE swapint(m,n)
      IMPLICIT NONE
      INTEGER, INTENT(inout) :: m,n
      INTEGER                :: tmp 
      tmp=m
      m=n
      n=tmp
      END SUBROUTINE swapint
!______________________________________________________

      SUBROUTINE spearcorr(C,D)
      IMPLICIT NONE
      DOUBLE PRECISION, dimension(:,:), INTENT(out) :: C
      integer, dimension(:,:),intent(in) :: D ! rank matrix
      double precision, dimension(:,:),allocatable :: DD !,DDT
      double precision, dimension(:),allocatable :: tmp
      INTEGER             :: N,M,ix,iy      
      DOUBLE PRECISION    :: dN      
! this procedure calculates spearmans correlation coefficient
! between the columns of D 
     
      N=size(D,dim=1);M=SIZE(D,dim=2)
      dN=dble(N)
      allocate(DD(1:N,1:M))
      DD=dble(D)
!      if (.false.) then ! old call
!         allocate(DDt(1:M,1:N))
!         DDT=transpose(DD)
!         C = matmul(DDt,DD)*12.d0/(dn*(dn*dn-1.d0)) 
!         C=(C-3.d0*(dn+1.d0)/(dn-1.d0))
!         deallocate(DDT)
!      else
      allocate(tmp(1:N))
      do  ix=1, m-1
         do iy=ix+1,m
            tmp= DD(1:N,ix)-DD(1:N,iy)         
            C(ix,iy)=1.d0-6.d0*SUM(tmp*tmp)/dn/(dn*dn-1.d0)  
            C(iy,ix)=C(ix,iy)
         enddo
         C(ix,ix) = 1.d0
      enddo
      C(m,m)=1.d0
      deallocate(tmp)
!      endif
      deallocate(DD)
      return
      END SUBROUTINE spearcorr
  
      SUBROUTINE SVDCMP(A,W,V)
      IMPLICIT NONE 
      DOUBLE PRECISION, DIMENSION(:  ), INTENT(out) :: W
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(inout)  :: A
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(OUT) :: V   
!LOCAL VARIABLES
      DOUBLE PRECISION, DIMENSION(:), allocatable :: RV1   
      DOUBLE PRECISION :: G,S,SCALE,ANORM,F,H,C,X,Y,Z   
      INTEGER M,N,NM,I,J,K,L,ITS
      
      !PARAMETER (NMAX=100)
C  Maximum anticipated values of  N

C  DIMENSION A(MP,NP),W(NP),V(NP,NP),RV1(NMAX)
C  Given a matrix  A, with logical dimensions  M  by  N  and physical
C  dimensions  MP  by  NP, this routine computes its singular value
C  decomposition,  A=U.W.V^T, see Numerical Recipes, by Press W.,H.
C  Flannery, B. P., Teukolsky S.A. and Vetterling W., T. Cambrige
C  University Press 1986, Chapter 2.9. The matrix  U  replaces A  on
C  output. The diagonal matrix of singular values  W  is output as a vector
C  W. The matrix  V (not the transpose  V^T) is output as  V.  M  must be
C  greater or equal to  N; if it is smaller, then  A  should be filled up
C  to square with zero rows.
C
      
       M=size(A,dim=1);N=size(A,dim=2)
       !Mp=M;Np=N
       allocate(RV1(1:N))
      IF (M.LT.N) then
!         Print *,'SVDCMP: You must augment  A  with extra zero rows.'
      endif
C  Householder reduction to bidiagonal form
       G=0.d0
       SCALE=0.d0
       ANORM=0.d0
       DO 25 I=1,N
          L=I+1
          RV1(I)=SCALE*G
          G=0.D0
          S=0.D0
          SCALE=0.D0
          IF (I.LE.M) THEN
             DO  K=I,M
               SCALE=SCALE+ABS(A(K,I))
             enddo
             IF (SCALE.NE.0.D0) THEN
                DO  K=I,M
                  A(K,I)=A(K,I)/SCALE
                  S=S+A(K,I)*A(K,I)
                enddo
                F=A(I,I)
                G=-SIGN(SQRT(S),F)
                H=F*G-S
                A(I,I)=F-G
                IF (I.NE.N) THEN
                   DO  J=L,N
                     S=0.D0
                     DO  K=I,M
                       S=S+A(K,I)*A(K,J)
                     enddo
                     F=S/H
                     DO  K=I,M
                       A(K,J)=A(K,J)+F*A(K,I)
                    enddo
                enddo
              ENDIF
              DO  K=I,M
                 A(K,I)=SCALE*A(K,I)
              enddo
           ENDIF
       ENDIF
       W(I)=SCALE*G
       G=0.d0
       S=0.d0
       SCALE=0.d0
       IF ((I.LE.M).AND.(I.NE.N)) THEN
           DO  K=L,N
               SCALE=SCALE+ABS(A(I,K))
           enddo
             IF (SCALE.NE.0.0) THEN
                DO  K=L,N
                  A(I,K)=A(I,K)/SCALE
                  S=S+A(I,K)*A(I,K)
                enddo
                F=A(I,L)
                G=-SIGN(SQRT(S),F)
                H=F*G-S
                A(I,L)=F-G
                DO  K=L,N
                  RV1(K)=A(I,K)/H
                enddo
                IF (I.NE.M) THEN
                   DO  J=L,M
                     S=0.D0
                     DO  K=L,N
                       S=S+A(J,K)*A(I,K)
                    enddo
                     DO  K=L,N
                       A(J,K)=A(J,K)+S*RV1(K)
                     enddo
                   enddo
              ENDIF
              DO  K=L,N
                 A(I,K)=SCALE*A(I,K)
              enddo
           ENDIF
       ENDIF
       ANORM=MAX(ANORM,(ABS(W(I))+ABS(RV1(I))))
25     CONTINUE
c        print *,'25'
C   Accumulation of right-hand transformations.
       DO  I=N,1,-1
       IF (I.LT.N) THEN
         IF (G.NE.0.d0) THEN
           DO  J=L,N
             V(J,I)=(A(I,J)/A(I,L))/G
C   Double division to avoid possible underflow.
           enddo
          DO  J=L,N
            S=0.d0
            DO  K=L,N
              S=S+A(I,K)*V(K,J)
            enddo
            DO  K=L,N
              V(K,J)=V(K,J)+S*V(K,I)
            enddo
          enddo
        ENDIF
        DO  J=L,N
          V(I,J)=0.d0
          V(J,I)=0.d0
        enddo
       ENDIF
       V(I,I)=1.d0
       G=RV1(I)
       L=I
       enddo
c        print *,'32'

C  Accumulation of the left-hang transformation
       DO 39 I=N,1,-1
         L=I+1
         G=W(I)
         IF (I.LT.N) THEN
           DO  J=L,N
             A(I,J)=0.d0
           enddo
         ENDIF
         IF (G.NE.0.d0) THEN
           G=1.d0/G
           IF (I.NE.N) THEN
             DO  J=L,N
               S=0.d0
               DO K=L,M
                 S=S+A(K,I)*A(K,J)
               enddo
               F=(S/A(I,I))*G
             DO  K=I,M
               A(K,J)=A(K,J)+F*A(K,I)
             enddo
           enddo
         ENDIF
        DO  J=I,M
          A(J,I)=A(J,I)*G
        enddo
       ELSE
         DO  J=I,M
           A(J,I)=0.d0
         enddo
       ENDIF
       A(I,I)=A(I,I)+1.d0
39     CONTINUE
c        print *,'39'

C   Diagonalization of the bidiagonal form
C   Loop over singular values
       DO 49 K=N,1,-1
C   Loop allowed iterations
         DO 48 ITS=1,30
C   Test for spliting
            DO  L=K,1,-1
              NM=L-1
C   Note that RV1(1) is always zero
! old call which may cause inconsistent results
!              IF((ABS(RV1(L))+ANORM).EQ.ANORM) GO TO 2
!              IF((ABS(W(NM))+ANORM).EQ.ANORM) GO TO 1
! NEW CALL
              IF (((ABS(RV1(L))+ANORM).GE.NEAREST(ANORM,-1.d0)).AND.
     &          ((ABS(RV1(L))+ANORM).LE.NEAREST(ANORM,1.d0)) ) GO TO 2
              IF (((ABS(W(NM))+ANORM).GE.NEAREST(ANORM,-1.d0)).AND.
     &          ((ABS(W(NM))+ANORM).LE.NEAREST(ANORM,1.d0)) ) GO TO 1

            enddo
c          print *,'41'
1         C=0.d0
          S=1.d0
          DO  I=L,K
            F=S*RV1(I)
! old call which may cause inconsistent results

            IF (((ABS(F)+ANORM).LT.ANORM).OR.
     &            ((ABS(F)+ANORM).GT.ANORM)) THEN
              G=W(I)
              H=SQRT(F*F+G*G)
              W(I)=H
              H=1.D0/H
              C= (G*H)
              S=-(F*H)
              DO  J=1,M
                Y=A(J,NM)
                Z=A(J,I)
                A(J,NM)=(Y*C)+(Z*S)
                A(J,I)=-(Y*S)+(Z*C)
              enddo
            ENDIF
          enddo
c          print *,'43'
2         Z=W(K)
          IF (L.EQ.K) THEN
C   Convergence
            IF (Z.LT.0.d0) THEN
C   Singular values are made nonnegative
              W(K)=-Z
              DO  J=1,N
                V(J,K)=-V(J,K)
              enddo
            ENDIF
            GO TO 3
          ENDIF
          IF (ITS.EQ.30) then
!             print *,'SVDCMP: No convergence in 30 iterations'
          endif
          X=W(L)
          NM=K-1
          Y=W(NM)
          G=RV1(NM)
          H=RV1(K)
          F=((Y-Z)*(Y+Z)+(G-H)*(G+H))/(2.d0*H*Y)
          G=SQRT(F*F+1.D0)
          F=((X-Z)*(X+Z)+H*((Y/(F+SIGN(G,F)))-H))/X
C   Next  QR  transformation
          C=1.d0
          S=1.d0
          DO 47 J=L,NM
            I=J+1
            G=RV1(I)
            Y=W(I)
            H=S*G
            G=C*G
            Z=SQRT(F*F+H*H)
            RV1(J)=Z
            C=F/Z
            S=H/Z
            F= (X*C)+(G*S)
            G=-(X*S)+(G*C)
            H=Y*S
            Y=Y*C
            DO  NM=1,N
              X=V(NM,J)
              Z=V(NM,I)
              V(NM,J)= (X*C)+(Z*S)
              V(NM,I)=-(X*S)+(Z*C)
            enddo
c            print *,'45',F,H
            Z=pythag(F,H)
            W(J)=Z
C   Rotation can be arbitrary if  Z=0.
            IF (Z.NE.0.d0) THEN
c            print *,1/Z
              Z=1.d0/Z
c              print *,'*'
              C=F*Z
              S=H*Z
            ENDIF
            F= (C*G)+(S*Y)
            X=-(S*G)+(C*Y)
            DO  NM=1,M
              Y=A(NM,J)
              Z=A(NM,I)
              A(NM,J)= (Y*C)+(Z*S)
              A(NM,I)=-(Y*S)+(Z*C)
            enddo
c          print *,'46'

47        CONTINUE
c          print *,'47'
          RV1(L)=0.D0
          RV1(K)=F
          W(K)=X
48      CONTINUE
3      CONTINUE
49     CONTINUE
c        print *,'49'
       deallocate(RV1)
       RETURN
       END SUBROUTINE SVDCMP

       FUNCTION pythag(a,b) RESULT (VALUE)
       DOUBLE PRECISION, INTENT(IN) :: a,b
       DOUBLE PRECISION :: VALUE
       DOUBLE PRECISION :: absa,absb
       absa=abs(a)
       absb=abs(b)
       IF (absa.GT.absb) THEN
          VALUE=absa*SQRT(1.d0+(absb/absa)**2)
       ELSE
          IF (absb.EQ.0) THEN
             VALUE=0.D0
          ELSE
             VALUE=absb*SQRT(1.d0+(absa/absb)**2)
          ENDIF
       ENDIF
       RETURN
       END FUNCTION PYTHAG
       END MODULE RCRUDEMOD








*  KRBVRCMOD is a module  containing a:
*
*  Automatic Multidimensional Integration Subroutine
*               
*         AUTHOR: Alan Genz
*                 Department of Mathematics
*                 Washington State University
*                 Pulman, WA 99164-3113
*                 Email: AlanGenz@wsu.edu
*
*         Last Change: 5/15/98
* revised pab 10.03.2000
*   - updated to f90 (i.e. changed to assumed shape arrays + changing integers to DBLE)
*   - put it into a module
*
*  KRBVRC computes an approximation to the integral
*
*      1  1     1
*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1)
*      0  0     0
*
*
*  KRBVRC uses randomized Korobov rules for the first 20 variables. 
*  The primary references are
*   "Randomization of Number Theoretic Methods for Multiple Integration"
*    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13, pp. 904-14,
*  and 
*   "Optimal Parameters for Multidimensional Integration", 
*    P. Keast, SIAM J Numer Anal, 10, pp.831-838.
*  If there are more than 20 variables, the remaining variables are
*  integrated using Richtmeyer rules. A reference is
*   "Methods of Numerical Integration", P.J. Davis and P. Rabinowitz, 
*    Academic Press, 1984, pp. 482-483.
*   
***************  Parameters for KRBVRC ********************************************
****** Input parameters
*  NDIM    Number of variables, must exceed 1, but not exceed 100
*  MINVLS  Integer minimum number of function evaluations allowed.
*          MINVLS must not exceed MAXVLS.  If MINVLS < 0 then the
*          routine assumes a previous call has been made with 
*          the same integrand and continues that calculation.
*  MAXVLS  Integer maximum number of function evaluations allowed.
*  FUNCTN  EXTERNALly declared user defined function to be integrated.
*          It must have parameters (NDIM,Z), where Z is a real array
*          of dimension NDIM.
*                                     
*  ABSEPS  Required absolute accuracy.
*  RELEPS  Required relative accuracy.
*
****** Output parameters
*
*  MINVLS  Actual number of function evaluations used.
*  ABSERR  Estimated absolute accuracy of FINEST.
*  FINEST  Estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when 
*                     ABSERR <= MAX(ABSEPS, RELEPS*ABS(FINEST))
*                  and 
*                     INTVLS <= MAXCLS.
*          INFORM = 1 If MAXVLS was too small to obtain the required 
*          accuracy. In this case a value FINEST is returned with 
*          estimated absolute accuracy ABSERR.
************************************************************************
! William H. Press, Saul Teukolsky, 
! William T. Wetterling and Brian P. Flannery (1997)
! "Numerical recipes in Fortran 77", Vol. 1, pp 299--305  (SOBSEQ)

! You may  initialize the random generator before you 
!  call KRBVRC by the following lines:
!
!      call random_seed(SIZE=seed_size) 
!      allocate(seed(seed_size)) 
!      call random_seed(GET=seed(1:seed_size))  ! get current seed
!      seed(1)=seed1                            ! change seed
!      call random_seed(PUT=seed(1:seed_size)) 
!      deallocate(seed)
!
      MODULE KRBVRCMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: KRBVRC
! 
      INTERFACE KRBVRC
      MODULE PROCEDURE KRBVRC
      END INTERFACE
!
      INTERFACE DKSMRC
      MODULE PROCEDURE DKSMRC
      END INTERFACE
!      
      INTERFACE  DKRCHT
      MODULE PROCEDURE  DKRCHT
      END INTERFACE
      
      INTERFACE  SOBSEQ
      MODULE PROCEDURE SOBSEQ
      END INTERFACE
!
      CONTAINS   

!***********************************************************
!    MAIN INTEGRATION ROUTINE KRBVRC
!***********************************************************  

      SUBROUTINE KRBVRC( NDIM, MINVLS, MAXVLS, FUNCTN, ABSEPS, RELEPS,
     &                   ABSERR, FINEST, INFORM )
*
*  Automatic Multidimensional Integration Subroutine
*               
*         AUTHOR: Alan Genz
*                 Department of Mathematics
*                 Washington State University
*                 Pulman, WA 99164-3113
*                 Email: AlanGenz@wsu.edu
*
*         Last Change: 5/15/98
*
*  KRBVRC computes an approximation to the integral
*
*      1  1     1
*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1)
*      0  0     0
*
*
*  KRBVRC uses randomized Korobov rules for the first 20 variables. 
*  The primary references are
*   "Randomization of Number Theoretic Methods for Multiple Integration"
*    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13, pp. 904-14,
*  and 
*   "Optimal Parameters for Multidimensional Integration", 
*    P. Keast, SIAM J Numer Anal, 10, pp.831-838.
*  If there are more than 20 variables, the remaining variables are
*  integrated using Richtmeyer rules. A reference is
*   "Methods of Numerical Integration", P.J. Davis and P. Rabinowitz, 
*    Academic Press, 1984, pp. 482-483.
*   
***************  Parameters ********************************************
****** Input parameters
*  NDIM    Number of variables, must exceed 1, but not exceed 100
*  MINVLS  Integer minimum number of function evaluations allowed.
*          MINVLS must not exceed MAXVLS.  If MINVLS < 0 then the
*          routine assumes a previous call has been made with 
*          the same integrand and continues that calculation.
*  MAXVLS  Integer maximum number of function evaluations allowed.
*  FUNCTN  EXTERNALly declared user defined function to be integrated.
*          It must have parameters (NDIM,Z), where Z is a real array
*          of dimension NDIM.
*                                     
*  ABSEPS  Required absolute accuracy.
*  RELEPS  Required relative accuracy.
****** Output parameters
*  MINVLS  Actual number of function evaluations used.
*  ABSERR  Estimated absolute accuracy of FINEST.
*  FINEST  Estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when 
*                     ABSERR <= MAX(ABSEPS, RELEPS*ABS(FINEST))
*                  and 
*                     INTVLS <= MAXCLS.
*          INFORM = 1 If MAXVLS was too small to obtain the required 
*                  accuracy. In this case a value FINEST is returned with 
*                  estimated absolute accuracy ABSERR.
*          INFORM = 2 If NDIM>100 or NDIM<1
************************************************************************
      INTEGER, INTENT(IN)    :: NDIM,  MAXVLS
      INTEGER, INTENT(INOUT) :: MINVLS
      INTEGER, INTENT(OUT)   :: INFORM
      DOUBLE PRECISION, INTENT(IN)  :: ABSEPS, RELEPS
      DOUBLE PRECISION, INTENT(OUT) :: FINEST, ABSERR 
      INTEGER :: NP,PLIM,NLIM,KLIM,KLIMI,SAMPLS,I,INTVLS,MINSMP,NK
      PARAMETER ( PLIM = 25, NLIM = 100, KLIM = 20, MINSMP = 8 )
      INTEGER , DIMENSION(PLIM) :: P
      INTEGER , DIMENSION(PLIM,KLIM-1) ::  C 
      DOUBLE PRECISION :: DIFINT,FINVAL,VARSQR,VAREST,VARPRD,VALUE
      DOUBLE PRECISION, PARAMETER :: ONE = 1.D0 , ZERO = 0.D0
      DOUBLE PRECISION, DIMENSION(2*NLIM) :: X  = 0.d0
      DOUBLE PRECISION, DIMENSION(KLIM  ) :: VK = 0.d0
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      DATA P / 31, 47, 73, 113, 173, 263, 397, 593, 907, 1361,
     &     2053, 3079, 4621, 6947, 10427, 15641, 23473, 35221, 
     &     52837, 79259, 118891, 178349, 267523, 401287, 601943/
      DATA (C( 1,I), I = 1, 19)/       12,      9,      9,
     &      13,     12,     12,     12,     12,     12,     12,     12,
     &      12,      3,      3,      3,     12,      7,      7,     12/
      DATA (C( 2,I), I = 1, 19)/        13,     11,     17,
     &      10,     15,     15,     15,     15,     15,     15,     22,
     &      15,     15,      6,      6,      6,     15,     15,      9/
      DATA (C( 3,I), I = 1, 19)/        27,     28,     10,
     &      11,     11,     20,     11,     11,     28,     13,     13,
     &      28,     13,     13,     13,     14,     14,     14,     14/
      DATA (C( 4,I), I = 1, 19)/        35,     27,     27,
     &      36,     22,     29,     29,     20,     45,      5,      5,
     &       5,     21,     21,     21,     21,     21,     21,     21/
      DATA (C( 5,I), I = 1, 19)/        64,     66,     28,
     &      28,     44,     44,     55,     67,     10,     10,     10,
     &      10,     10,     10,     38,     38,     10,     10,     10/
      DATA (C( 6,I), I = 1, 19)/       111,     42,     54,
     &     118,     20,     31,     31,     72,     17,     94,     14,
     &      14,     11,     14,     14,     14,     94,     10,     10/
      DATA (C( 7,I), I = 1, 19)/       163,    154,     83,
     &      43,     82,     92,    150,     59,     76,     76,     47,
     &      11,     11,    100,    131,    116,    116,    116,    116/
      DATA (C( 8,I), I = 1, 19)/      246,    189,    242,
     &     102,    250,    250,    102,    250,    280,    118,    196,
     &     118,    191,    215,    121,    121,     49,     49,     49/
      DATA (C( 9,I), I = 1, 19)/      347,    402,    322,
     &     418,    215,    220,    339,    339,    339,    337,    218,
     &     315,    315,    315,    315,    167,    167,    167,    167/
      DATA (C(10,I), I = 1, 19)/      505,    220,    601,
     &     644,    612,    160,    206,    206,    206,    422,    134,
     &     518,    134,    134,    518,    652,    382,    206,    158/
      DATA (C(11,I), I = 1, 19)/     794,    325,    960,
     &     528,    247,    247,    338,    366,    847,    753,    753,
     &     236,    334,    334,    461,    711,    652,    381,    381/
      DATA (C(12,I), I = 1, 19)/     1189,    888,    259,
     &    1082,    725,    811,    636,    965,    497,    497,   1490,
     &    1490,    392,   1291,    508,    508,   1291,   1291,    508/
      DATA (C(13,I), I = 1, 19)/     1763,   1018,   1500,
     &     432,   1332,   2203,    126,   2240,   1719,   1284,    878,
     &    1983,    266,    266,    266,    266,    747,    747,    127/
      DATA  (C(14,I), I = 1, 19)/     2872,   3233,   1534,
     &    2941,   2910,    393,   1796,    919,    446,    919,    919,
     &    1117,    103,    103,    103,    103,    103,    103,    103/
      DATA  (C(15,I), I = 1, 19)/    4309,   3758,   4034,
     &    1963,    730,    642,   1502,   2246,   3834,   1511,   1102,
     &    1102,   1522,   1522,   3427,   3427,   3928,    915,    915/
      DATA  (C(16,I), I = 1, 19)/     6610,   6977,   1686,
     &    3819,   2314,   5647,   3953,   3614,   5115,    423,    423,
     &    5408,   7426,    423,    423,    487,   6227,   2660,   6227/
      DATA  (C(17,I), I = 1, 19)/     9861,   3647,   4073,
     &    2535,   3430,   9865,   2830,   9328,   4320,   5913,  10365,
     &    8272,   3706,   6186,   7806,   7806,   7806,   8610,   2563/
      DATA  (C(18,I), I = 1, 19)/   10327,   7582,   7124,
     &    8214,   9600,  10271,  10193,  10800,   9086,   2365,   4409,
     &   13812,   5661,   9344,   9344,  10362,   9344,   9344,   8585/
      DATA (C(19,I), I = 1, 19)/   19540,  19926,  11582,
     &   11113,  24585,   8726,  17218,    419,   4918,   4918,   4918,
     &   15701,  17710,   4037,   4037,  15808,  11401,  19398,  25950/
      DATA  (C(20,I), I = 1, 19)/    34566,   9579,  12654,
     &   26856,  37873,  38806,  29501,  17271,   3663,  10763,  18955,
     &    1298,  26560,  17132,  17132,   4753,   4753,   8713,  18624/
      DATA  (C(21,I), I = 1, 19)/   31929,  49367,  10982,
     &    3527,  27066,  13226,  56010,  18911,  40574,  20767,  20767,
     &    9686,  47603,  47603,  11736,  11736,  41601,  12888,  32948/
      DATA (C(22,I), I = 1, 19)/   40701,  69087,  77576,
     &   64590,  39397,  33179,  10858,  38935,  43129,  35468,  35468,
     &    2196,  61518,  61518,  27945,  70975,  70975,  86478,  86478/
      DATA  (C(23,I), I = 1, 19)/  103650, 125480,  59978,
     &   46875,  77172,  83021, 126904,  14541,  56299,  43636,  11655,
     &   52680,  88549,  29804, 101894, 113675,  48040, 113675,  34987/
      DATA (C(24,I), I = 1, 19)/  165843,  90647,  59925,
     &  189541,  67647,  74795,  68365, 167485, 143918,  74912, 167289,
     &   75517,   8148, 172106, 126159,  35867,  35867,  35867, 121694/
      DATA (C(25,I), I = 1, 19)/  130365, 236711, 110235,
     &  125699,  56483,  93735, 234469,  60549,   1291,  93937, 245291,
     &  196061, 258647, 162489, 176631, 204895,  73353, 172319,  28881/
*
      SAVE P, C, SAMPLS, NP, VAREST
      IF ( NDIM .GT. NLIM .OR. NDIM .LT. 1 ) THEN
         INFORM = 2
         FINEST = ZERO
         ABSERR = ONE
         RETURN
      ENDIF
      INFORM = 1
      INTVLS = 0
      KLIMI = KLIM
      IF ( MINVLS .GE. 0 ) THEN
         FINEST = ZERO
         VAREST = ZERO
         SAMPLS = MINSMP 
         DO I = 1, PLIM
            NP = I
            IF ( MINVLS .LT. 2*SAMPLS*P(I) ) GO TO 10
         END DO
         SAMPLS = MAX( MINSMP, MINVLS/( 2*P(NP) ) )
      ENDIF
 10   VK(1) = ONE/DBLE(P(NP))
      NK = MIN( NDIM, KLIM )
      DO I = 2, NK
         VK(I) = MOD(DBLE(C(NP,NK-1))*VK(I-1), ONE )
      END DO
      FINVAL = ZERO
      VARSQR = ZERO
      DO I = 1, SAMPLS
         CALL DKSMRC( NDIM, KLIMI, VALUE, P(NP), VK, FUNCTN, X )
         DIFINT = ( VALUE - FINVAL )/DBLE(I)
         FINVAL = FINVAL + DIFINT
         VARSQR = DBLE( I - 2 )*VARSQR/DBLE(I) + DIFINT*DIFINT
      END DO
      INTVLS = INTVLS + 2*SAMPLS*P(NP)
      VARPRD = VAREST*VARSQR
      FINEST = FINEST + ( FINVAL - FINEST )/( ONE + VARPRD )
      IF ( VARSQR .GT. ZERO ) VAREST = ( ONE + VARPRD )/VARSQR
      ABSERR = 3.d0*SQRT( VARSQR/( ONE + VARPRD ) )
      IF ( ABSERR .GT. MAX( ABSEPS, ABS(FINEST)*RELEPS ) ) THEN
         IF ( NP .LT. PLIM ) THEN
            NP = NP + 1
         ELSE
            SAMPLS = MIN( 3*SAMPLS/2, ( MAXVLS - INTVLS )/( 2*P(NP) ) ) 
            SAMPLS = MAX( MINSMP, SAMPLS )
         ENDIF
         IF ( INTVLS + 2*SAMPLS*P(NP) .LE. MAXVLS ) GO TO 10
      ELSE
         INFORM = 0
      ENDIF
      MINVLS = INTVLS
*
      END  SUBROUTINE KRBVRC
*
      SUBROUTINE DKSMRC( NDIM, KLIM, SUMKRO, PRIME, VK, FUNCTN, X )
      INTEGER, INTENT(IN):: NDIM, KLIM, PRIME
      DOUBLE PRECISION, INTENT(OUT) :: SUMKRO
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) :: VK,X
      INTEGER :: K, J, JP, NK
      DOUBLE PRECISION ::  ONE, XT, MVNUNI
      PARAMETER ( ONE = 1.d0 )
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE      
      SUMKRO = 0.d0
*
*     Randomize Variable Order
*
      NK = MIN( NDIM, KLIM )
      DO J = 1, NK-1
         CALL random_number(MVNUNI)
!         JP = J + NINT(MVNUNI*DBLE( NK + 1 - J )) 
         JP = J + NINT(MVNUNI*DBLE( NK - J ))   ! pab 21.11.2000
         XT = VK(J)
         VK(J) = VK(JP)
         VK(JP) = XT
      END DO
*
*     Determine Random Shifts for each Variable
*
      CALL random_number(X(NDIM+1:2*NDIM))
*
*     Compute periodized and symmetrized  lattice rule sum
*
      DO K = 1, PRIME
         X(1:NK) = MOD( DBLE(K)*VK(1:NK), ONE )
         IF ( NDIM. GT. KLIM ) CALL DKRCHT(KLIM, NDIM-KLIM, X) !X(KLIM+1:NDIM) )
         DO J = 1, NDIM
            XT = X(J) + X(NDIM+J)
            IF ( XT .GT. ONE ) XT = XT - 1.d0
            X(J) = ABS( 2.d0*XT - 1.d0 )
         END DO
         SUMKRO = SUMKRO+(FUNCTN(NDIM,X)-SUMKRO)/DBLE(2*K-1)
         X(1:NDIM) = 1.d0 - X(1:NDIM)
         SUMKRO = SUMKRO+(FUNCTN(NDIM,X)-SUMKRO)/DBLE(2*K)
      END DO
      END  SUBROUTINE DKSMRC 
*
      SUBROUTINE DKRCHT(KLIM, S, QUASI )
*
*     This subroutine generates a new quasi-random Richtmeyer vector. 
*     A reference is
*      "Methods of Numerical Integration", P.J. Davis and P. Rabinowitz, 
*       Academic Press, 1984, pp. 482-483.
*
*       INPUTS:
*      KLIM - Lower start value
*         S - the number of dimensions; 
*             DKRCHT is initialized for each new S or S < 1.
*
*       OUTPUTS:
*         QUASI - a new quasi-random S-vector
*
* revised pab 28.05.2003
* - added klim in order to avoid copying of arrays in and out
* revised pab 01.11.1999
* updated to fortran 90
      INTEGER, INTENT(IN) :: S,KLIM
      DOUBLE PRECISION , DIMENSION(:) :: QUASI
      INTEGER :: MXDIM, MXHSUM, B
      PARAMETER ( MXDIM = 80, MXHSUM = 48, B = 2 )
      INTEGER :: HISUM, I,  OLDS 
      DOUBLE PRECISION , DIMENSION(MXDIM) :: PSQT
      INTEGER, DIMENSION(MXDIM   )  :: PRIME
      INTEGER, DIMENSION(0:MXHSUM)  :: N
        
     
      DOUBLE PRECISION ::  ONE, RN
      PARAMETER ( ONE = 1.D0 )
      PARAMETER ( PRIME = (/ 
     &     2,    3,    5,    7,   11,   13,   17,   19,   23,   29,
     &    31,   37,   41,   43,   47,   53,   59,   61,   67,   71,
     &    73,   79,   83,   89,   97,  101,  103,  107,  109,  113,
     &   127,  131,  137,  139,  149,  151,  157,  163,  167,  173,
     &   179,  181,  191,  193,  197,  199,  211,  223,  227,  229,
     &   233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
     &   283,  293,  307,  311,  313,  317,  331,  337,  347,  349,
     &   353,  359,  367,  373,  379,  383,  389,  397,  401,  409/))
* Primes to continue
* 419  421   431   433   439   443   449   457   461   463   467   479   487   491   499
* 503   509   521   523   541   547   557   563   569   571   577   587   593   599
      SAVE OLDS, PSQT, HISUM, N
      DATA OLDS / 0 /
      IF ( S .NE. OLDS .OR. S .LT. 1 ) THEN                          
         OLDS = ABS(S)                             ! pab 14.03.2000
         N(0) = 0
         HISUM = 0
         DO I = 1, OLDS
            RN = DBLE(PRIME(I))
            PSQT(I) = SQRT( RN )
         END DO
      END IF
      DO I = 0, HISUM 
         N(I) = N(I) + 1
         IF ( N(I) .LT. B ) GO TO 10
         N(I) = 0
      END DO
      HISUM = HISUM + 1
      IF ( HISUM .GT. MXHSUM ) HISUM = 0
      N(HISUM) = 1
 10   RN = 0.d0
      DO I = HISUM, 0, -1
         RN = DBLE(N(I)) + DBLE(B)*RN
      END DO
      DO I = 1, OLDS
         QUASI(KLIM+I) = MOD( RN*PSQT(I), ONE )
      END DO
      END SUBROUTINE DKRCHT
!
! SOBSEQ is not taken in to use:
!
      SUBROUTINE SOBSEQ(N,X)
      IMPLICIT NONE
      DOUBLE PRECISION,DIMENSION(:), INTENT(OUT):: X
      INTEGER, INTENT(IN) :: N
      INTEGER,PARAMETER ::MAXBIT=30,MAXDIM=6
      INTEGER :: I,IM, IN,IPP,J,K,L, OLDN
      INTEGER, DIMENSION(MAXDIM) :: IP,MDEG,IX
      INTEGER, DIMENSION(MAXDIM,MAXBIT) ::IU
      INTEGER, DIMENSION(MAXDIM*MAXBIT) ::IV
      DOUBLE PRECISION :: FAC
      SAVE IP,MDEG,IX,IV,IN,FAC, OLDN
      DATA OLDN / 0 /
      DATA IP /0,1,1,2,1,4 /, MDEG /1,2,3,3,4,4 /
      DATA IX /0,0,0,0,0,0 /
      DATA IV /1,1,1,1,1,1,3,1,3,3,1,1,5,
     &     7,7,3,3,5,15,11,5,15,13,9,156*0/
      !(MAXDIM*MAXBIT-24)
      EQUIVALENCE (IV,IU)       ! to allow both 1D and 2D addressing
! returns sobols sequence of quasi-random numbers between 0 1
! When n is new or is negative, internally initializes a set of MAXBIT
! direction numbers for each of MAXDIM different sobol
! sequences. When n is positive (but < MAXDIM)
! returns as the vector x(1:n) the next values from n of these sequences 
! (n must not be changed between initializations)
!
! This routine is initialised for maximum of n=6 dimensions
! and a word length of 30 bits. These parameter may be increased by 
!changing MAXBIT and MAXDIM and add more initializing data to 
! ip (primitive polynomials), mdeg (their degrees) and iv 
! (the starting value for the recurrence relation)
 
!reference
! William H. Press, Saul Teukolsky, William T. Wetterling and Brian P. Flannery (1997)
! "Numerical recipes in Fortran 77", Vol. 1, pp 299--305
      
      
      IF (N.LT.0 .OR. OLDN.NE.N ) THEN          ! INITIALIZE, DO NOT RETURN VECTOR
         OLDN = ABS(N)
         IX=0
         IN=0  ! RANDOM STARTPOINT: CALL RANDOM_NUMBER(P); IN=P*2^MAXBIT
               ! AND REMOVE WARNING MESSAGE BELOW
         !IF (IV(1).NE.1) RETURN
 
         IF (IV(1).EQ.1) THEN
            FAC=1.D0/2.D0**MAXBIT
            DO K=1,MAXDIM
               DO J=1,MDEG(K)   ! STORED VALUES NEED NORMALIZATION
                  IU(K,J)=IU(K,J)*2**(MAXBIT-J)
               ENDDO
               DO J=1,MDEG(K)+1,MAXBIT ! USE RECCURENCE TO GET OTHER VALUES
                  IPP=IP(K)
                  I=IU(K,J-MDEG(K))
                  I=IEOR(I,I/2**MDEG(K))
                  DO L=MDEG(K)-1,1,-1
                     IF (IAND(IPP,1).NE.0) I=IEOR(I,IU(K,J-L))
                     IPP=IPP/2
                  ENDDO
                  IU(K,J)=I
               ENDDO      
            ENDDO
         ENDIF
      ENDIF                      ! CALCULATE THE NEXT VECTOR IN THE SEQUENCE
         IM=IN
         DO J=1,MAXBIT          ! FIND THE RIGHTMOST ZERO BIT
            IF (IAND(IM,1).EQ.0) GOTO 1
            IM=IM/2
         ENDDO
!         PRINT *,'MAXBIT TOO SMALL IN SOBSEQ'
 1       IM=(J-1)*MAXDIM
         DO K=1,MIN(OLDN,MAXDIM)   !XOR THE 
            IX(K)=IEOR(IX(K),IV(IM+K))
            X(K)=IX(K)*FAC
         ENDDO
         IN=IN+1                ! INCREMENT COUNTER
      
      RETURN
      END SUBROUTINE SOBSEQ
      END MODULE KRBVRCMOD

      MODULE DKBVRCMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: DKBVRC
! 
      INTERFACE DKBVRC
      MODULE PROCEDURE DKBVRC
      END INTERFACE
!
      INTERFACE DKSMRC 
      MODULE PROCEDURE DKSMRC
      END INTERFACE
!      
      CONTAINS
      SUBROUTINE DKBVRC( NDIM, MINVLS, MAXVLS, FUNCTN, ABSEPS, RELEPS,
     &     ABSERR, FINEST, INFORM )
*
*  Automatic Multidimensional Integration Subroutine
*               
*         AUTHOR: Alan Genz
*                 Department of Mathematics
*                 Washington State University
*                 Pulman, WA 99164-3113
*                 Email: AlanGenz@wsu.edu
*
*         Last Change: 1/15/03
*
! revised pab June 2004
! updated to F90
*
*  DKBVRC computes an approximation to the integral
*
*      1  1     1
*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1)
*      0  0     0
*
*
*  DKBVRC uses randomized Korobov rules for the first 100 variables. 
*  The primary references are
*   "Randomization of Number Theoretic Methods for Multiple Integration"
*    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13, pp. 904-14,
*  and 
*   "Optimal Parameters for Multidimensional Integration", 
*    P. Keast, SIAM J Numer Anal, 10, pp.831-838.
*  If there are more than 100 variables, the remaining variables are
*  integrated using the rules described in the reference
*   "On a Number-Theoretical Integration Method"
*   H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11.
*   
***************  Parameters ********************************************
****** Input parameters
*  NDIM    Number of variables, must exceed 1, but not exceed 1000
*  MINVLS  Integer minimum number of function evaluations allowed.
*          MINVLS must not exceed MAXVLS.  If MINVLS < 0 then the
*          routine assumes a previous call has been made with 
*          the same integrand and continues that calculation.
*  MAXVLS  Integer maximum number of function evaluations allowed.
*  FUNCTN  EXTERNALly declared user defined function to be integrated.
*          It must have parameters (NDIM,Z), where Z is a real array
*          of dimension NDIM.
*                                     
*  ABSEPS  Required absolute accuracy.
*  RELEPS  Required relative accuracy.
****** Output parameters
*  MINVLS  Actual number of function evaluations used.
*  ABSERR  Estimated absolute accuracy of FINEST.
*  FINEST  Estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when 
*                     ABSERR <= MAX(ABSEPS, RELEPS*ABS(FINEST))
*                  and 
*                     INTVLS <= MAXCLS.
*          INFORM = 1 If MAXVLS was too small to obtain the required 
*          accuracy. In this case a value FINEST is returned with 
*          estimated absolute accuracy ABSERR.
*          INFORM = 2 If NDIM>1000 or NDIM<1
************************************************************************
      INTEGER, INTENT(IN)    :: NDIM,  MAXVLS
      INTEGER, INTENT(INOUT) :: MINVLS
      INTEGER, INTENT(OUT)   :: INFORM
      DOUBLE PRECISION, INTENT(IN)  :: ABSEPS, RELEPS
      DOUBLE PRECISION, INTENT(OUT) :: FINEST, ABSERR 
      INTEGER :: NP,PLIM,NLIM,KLIM,KLIMI,SAMPLS,I,INTVLS,MINSMP
      PARAMETER ( PLIM = 28, NLIM = 1000, KLIM = 100, MINSMP = 8 )
      INTEGER P(PLIM), C(PLIM,KLIM-1) 
      DOUBLE PRECISION :: DIFINT, FINVAL, VARSQR, VAREST, VARPRD, VALUE
      DOUBLE PRECISION, PARAMETER :: ONE= 1.D0,ZERO = 0.D0
      DOUBLE PRECISION X(2*NLIM), VK(NLIM)
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      SAVE P, C, SAMPLS, NP, VAREST
      IF ( NDIM .GT. NLIM .OR. NDIM .LT. 1 ) THEN
         INFORM = 2
         FINEST = ZERO
         ABSERR = ONE
         RETURN
      ENDIF
      INFORM = 1
      INTVLS = 0
      KLIMI = KLIM
      IF ( MINVLS .GE. 0 ) THEN
         FINEST = ZERO
         VAREST = ZERO
         SAMPLS = MINSMP 
         DO I = 1, PLIM
            NP = I
            IF ( MINVLS .LT. 2*SAMPLS*P(I) ) GO TO 10
         END DO
         SAMPLS = MAX( MINSMP, MINVLS/( 2*P(NP) ) )
      ENDIF
 10   VK(1) = ONE/P(NP)
      DO I = 2, NDIM
         IF ( I .LE. KLIM ) THEN
            VK(I) = MOD( C(NP, MIN(NDIM-1,KLIM-1))*VK(I-1), ONE )
         ELSE
            VK(I) = INT( P(NP)*2**(DBLE(I-KLIM)/(NDIM-KLIM+1)) ) 
            VK(I) = MOD( VK(I)/P(NP), ONE ) 
         END IF
      END DO
      FINVAL = ZERO
      VARSQR = ZERO
      DO I = 1, SAMPLS
         CALL DKSMRC( NDIM, KLIMI, VALUE, P(NP), VK, FUNCTN, X )
         DIFINT = ( VALUE - FINVAL )/DBLE(I)
         FINVAL = FINVAL + DIFINT
         VARSQR = DBLE( I - 2 )*VARSQR/DBLE(I) + DIFINT**2
      END DO
      INTVLS = INTVLS + 2*SAMPLS*P(NP)
      VARPRD = VAREST*VARSQR
      FINEST = FINEST + ( FINVAL - FINEST )/( ONE + VARPRD )
      IF ( VARSQR .GT. ZERO ) VAREST = ( ONE + VARPRD )/VARSQR
      ABSERR = 3.0D0*SQRT( VARSQR/( ONE + VARPRD ) )
      IF ( ABSERR .GT. MAX( ABSEPS, ABS(FINEST)*RELEPS ) ) THEN
         IF ( NP .LT. PLIM ) THEN
            NP = NP + 1
         ELSE
            SAMPLS = MIN( 3*SAMPLS/2, ( MAXVLS - INTVLS )/( 2*P(NP) ) ) 
            SAMPLS = MAX( MINSMP, SAMPLS )
         ENDIF
         IF ( INTVLS + 2*SAMPLS*P(NP) .LE. MAXVLS ) GO TO 10
      ELSE
         INFORM = 0
      ENDIF
      MINVLS = INTVLS
*
*    Optimal Parameters for Lattice Rules
*
      DATA P( 1),(C( 1,I),I = 1,99)/     31, 12, 2*9, 13, 8*12, 3*3, 12,
     & 2*7, 9*12, 3*3, 12, 2*7, 9*12, 3*3, 12, 2*7, 9*12, 3*3, 12, 2*7,
     & 8*12, 7, 3*3, 3*7, 21*3/
      DATA P( 2),(C( 2,I),I = 1,99)/    47, 13, 11, 17, 10, 6*15,
     & 22, 2*15, 3*6, 2*15, 9, 13, 3*2, 13, 2*11, 10, 9*15, 3*6, 2*15,
     & 9, 13, 3*2, 13, 2*11, 10, 9*15, 3*6, 2*15, 9, 13, 3*2, 13, 2*11,
     & 2*10, 8*15, 6, 2, 3, 2, 3, 12*2/
      DATA P( 3),(C( 3,I),I = 1,99)/    73, 27, 28, 10, 2*11, 20,
     & 2*11, 28, 2*13, 28, 3*13, 16*14, 2*31, 3*5, 31, 13, 6*11, 7*13,
     & 16*14, 2*31, 3*5, 11, 13, 7*11, 2*13, 11, 13, 4*5, 14, 13, 8*5/
      DATA P( 4),(C( 4,I),I = 1,99)/   113, 35, 2*27, 36, 22, 2*29,
     & 20, 45, 3*5, 16*21, 29, 10*17, 12*23, 21, 27, 3*3, 24, 2*27,
     & 17, 3*29, 17, 4*5, 16*21, 3*17, 6, 2*17, 6, 3, 2*6, 5*3/
      DATA P( 5),(C( 5,I),I = 1,99)/   173, 64, 66, 2*28, 2*44, 55,
     & 67, 6*10, 2*38, 5*10, 12*49, 2*38, 31, 2*4, 31, 64, 3*4, 64,
     & 6*45, 19*66, 11, 9*66, 45, 11, 7, 3, 3*2, 27, 5, 2*3, 2*5, 7*2/
      DATA P( 6),(C( 6,I),I = 1,99)/   263, 111, 42, 54, 118, 20,
     & 2*31, 72, 17, 94, 2*14, 11, 3*14, 94, 4*10, 7*14, 3*11, 7*8,
     & 5*18, 113, 2*62, 2*45, 17*113, 2*63, 53, 63, 15*67, 5*51, 12,
     & 51, 12, 51, 5, 2*3, 2*2, 5/
      DATA P( 7),(C( 7,I),I = 1,99)/   397, 163, 154, 83, 43, 82,
     & 92, 150, 59, 2*76, 47, 2*11, 100, 131, 6*116, 9*138, 21*101,
     & 6*116, 5*100, 5*138, 19*101, 8*38, 5*3/
      DATA P( 8),(C( 8,I),I = 1,99)/   593, 246, 189, 242, 102,
     & 2*250, 102, 250, 280, 118, 196, 118, 191, 215, 2*121,
     & 12*49, 34*171, 8*161, 17*14, 6*10, 103, 4*10, 5/
      DATA P( 9),(C( 9,I),I = 1,99)/   907, 347, 402, 322, 418,
     & 215, 220, 3*339, 337, 218, 4*315, 4*167, 361, 201, 11*124,
     & 2*231, 14*90, 4*48, 23*90, 10*243, 9*283, 16, 283, 16, 2*283/
      DATA P(10),(C(10,I),I = 1,99)/  1361, 505, 220, 601, 644,
     & 612, 160, 3*206, 422, 134, 518, 2*134, 518, 652, 382,
     & 206, 158, 441, 179, 441, 56, 2*559, 14*56, 2*101, 56,
     & 8*101, 7*193, 21*101, 17*122, 4*101/
      DATA P(11),(C(11,I),I = 1,99)/  2053, 794, 325, 960, 528,
     & 2*247, 338, 366, 847, 2*753, 236, 2*334, 461, 711, 652,
     & 3*381, 652, 7*381, 226, 7*326, 126, 10*326, 2*195, 19*55,
     & 7*195, 11*132, 13*387/
      DATA P(12),(C(12,I),I = 1,99)/  3079, 1189, 888, 259, 1082, 725,      
     & 811, 636, 965, 2*497, 2*1490, 392, 1291, 2*508, 2*1291, 508,
     & 1291, 2*508, 4*867, 934, 7*867, 9*1284, 4*563, 3*1010, 208,
     & 838, 3*563, 2*759, 564, 2*759, 4*801, 5*759, 8*563, 22*226/
      DATA P(13),(C(13,I),I = 1,99)/  4621, 1763, 1018, 1500, 432,
     & 1332, 2203, 126, 2240, 1719, 1284, 878, 1983, 4*266,
     & 2*747, 2*127, 2074, 127, 2074, 1400, 10*1383, 1400, 7*1383,
     & 507, 4*1073, 5*1990, 9*507, 17*1073, 6*22, 1073, 6*452, 318,
     & 4*301, 2*86, 15/
      DATA P(14),(C(14,I),I = 1,99)/  6947, 2872, 3233, 1534, 2941,
     & 2910, 393, 1796, 919, 446, 2*919, 1117, 7*103, 2311, 3117, 1101,
     & 2*3117, 5*1101, 8*2503, 7*429, 3*1702, 5*184, 34*105, 13*784/
      DATA P(15),(C(15,I),I = 1,99)/ 10427, 4309, 3758, 4034, 1963,
     & 730, 642, 1502, 2246, 3834, 1511, 2*1102, 2*1522, 2*3427,
     & 3928, 2*915, 4*3818, 3*4782, 3818, 4782, 2*3818, 7*1327, 9*1387,
     & 13*2339, 18*3148, 3*1776, 3*3354, 925, 2*3354, 5*925, 8*2133/
      DATA P(16),(C(16,I),I = 1,99)/ 15641, 6610, 6977, 1686, 3819,
     & 2314, 5647, 3953, 3614, 5115, 2*423, 5408, 7426, 2*423,
     & 487, 6227, 2660, 6227, 1221, 3811, 197, 4367, 351,
     & 1281, 1221, 3*351, 7245, 1984, 6*2999, 3995, 4*2063, 1644,
     & 2063, 2077, 3*2512, 4*2077, 19*754, 2*1097, 4*754, 248, 754,
     & 4*1097, 4*222, 754,11*1982/
      DATA P(17),(C(17,I),I = 1,99)/ 23473, 9861, 3647, 4073, 2535,
     & 3430, 9865, 2830, 9328, 4320, 5913, 10365, 8272, 3706, 6186,
     & 3*7806, 8610, 2563, 2*11558, 9421, 1181, 9421, 3*1181, 9421,
     & 2*1181, 2*10574, 5*3534, 3*2898, 3450, 7*2141, 15*7055, 2831,
     & 24*8204, 3*4688, 8*2831/
      DATA P(18),(C(18,I),I = 1,99)/ 35221, 10327, 7582, 7124, 8214,
     & 9600, 10271, 10193, 10800, 9086, 2365, 4409, 13812,
     & 5661, 2*9344, 10362, 2*9344, 8585, 11114, 3*13080, 6949,
     & 3*3436, 13213, 2*6130, 2*8159, 11595, 8159, 3436, 18*7096,
     & 4377, 7096, 5*4377, 2*5410, 32*4377, 2*440, 3*1199/
      DATA P(19),(C(19,I),I = 1,99)/ 52837, 19540, 19926, 11582,
     & 11113, 24585, 8726, 17218, 419, 3*4918, 15701, 17710,
     & 2*4037, 15808, 11401, 19398, 2*25950, 4454, 24987, 11719,
     & 8697, 5*1452, 2*8697, 6436, 21475, 6436, 22913, 6434, 18497,
     & 4*11089, 2*3036, 4*14208, 8*12906, 4*7614, 6*5021, 24*10145,
     & 6*4544, 4*8394/    
      DATA P(20),(C(20,I),I = 1,99)/ 79259, 34566, 9579, 12654,
     & 26856, 37873, 38806, 29501, 17271, 3663, 10763, 18955,
     & 1298, 26560, 2*17132, 2*4753, 8713, 18624, 13082, 6791,
     & 1122, 19363, 34695, 4*18770, 15628, 4*18770, 33766, 6*20837,
     & 5*6545, 14*12138, 5*30483, 19*12138, 9305, 13*11107, 2*9305/
      DATA P(21),(C(21,I),I = 1,99)/118891, 31929, 49367, 10982, 3527,
     & 27066, 13226, 56010, 18911, 40574, 2*20767, 9686, 2*47603, 
     & 2*11736, 41601, 12888, 32948, 30801, 44243, 2*53351, 16016, 
     & 2*35086, 32581, 2*2464, 49554, 2*2464, 2*49554, 2464, 81, 27260, 
     & 10681, 7*2185, 5*18086, 2*17631, 3*18086, 37335, 3*37774, 
     & 13*26401, 12982, 6*40398, 3*3518, 9*37799, 4*4721, 4*7067/
      DATA P(22),(C(22,I),I = 1,99)/178349, 40701, 69087, 77576, 64590, 
     & 39397, 33179, 10858, 38935, 43129, 2*35468, 5279, 2*61518, 27945,
     & 2*70975, 2*86478, 2*20514, 2*73178, 2*43098, 4701,
     & 2*59979, 58556, 69916, 2*15170, 2*4832, 43064, 71685, 4832,
     & 3*15170, 3*27679, 2*60826, 2*6187, 5*4264, 45567, 4*32269,
     & 9*62060, 13*1803, 12*51108, 2*55315, 5*54140, 13134/
      DATA P(23),(C(23,I),I = 1,99)/267523, 103650, 125480, 59978,
     & 46875, 77172, 83021, 126904, 14541, 56299, 43636, 11655,
     & 52680, 88549, 29804, 101894, 113675, 48040, 113675,
     & 34987, 48308, 97926, 5475, 49449, 6850, 2*62545, 9440,
     & 33242, 9440, 33242, 9440, 33242, 9440, 62850, 3*9440,
     & 3*90308, 9*47904, 7*41143, 5*36114, 24997, 14*65162, 7*47650,
     & 7*40586, 4*38725, 5*88329/
      DATA P(24),(C(24,I),I = 1,99)/401287, 165843, 90647, 59925,
     & 189541, 67647, 74795, 68365, 167485, 143918, 74912,
     & 167289, 75517, 8148, 172106, 126159,3*35867, 121694,
     & 52171, 95354, 2*113969, 76304, 2*123709, 144615, 123709,
     & 2*64958, 32377, 2*193002, 25023, 40017, 141605, 2*189165,
     & 141605, 2*189165, 3*141605, 189165, 20*127047, 10*127785,
     & 6*80822, 16*131661, 7114, 131661/
      DATA P(25),(C(25,I),I = 1,99)/601943, 130365, 236711, 110235,
     & 125699, 56483, 93735, 234469, 60549, 1291, 93937,
     & 245291, 196061, 258647, 162489, 176631, 204895, 73353,
     & 172319, 28881, 136787,2*122081, 275993, 64673, 3*211587,
     & 2*282859, 211587, 242821, 3*256865, 122203, 291915, 122203,
     & 2*291915, 122203, 2*25639, 291803, 245397, 284047,
     & 7*245397, 94241, 2*66575, 19*217673, 10*210249, 15*94453/
      DATA P(26),(C(26,I),I = 1,99)/902933, 333459, 375354, 102417,            
     & 383544, 292630, 41147, 374614, 48032, 435453, 281493, 358168, 
     & 114121, 346892, 238990, 317313, 164158, 35497, 2*70530, 434839,  
     & 3*24754, 393656, 2*118711, 148227, 271087, 355831, 91034, 
     & 2*417029, 2*91034, 417029, 91034, 2*299843, 2*413548, 308300,  
     & 3*413548, 3*308300, 413548, 5*308300, 4*15311, 2*176255, 6*23613, 
     & 172210, 4* 204328, 5*121626, 5*200187, 2*121551, 12*248492, 
     & 5*13942/
      DATA P(27), (C(27,I), I = 1,99)/ 1354471, 500884, 566009, 399251,
     & 652979, 355008, 430235, 328722, 670680, 2*405585, 424646, 
     & 2*670180, 641587, 215580, 59048, 633320, 81010, 20789, 2*389250,  
     & 2*638764, 2*389250, 398094, 80846, 2*147776, 296177, 2*398094,  
     & 2*147776, 396313, 3*578233, 19482, 620706, 187095, 620706, 
     & 187095, 126467, 12*241663, 321632, 2*23210, 3*394484, 3*78101, 
     & 19*542095, 3*277743, 12*457259/
      DATA P(28), (C(28,I), I = 1, 99)/ 2031713, 858339, 918142, 501970, 
     & 234813, 460565, 31996, 753018, 256150, 199809, 993599, 245149,      
     & 794183, 121349, 150619, 376952, 2*809123, 804319, 67352, 969594, 
     & 434796, 969594, 804319, 391368, 761041, 754049, 466264, 2*754049,
     & 466264, 2*754049, 282852, 429907, 390017, 276645, 994856, 250142, 
     & 144595, 907454, 689648, 4*687580, 978368, 687580, 552742, 105195, 
     & 942843, 768249, 4*307142, 7*880619, 11*117185, 11*60731,  
     & 4*178309, 8*74373, 3*214965/
*
      END SUBROUTINE DKBVRC
*
      SUBROUTINE DKSMRC( NDIM, KLIM, SUMKRO, PRIME, VK, FUNCTN, X )
      INTEGER, INTENT(IN):: NDIM, KLIM, PRIME
      DOUBLE PRECISION, INTENT(OUT) :: SUMKRO
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) :: VK,X
      INTEGER :: K, J, JP, NK
      DOUBLE PRECISION ::  ONE, XT, MVNUNI
      PARAMETER ( ONE = 1.d0 )
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE    
      SUMKRO = 0.D0
*
*     Randomize Variable Order
*      
      NK = MIN( NDIM, KLIM )
      DO J = 1, NK - 1
         CALL random_number(MVNUNI)
!     JP = J + MVNUNI()*( NK + 1 - J ) 
         JP = J +  NINT(MVNUNI*DBLE( NK - J )) ! pab 12 May 2004
        
         XT     = VK(J)
         VK(J)  = VK(JP)
         VK(JP) = XT
      END DO
*
*     Determine Random Shifts for each Variable
*
      CALL random_number(X(NDIM+1:2*NDIM))
      DO K = 1, PRIME
         X(1:NDIM) = ABS( 2.d0*MOD( DBLE(K)*VK(1:NDIM) + 
     &        X(NDIM+1:2*NDIM), ONE ) - ONE )
!         DO J = 1, NDIM
!            X(J) = ABS( 2*MOD( K*VK(J) + X(NDIM+J), ONE ) - ONE )
!         END DO
         SUMKRO = SUMKRO + ( FUNCTN(NDIM,X) - SUMKRO )/DBLE( 2*K - 1 )
         X(1:NDIM) = ONE - X(1:NDIM)
         SUMKRO = SUMKRO + ( FUNCTN(NDIM,X) - SUMKRO )/DBLE( 2*K )
      END DO
      END SUBROUTINE DKSMRC
      END MODULE DKBVRCMOD

      MODULE PRECISIONMOD
      IMPLICIT NONE
      PUBLIC
!     Note double precision is the fastest choice for x86 machines
!     double (15,307) single  (6,37) precision constants
      INTEGER, PARAMETER :: gP = SELECTED_REAL_KIND(15,307) 
      END MODULE PRECISIONMOD

      MODULE SSOBOLMOD
      USE PRECISIONMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: initSobol, sobolSeq, sobnied
      
!      BLOCK DATA BDSOBL
!
!     INITIALIZES LABELLED COMMON /SOBDAT/
!     FOR "INSOBL".
!
!     THE ARRAY POLY GIVES SUCCESSIVE PRIMITIVE
!     POLYNOMIALS CODED IN BINARY, E.G.
!          45 = 100101
!     HAS BITS 5, 2, AND 0 SET (COUNTING FROM THE
!     RIGHT) AND THEREFORE REPRESENTS
!          X**5 + X**2 + X**0
!
!     THESE  POLYNOMIALS ARE IN THE ORDER USED BY
!     SOBOL IN USSR COMPUT. MATHS. MATH. PHYS. 16 (1977),
!     236-242. A MORE COMPLETE TABLE IS GIVEN IN SOBOL AND
!     LEVITAN, THE PRODUCTION OF POINTS UNIFORMLY
!     DISTRIBUTED IN A MULTIDIMENSIONAL CUBE (IN RUSSIAN),
!     PREPRINT IPM AKAD. NAUK SSSR, NO. 40, MOSCOW 1976.
!
!     THE INITIALIZATION OF THE ARRAY mVINIT IS FROM THE
!     LATTER PAPER. FOR A POLYNOMIAL OF DEGREE M, M INITIAL
!     VALUES ARE NEEDED :  THESE ARE THE VALUES GIVEN HERE.
!     SUBSEQUENT VALUES ARE CALCULATED IN "INSOBL".
!
!      ASSUME WE ARE WORKING ON A COMPUTER WITH
!     WORD LENGTH AT LEAST mMaxBit BITS EXCLUDING SIGN.
      integer :: mI
      integer, parameter :: mMaxBit = 31
      integer, parameter :: mMaxDim = 40
      integer, parameter :: mMaxAtMost = 2**(mMaxBit-1)
!COMMON SOBOL
!         mSV       TABLE OF DIRECTION NUMBERS
!         mS        DIMENSION
!         mMAXCOL   LAST COLUMN OF V TO BE USED
!         mCOUNT    SEQUENCE NUMBER OF THIS CALL
!         mLASTQ    NUMERATORS FOR LAST VECTOR GENERATED
!         mRECIPD   (1/DENOMINATOR) FOR THESE NUMERATORS      
      INTEGER, dimension(mMaxDim,mMaxBit), SAVE :: mSV
      INTEGER, dimension(mMaxDim),         SAVE :: mLASTQ
      INTEGER, SAVE :: mS,mMAXCOL,mCOUNT,mATMOST
      REAL(KIND=gP), save :: mRECIPD
! COMMON SOBDAT
      INTEGER, save, dimension(2:mMaxDim)   ::  mPOLY
      integer, save, dimension(2:mMaxDim,8) ::  mVINIT

      DATA mPOLY    /3,7,11,13,19,25,37,59,47,
     &     61,55,41,67,97,91,109,103,115,131,
     &     193,137,145,143,241,157,185,167,229,171,
     &     213,191,253,203,211,239,247,285,369,299/
!
      DATA (mVINIT(mI,1),mI=2,40)  /39*1/
      DATA (mVINIT(mI,2),mI=3,40)  /1,3,1,3,1,3,3,1,
     &                           3,1,3,1,3,1,1,3,1,3,
     &                           1,3,1,3,3,1,3,1,3,1,
     &                           3,1,1,3,1,3,1,3,1,3/
      DATA (mVINIT(mI,3),mI=4,40)  /7,5,1,3,3,7,5,
     &                           5,7,7,1,3,3,7,5,1,1,
     &                           5,3,3,1,7,5,1,3,3,7,
     &                           5,1,1,5,7,7,5,1,3,3/
      DATA (mVINIT(mI,4),mI=6,40)  /1,7,9,13,11,
     &                           1,3,7,9,5,13,13,11,3,15,
     &                           5,3,15,7,9,13,9,1,11,7,
     &                           5,15,1,15,11,5,3,1,7,9/
      DATA (mVINIT(mI,5),mI=8,40)  /9,3,27,
     &                           15,29,21,23,19,11,25,7,13,17,
     &                           1,25,29,3,31,11,5,23,27,19,
     &                           21,5,1,17,13,7,15,9,31,9/
      DATA (mVINIT(mI,6),mI=14,40) /37,33,7,5,11,39,63,
     &                           27,17,15,23,29,3,21,13,31,25,
     &                           9,49,33,19,29,11,19,27,15,25/
      DATA (mVINIT(mI,7),mI=20,40) /13,
     &                           33,115,41,79,17,29,119,75,73,105,
     &                           7,59,65,21,3,113,61,89,45,107/
      DATA (mVINIT(mI,8),mI=38,40) /7,23,39/

      INTERFACE getMSBP
      MODULE PROCEDURE getMSBP
      END INTERFACE
      
      INTERFACE initSobol
      MODULE PROCEDURE initSobol
      END INTERFACE
      
      INTERFACE GENSCRML
      MODULE PROCEDURE GENSCRML
      END INTERFACE

      INTERFACE GENSCRMU
      MODULE PROCEDURE GENSCRMU
      END INTERFACE

      INTERFACE sobolSeq
      MODULE PROCEDURE  sobolSeq
      END INTERFACE

      INTERFACE sobnied
      MODULE PROCEDURE  sobnied
      END INTERFACE

      INTERFACE dksmrc
      MODULE PROCEDURE  dksmrc
      END INTERFACE

      INTERFACE uni
      MODULE PROCEDURE  uni
      END INTERFACE
            
      CONTAINS

      FUNCTION getMSBP(J) result (nb)
!getMSBP Returns the Most Significant Bit position
!
! CALL ix = getMSBP(x);
!
! ix = Most Significant Bit position
! x  = number
!
! getMSBP  calculates the most significant bit position in X that contains a
! one, i.e., 
!     MSB(X) = max(i|2^i<=x) for X~=0
!            
      integer, intent(in)  :: J
      integer :: nb
      integer :: I
      nb = 0
      I  = J/2
      DO WHILE (I>0)
         nb = nb + 1
         I  = I / 2
      ENDDO
      end function getMSBP
      
      SUBROUTINE initSobol(INFORM,TAUS,NDIM, ATMOST, 
     *                    NUMDS,IFLAG)
! InitSobol Initializes the sobol sequence
! Inputs:
!        NDIM   : Number of dimensions
!        ATMOST : Maximum sequence length, i.e., upper bound on the number
!                 of calls the user intends to make on "ssobseq" 
!        NUMDS  : Number of Digits to Scramble if IFLAG==1 or IFLAG==3
!        IFLAG  : integer defining scrambling of sequences:
!                 0 : No Scrambling
!                 1 : Owen type Scrambling
!                 2 : Faure-Tezuka type Scrambling
!                 3 : Owen + Faure-Tezuka type Scrambling
!       Uses the member variables:  mPOLY and mVINIT
! Outputs:
!     INFORM = 0 If no error occurred otherwise 
!              2 If NDIM < 1 .OR. mMaxDim < NDIM
!              3 If ATMOST < 1 .OR. mMaxAtMost <= ATMOST
!              4 If ((IFLAG==1 OR IFLAG==3) AND (mMaxBit < NUMDS)) 
!      TAUS  =  Defines "FAVORABLE" values as 
!               discussed in BRATLEY/FOX. These have the form
!               N = 2**K WHERE K .GE. (TAUS+NDIM-1) for integration
!               and k .gt. taus for global optimization.
!               If NDIM>12 then TAUS = -1
!   Initializes the member variables:
!         mSV, mS, mMAXCOL, mCOUNT, mLASTQ, mRECIPD, mATMOST
!        Used in SOBOLSEQ    
!
! InitSobol initializes member variables for scrambled sobol sequence
!
!   
!     THIS IS MODIFIED ROUTINE OF "INSOBL".
!
!     NEXT CHECK "ATMOST", AN UPPER BOUND ON THE NUMBER
!     OF CALLS THE USER INTENDS TO MAKE ON "GOSOBL".  IF
!     THIS IS POSITIVE AND LESS THAN mMaxAtMost = 2**(mMaxBit-1), 
!      THEN FLAG(2) = .TRUE.
!     (WE ASSUME WE ARE WORKING ON A COMPUTER WITH
!     WORD LENGTH AT LEAST mMaxBit BITS EXCLUDING SIGN.)
!     THE NUMBER OF COLUMNS OF THE ARRAY V WHICH
!     ARE INITIALIZED IS
!          mMAXCOL = NUMBER OF BITS IN ATMOST.
!     IN "GOSOBL" WE CHECK THAT THIS IS NOT EXCEEDED.
!
!     THE LEADING ELEMENTS OF EACH ROW OF V ARE
!     INITIALIZED USING "mVINIT" FROM "BDSOBL".
!     EACH ROW CORRESPONDS TO A PRIMITIVE POLYNOMIAL
!     (AGAIN, SEE "BDSOBL").  IF THE POLYNOMIAL HAS
!     DEGREE M, ELEMENTS AFTER THE FIRST M ARE CALCULATED.
!
!     THE NUMBERS IN V ARE ACTUALLY BINARY FRACTIONS.
!     LSM ARE LOWER TRIAUGULAR SCRAMBLING MATRICES.
!     USM ARE UPPER TRIAUGULAR SCRMABLING MATRIX.
!     mSV ARE SCAMBLING GENERATING MATRICES AND THE NUMBERS
!     ARE BINARY FRACTIONS.
!     "mRECIPD" HOLDS 1/(THE COMMON DENOMINATOR OF ALL
!     OF THEM).
!
!
!     "INSSOBL" IMPLICITLY COMPUTES THE FIRST SHIFTED 
!     VECTOR "mLASTQ", AND RETURN IT TO THE CALLING
!     PROGRAM. SUBSEQUENT VECTORS COME FROM "GOSSOBL".
!     "mLASTQ" HOLDS NUMERATORS OF THE LAST VECTOR GENERATED.
!
!
      integer, intent(in) :: NDIM,ATMOST,NUMDS,IFLAG
      INTEGER, INTENT(OUT) :: INFORM,TAUS
!      REAL(kind=gP), dimension(:), intent(out) :: QUASI
      INTEGER, dimension(mMaxDim,mMaxBit) ::  V, LSM
      integer, dimension(mMaxBit,mMaxBit) ::  USM
      integer, dimension(mMaxDim,mMaxBit,mMaxBit) :: TV
      integer, dimension(mMaxDim) :: SHIFT
      integer, dimension(mMaxBit) :: USHIFT
      INTEGER, dimension(13),save :: TAU
      INTEGER  I,J,K,P,M,NEWV,L,PP
      INTEGER  TEMP1,TEMP2,TEMP3,TEMP4,MAXX
      REAL(KIND=gP) ::   LL
      LOGICAL, dimension(8) ::  INCLUD
!      EXTERNAL IEOR
!      COMMON   /SOBDAT/ mPOLY,mVINIT
!      COMMON   /SOBOL/  mS,mMAXCOL,mSV,mCOUNT,mLASTQ,mRECIPD
!      SAVE     /SOBDAT/,/SOBOL/
      DATA TAU /0,0,1,3,5,8,11,15,19,23,27,31,35/
      inform  = 0
      mMAXCOL = 0
      mS      = NDIM
      mATMOST = ATMOST
      IF (mS < 1 .OR. mMaxDim < mS) THEN
         INFORM = 2
         RETURN
      ENDIF
      IF ( mATMOST < 1 .OR. mMaxAtMost <= mATMOST) THEN
         INFORM = 3
         RETURN
      ENDIF
      if ((IFLAG.EQ.1 .or. IFLAG.EQ.3) .AND. (mMaxBit < NUMDS)) then
         INFORM = 4
         return
      endif
      IF (mS .LE. 13) THEN
        TAUS = TAU(mS)
      ELSE
        TAUS = -1
!     RETURN A DUMMY VALUE TO THE CALLING PROGRAM
      ENDIF

!     FIND NUMBER OF BITS IN ATMOST
      mMAXCOL = getMSBP(mATMOST)+1  
      

!     INITIALIZE V
      V(1,1:mMAXCOL) = 1
      DO I = 2, mS ! 100
!     FIND DEGREE OF POLYNOMIAL I FROM BINARY ENCODING

        J = mPOLY(I)
        M = getMSBP(J)

!     WE EXPAND THIS BIT PATTERN TO SEPARATE COMPONENTS
!     OF THE LOGICAL ARRAY INCLUD.

        DO K = M, 1, -1
           INCLUD(K) = (MOD(J,2) .EQ. 1)
           J = J / 2
        enddo                   ! K

!     THE LEADING ELEMENTS OF ROW I COME FROM mVINIT
          V(I,1:M) = mVINIT(I, 1:M)
!
!     CALCULATE REMAINING ELEMENTS OF ROW I AS EXPLAINED
!     IN BRATLEY AND FOX, SECTION 2
        DO J = M+1, mMAXCOL
           NEWV = V(I, J-M)
           L = 1
           DO  K = 1, M
              L = 2 * L
              IF (INCLUD(K)) NEWV = IEOR(NEWV, L * V(I, J-K))
           enddo                ! K
           V(I,J) = NEWV
        enddo                   ! J
      enddo                     ! I 
!
!     MULTIPLY COLUMNS OF V BY APPROPRIATE POWER OF 2
!
      L = 1
      DO J = mMAXCOL-1, 1, -1
         L = 2 * L
         V(1:mS,J) = V(1:mS,J) * L
      enddo                     ! J
! 
! COMPUTING GENERATOR MATRICES OF USER CHOICE
!
      IF (IFLAG .EQ. 0) THEN
         FORALL (I = 1:mS, J = 1:mMAXCOL) mSV(I,J) = V(I,J)         
         SHIFT(1:mS) = 0
         LL = DBLE(2**(mMAXCOL))
      ELSE             
        IF ((IFLAG .EQ. 1) .OR. (IFLAG .EQ. 3)) THEN
         CALL GENSCRML(NUMDS,LSM,SHIFT)
         DO  I = 1,mS
           DO J = 1,mMAXCOL
             L = 1
             TEMP2 = 0
             DO P = NUMDS,1,-1
                TEMP1 = 0
                DO K = 1,mMAXCOL
                   TEMP1 = TEMP1+ 
     &                  (IBITS(LSM(I,P),K-1,1)*IBITS(V(I,J),K-1,1))
                enddo           ! K
                TEMP1 = MOD(TEMP1,2)
                TEMP2 = TEMP2+TEMP1*L   
                L = 2 * L
             enddo              ! P
              mSV(I,J) = TEMP2
           enddo                ! J 
        enddo                   ! I 
         LL= DBLE(2**(NUMDS))
       ENDIF
       IF ((IFLAG .EQ. 2) .OR. (IFLAG .EQ. 3)) THEN
          CALL GENSCRMU(USM,USHIFT) 
          IF (IFLAG .EQ. 2) THEN
             MAXX = mMAXCOL
          ELSE
             MAXX = NUMDS
          ENDIF    
          DO I = 1,mS
             DO  J = 1,mMAXCOL
                P = MAXX
                DO  K = 1,MAXX
                   IF (IFLAG .EQ. 2) THEN
                      TV(I,P,J) = IBITS(V(I,J),K-1,1)
                   ELSE
                      TV(I,P,J) = IBITS(mSV(I,J),K-1,1) 
                   ENDIF 
                   P = P-1
                enddo           ! K 
             enddo              ! J        
             DO PP = 1,mMAXCOL 
                TEMP2 = 0 
                TEMP4 = 0
                L = 1
                DO J = MAXX,1,-1
                   TEMP1 = 0
                   TEMP3 = 0
                   DO P = 1,mMAXCOL
                      TEMP1 = TEMP1 + TV(I,J,P)*USM(P,PP)
                      IF (PP .EQ. 1) THEN
                         TEMP3 = TEMP3 + TV(I,J,P)*USHIFT(P)
                      ENDIF 
                   enddo        ! P 
                   TEMP1 = MOD(TEMP1,2)
                   TEMP2 = TEMP2 + TEMP1*L
                   IF (PP .EQ. 1) THEN 
                      TEMP3  = MOD(TEMP3,2)
                      TEMP4 = TEMP4 + TEMP3*L
                   ENDIF  
                   L = 2*L
                enddo           ! J
                mSV(I,PP) = TEMP2
                IF (PP .EQ. 1) THEN
                   IF (IFLAG .EQ. 3) THEN
                      SHIFT(I) = IEOR(TEMP4, SHIFT(I))           
                   ELSE
                      SHIFT(I) = TEMP4
                   ENDIF  
                ENDIF
             enddo              ! PP
          enddo                 ! I
          LL = DBLE(2**(MAXX))
       ENDIF
      ENDIF 
!
!     mRECIPD IS 1/(COMMON DENOMINATOR OF THE ELEMENTS IN V)
!
      mRECIPD = 1.0_gP / LL

!     SET UP FIRST VECTOR AND VALUES FOR "SOBOLSEQ"
      mCOUNT = 0
      mLASTQ(1:mS) = SHIFT(1:mS)
!      QUASI(1:mS)  = DBLE(mLASTQ(1:mS))*mRECIPD
      RETURN
      END subroutine initSobol
      FUNCTION UNI() result (val)
*
*     Random number generator, adapted from F. James
*     "A Review of Random Number Generators"
*      Comp. Phys. Comm. 60(1990), pp. 329-344.
*
      real(kind=gP) SEEDS(24), TWOM24, CARRY,val
      PARAMETER ( TWOM24 = 1.0_gP/16777216.0_gP )
      INTEGER I, J
      SAVE I, J, CARRY, SEEDS
      DATA I, J, CARRY / 24, 10, 0.0 /
      DATA SEEDS / 
     & 0.8804418, 0.2694365, 0.0367681, 0.4068699, 0.4554052, 0.2880635,
     & 0.1463408, 0.2390333, 0.6407298, 0.1755283, 0.7132940, 0.4913043,
     & 0.2979918, 0.1396858, 0.3589528, 0.5254809, 0.9857749, 0.4612127,
     & 0.2196441, 0.7848351, 0.4096100, 0.9807353, 0.2689915, 0.5140357/
!     & 0.8804418_gP, 0.2694365_gP, 0.0367681_gP, 0.4068699_gP,
!     & 0.4554052_gP, 0.2880635_gP,
!     & 0.1463408_gP, 0.2390333_gP, 0.6407298_gP, 0.1755283_gP,
!     & 0.7132940_gP, 0.4913043_gP,
!     & 0.2979918_gP, 0.1396858_gP, 0.3589528_gP, 0.5254809_gP,
!     & 0.9857749_gP, 0.4612127_gP,
!     & 0.2196441_gP, 0.7848351_gP, 0.4096100_gP, 0.9807353_gP,
!     & 0.2689915_gP, 0.5140357_gP/
      
      CALL random_number(val)
      return
      val = SEEDS(I) - SEEDS(J) - CARRY
      IF ( val .LT. 0.0_gP ) THEN 
         val = val + 1.0_gP
         CARRY = TWOM24
      ELSE 
         CARRY = 0.0_gP
      ENDIF
      SEEDS(I) = val
      I = 24 - MOD( 25-I, 24 )
      J = 24 - MOD( 25-J, 24 )
      RETURN
      END function uni     
      SUBROUTINE GENSCRML(NUMDS,LSM,SHIFT)
!     GENERATING LOWER TRIANGULAR SCRMABLING MATRICES AND SHIFT VECTORS.
!     INPUTS :
!       FROM INSSOBL : NUMDS
!       FROM BLOCK DATA "SOBOL" : mS, mMAXCOL,
!
!     OUTPUTS :
!       TO initSobol : LSM, SHIFT
      integer,intent(in) :: NUMDS
      integer, dimension(mMaxDim,mMaxBit), intent(inout) :: LSM
      integer, dimension(mMaxDim), intent(inout) :: SHIFT
      INTEGER  :: P,I,J,TEMP,STEMP,L,LL
!      REAL(KIND=gP) :: UNI
!      COMMON /SOBOL/ mS,mMAXCOL
!      SAVE /SOBOL/
      
      DO 10 P = 1,mS
               SHIFT(P) = 0
               L = 1
         DO 20 I = NUMDS,1,-1
               LSM(P,I) = 0
!               CALL random_number(UNI)
               STEMP =  MOD((int(UNI()*1000.0_gP)),2)
               SHIFT(P) = SHIFT(P)+STEMP*L
               L = 2 * L
               LL = 1
            DO 30 J = mMAXCOL,1,-1
               IF (J .EQ. I) THEN
                  TEMP = 1
               ELSE IF (J .LT. I)  THEN 
!                  CALL random_number(UNI)
                  TEMP = MOD((int(UNI()*1000.0_gP)),2)
               ELSE
                  TEMP = 0
               ENDIF
               LSM(P,I) = LSM(P,I) + TEMP*LL
               LL       = 2 * LL            
 30           CONTINUE
 20        CONTINUE 
 10      CONTINUE
      RETURN
      END  SUBROUTINE GENSCRML 
      
      SUBROUTINE GENSCRMU(USM,USHIFT)

!     GENERATING UPPER TRIANGULAR SCRMABLING MATRICES AND 
!     SHIFT VECTORS.
!     INPUTS :
!       FROM BLOCK DATA "SOBOL" : mS, mMAXCOL,
!
!     OUTPUTS :
!       TO INSSOBL : USM, USHIFT
      integer, dimension(mMaxBit,mMaxBit), intent(inout) :: USM
      integer, dimension(mMaxBit), intent(inout) :: USHIFT
      INTEGER I,J,TEMP
!      REAL(KIND=gP) :: UNI
!      COMMON /SOBOL/ mS,mMAXCOL
!      SAVE /SOBOL/
      
      DO 20 I = 1,mMAXCOL
!         CALL random_number(UNI)
         USHIFT(I) = MOD((int(UNI()*1000.0_gP)),2)
         DO 30 J = 1,mMAXCOL
            IF (J .EQ. I) THEN
               TEMP = 1
            ELSE IF (J .GT. I)  THEN 
!               CALL random_number(UNI)
               TEMP = MOD((int(UNI()*1000.0_gP)),2)
            ELSE
               TEMP = 0
            ENDIF
            USM(I,J) = TEMP        
 30      CONTINUE
 20   CONTINUE 
      RETURN
      END SUBROUTINE GENSCRMU  
      
      SUBROUTINE sobolSeq(QUASI,INFORM)
!SOBOLSEQ GENERATES A NEW QUASIRANDOM VECTOR WITH EACH CALL
!
!     IT ADAPTS THE IDEAS OF ANTONOV AND SALEEV,
!     USSR COMPUT. MATHS. MATH. PHYS. 19 (1980),
!     252 - 256
!
!     The user must call "initSobol" before calling
!     "sobolSeq".  After calling "initsobol", test
!     if inform == 0. if inform>0 then
!     do not call "sobolSeq".  
!     "sobolSeq" checks that the user does not make more calls
!     than he said he would : see the comments
!     to "initSobol".
!
!     INPUTS:
!       FROM USER'S CALLING PROGRAM:
!         NONE
!
!       FROM LABELLED COMMON /SOBOL/:
!         mSV       TABLE OF DIRECTION NUMBERS
!         mS         DIMENSION
!         mMAXCOL   LAST COLUMN OF mSV TO BE USED
!         mCOUNT    SEQUENCE NUMBER OF THIS CALL
!         mLASTQ    NUMERATORS FOR LAST VECTOR GENERATED
!         mRECIPD   (1/DENOMINATOR) FOR THESE NUMERATORS
!
      REAL(KIND=gP), dimension(:), intent(out) :: QUASI
      integer, intent(inout) :: inform
      INTEGER ::  I,L
!      INTEGER  mSV(40,31),mS,mMAXCOL,mCOUNT,mLASTQ(40)
!      COMMON   /SOBOL/ S,mMAXCOL,mSV,mCOUNT,mLASTQ,mRECIPD
!      SAVE     /SOBOL/
!      
      
!      FORALL ( I = 1:mS)
!        QUASI(I)  = DBLE(mLASTQ(I)) * mRECIPD
!      END FORALL
      QUASI(1:mS)  = DBLE(mLASTQ(1:mS))*mRECIPD
!     FIND POSITION OF RIGHTMOST ZERO  BIT IN mCOUNT
      L = 1
      I = mCOUNT      
      do while (MOD(I,2) .EQ. 1)
         I = I / 2
         L = L + 1
      ENDDO
!     CHECK THAT THE USER IS NOT CHEATING 
      IF (L > mMAXCOL) THEN
         INFORM = 4
!     WARNING: Reached the end of the sobol sequence
!     Next call will wrap around and return the same numbers 
!     as for mCOUNT = 0
!     Call initSobol to increase mATMOST before calling sobolseq again.
      else
         INFORM = 0
!     Calculate the new components of quasi,
!     first the numerators
         FORALL ( I = 1:mS)
            mLASTQ(I) = IEOR(mLASTQ(I), mSV(I,L))
         END FORALL                    
         mCOUNT = mCOUNT + 1
      ENDIF
      RETURN
      END SUBROUTINE sobolSeq      
      
      
!***********************************************************
!    MAIN INTEGRATION ROUTINE SOBNIED
!***********************************************************  

      SUBROUTINE SOBNIED( NDIM, MINVLS, MAXVLS, FUNCTN, ABSEPS, RELEPS,
     &                   ABSERR, FINEST, INFORM )
      use precisionmod
      implicit none
*
*  Automatic Multidimensional Integration Subroutine
*               
*  AUTHOR: Per A. Brodtkorb
!  Norwegian Defence Research Establishment
!  P.O. Box 115
!  N-3191 Horten
!  Norway
!  Email: Per.Brodtkorb@ffi.no
!
*         Last Change: 6/19/2004
*
*  SOBNIED computes an approximation to the integral
*
*      1  1     1
*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1)
*      0  0     0
*
*
*  SOBNIED uses scrambled SOBOL sequences for the first 40 variables. 
*  The primary reference is

*  If there are more than 40 variables, the remaining variables are
*  integrated using  the rule described in the reference
*   "On a Number-Theoretical Integration Method"
*   H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11.
*   
***************  Parameters ********************************************
****** Input parameters
*  NDIM    Number of variables, must exceed 1, but not exceed 100
*  MINVLS  Integer minimum number of function evaluations allowed.
*          MINVLS must not exceed MAXVLS.  If MINVLS < 0 then the
*          routine assumes a previous call has been made with 
*          the same integrand and continues that calculation.
*  MAXVLS  Integer maximum number of function evaluations allowed.
*  FUNCTN  EXTERNALly declared user defined function to be integrated.
*          It must have parameters (NDIM,Z), where Z is a real array
*          of dimension NDIM.
*                                     
*  ABSEPS  Required absolute accuracy.
*  RELEPS  Required relative accuracy.
****** Output parameters
*  MINVLS  Actual number of function evaluations used.
*  ABSERR  Estimated absolute accuracy of FINEST.
*  FINEST  Estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when 
*                     ABSERR <= MAX(ABSEPS, RELEPS*ABS(FINEST))
*                  and 
*                     INTVLS <= MAXCLS.
*          INFORM = 1 If MAXVLS was too small to obtain the required 
*                  accuracy. In this case a value FINEST is returned with 
*                  estimated absolute accuracy ABSERR.
*          INFORM = 2 If NDIM>1040 or NDIM<1
************************************************************************
      INTEGER, INTENT(IN)    :: NDIM,  MAXVLS
      INTEGER, INTENT(INOUT) :: MINVLS
      INTEGER, INTENT(OUT)   :: INFORM
      REAL(KIND=gP), INTENT(IN)  :: ABSEPS, RELEPS
      REAL(KIND=gP), INTENT(OUT) :: FINEST, ABSERR 
      INTEGER :: NP,PLIM,NLIM,KLIM,KLIMI,SAMPLS,I,INTVLS,MINSMP,NK
      integer  :: numRep, J, TAUS
      INTEGER, parameter :: NUMDS=30,IFLAG=1
      PARAMETER ( PLIM = 28, NLIM = 1040, KLIM = mMaxDim, MINSMP = 8 )
      INTEGER , DIMENSION(PLIM) :: P
      REAL(KIND=gP) :: DIFINT,FINVAL,VARSQR,VAREST,VARPRD,VALUE
      REAL(KIND=gP), PARAMETER :: ONE = 1.D0 , ZERO = 0.D0
      REAL(KIND=gP), DIMENSION(2*NLIM) :: X  = 0.d0
      REAL(KIND=gP), DIMENSION(NLIM)   :: VK  = 0.d0
      logical :: NPtooSmall,errorTooLarge,numSamplesOk
      INTERFACE
         REAL(KIND=gP) FUNCTION FUNCTN(N,Z)
         use precisionmod
         REAL(KIND=gP),DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      DATA P / 31, 47, 73, 113, 173, 263, 397, 593, 907, 1361,
     &     2053, 3079, 4621, 6947, 10427, 15641, 23473, 35221, 
     &     52837, 79259, 118891, 178349, 267523, 401287, 601943,
     &    902933,1354471,2031713/
      SAVE P, SAMPLS, NP, VAREST
      IF ( NDIM .GT. NLIM .OR. NDIM .LT. 1 ) THEN
         INFORM = 2
         FINEST = ZERO
         ABSERR = ONE
         RETURN
      ENDIF
      NK = MIN( NDIM, KLIM )
      
      
      IF ( MINVLS >= 0 ) THEN
         FINEST = ZERO
         VAREST = ZERO
         SAMPLS = MINSMP 
         NP = 1
         NPtooSmall = ( MINVLS >= 2*SAMPLS*P(NP) )
         do while(NPtooSmall .AND. NP<PLIM)
            NP = NP + 1
            NPtooSmall = ( MINVLS >= 2*SAMPLS*P(NP) )
         enddo
         if (NPtooSmall) then
            SAMPLS = MAX( MINSMP, MINVLS/( 2*P(NP) ) )
         endif
      ENDIF
      numRep = 1 !max(1,nint(MAXVLS/mMaxAtMost))
      
      INFORM = 1
      INTVLS = 0
      KLIMI = KLIM
      errorTooLarge = .TRUE.
      do J = 1,numRep
         CALL initSobol(inform,TAUS,NK,MAXVLS/numRep,NUMDS,IFLAG)
         if (inform.ne.0) then
            FINEST = ZERO
            ABSERR = ONE
            RETURN
         endif
         INFORM = 1 
         numSamplesOk = ( INTVLS + 2*SAMPLS*P(NP) <= MAXVLS )
         do while (errorTooLarge .and. numSamplesOk)
            DO I = 1, NDIM-NK
               VK(I) = INT( P(NP)*2**(DBLE(I)/(NDIM-KLIM+1)) ) 
               VK(I) = MOD( VK(I)/P(NP), ONE ) 
            END DO
            FINVAL = ZERO
            VARSQR = ZERO
            DO I = 1, SAMPLS
               CALL DKSMRC( NDIM, KLIMI, VALUE, P(NP),VK, FUNCTN, X )
               DIFINT = ( VALUE - FINVAL )/DBLE(I)
               FINVAL = FINVAL + DIFINT
               VARSQR = DBLE( I - 2 )*VARSQR/DBLE(I) + DIFINT*DIFINT
            END DO
            INTVLS = INTVLS + 2*SAMPLS*P(NP)
            VARPRD = VAREST*VARSQR
            FINEST = FINEST + ( FINVAL - FINEST )/( ONE + VARPRD )
            IF ( VARSQR > ZERO ) VAREST = ( ONE + VARPRD )/VARSQR
            ABSERR = 3.0_gP*SQRT( VARSQR/( ONE + VARPRD ) )
            errorTooLarge = (ABSERR > MAX(ABSEPS, ABS(FINEST)*RELEPS))
            IF ( errorTooLarge ) THEN
               IF ( NP < PLIM ) THEN
                  NP = NP + 1
               ELSE
                  SAMPLS = MIN(3*SAMPLS/2, (MAXVLS - INTVLS)/(2*P(NP)))
                  SAMPLS = MAX( MINSMP, SAMPLS )
               ENDIF
               numSamplesOk = ( INTVLS + 2*SAMPLS*P(NP) <= MAXVLS )
            ELSE
               INFORM = 0
            ENDIF
         enddo
      enddo
      MINVLS = INTVLS
      END  SUBROUTINE SOBNIED
      SUBROUTINE DKSMRC( NDIM, KLIM, SUMKRO, PRIME, VK,FUNCTN, X )
      use precisionmod
      implicit none
      INTEGER, INTENT(IN):: NDIM, KLIM, PRIME
      REAL(KIND=gP), INTENT(OUT) :: SUMKRO
      REAL(KIND=gP), DIMENSION(:), INTENT(INOUT) :: VK,X
      INTEGER :: K, NK, inform
      REAL(KIND=gP) ::  ONE, XT, MVNUNI
      PARAMETER ( ONE = 1.0_gP )
      INTERFACE
         REAL(KIND=gP) FUNCTION FUNCTN(N,Z)
         use precisionmod
         REAL(KIND=gP),DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE    
      SUMKRO = 0.0_gP
      NK = MIN( NDIM, KLIM )
*     Determine Random Shifts for each Variable
      if (NK<NDIM) THEN
         CALL random_number(X(NDIM+NK:2*NDIM))
      ENDIF
      DO K = 1, PRIME
         CALL sobolSeq(X,inform)
         if (NK<NDIM) THEN
            X(NK+1:NDIM) = ABS( 2.0_gP*MOD( DBLE(K)*VK(1:NDIM-NK) + 
     &        X(NDIM+1+NK:2*NDIM), ONE ) - ONE )
         endif
         SUMKRO = SUMKRO + ( FUNCTN(NDIM,X) - SUMKRO )/DBLE( 2*K - 1 )
!         X(1:NDIM) = ONE - X(1:NDIM)
         CALL sobolSeq(X,inform)
         if (NK<NDIM) THEN
            X(NK+1:NDIM) = ONE - X(NK+1:NDIM)
         endif
         SUMKRO = SUMKRO + ( FUNCTN(NDIM,X) - SUMKRO )/DBLE( 2*K )
      END DO
      END SUBROUTINE DKSMRC
      END MODULE SSOBOLMOD

*  KROBOVMOD is a module  containing a:
*
*  Automatic Multidimensional Integration Subroutine
*               
*         AUTHOR: Alan Genz
*                 Department of Mathematics
*                 Washington State University
*                 Pulman, WA 99164-3113
*                 Email: AlanGenz@wsu.edu
*
*         Last Change: 4/15/98
*
* revised pab 10.03.2000
*   - updated to f90 (i.e. changed to assumed shape arrays + changing integers to DBLE)
*   - put it into a module
*
*  KROBOV computes an approximation to the integral
*
*      1  1     1
*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1)
*      0  0     0
*
*
*  KROBOV uses randomized Korobov rules. The primary references are
*  "Randomization of Number Theoretic Methods for Multiple Integration"
*   R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13, pp. 904-14,
*  and 
*   "Optimal Parameters for Multidimensional Integration", 
*    P. Keast, SIAM J Numer Anal, 10, pp.831-838.
*   
***************  Parameters ********************************************
****** Input parameters
*  NDIM    Number of variables, must exceed 1, but not exceed 100
*  MINVLS  Integer minimum number of function evaluations allowed.
*          MINVLS must not exceed MAXVLS.  If MINVLS < 0 then the
*          routine assumes a previous call has been made with 
*          the same integrand and continues that calculation.
*  MAXVLS  Integer maximum number of function evaluations allowed.
*  FUNCTN  EXTERNALly declared user defined function to be integrated.
*          It must have parameters (NDIM,Z), where Z is a real array
*          of dimension NDIM.
*  ABSEPS  Required absolute accuracy.
*  RELEPS  Required relative accuracy.
****** Output parameters
*  MINVLS  Actual number of function evaluations used.
*  ABSERR  Estimated absolute accuracy of FINEST.
*  FINEST  Estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when 
*                     ABSERR <= MAX(ABSEPS, RELEPS*ABS(FINEST))
*                  and 
*                     INTVLS <= MAXCLS.
*          INFORM = 1 If MAXVLS was too small to obtain the required 
*          accuracy. In this case a value FINEST is returned with 
*          estimated absolute accuracy ABSERR.
************************************************************************
! You may  initialize the random generator before you 
!  call KROBOV by the following lines:
!
!      call random_seed(SIZE=seed_size) 
!      allocate(seed(seed_size)) 
!      call random_seed(GET=seed(1:seed_size))  ! get current seed
!      seed(1)=seed1                            ! change seed
!      call random_seed(PUT=seed(1:seed_size)) 
!      deallocate(seed)


      MODULE KROBOVMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: KROBOV
 
      INTERFACE KROBOV
      MODULE PROCEDURE KROBOV
      END INTERFACE

      INTERFACE KROSUM
      MODULE PROCEDURE KROSUM
      END INTERFACE

      CONTAINS   

!***********************************************************
!    MAIN INTEGRATION ROUTINE KROBOV
!***********************************************************  
      SUBROUTINE KROBOV( NDIM, MINVLS, MAXVLS, FUNCTN, ABSEPS, RELEPS,
     &                   ABSERR, FINEST, INFORM )
*
*  Automatic Multidimensional Integration Subroutine
*               
*         AUTHOR: Alan Genz
*                 Department of Mathematics
*                 Washington State University
*                 Pulman, WA 99164-3113
*                 Email: AlanGenz@wsu.edu
*
*         Last Change: 4/15/98
*
*  KROBOV computes an approximation to the integral
*
*      1  1     1
*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1)
*      0  0     0
*
*
*  KROBOV uses randomized Korobov rules. The primary references are
*  "Randomization of Number Theoretic Methods for Multiple Integration"
*   R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13, pp. 904-14,
*  and 
*   "Optimal Parameters for Multidimensional Integration", 
*    P. Keast, SIAM J Numer Anal, 10, pp.831-838.
*   
***************  Parameters ********************************************
****** Input parameters
*  NDIM    Number of variables, must exceed 1, but not exceed 100
*  MINVLS  Integer minimum number of function evaluations allowed.
*          MINVLS must not exceed MAXVLS.  If MINVLS < 0 then the
*          routine assumes a previous call has been made with 
*          the same integrand and continues that calculation.
*  MAXVLS  Integer maximum number of function evaluations allowed.
*  FUNCTN  EXTERNALly declared user defined function to be integrated.
*          It must have parameters (NDIM,Z), where Z is a real array
*          of dimension NDIM.
*  ABSEPS  Required absolute accuracy.
*  RELEPS  Required relative accuracy.
****** Output parameters
*  MINVLS  Actual number of function evaluations used.
*  ABSERR  Estimated absolute accuracy of FINEST.
*  FINEST  Estimated value of integral.
*  INFORM  INFORM = 0 for normal exit, when 
*                     ABSERR <= MAX(ABSEPS, RELEPS*ABS(FINEST))
*                  and 
*                     INTVLS <= MAXCLS.
*          INFORM = 1 If MAXVLS was too small to obtain the required 
*                   accuracy. In this case a value FINEST is returned with 
*                   estimated absolute accuracy ABSERR.
*          INFORM = 2 If NDIM>100 or NDIM<1
************************************************************************
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, MAXVLS 
      INTEGER, INTENT(INOUT) ::MINVLS
      INTEGER, INTENT(OUT) ::INFORM
      DOUBLE PRECISION, INTENT(IN)  :: ABSEPS, RELEPS
      DOUBLE PRECISION, INTENT(OUT) :: FINEST, ABSERR
!     Local variables:	 
      INTEGER :: NP, PLIM, NLIM, SAMPLS, I, INTVLS, MINSMP
      PARAMETER ( PLIM = 20, NLIM = 100, MINSMP = 6 )
      INTEGER, DIMENSION(PLIM,NLIM) :: C
      INTEGER, DIMENSION(PLIM)      :: P
      DOUBLE PRECISION :: DIFINT, FINVAL, VARSQR, VAREST, VARPRD, VALUE
      DOUBLE PRECISION, DIMENSION(NLIM) :: ALPHA, X, VK
      DOUBLE PRECISION :: ONE
      PARAMETER ( ONE = 1.d0 )
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      DATA P /113, 173, 263,397,593,907,1361,2053,3079,4621,6947,
     &     10427, 15641,23473, 35221, 52837, 79259,
     &     118891, 178349, 267523 /
        DATA ( C( 1,I), I = 1, 99 ) /   
     &     42,    54,    55,    32,    13,    26,    26,    13,    26,
     &     14,    13,    26,    35,     2,     2,     2,     2,    56,
     &     28,     7,     7,    28,     4,    49,     4,    40,    48,
     &      5,    35,    27,    16,    16,     2,     2,     7,    28,
     &      4,    49,     4,    56,     8,     2,     2,    56,     7,
     &     16,    28,     7,     7,    28,     4,    49,     4,    37,
     &     55,    21,    33,    40,    16,    16,    28,     7,    16,
     &     28,     4,    49,     4,    56,    35,     2,     2,     2,
     &     16,    16,    28,     4,    16,    28,     4,    49,     4,
     &     40,    40,     5,    42,    27,    16,    16,    28,     4,
     &     16,    28,     4,    49,     4,     8,     8,     2,     2/
      DATA  ( C( 2,I), I = 1, 99 ) /    
     &     64,    34,    57,     9,    72,    86,    16,    75,    75,
     &     70,    42,     2,    86,    62,    62,    30,    30,     5,
     &     42,    70,    70,    70,    53,    70,    70,    53,    42,
     &     62,    53,    53,    53,    69,    75,     5,    53,    86,
     &      2,     5,    30,    75,    59,     2,    69,     5,     5,
     &     63,    62,     5,    69,    30,    44,    30,    86,    86,
     &      2,    69,     5,     5,     2,     2,    61,    69,    17,
     &      2,     2,     2,    53,    69,     2,     2,    86,    69,
     &     13,     2,     2,    37,    43,    65,     2,     2,    30,
     &     86,    45,    16,    32,    18,    86,    86,    86,     9,
     &     63,    63,    11,    76,    76,    76,    63,    60,    70/
      DATA  ( C( 3,I), I = 1, 99 ) /   
     &    111,    67,    98,    36,    48,   110,     2,   131,     2,
     &      2,   124,   124,    48,     2,     2,   124,   124,    70,
     &     70,    48,   126,    48,   126,    56,    65,    48,    48,
     &     70,     2,    92,   124,    92,   126,   131,   124,    70,
     &     70,    70,    20,   105,    70,     2,     2,    27,   108,
     &     27,    39,     2,   131,   131,    92,    92,    48,     2,
     &    126,    20,   126,     2,     2,   131,    38,   117,     2,
     &    131,    68,    58,    38,    90,    38,   108,    38,     2,
     &    131,   131,   131,    68,    14,    94,   131,   131,   131,
     &    108,    18,   131,    56,    85,   117,   117,     9,   131,
     &    131,    55,    92,    92,    92,   131,   131,    48,    48/
      DATA  ( C( 4,I), I = 1, 99 ) /    
     &    151,   168,    46,   197,    69,    64,     2,   198,   191,
     &    134,   134,   167,   124,    16,   124,   124,   124,   124,
     &    141,   134,   128,     2,     2,    32,    32,    32,    31,
     &     31,    64,    64,    99,     4,     4,   167,   124,   124,
     &    124,   124,   124,   124,   107,    85,    79,    85,   111,
     &     85,   128,    31,    31,    31,    31,    64,   167,     4,
     &    107,   167,   124,   124,   124,   124,   124,   124,   107,
     &    183,     2,     2,     2,    62,    32,    31,    31,    31,
     &     31,    31,   167,     4,   107,   167,   124,   124,   124,
     &    124,   124,   124,   107,   142,   184,   184,    65,    65,
     &    183,    31,    31,    31,    31,    31,   167,     4,   107/
      DATA  ( C( 5,I), I = 1, 99 ) /   
     &    229,    40,   268,    42,   153,   294,    71,     2,   130,
     &    199,   199,   199,   149,   199,   149,   153,   130,   149,
     &    149,    15,   119,   294,    31,    82,   260,   122,   209,
     &    209,   122,   296,   130,   130,   260,   260,    30,   206,
     &     94,   209,    94,   122,   209,   209,   122,   122,   209,
     &    130,     2,   130,   130,    38,    38,    79,    82,    94,
     &     82,   122,   122,   209,   209,   122,   122,   168,   220,
     &     62,    60,   168,   282,   282,    82,   209,   122,    94,
     &    209,   122,   122,   122,   122,   258,   148,   286,   256,
     &    256,    62,    62,    82,   122,    82,    82,   122,   122,
     &    122,   209,   122,    15,    79,    79,    79,    79,   168/
      DATA  ( C( 6,I), I = 1, 99 ) /   
     &    264,   402,   406,   147,   452,   153,   224,     2,     2,
     &    224,   224,   449,   101,   182,   449,   101,   451,   181,
     &    181,   101,   101,   377,    85,   453,   453,   453,    85,
     &    197,   451,     2,     2,   101,   449,   449,   449,   173,
     &    173,     2,   453,   453,     2,   426,    66,   367,   426,
     &    101,   453,     2,    32,    32,    32,   101,     2,     2,
     &    453,   223,   147,   449,   290,     2,   453,     2,    83,
     &    223,   101,   453,     2,    83,    83,   147,     2,   453,
     &    147,   147,   147,   147,   147,   147,   147,   453,   153,
     &    153,   147,     2,   224,   290,   320,   453,   147,   431,
     &    383,   290,   290,     2,   162,   162,   147,     2,   162/
      DATA ( C( 7,I), I = 1, 99 ) /   
     &    505,   220,   195,   410,   199,   248,   460,   471,     2,
     &    331,   662,   547,   209,   547,   547,   209,     2,   680,
     &    680,   629,   370,   574,    63,    63,   259,   268,   259,
     &    547,   209,   209,   209,   547,   547,   209,   209,   547,
     &    547,   108,    63,    63,   108,    63,    63,   108,   259,
     &    268,   268,   547,   209,   209,   209,   209,   547,   209,
     &    209,   209,   547,   108,    63,    63,    63,   405,   285,
     &    234,   259,   259,   259,   259,   209,   209,   209,   209,
     &    209,   209,   209,   209,   547,   289,   289,   234,   285,
     &    316,     2,   410,   259,   259,   259,   268,   209,   209,
     &    209,   209,   547,   547,   209,   209,   209,   285,   316/
      DATA ( C( 8,I), I = 1, 99 ) /   
     &    468,   635,   849,   687,   948,    37,  1014,   513,     2,
     &      2,     2,     2,     2,  1026,     2,     2,  1026,   201,
     &    201,     2,  1026,   413,  1026,  1026,     2,     2,   703,
     &    703,     2,     2,   393,   393,   678,   413,  1026,     2,
     &      2,  1026,  1026,     2,   405,   953,     2,  1026,   123,
     &    123,   953,   953,   123,   405,   794,   123,   647,   613,
     &   1026,   647,   768,   953,   405,   953,   405,   918,   918,
     &    123,   953,   953,   918,   953,   536,   405,    70,   124,
     &   1005,   529,   207,   405,   405,   953,   953,   123,   918,
     &    918,   953,   405,   918,   953,   468,   405,   794,   794,
     &    647,   613,   548,   405,   953,   405,   953,   123,   918/
      DATA ( C( 9,I), I = 1, 99 ) /   
     &   1189,  1423,   287,   186,   341,    77,   733,   733,  1116,
     &      2,  1539,     2,     2,     2,     2,     2,  1116,   847,
     &   1174,     2,   827,   713,   910,   944,   139,  1174,  1174,
     &   1539,  1397,  1397,  1174,   370,    33,  1210,     2,   370,
     &   1423,   370,   370,  1423,  1423,  1423,   434,  1423,   901,
     &    139,  1174,   427,   427,   200,  1247,   114,   114,  1441,
     &    139,   728,  1116,  1174,   139,   113,   113,   113,  1406,
     &   1247,   200,   200,   200,   200,  1247,  1247,    27,   427,
     &    427,  1122,  1122,   696,   696,   427,  1539,   435,  1122,
     &    758,  1247,  1247,  1247,   200,   200,   200,  1247,   114,
     &     27,   118,   118,   113,   118,   453,   453,  1084,  1406/
      DATA ( C(10,I), I = 1, 99 ) /   
     &   1764,  1349,  1859,   693,    78,   438,   531,    68,  2234,
     &   2310,  2310,  2310,     2,  2310,  2310,  2102,  2102,   178,
     &    314,   921,  1074,  1074,  1074,  2147,   314,  1869,   178,
     &    178,  1324,  1324,   510,  2309,  1541,  1541,  1541,  1541,
     &    342,  1324,  1324,  1324,  1324,   510,   570,   570,  2197,
     &    173,  1202,   998,  1324,  1324,   178,  1324,  1324,  1541,
     &   1541,  1541,   342,  1541,   886,   178,  1324,  1324,  1324,
     &    510,   784,   784,   501,   652,  1541,  1541,  1324,   178,
     &   1324,   178,  1324,  1541,   342,  1541,  2144,   784,  2132,
     &   1324,  1324,  1324,  1324,   510,   652,  1804,  1541,  1541,
     &   1541,  2132,  1324,  1324,  1324,   178,   510,  1541,   652/
      DATA  ( C(11,I), I = 1, 99 ) /   
     &   2872,  1238,   387,  2135,   235,  1565,   221,  1515,  2950,
     &    486,  3473,     2,  2950,   982,  2950,  3122,  2950,  3172,
     &   2091,  2091,     9,  3449,  3122,  2846,  3122,  3122,  1947,
     &   2846,  3122,   772,  1387,  2895,  1387,     3,     3,     3,
     &   1320,  1320,  2963,  2963,  1320,  1320,  2380,   108,  1284,
     &    702,  1429,   907,  3220,  3125,  1320,  2963,  1320,  1320,
     &   2963,  1320,  1639,  3168,  1660,  2895,  2895,  2895,  2895,
     &   1639,  1297,  1639,   404,  3168,  2963,  2943,  2943,   550,
     &   1387,  1387,  2895,  2895,  2895,  1387,  2895,  1387,  2895,
     &   1320,  1320,  2963,  1320,  1320,  1320,  2963,  1320,     2,
     &   3473,     2,  3473,   772,  2550,     9,  1320,  2963,  1320/
      DATA ( C(12,I), I = 1, 99 ) /  
     &   4309,  2339,  4154,  4480,  4967,   630,  5212,  2592,  4715,
     &   1808,  1808,  5213,     2,   216,  4014,  3499,  3499,  4204,
     &   2701,  2701,  5213,  4157,  1209,  4157,  4460,   335,  4460,
     &   1533,  4575,  4013,  4460,  1881,  2701,  4030,  4030,  1881,
     &   4030,  1738,   249,   335,    57,  2561,  2561,  2561,  1533,
     &   1533,  1533,  4013,  4013,  4013,  4013,  4013,  1533,   856,
     &    856,   468,   468,   468,  2561,   468,  2022,  2022,  2434,
     &    138,  4605,  1100,  2561,  2561,    57,    57,  3249,   468,
     &    468,   468,    57,   468,  1738,   313,   856,     6,  3877,
     &    468,   557,   468,    57,   468,  4605,  2022,     2,  4605,
     &    138,  1100,    57,  2561,    57,    57,  2022,  5213,  3249/
      DATA  ( C(13,I), I = 1, 99 ) /  
     &   6610,  1658,  3022,  2603,  5211,   265,  4985,     3,  4971,
     &   2127,  1877,  1877,     2,  2925,  3175,  3878,  1940,  1940,
     &   1940,  5117,  5117,  5771,  5117,  5117,  5117,  5117,  5117,
     &   5771,  5771,  5117,  3658,  3658,  3658,  3658,  3658,  3658,
     &   5255,  2925,  2619,  1714,  4100,  6718,  6718,  4100,  2322,
     &    842,  4100,  6718,  5119,  4728,  5255,  5771,  5771,  5771,
     &   5117,  5771,  5117,  5117,  5117,  5117,  5117,  5117,  5771,
     &   5771,  1868,  4483,  4728,  3658,  5255,  3658,  5255,  3658,
     &   3658,  5255,  5255,  3658,  6718,  6718,   842,  2322,  6718,
     &   4100,  6718,  4100,  4100,  5117,  5771,  5771,  5117,  5771,
     &   5771,  5771,  5771,  5117,  5117,  5117,  5771,  5771,  1868/
      DATA  ( C(14,I), I = 1, 99 ) /  
     &   9861,  7101,  6257,  7878, 11170, 11638,  7542,  2592,  2591,
     &   6074,  1428,  8925, 11736,  8925,  5623,  5623,  1535,  6759,
     &   9953,  9953, 11459,  9953,  7615,  7615, 11377, 11377,  2762,
     &  11734, 11459,  6892,  1535,  6759,  4695,  1535,  6892,     2,
     &      2,  6892,  6892,  4177,  4177,  6339,  6950,  1226,  1226,
     &   1226,  4177,  6892,  6890,  3640,  3640,  1226, 10590, 10590,
     &   6950,  6950,  6950,  1226,  6950,  6950,  7586,  7586,  7565,
     &   7565,  3640,  3640,  6950,  7565,  6950,  3599,  3599,  3599,
     &   2441,  4885,  4885,  4885,  7565,  7565,  1226,  1226,  1226,
     &   6950,  7586,  1346,  2441,  6339,  3640,  6950, 10590,  6339,
     &   6950,  6950,  6950,  1226,  1226,  6950,   836,  6891,  7565/
      DATA  ( C(15,I), I = 1, 99 ) /  
     &  13482,  5629,  6068, 11974,  4732, 14946, 12097, 17609, 11740,
     &  15170, 10478, 10478, 17610,     2,     2,  7064,  7064,  7064,
     &   5665,  1771,  2947,  4453, 12323, 17610, 14809, 14809,  5665,
     &   5665,  2947,  2947,  2947,  2947, 12323, 12323,  4453,  4453,
     &   2026, 11772,  2026, 11665, 12323, 12323,  3582,  2940,  2940,
     &   6654,  4449,  9254, 11470,   304,   304, 11470,   304, 11470,
     &   6156,  9254, 11772,  6654, 11772,  6156, 11470, 11470, 11772,
     &  11772, 11772, 11470, 11470,   304, 11470, 11470,   304, 11470,
     &    304, 11470,   304,   304,   304,  6654, 11508,   304,   304,
     &   6156,  3582, 11470, 11470, 11470, 17274,  6654,  6654,  6744,
     &   6711,  6654,  6156,  3370,  6654, 12134,  3370,  6654,  3582/
      DATA  ( C(16,I), I = 1, 99 ) /  
     &  13482,  5629,  6068, 11974,  4732, 14946, 12097, 17609, 11740,
     &  15170, 10478, 10478, 17610,     2,     2,  7064,  7064,  7064,
     &   5665,  1771,  2947,  4453, 12323, 17610, 14809, 14809,  5665,
     &   5665,  2947,  2947,  2947,  2947, 12323, 12323,  4453,  4453,
     &   2026, 11772,  2026, 11665, 12323, 12323,  3582,  2940,  2940,
     &   6654,  4449,  9254, 11470,   304,   304, 11470,   304, 11470,
     &   6156,  9254, 11772,  6654, 11772,  6156, 11470, 11470, 11772,
     &  11772, 11772, 11470, 11470,   304, 11470, 11470,   304, 11470,
     &    304, 11470,   304,   304,   304,  6654, 11508,   304,   304,
     &   6156,  3582, 11470, 11470, 11470, 17274,  6654,  6654,  6744,
     &   6711,  6654,  6156,  3370,  6654, 12134,  3370,  6654,  3582/
      DATA  ( C(17,I), I = 1, 99 ) /  
     &  34566, 38838, 23965, 17279, 35325, 33471,   330, 36050, 26419,
     &   3012, 38428, 36430, 36430, 36755, 39629,  5749,  5749, 36755,
     &   5749, 14353, 14353, 14353, 32395, 32395, 32395, 32395, 32396,
     &  32396, 32396, 32396, 27739, 14353, 36430, 36430, 36430, 15727,
     &  38428, 28987, 28987, 27739, 38428, 27739, 18786, 14353, 15727,
     &  28987, 19151, 19757, 19757, 19757, 14353, 22876, 19151, 24737,
     &  24737,  4412, 30567, 30537, 19757, 30537, 19757, 30537, 30537,
     &   4412, 24737, 28987, 19757, 19757, 19757, 30537, 30537, 33186,
     &   4010,  4010,  4010, 17307, 15217, 32789, 37709,  4010,  4010,
     &   4010, 33186, 33186,  4010, 11057, 39388, 33186,  1122, 15089,
     &  39629,     2,     2, 23899, 16466, 16466, 17038,  9477,  9260/
      DATA ( C(18,I), I = 1, 99 ) / 
     &  31929, 40295,  2610,  5177, 17271, 23770,  9140,   952, 39631,
     &      3, 11424, 49719, 38267, 25172,     2,     2, 59445,     2,
     &  59445, 38267, 44358, 14673, 53892, 14674, 14673, 14674, 41368,
     &  17875, 17875, 30190, 20444, 55869, 15644, 25499, 15644, 20983,
     &  44358, 15644, 15644,   485, 41428,   485,   485,   485, 41428,
     &  53798, 50230, 53798, 50253, 50253, 35677, 35677, 17474,  7592,
     &   4098, 17474,   485, 41428,   485, 41428,   485, 41428,   485,
     &  41428, 41428, 41428, 41428, 41428,  9020, 22816,  4098,  4098,
     &   4098,  7592, 42517,   485, 50006, 50006, 22816, 22816,  9020,
     &    485, 41428, 41428, 41428, 41428, 50006,   485, 41428, 41428,
     &  41428, 41428, 22816, 41428, 41428,   485,   485,   485,  9020/
      DATA  ( C(19,I), I = 1, 99 ) / 
     &  73726, 16352, 16297, 74268, 60788,  8555,  1077, 25486, 86595,
     &  59450, 19958, 62205, 62205,  4825,  4825, 89174, 89174, 62205,
     &  19958, 62205, 19958, 27626, 63080, 62205, 62205, 62205, 19958,
     &   8914, 83856, 30760, 47774, 47774, 19958, 62205, 39865, 39865,
     &  74988, 75715, 75715, 74988, 34522, 74988, 74988, 25101, 44621,
     &  44621, 44621, 25101, 25101, 25101, 44621, 47768, 41547, 44621,
     &  10273, 74988, 74988, 74988, 74988, 74988, 74988, 34522, 34522,
     &  67796, 67796, 30208,     2, 67062, 18500, 29251, 29251,     2,
     &  67796, 67062, 38649, 59302,  6225, 67062,  6475,  6225, 46772,
     &  38649, 67062, 46772, 46772, 67062, 46772, 25372, 67062,  6475,
     &  25372, 67062, 67062, 67062,  6225, 67062, 67062, 68247, 80676/
      DATA ( C(20,I), I = 1, 99 )/ 
     & 103650, 50089, 70223, 41805, 74847,112775, 40889, 64866, 44053,
     &   1754,129471, 13630, 53467, 53467, 61378,133761,     2,133761,
     &      2,133761,133761, 65531, 65531, 65531, 38080,133761,133761,
     & 131061,  5431, 65531, 78250, 11397, 38841, 38841,107233,107233,
     & 111286, 19065, 38841, 19065, 19065, 16099,127638, 82411, 96659,
     &  96659, 82411, 96659, 82411, 51986,101677, 39264, 39264,101677,
     &  39264, 39264, 47996, 96659, 82411, 47996, 10971, 10004, 82411,
     &  96659, 82411, 82411, 82411, 96659, 96659, 96659, 82411, 96659,
     &  51986,110913, 51986, 51986,110913, 82411, 54713, 54713, 22360,
     & 117652, 22360, 78250, 78250, 91996, 22360, 91996, 97781, 91996,
     &  97781, 91996, 97781, 97781, 91996, 97781, 97781, 36249, 39779/
      SAVE P, C, SAMPLS, NP, VAREST
      IF ( NDIM .GT. NLIM .OR. NDIM .LT. 1 ) THEN
         INFORM = 2
         FINEST = 0.d0
         ABSERR = 1.d0
         RETURN
      ENDIF
      INFORM = 1
      INTVLS = 0
      IF ( MINVLS .GE. 0 ) THEN
         FINEST = 0.d0
         VAREST = 0.d0
         SAMPLS = MINSMP 
         DO I = 1, PLIM
            NP = I
            IF ( MINVLS .LT. 2*SAMPLS*P(I) ) GO TO 10
         END DO
         SAMPLS = MAX( MINSMP, INT(MINVLS/( 2*P(NP)) ) )
      ENDIF
 10   VK(1) = ONE/DBLE(P(NP))
      DO I = 2, NDIM
         VK(I) = MOD( DBLE(C(NP,NDIM-1))*VK(I-1), ONE )
      END DO
      FINVAL = 0.d0
      VARSQR = 0.d0
*
*     Compute mean and standard error for SAMPLS randomized lattice rules
*
      DO I = 1, SAMPLS
         CALL KROSUM( NDIM, VALUE, P(NP), VK, FUNCTN, ALPHA, X )
         DIFINT = ( VALUE - FINVAL )/DBLE(I)
         FINVAL = FINVAL + DIFINT
         VARSQR = DBLE(I - 2)*VARSQR/DBLE(I) + DIFINT*DIFINT
      END DO
      INTVLS = INTVLS + 2*SAMPLS*P(NP)
      VARPRD = VAREST*VARSQR
      FINEST = FINEST + ( FINVAL - FINEST )/( 1.d0 + VARPRD )
      IF ( VARSQR .GT. 0.d0 ) VAREST = ( 1.d0 + VARPRD )/VARSQR
      ABSERR = 3.d0*SQRT( VARSQR/( 1.d0 + VARPRD ) )
      IF ( ABSERR .GT. MAX( ABSEPS, ABS(FINEST)*RELEPS ) ) THEN
         IF ( NP .LT. PLIM ) THEN
            NP = NP + 1
         ELSE
            SAMPLS = MIN( 3*SAMPLS/2, ( MAXVLS - INTVLS )/( 2*P(NP) ) ) 
            SAMPLS = MAX( MINSMP, SAMPLS )
         ENDIF
         IF ( INTVLS + 2*SAMPLS*P(NP) .LE. MAXVLS ) GO TO 10
      ELSE
         INFORM = 0
      ENDIF
      MINVLS = INTVLS
      END SUBROUTINE KROBOV
*
      SUBROUTINE KROSUM( NDIM, SUMKRO, PRIME, VK, FUNCTN, ALPHA, X )
      INTEGER, INTENT(IN):: NDIM, PRIME
      DOUBLE PRECISION, INTENT(OUT) :: SUMKRO
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) :: ALPHA,X     ! size NDIM
      INTEGER :: K                 !, J
      DOUBLE PRECISION :: ONE        
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: VK
      INTERFACE
         DOUBLE PRECISION FUNCTION FUNCTN(N,Z)
         DOUBLE PRECISION,DIMENSION(:), INTENT(IN) :: Z
         INTEGER, INTENT(IN) :: N
         END FUNCTION FUNCTN
      END INTERFACE
      PARAMETER ( ONE = 1.d0 )
      SUMKRO = 0.d0
      CALL random_number(ALPHA(1:NDIM))
      DO K = 1, PRIME
         X(1:NDIM) = MOD( DBLE(K)*VK(1:NDIM) + ALPHA(1:NDIM), ONE )
         X(1:NDIM) = ABS( 2.d0*X(1:NDIM) - ONE )
!         PRINT *,'KROSUM W=',X(1:NDIM)
         SUMKRO = SUMKRO+(FUNCTN(NDIM,X)-SUMKRO)/DBLE(2*K-1)
         X(1:NDIM) = ONE - X(1:NDIM)
         SUMKRO = SUMKRO+(FUNCTN(NDIM,X)-SUMKRO)/DBLE(2*K)
      END DO
      END SUBROUTINE KROSUM
      END MODULE KROBOVMOD

