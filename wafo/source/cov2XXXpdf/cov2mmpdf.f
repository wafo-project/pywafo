      PROGRAM cov2mmpdf
C*******************************************************************************
C     This program computes joint density of maximum and the following minimum * 
C*******************************************************************************
      use GLOBALDATA, only : Nt,Nj,Nd,Nc,Ntd,Ntdc,NI,Mb,
     &NIT,Nx,TWOPI,XSPLT,SCIS,NSIMmax,COV
      use rind
      IMPLICIT NONE
      double precision, dimension(:,:),allocatable :: BIG
      double precision, dimension(:,:),allocatable :: ansr
      double precision, dimension(:  ),allocatable :: ex
      double precision, dimension(:,:),allocatable :: xc
      double precision, dimension(:  ),allocatable :: fxind,h
      double precision, dimension(:  ),allocatable :: R0,R1,R2,R3,R4
      double precision             ::CC,U,XddInf,XdInf,XtInf  
      double precision, dimension(:,:),allocatable  :: a_up,a_lo
      integer         , dimension(:  ),allocatable :: seed
      integer ,dimension(7) :: indI
      integer :: Nstart,Ntime,tn,ts,speed,seed1,seed_size
      integer :: status,i,j,ij,Nx1
      double precision :: ds,dT ! lag spacing for covariances
 
! f90  cov2mmpdf.f rind51.f 

      CALL INIT_LEVELS(Ntime,Nstart,NIT,speed,Nx1,dT) 
      Nx=Nx1*(Nx1-1)/2 
      if (SCIS.GT.0) then
        allocate(COV(1:Nx))
        call random_seed(SIZE=seed_size) 
        allocate(seed(seed_size)) 
        call random_seed(GET=seed(1:seed_size))  ! get current seed
        seed(1)=seed1                            ! change seed
        call random_seed(PUT=seed(1:seed_size)) 
        deallocate(seed)
      endif
 
      CALL INITDATA(speed)  
 
      allocate(R0(1:Ntime+1))
      allocate(R1(1:Ntime+1))
      allocate(R2(1:Ntime+1))
      allocate(R3(1:Ntime+1))
      allocate(R4(1:Ntime+1))
      allocate(h(1:Nx1)) 

      CALL INIT_AMPLITUDES(h,Nx1)
      CALL INIT_COVARIANCES(Ntime,R0,R1,R2,R3,R4) 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Y=  X'(t2)...X'(tn-1)||X''(t1) X''(tn)|| X'(t1) X'(tn) X(t1) X(tn)         !!
! = [       Xt                   Xd                    Xc            ]       !!
!                                                                            !!
! Nt=tn-2, Nd=2, Nc=4                                                        !!
!                                                                            !!
! Xt= contains Nt time points in the indicator function                      !!
! Xd=    "     Nd    derivatives                                             !!
! Xc=    "     Nc    variables to condition on                               !!
!                                                                            !!
! There are 3 ( NI=4) regions with constant bariers:                         !!
! (indI(1)=0);     for i\in (indI(1),indI(2)]    Y(i)<0.                     !!
! (indI(2)=Nt)  ;  for i\in (indI(2)+1,indI(3)], Y(i)<0 (deriv. X''(t1))     !!
! (indI(3)=Nt+1);  for i\in (indI(3)+1,indI(4)], Y(i)>0 (deriv. X''(tn))     !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      NI=4; Nd=2
      Nc=4; Mb=1
      
      Nj=0
      indI(1)=0
C ***** The bound 'infinity' is set to 10*sigma *****
      XdInf=10.d0*SQRT(R4(1))
      XtInf=10.d0*SQRT(-R2(1))
                                                      ! normalizing constant
      CC=TWOPI*SQRT(-R2(1)/R4(1))
 
      allocate(BIG(1:Ntime+Nc,1:Ntime+Nc),stat=status) 
         if (status.ne.0) then 
             print *,'can not allocate BIG' 
         end if 
      allocate(ex(1:Ntime+Nc),stat=status)          
         if (status.ne.0) then 
             print *,'can not allocate ex' 
         end if
      if (Nx.gt.1) then
         allocate(ansr(1:Nx1,1:Nx1))
        else
         allocate(ansr(1,1:Ntime))
      end if
      ansr=0.d0
      allocate(fxind(1:Nx))    
      fxind=0.d0 !this is not needed
      allocate(xc(1:Nc,1:Nx))
 
 
      allocate(a_up(Mb,NI-1)) 
      allocate(a_lo(Mb,NI-1)) 
 
        a_up=0.d0 
        a_lo=0.d0 
 
        ij=0 
        do i=2,Nx1 
           do j=1,i-1 
             ij=ij+1
             xc(3,ij)=h(i)
             xc(4,ij)=h(j) 
           enddo 
        enddo
        xc(1,1:Nx)=0.d0
        xc(2,1:Nx)=0.d0 
        
        a_lo(1,1)=-Xtinf
        a_lo(1,2)=-XdInf
        a_up(1,3)=+XdInf 
             
 
      Nstart=MAX(2,Nstart)  
      
     
       if (SCIS.GT.0) then
         open (unit=11, file='COV.out',  STATUS='unknown') 
         write(11,*) 0.d0
      endif 
 
      do Ntd=Nstart,Ntime 
 
         Ntdc=Ntd+Nc
         ex=0.d0
         BIG=0.d0 
         CALL COV_INPUT(BIG(1:Ntdc,1:Ntdc),Ntd,R0,R1,R2,R3,R4) ! positive wave period
 
         Nt=Ntd-Nd;
         indI(2)=Nt;
         indI(3)=Nt+1;
         indI(4)=Ntd; 

          CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex,xc,indI,a_lo,a_up) 
          ij=0
          if (Nx .gt. 1) then
          do i=2,Nx1 
             do j=1,i-1 
                 ij=ij+1  
                 ansr(i,j)=ansr(i,j)+fxind(ij)*CC*dt
             enddo 
          enddo
          else
             ansr(1,Ntd)=fxind(1)*CC
          end if
 
          if (SCIS.GT.0) then
            write(11,*)  COV(1) ! save coefficient of variation
          endif       
          print *,'Ready: ',Ntd,' of ',Ntime
      enddo 
      goto 300
 300  open (unit=11, file='dens.out',  STATUS='unknown')
      if (Nx.gt.1) then
      do i=1,Nx1 
        do j=1,Nx1 
           write(11,*)  ansr(i,j)
        enddo 
      enddo
      else
        do j=1,Ntime
           write(11,*)  ansr(1,j)
        enddo
      end if
      close(11)
 900  continue
      deallocate(BIG) 
      deallocate(ex)
      deallocate(fxind)
      deallocate(ansr)
      deallocate(xc)
      deallocate(R0)
      deallocate(R1)
      deallocate(R2)  
      deallocate(R3)
      deallocate(R4)
      deallocate(h)
 
      if (allocated(COV) ) then
         deallocate(COV)
      endif
      stop
      !return

      CONTAINS
      
    
            
      SUBROUTINE INIT_LEVELS
     & (Ntime,Nstart,NIT,speed,Nx,dT)
      IMPLICIT NONE
      integer, intent(out):: Ntime,Nstart,NIT,speed,Nx
      double precision ,intent(out) :: dT
     

      OPEN(UNIT=14,FILE='reflev.in',STATUS= 'UNKNOWN')
      READ (14,*) Ntime
      READ (14,*) Nstart
      READ (14,*) NIT
      READ (14,*) speed 
      READ (14,*) SCIS   
      READ (14,*) seed1   
      READ (14,*) Nx
      READ (14,*) dT
     
      if (Ntime.lt.2) then
           print *,'The number of wavelength points is too small, stop'
           stop
      end if
      CLOSE(UNIT=14)
       
      RETURN
      END SUBROUTINE INIT_LEVELS
 
C******************************************************
      SUBROUTINE INIT_AMPLITUDES(h,Nx)      
      IMPLICIT NONE
      double precision, dimension(:), intent(out) :: h    
      integer, intent(in) :: Nx
      integer :: ix
      
 
      OPEN(UNIT=4,FILE='h.in',STATUS= 'UNKNOWN')

C
C Reading in amplitudes
C
      do ix=1,Nx
        READ (4,*) H(ix)
      enddo
      CLOSE(UNIT=4)
      
      RETURN
      END SUBROUTINE INIT_AMPLITUDES

C**************************************************

C***********************************************************************
C***********************************************************************
       
      SUBROUTINE INIT_COVARIANCES(Ntime,R0,R1,R2,R3,R4)
      IMPLICIT NONE
      double precision, dimension(:),intent(out) :: R0,R1,R2
      double precision, dimension(:),intent(out) :: R3,R4
      integer,intent(in) :: Ntime
      integer :: i
      open (unit=1, file='Cd0.in',STATUS='unknown')
      open (unit=2, file='Cd1.in',STATUS='unknown')
      open (unit=3, file='Cd2.in',STATUS='unknown')
      open (unit=4, file='Cd3.in',STATUS='unknown')
      open (unit=5, file='Cd4.in',STATUS='unknown')
      
      do i=1,Ntime
         read(1,*) R0(i)
         read(2,*) R1(i)
         read(3,*) R2(i)
         read(4,*) R3(i)
         read(5,*) R4(i)
       enddo
      close(1)
      close(2)
      close(3)
      close(3)
      close(5)
      return
      END SUBROUTINE INIT_COVARIANCES

C**********************************************************************

      SUBROUTINE COV_INPUT(BIG,tn,R0,R1,R2,R3,R4)
      IMPLICIT NONE
      double precision, dimension(:,:),intent(inout) :: BIG
      double precision, dimension(:),intent(in) :: R0,R1,R2
      double precision, dimension(:),intent(in) :: R3,R4
      integer ,intent(in) :: tn
      integer :: i,j,N 
      double precision :: tmp 
! the order of the variables in the covariance matrix
! are organized as follows: 
!    X'(t2)..X'(ts),...,X'(tn-1) X''(t1),X''(tn) X'(t1),X'(tn),X(t1),X(tn) 
! = [          Xt               |      Xd       |          Xc             ]
!
! where 
!
! Xt= time points in the indicator function
! Xd= derivatives
! Xc=variables to condition on
 
! Computations of all covariances follows simple rules: Cov(X(t),X(s))=r(t,s),
! then  Cov(X'(t),X(s))=dr(t,s)/dt.  Now for stationary X(t) we have
! a function r(tau) such that Cov(X(t),X(s))=r(s-t) (or r(t-s) will give the same result).
!
! Consequently  Cov(X'(t),X(s))    = -r'(s-t)    = -sign(s-t)*r'(|s-t|)
!               Cov(X'(t),X'(s))   = -r''(s-t)   = -r''(|s-t|)
!               Cov(X''(t),X'(s))  =  r'''(s-t)  =  sign(s-t)*r'''(|s-t|)
!               Cov(X''(t),X(s))   =  r''(s-t)   =   r''(|s-t|)
!               Cov(X''(t),X''(s)) =  r''''(s-t) = r''''(|s-t|)
       
      N=tn+4
      do i=1,tn-2
      !cov(Xt)
         do j=i,tn-2
           BIG(i,j) = -R2(j-i+1) ! cov(X'(ti+1),X'(tj+1))
         enddo
      !cov(Xt,Xc)
         BIG(i      ,tn+3) =  R1(i+1)         !cov(X'(ti+1),X(t1))  
         BIG(tn-1-i ,tn+4) = -R1(i+1)         !cov(X'(ti+1),X(tn))  
         BIG(i      ,tn+1) = -R2(i+1)         !cov(X'(ti+1),X'(t1))  
         BIG(tn-1-i ,tn+2) = -R2(i+1)         !cov(X'(ti+1),X'(tn))  
      !Cov(Xt,Xd)
         BIG(i,tn-1)       = R3(i+1)          !cov(X'(ti+1),X''(t1))  
         BIG(tn-1-i,tn)    =-R3(i+1)          !cov(X'(ti+1),X''(tn)) 
      enddo
      
!cov(Xd)
      BIG(tn-1  ,tn-1  ) = R4(1)
      BIG(tn-1,tn      ) = R4(tn)     !cov(X''(t1),X''(tn))
      BIG(tn    ,tn    ) = R4(1)

!cov(Xc)
      BIG(tn+3,tn+3) = R0(1)        ! cov(X(t1),X(t1))
      BIG(tn+3,tn+4) = R0(tn)       ! cov(X(t1),X(tn))
      BIG(tn+1,tn+3) = 0.d0         ! cov(X(t1),X'(t1))
      BIG(tn+2,tn+3) = R1(tn)       ! cov(X(t1),X'(tn))
      BIG(tn+4,tn+4) = R0(1)        ! cov(X(tn),X(tn))
      BIG(tn+1,tn+4) =-R1(tn)       ! cov(X(tn),X'(t1))
      BIG(tn+2,tn+4) = 0.d0         ! cov(X(tn),X'(tn)) 
      BIG(tn+1,tn+1) =-R2(1)        ! cov(X'(t1),X'(t1))
      BIG(tn+1,tn+2) =-R2(tn)       ! cov(X'(t1),X'(tn))
      BIG(tn+2,tn+2) =-R2(1)       ! cov(X'(tn),X'(tn))
!Xc=X(t1),X(tn),X'(t1),X'(tn) 
!Xd=X''(t1),X''(tn)
!cov(Xd,Xc)
      BIG(tn-1  ,tn+3) = R2(1)           !cov(X''(t1),X(t1))     
      BIG(tn-1  ,tn+4) = R2(tn)          !cov(X''(t1),X(tn))     
      BIG(tn-1  ,tn+1) = 0.d0            !cov(X''(t1),X'(t1))     
      BIG(tn-1  ,tn+2) = R3(tn)          !cov(X''(t1),X'(tn))     
      BIG(tn    ,tn+3) = R2(tn)          !cov(X''(tn),X(t1))     
      BIG(tn    ,tn+4) = R2(1)           !cov(X''(tn),X(tn))     
      BIG(tn    ,tn+1) =-R3(tn)          !cov(X''(tn),X'(t1))     
      BIG(tn    ,tn+2) = 0.d0            !cov(X''(tn),X'(tn))     
      ! make lower triangular part equal to upper 
      do j=1,N-1
        do i=j+1,N
           tmp =BIG(j,i)
           BIG(i,j)=tmp
        enddo
      enddo
      RETURN
      END  SUBROUTINE COV_INPUT


      END  PROGRAM  cov2mmpdf
       