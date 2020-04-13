      PROGRAM sp2Acdf1
C***********************************************************************
C     This program computes upper and lower bounds for:                *
C                                                                      *
C     density of     T_i, for Ac <=h, in a gaussian process i.e.       *
C                                                                      *
C     half wavelength (up-crossing to downcrossing) for crests <h      *
C    or  half wavelength (down-crossing to upcrossing) for trough >h   * 
C    I.R. 27 Dec. 1999                                                 *
C***********************************************************************
      use GLOBALDATA, only : Nt,Nj,Nd,Nc,Ntd,Ntdc,NI,Mb,
     &NIT,Nx,TWOPI,XSPLT,SCIS,NSIMmax,COV
      use rind
      IMPLICIT NONE
      double precision, dimension(:,:),allocatable :: BIG
      double precision, dimension(:,:),allocatable :: ansrup
      double precision, dimension(:,:),allocatable :: ansrlo
      double precision, dimension(:  ),allocatable :: ex,CY
      double precision, dimension(:,:),allocatable :: xc,fxind
      double precision, dimension(:  ),allocatable :: h
      double precision, dimension(:  ),allocatable :: R0,R1,R2,R3,R4
      double precision             ::CC,U,XddInf,XdInf,XtInf  
      double precision, dimension(:,:),allocatable  :: a_up,a_lo
      integer         , dimension(:  ),allocatable :: seed
      integer ,dimension(7) :: indI
      integer :: Nstart,Ntime,tn,ts,speed,ph,def,seed1,seed_size,icy
      integer ::it1,it2,status
      double precision :: ds,dT ! lag spacing for covariances
! f90  sp2Acdf1.f rind50.f

      CALL INIT_LEVELS(U,def,Ntime,Nstart,NIT,speed,Nx,dT)
      !print *,'U,def,Ntime,Nstart,NIT,speed,SCIS,seed1,Nx,dT'
      !print *,U,def,Ntime,Nstart,NIT,speed,SCIS,seed1,Nx,dT
      !XSPLT=1.5d0
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
      !print *,ntime,speed,u,NIT
      allocate(R0(1:Ntime+1))
      allocate(R1(1:Ntime+1))
      allocate(R2(1:Ntime+1))
      if (abs(def).GT.1) THEN
        allocate(R3(1:Ntime+1))
        allocate(R4(1:Ntime+1))
        !CALL INIT_AMPLITUDES(h,def,Nx)
      endif
      allocate(h(1:Nx))
      CALL INIT_AMPLITUDES(h,def,Nx)
      CALL INIT_COVARIANCES(Ntime,def,R0,R1,R2,R3,R4)
   
      NI=4; Nd=2
      Nc=3; Mb=2
      
      Nj=0
      indI(1)=0
C ***** The bound 'infinity' is set to 10*sigma *****
      XdInf=10.d0*SQRT(-R2(1))
      XtInf=10.d0*SQRT(R0(1))
                                                      ! normalizing constant
      CC=TWOPI*SQRT(-R0(1)/R2(1))*exp(u*u/(2.d0*R0(1)) )
      
      allocate(CY(1:Nx)) 
      do icy=1,Nx
        CY(icy)=exp(-0.5*h(icy)*h(icy)/100)/(10*sqrt(twopi))
      enddo 
      allocate(BIG(1:Ntime+Nc,1:Ntime+Nc),stat=status) 
         if (status.ne.0) then 
             print *,'can not allocate BIG' 
         end if 
      allocate(ex(1:Ntime+Nc),stat=status)          
         if (status.ne.0) then 
             print *,'can not allocate ex' 
         end if
      allocate(ansrup(1:Ntime,1:Nx))
      allocate(ansrlo(1:Ntime,1:Nx))
      ansrup=0.d0
      ansrlo=0.d0
      allocate(fxind(1:Nx,1:2))    
      fxind=0.d0 !this is not needed
      allocate(xc(1:Nc,1:Nx))
 
 
      allocate(a_up(Mb,NI-1)) 
      allocate(a_lo(Mb,NI-1)) 
        a_up=0.d0 
        a_lo=0.d0
        xc(1,1:Nx)=h(1:Nx)
        xc(2,1:Nx)=u
        xc(3,1:Nx)=u 
        
      if (def.GT.0) then
         a_up(1,1)=0.d0 
         a_lo(1,1)=u
         a_up(1,2)=XdInf
         a_lo(1,3)=-XdInf 
         a_up(2,1)=1.d0      
      else
         a_up(1,1)=u 
         a_lo(1,1)=0.d0
         a_lo(1,2)=-XdInf
         a_up(1,3)= XdInf  
         a_lo(2,1)=1.d0 
      endif
      !print *,'Nstart',Nstart
      Nstart=MAX(3,Nstart)  
      
     
       if (SCIS.GT.0) then
         open (unit=11, file='COV.out',  STATUS='unknown') 
         write(11,*) 0.d0
      endif 
 
      !print *,'loop starts' 
      do Ntd=Nstart,Ntime 
 
         Ntdc=Ntd+Nc
         ex=0.d0
         BIG=0.d0 
         CALL COV_INPUT(BIG(1:Ntdc,1:Ntdc),Ntd,-1,R0,R1,R2,R3,R4) ! positive wave period
 
         Nt=Ntd-Nd;
         indI(2)=Nt;
         indI(3)=Nt+1;
         indI(4)=Ntd; 

          CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex,xc,indI,a_lo,a_up) 
          !print *,'test',fxind/CY(1:Nx) 
 
          do icy=1,Nx        
            ansrup(Ntd,icy)=fxind(icy,1)*CC/CY(icy)
            ansrlo(Ntd,icy)=fxind(icy,2)*CC/CY(icy)
          enddo
          if (SCIS.GT.0) then
            write(11,*)  COV(1) ! save coefficient of variation
          endif
      if((Nx.gt.4).or.NIT.gt.4) print *,'Ready: ',Ntd,' of ',Ntime
      enddo 
      goto 300
 300  open (unit=11, file='dens.out',  STATUS='unknown')

      do ts=1,Ntime
         do ph=1,Nx
            write(11,*)  ansrup(ts,ph),ansrlo(ts,ph)
         enddo
      enddo
 !111  FORMAT(2x,F12.8)  
      close(11)
 900  continue
      deallocate(BIG) 
      deallocate(ex)
      deallocate(fxind)
      deallocate(ansrup)
      deallocate(ansrlo)
      deallocate(xc)
      deallocate(R0)
      deallocate(R1)
      deallocate(R2)  
      if (allocated(COV) ) then
         deallocate(COV)
      endif

      if (allocated(R3)) then
        deallocate(R3)
        deallocate(R4)
        deallocate(h)
      ENDIF
      stop
      !return

      CONTAINS
      
    
            
      SUBROUTINE INIT_LEVELS
     & (U,def,Ntime,Nstart,NIT,speed,Nx,dT)
      IMPLICIT NONE
      integer, intent(out):: def,Ntime,Nstart,NIT,speed,Nx
      double precision ,intent(out) :: U,dT
     

      OPEN(UNIT=14,FILE='reflev.in',STATUS= 'UNKNOWN')
      READ (14,*) U
      READ (14,*) def
      READ (14,*) Ntime
      READ (14,*) Nstart
      READ (14,*) NIT
      READ (14,*) speed 
      READ (14,*) SCIS   
      READ (14,*) seed1   
      READ (14,*) Nx
     
      if (abs(def).GT.1) then
        READ (14,*) dT
        if (Ntime.lt.3) then
           print *,'The number of wavelength points is too small, stop'
           stop
        end if
      else
        if (Ntime.lt.2) then
           print *,'The number of wavelength points is too small, stop'
           stop
        end if
      endif
      CLOSE(UNIT=14)
       
      RETURN
      END SUBROUTINE INIT_LEVELS
 
C******************************************************
      SUBROUTINE INIT_AMPLITUDES(h,def,Nx)      
      IMPLICIT NONE
      double precision, dimension(:), intent(out) :: h    
      integer, intent(in) :: def
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
      !if (def.LT.0) THEN
      !   H=-H
      !endif      
      
      RETURN
      END SUBROUTINE INIT_AMPLITUDES

C**************************************************

C***********************************************************************
C***********************************************************************
       
      SUBROUTINE INIT_COVARIANCES(Ntime,def,R0,R1,R2,R3,R4)
      IMPLICIT NONE
      double precision, dimension(:),intent(out) :: R0,R1,R2
      double precision, dimension(:),intent(out) :: R3,R4
      integer,intent(in) :: Ntime,def
      integer :: i
      open (unit=1, file='Cd0.in',STATUS='unknown')
      open (unit=2, file='Cd1.in',STATUS='unknown')
      open (unit=3, file='Cd2.in',STATUS='unknown')
      
      do i=1,Ntime
         read(1,*) R0(i)
         read(2,*) R1(i)
         read(3,*) R2(i)
       enddo
      close(1)
      close(2)
      close(3)
      
      if (abs(def).GT.1) then      
        open (unit=4, file='Cd3.in',STATUS='unknown')
        open (unit=5, file='Cd4.in',STATUS='unknown')

        do i=1,Ntime
          read(4,*) R3(i)
          read(5,*) R4(i)
        enddo

        close(4)
        close(5)
      endif
      return
      END SUBROUTINE INIT_COVARIANCES
      
C***********************************************************************
C***********************************************************************

C**********************************************************************

      SUBROUTINE COV_INPUT(BIG,tn,ts, R0,R1,R2,R3,R4)
      IMPLICIT NONE
      double precision, dimension(:,:),intent(inout) :: BIG
      double precision, dimension(:),intent(in) :: R0,R1,R2
      double precision, dimension(:),intent(in) :: R3,R4
      integer ,intent(in) :: tn,ts
      integer :: i,j,shft,Ntd1,N !=Ntdc
      double precision :: tmp 
! the order of the variables in the covariance matrix
! are organized as follows: 
! For ts>1:
! X(t2)..X(ts),..X(tn-1) X''(ts) X'(t1) X'(tn) X(ts) X(t1) X(tn) X'(ts) 
! = [Xt                          Xd                    Xc]
!
! For ts<=1: 
! X(t2)..,..X(tn-1)  X'(t1) X'(tn) Y X(t1) X(tn)  
! = [Xt              Xd            Xc]
!Add Y Condition : Y=h

! where 
!
! Xt= time points in the indicator function
! Xd= derivatives
! Xc=variables to condition on

      if (ts.LE.1) THEN
	       Ntd1=tn
          N=Ntd1+Nc;
          shft=0  ! def=1 want only crest period Tc
      else
          Ntd1=tn+1
          N=Ntd1+4
          shft=1  ! def=2 or 3 want Tc Ac or Tcf, Ac
      endif

      do i=1,tn-2
      !cov(Xt)
         do j=i,tn-2
           BIG(i,j) = R0(j-i+1) ! cov(X(ti+1),X(tj+1))
         enddo
      !cov(Xt,Xc)
	 BIG(i      ,Ntd1+1+shft) = 0.d0            !cov(X(ti+1),Y)   
         BIG(i      ,Ntd1+2+shft) = R0(i+1)         !cov(X(ti+1),X(t1))  
         BIG(tn-1-i ,Ntd1+3+shft) = R0(i+1)         !cov(X(t.. ),X(tn))  
      !Cov(Xt,Xd)=cov(X(ti+1),x(tj)
         BIG(i,Ntd1-1)         =-R1(i+1)         !cov(X(ti+1),X' (t1))  
         BIG(tn-1-i,Ntd1)= R1(i+1)         !cov(X(ti+1),X' (tn)) 
      enddo
      !call echo(big(1:tn,1:tn),tn)
!cov(Xd)
      BIG(Ntd1  ,Ntd1  ) = -R2(1)
      BIG(Ntd1-1,Ntd1  ) = -R2(tn)     !cov(X'(t1),X'(tn))
      BIG(Ntd1-1,Ntd1-1) = -R2(1)

!cov(Xc)
      !print *,'t'
      BIG(Ntd1+1+shft,Ntd1+1+shft) = 100.d0!100.d0         ! cov(Y,Y) 
      BIG(Ntd1+1+shft,Ntd1+2+shft) = 0.d0
      BIG(Ntd1+1+shft,Ntd1+3+shft) = 0.d0
      BIG(Ntd1+2+shft,Ntd1+2+shft) = R0(1)        ! cov(X(t1),X (t1)) 
      BIG(Ntd1+2+shft,Ntd1+3+shft) = R0(tn)       ! cov(X(t1),X (tn))
      BIG(Ntd1+3+shft,Ntd1+3+shft) = R0(1)        ! cov(X(tn),X (tn))
!cov(Xd,Xc)
      BIG(Ntd1  ,Ntd1+1+shft) = 0.d0            !cov(X'(tn),Y)     
      BIG(Ntd1  ,Ntd1+2+shft) = R1(tn)       !cov(X'(tn),X(t1))     
      BIG(Ntd1  ,Ntd1+3+shft) = 0.d0         !cov(X'(tn),X(tn))
      BIG(Ntd1-1,Ntd1+1+shft) = 0.d0         !cov(X'(t1),Y)
      BIG(Ntd1-1,Ntd1+2+shft) = 0.d0         !cov(X'(t1),X(t1))
      BIG(Ntd1-1,Ntd1+3+shft) =-R1(tn)       !cov(X'(t1),X(tn))


      !call echo(big(1:N,1:N),N) 
      ! make lower triangular part equal to upper 
      do j=1,N-1
        do i=j+1,N
           tmp =BIG(j,i)
           BIG(i,j)=tmp
        enddo
        !call echo(big(1:N,1:N),N) 
      enddo
      !if (tn.eq.3) then
      !do j=1,N
      !  do i=j,N
      !      print *,'test',j,i,BIG(j,i)
      !  enddo
        !call echo(big(1:N,1:N),N) 
      !enddo
      !endif
      !call echo(big(1:N,1:N),N) 

C     write (*,10) ((BIG(j,i),i=N+1,N+6),j=N+1,N+6)
C 10  format(6F8.4)
      RETURN
      END  SUBROUTINE COV_INPUT

      SUBROUTINE COV_INPUT2(BIG,pt, R0,R1,R2)
      IMPLICIT NONE
      double precision, dimension(:,:), intent(out) :: BIG
      double precision, dimension(:), intent(in) ::  R0,R1,R2
      integer :: pt,i,j
! the order of the variables in the covariance matrix
! are organized as follows;
! X(t2)...X(tn-1) X'(t1) X'(tn) X(t1) X(tn) = [Xt Xd Xc]
! 
! where Xd is the derivatives 
!
! Xt= time points in the indicator function
! Xd= derivatives
! Xc=variables to condition on

!cov(Xc)
      BIG(pt+2,pt+2) = R0(1)
      BIG(pt+1,pt+1) = R0(1)
      BIG(pt+1,pt+2) = R0(pt)
!cov(Xd)
      BIG(pt,pt)     = -R2(1)
      BIG(pt-1,pt-1) = -R2(1)
      BIG(pt-1,pt)   = -R2(pt)
!cov(Xd,Xc)
      BIG(pt,pt+2)   =  0.d0
      BIG(pt,pt+1)   = R1(pt)
      BIG(pt-1,pt+2) = -R1(pt)
      BIG(pt-1,pt+1) =  0.d0
 
      if (pt.GT.2) then
!cov(Xt)
         do i=1,pt-2
           do j=i,pt-2
              BIG(i,j) = R0(j-i+1)
           enddo
        enddo
!cov(Xt,Xc)
        do i=1,pt-2
           BIG(i,pt+1)      = R0(i+1)
           BIG(pt-1-i,pt+2) = R0(i+1)
        enddo
!Cov(Xt,Xd)=cov(X(ti+1),x(tj))
        do i=1,pt-2
           BIG(i,pt-1)   = -R1(i+1)
           BIG(pt-1-i,pt)=  R1(i+1)
        enddo
      endif


      ! make lower triangular part equal to upper 
      do j=1,pt+1
         do i=j+1,pt+2
            BIG(i,j)=BIG(j,i)
         enddo
      enddo
C      write (*,10) ((BIG(j,i),i=N+1,N+6),j=N+1,N+6)
C 10   format(6F8.4)
      RETURN
      END  SUBROUTINE COV_INPUT2 


      END  PROGRAM  sp2Acdf1
       








