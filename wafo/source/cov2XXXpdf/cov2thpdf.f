      PROGRAM sp2thpdf
!***********************************************************************
!     This program computes:                                           *
!                                                                      *
!     density of         S_i,Hi,T_i in a gaussian process i.e.         *
!                                                                      *
!      quart wavelength (up-crossing to crest) and crest amplitude     *    
!  
! def = 1,  gives half wave period, Tc (default).
!      -1,  gives half wave period, Tt.
!       2,  gives half wave period and wave crest amplitude (Tc,Ac).
!      -2,  gives half wave period and wave trough amplitude (Tt,At).
!       3,  gives crest front period and wave crest amplitude (Tcf,Ac).
!      -3,  gives trough back period and wave trough amplitude (Ttb,At).
!       4,  gives minimum of crest front/back period and wave crest 
!                                           amplitude (max(Tcf,Tcb),Ac).
!      -4,  gives minimum of trough front/back period and wave trough 
!                                           amplitude (max(Ttf,Ttb),At).
!***********************************************************************
      use GLOBALDATA, only : Nt,Nj,Nd,Nc,Ntd,Ntdc,NI,Mb,
     &   NIT,Nx,TWOPI,XSPLT,SCIS,NSIMmax,COV
      use rind
      IMPLICIT NONE
      double precision, dimension(:,:),allocatable :: BIG
      double precision, dimension(:,:),allocatable :: ansr
      double precision, dimension(:  ),allocatable :: ex
      double precision, dimension(:,:),allocatable :: xc
      double precision, dimension(:  ),allocatable :: fxind,h
      double precision, dimension(:  ),allocatable :: R0,R1,R2,R3,R4
      double precision             ::CC,U,XddInf,XdInf,XtInf  
      double precision, dimension(2,6) :: a_up=0.d0,a_lo=0.d0
      integer         , dimension(:  ),allocatable :: seed
      integer ,dimension(7) :: indI
      integer :: Nstart,Ntime,tn,ts,speed,ph,def,seed1,seed_size
      double precision :: ds,dT ! lag spacing for covariances 
! DIGITAL:
! f90 -g2 -C -automatic -o ../wave/alpha/sp2thpdf.exe rind44.f sp2thpdf.f
! SOLARIS:
!f90 -g -O -w3 -Bdynamic -fixed -o ../wave/sol2/sp2thpdf.exe rind44.f sp2thpdf.f
! linux:
! f90 -gline -Nl126 -C -o sp2thpdf.exe rind45.f sp2thpdf.f
! HP700
!f90 -g -C -o ../exec/hp700/sp2thpdf.exe rind45.f sp2thpdf.f
!f90 -g -C +check=all +FPVZID -o ../exec/hp700/sp2thpdf2.exe rind45.f sp2thpdf.f


      !print *,'enter sp2thpdf'
      CALL INIT_LEVELS(U,def,Ntime,Nstart,NIT,speed,SCIS,seed1,Nx,dT)
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
        allocate(h(1:Nx))
        allocate(R3(1:Ntime+1))
        allocate(R4(1:Ntime+1))
        
        CALL INIT_AMPLITUDES(h,def,Nx)
      endif
      CALL INIT_COVARIANCES(Ntime,def,R0,R1,R2,R3,R4)

      !print *,'Nx',Nx
      
      Nj=0
      indI(1)=0
C ***** The bound 'infinity' is set to 10*sigma *****
      XdInf=10.d0*SQRT(-R2(1))
      XtInf=10.d0*SQRT(R0(1))
      !print *,'XdInf,XtInf'
      !print *,XdInf,XtInf
      ! normalizing constant
      CC=TWOPI*SQRT(-R0(1)/R2(1))*exp(u*u/(2.d0*R0(1)) )
      if (abs(def).EQ.4) CC=2.d0*CC
      allocate(ansr(1:Ntime,1:Nx))
      ansr=0.d0
      allocate(fxind(1:Nx))    
      !fxind=0.d0 this is not needed
   
      if (abs(def).GT.1) then 
        GOTO 200
      endif
      NI=4; Nd=2
      Nc=2; Mb=1
      Nx=1
      allocate(BIG(1:Ntime+Nc,1:Ntime+Nc))
      allocate(xc(1:Nc,1:Nx))
      allocate(ex(1:Ntime+Nc)) 
      ex=0.d0 
      xc(1,1)=u
      xc(2,1)=u

      if (def.GT.0) then
         a_up(1,1)=u+XtInf 
         a_lo(1,1)=u
         a_up(1,2)=XdInf
         a_lo(1,3)=-XdInf       
      else
         a_up(1,1)=u 
         a_lo(1,1)=u-XtInf
         a_lo(1,2)=-XdInf
         a_up(1,3)= XdInf      
      endif
      !print *,'Nstart',Nstart
      Nstart=MAX(2,Nstart)  
      !print *,'Nstart',Nstart
      if (SCIS.GT.0) then
         open (unit=11, file='COV.out',  STATUS='unknown') 
         write(11,*) 0.d0
      endif
      do Ntd=Nstart,Ntime
         !CALL COV_INPUT2(BIG,Ntd, R0,R1,R2)
         CALL COV_INPUT(BIG,Ntd,-1,R0,R1,R2,R3,R4) ! positive wave period
         Nt=Ntd-Nd;
         indI(2)=Nt;
         indI(3)=Nt+1;
         indI(4)=Ntd;
         Ntdc=Ntd+Nc;
         !if (SCIS.gt.0) then
         !  if (SCIS.EQ.2) then           
         !     Nj=max(Nt,0)
         !  else
         !     Nj=min(max(Nt-5, 0),0)
         !  endif
         !endif
         !Ex=0.d0
         !CALL echo(BIG(1:Ntdc,1:min(7,Ntdc)),Ntdc)
         CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex(1:Ntdc),
     &              xc,indI,a_lo,a_up)         
         ansr(Ntd,1)=fxind(1)*CC
         if (SCIS.GT.0) then
            write(11,*)  COV(1) ! save coefficient of variation
         endif       
         print *,'Ready: ',Ntd,' of ',Ntime
      enddo
      if (SCIS.GT.0) then
        close(11) 
      endif
      goto 300
200   continue 
      XddInf=10.d0*SQRT(R4(1))
      NI=7; Nd=3
      Nc=4; Mb=2
      allocate(BIG(1:Ntime+Nc+1,1:Ntime+Nc+1)) 
      ALLOCATE(xc(1:Nc,1:Nx))
      allocate(ex(1:Ntime+Nc+1))
      
      ex=0.d0 
      xc(1,1:Nx)=h
      xc(2,1:Nx)=u
      xc(3,1:Nx)=u
      xc(4,1:Nx)=0.d0
      
      if (def.GT.0) then
        a_up(2,1)=1.d0   !*h 
        a_lo(1,1)=u
        a_up(1,2)=XtInf   ! X(ts) is redundant  
        a_lo(1,2)=-Xtinf 
        a_up(2,2)=1.d0   ! *h
        a_lo(2,2)=1.d0   ! *h
        a_up(2,3)=1.d0   !*h 
        a_lo(1,3)=u

        a_lo(1,4)=-XddInf
        a_up(1,5)= XdInf
        a_lo(1,6)=-XdInf       
      else !def<0
        a_up(1,1)=u   
        a_lo(2,1)=1.d0 !*h
        a_up(1,2)=XtInf   ! X(ts) is redundant  
        a_lo(1,2)=-Xtinf 
        a_up(2,2)=1.d0          ! *h
        a_lo(2,2)=1.d0          ! *h
        a_up(1,3)=u   
        a_lo(2,3)=1.d0 !*h

        a_up(1,4)=XddInf
        a_lo(1,5)=-XdInf
        a_up(1,6)=XdInf
      endif 
      
      Nstart=MAX(Nstart,3)
      do tn=Nstart,Ntime,1
        Ntd=tn+1
        Nt=Ntd-Nd
        Ntdc=Ntd+Nc
        indI(4)=Nt
        indI(5)=Nt+1
        indI(6)=Nt+2
        indI(7)=Ntd
        if (SCIS.gt.0) then
           if (SCIS.EQ.2) then           
              Nj=max(Nt,0)
           else
              Nj=min(max(Nt-5, 0),0)
           endif
        endif
        do ts=2,FLOOR(DBLE(tn+1)/2.d0)
         !print *,'ts,tn' ,ts,tn
         CALL COV_INPUT(Big(1:Ntdc,1:Ntdc),tn,ts,R0,R1,R2,R3,R4) ! positive wave period
         indI(2)=ts-2
         indI(3)=ts-1          
         !CALL echo(BIG(1:Ntdc,1:min(7,Ntdc)),Ntdc)
         !print *,'sp call rind'
         CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex(1:Ntdc),
     &              xc,indI,a_lo,a_up)
         !CALL echo(BIG(1:Ntdc,1:min(7,Ntdc)),Ntdc)
         !print *,'sp rind finished',fxind
         !goto 900
         if (abs(def).LT.3) THEN
            if (ts .EQ.tn-ts+1) then
               ds=dt
            else 
               ds=2.d0*dt
            endif
            ansr(tn,1:Nx)=ansr(tn,1:Nx)+fxind*CC*ds
         else 
            ansr(ts,1:Nx)=ansr(ts,1:Nx)+fxind*CC*dT
           if ((ts.LT.tn-ts+1).and. (abs(def).lt.4)) THEN
             ansr(tn-ts+1,1:Nx)=ansr(tn-ts+1,1:Nx)+fxind*CC*dT ! exploiting the symmetry
           endif
         endif
        enddo ! ts
        print *,'Ready: ',tn,' of ',Ntime

      enddo !tn
      !print *,'ansr',ansr
 300  open (unit=11, file='dens.out',  STATUS='unknown')
      !print *, ansr
      do ts=1,Ntime
         do ph=1,Nx
            write(11,*)  ansr(ts,ph)
            ! write(11,111)  ansr(ts,ph)
         enddo
      enddo
 !111  FORMAT(2x,F12.8)  
      close(11)
 900  deallocate(big)
      deallocate(fxind)
      deallocate(ansr)
      deallocate(xc)
      deallocate(ex)
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
     & (U,def,Ntime,Nstart,NIT,speed,SCIS,seed1,Nx,dT)
      IMPLICIT NONE
      integer, intent(out):: def,Ntime,Nstart,NIT,speed,Nx,SCIS,seed1
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

      if (abs(def).GT.1) then
        READ (14,*) Nx
        READ (14,*) dT
        if (Ntime.lt.3) then
           print *,'The number of wavelength points is too small, stop'
           stop
        end if
      else
        Nx=1
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
! ||X(t2)..X(ts),..X(tn-1)||X''(ts) X'(t1) X'(tn)||X(ts) X(t1) X(tn) X'(ts)|| 
! = [Xt                          Xd                    Xc]
!
! For ts<=1: 
! ||X(t2)..,..X(tn-1)||X'(t1) X'(tn)||X(t1) X(tn)||  
! = [Xt              Xd            Xc]

! where 
!
! Xt= time points in the indicator function
! Xd= derivatives
! Xc=variables to condition on

      if (ts.LE.1) THEN
	  Ntd1=tn
          N=Ntd1+2;
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
         BIG(i      ,Ntd1+1+shft) = R0(i+1)         !cov(X(ti+1),X(t1))  
         BIG(tn-1-i ,Ntd1+2+shft) = R0(i+1)         !cov(X(t.. ),X(tn))  
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
      BIG(Ntd1+1+shft,Ntd1+1+shft) = R0(1)        ! cov(X(t1),X (t1)) 
      BIG(Ntd1+1+shft,Ntd1+2+shft) = R0(tn)      ! cov(X(t1),X (tn))
      BIG(Ntd1+2+shft,Ntd1+2+shft) = R0(1)        ! cov(X(tn),X (tn))
!cov(Xd,Xc)
      BIG(Ntd1  ,Ntd1+1+shft) = R1(tn)       !cov(X'(tn),X(t1))     
      BIG(Ntd1  ,Ntd1+2+shft) = 0.d0         !cov(X'(tn),X(tn))
      BIG(Ntd1-1,Ntd1+1+shft) = 0.d0         !cov(X'(t1),X(t1))
      BIG(Ntd1-1,Ntd1+2+shft) =-R1(tn)      !cov(X'(t1),X(tn))


      if (ts.GT.1) then 

! 
!cov(Xc)
        BIG(Ntd1+1,Ntd1+1) = R0(1)        ! cov(X(ts),X (ts)
        BIG(Ntd1+1,Ntd1+2) = R0(ts)       ! cov(X(ts),X (t1))
        BIG(Ntd1+1,Ntd1+3) = R0(tn+1-ts)  ! cov(X(ts),X (tn))
        BIG(Ntd1+1,Ntd1+4) = 0.d0         ! cov(X(ts),X'(ts))

        BIG(Ntd1+2,Ntd1+4) = R1(ts)       ! cov(X(t1),X'(ts))
        BIG(Ntd1+3,Ntd1+4) = -R1(tn+1-ts)  !cov(X(tn),X'(ts))
        BIG(Ntd1+4,Ntd1+4) = -R2(1)       ! cov(X'(ts),X'(ts))

!cov(Xd)
        BIG(Ntd1-2,Ntd1-1) = -R3(ts)      !cov(X''(ts),X'(t1))
        BIG(Ntd1-2,Ntd1-2) = R4(1)
        BIG(Ntd1-2,Ntd1  ) = R3(tn+1-ts)   !cov(X''(ts),X'(tn))
!cov(Xd,Xc)
        BIG(Ntd1  ,Ntd1+4) =-R2(tn+1-ts)   !cov(X'(tn),X'(ts))  
        BIG(Ntd1  ,Ntd1+1) = R1(tn+1-ts)   !cov(X'(tn),X (ts))  

        BIG(Ntd1-1,Ntd1+4) =-R2(ts)       !cov(X'(t1),X'(ts))     
        BIG(Ntd1-1,Ntd1+1) =-R1(ts)       !cov(X'(t1),X (ts))  
 
        BIG(Ntd1-2,Ntd1+1) = R2(1)        !cov(X''(ts),X (ts)
        BIG(Ntd1-2,Ntd1+2) = R2(ts)       !cov(X''(ts),X (t1))
        BIG(Ntd1-2,Ntd1+3) = R2(tn+1-ts)   !cov(X''(ts),X (tn))
        BIG(Ntd1-2,Ntd1+4) = 0.d0         !cov(X''(ts),X'(ts))
!cov(Xt,Xc)
        do i=1,tn-2
          j=abs(i+1-ts)
          BIG(i,Ntd1+1)   = R0(j+1) !cov(X(ti+1),X(ts))  
          BIG(i,Ntd1+4)   = sign(R1(j+1),R1(j+1)*dble(ts-i-1)) !cov(X(ti+1),X'(ts))   ! check this
        
!Cov(Xt,Xd)=cov(X(ti+1),X(ts))       
          BIG(i,Ntd1-2)    = R2(j+1) !cov(X(ti+1),X''(ts))  
        enddo
      endif ! ts>1

      !call echo(big(1:N,1:N),N) 
      ! make lower triangular part equal to upper 
      do j=1,N-1
        do i=j+1,N
           tmp =BIG(j,i)

           BIG(i,j)=tmp
        enddo
        !call echo(big(1:N,1:N),N) 

      enddo
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


      END  PROGRAM  sp2thpdf
       








