      PROGRAM sp2mmt
C*******************************************************************************
C     This program computes joint density of the  maximum and the following    *
C     minimum or level u separated maxima and minima + period/wavelength       * 
C*******************************************************************************
      use GLOBALDATA, only : Nt,Nj,Nd,Nc,Ntd,Ntdc,NI,Mb,
     &NIT,Nx,TWOPI,XSPLT,SCIS,NSIMmax,COV
      use rind
      IMPLICIT NONE
      double precision, dimension(:,:),  allocatable :: BIG
      double precision, dimension(:,:,:),allocatable :: ansr
      double precision, dimension(:  ),  allocatable :: ex
      double precision, dimension(:,:),  allocatable :: xc
      double precision, dimension(:  ),  allocatable :: fxind,h
      double precision, dimension(:  ),  allocatable :: R0,R1,R2,R3,R4
      double precision             :: CC,U,XdInf,XtInf  
      double precision, dimension(1,4)               :: a_up,a_lo      ! size Mb X NI-1
      integer         , dimension(:  ),  allocatable :: seed
      integer ,dimension(5) :: indI = 0                                ! length NI
      integer :: Nstart,Ntime,ts,tn,speed,seed1,seed_size
      integer :: status,i,j,ij,Nx0,Nx1,DEF,isOdd !,TMP
      LOGICAL :: SYMMETRY=.FALSE.
      double precision :: dT ! lag spacing for covariances
 
! f90  -gline -fieee -Nl126 -C -o  intmodule.f rind60.f sp2mmt.f

      CALL INIT_LEVELS(Ntime,Nstart,NIT,speed,SCIS,SEED1,Nx1,dT,u,def) 
      CALL INITDATA(speed)  
      
      if (SCIS.GT.0) then
        !allocate(COV(1:Nx))
        call random_seed(SIZE=seed_size) 
        allocate(seed(seed_size)) 
        call random_seed(GET=seed(1:seed_size))  ! get current seed
        seed(1)=seed1                            ! change seed
        call random_seed(PUT=seed(1:seed_size)) 
        deallocate(seed)
        if (ALLOCATED(COV)) then
           open (unit=11, file='COV.out',  STATUS='unknown') 
           write(11,*) 0.d0
        endif 
      endif

      allocate(R0(1:Ntime+1))
      allocate(R1(1:Ntime+1))
      allocate(R2(1:Ntime+1))
      allocate(R3(1:Ntime+1))
      allocate(R4(1:Ntime+1))

      Nx0 = Nx1 ! just plain Mm
      IF (def.GT.1) Nx0=2*Nx1   ! level v separated max2min densities wanted     
      
       
      allocate(h(1:Nx0)) 
      
      CALL INIT_AMPLITUDES(h,Nx0)
      CALL INIT_COVARIANCES(Ntime,R0,R1,R2,R3,R4) 
! For DEF = 0,1 : (Maxima, Minima and period/wavelength)
!         = 2,3 : (Level v separated Maxima and Minima and period/wavelength between them)
!      If Nx==1 then the conditional  density for  period/wavelength between Maxima and Minima 
!      given the Max and Min is returned
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
! Y=  X'(t2)..X'(ts)..X'(tn-1)||X''(t1) X''(tn)|| X'(t1) X'(tn)  X(t1) X(tn) 
! = [       Xt                   Xd                    Xc            ]  
!                                                                       
! Nt = tn-2, Nd = 2, Nc = 4
!                                                                       
! Xt= contains Nt time points in the indicator function                 
! Xd=    "     Nd    derivatives in Jacobian
! Xc=    "     Nc    variables to condition on                          
!                                                                       
! There are 3 (NI=4) regions with constant barriers:                    
! (indI(1)=0);     for i\in (indI(1),indI(2)]    Y(i)<0.                
! (indI(2)=Nt)  ;  for i\in (indI(2)+1,indI(3)], Y(i)<0 (deriv. X''(t1)) 
! (indI(3)=Nt+1);  for i\in (indI(3)+1,indI(4)], Y(i)>0 (deriv. X''(tn))  
! 
!
! For DEF = 4,5 (Level v separated Maxima and Minima and period/wavelength from Max to crossing)
!     If Nx==1 then the conditional joint density for  period/wavelength between Maxima, Minima and Max to 
!              level v crossing given the Max and the min is returned
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! Y=  X'(t2)..X'(ts)..X'(tn-1)||X''(t1) X''(tn) X'(ts)|| X'(t1) X'(tn)  X(t1) X(tn) X(ts)
! = [       Xt                      Xd                     Xc            ]
!                                                                         
! Nt = tn-2, Nd = 3, Nc = 5
!                                                                         
! Xt= contains Nt time points in the indicator function                      
! Xd=    "     Nd    derivatives                                             
! Xc=    "     Nc    variables to condition on                               
!                                                                            
! There are 4 (NI=5) regions with constant barriers:                        
! (indI(1)=0);     for i\in (indI(1),indI(2)]    Y(i)<0.                    
! (indI(2)=Nt)  ;  for i\in (indI(2)+1,indI(3)], Y(i)<0 (deriv. X''(t1))    
! (indI(3)=Nt+1);  for i\in (indI(3)+1,indI(4)], Y(i)>0 (deriv. X''(tn))
! (indI(4)=Nt+2);  for i\in (indI(4)+1,indI(5)], Y(i)<0 (deriv. X'(ts))    
! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!Revised pab 22.04.2000
! - added mean separated min/max + (Tdm, TMd) period distributions
! - added scis 


C ***** The bound 'infinity' is set to 10*sigma *****
      XdInf = 10.d0*SQRT(R4(1))
      XtInf = 10.d0*SQRT(-R2(1))

      Nc = 4
      NI=4; Nd=2; 
      Mb=1 ; 
      Nj = 0
      indI(1) = 0
      Nstart=MAX(2,Nstart) 

      isOdd = MOD(Nx1,2)
      IF (def.LE.1) THEN ! just plain Mm 
         Nx = Nx1*(Nx1-1)/2
         IJ = (Nx1+isOdd)/2         
         IF (H(1)+H(Nx1).EQ.0.AND.
     &        (H(IJ).EQ.0.OR.H(IJ)+H(IJ+1).EQ.0) ) THEN
            SYMMETRY=.FALSE.
            PRINT *,' Integration region symmetric'
            ! May save Nx1-isOdd integrations in each time step 
            ! This is not implemented yet.
            !Nx = Nx1*(Nx1-1)/2-Nx1+isOdd
         ENDIF
        
         CC = TWOPI*SQRT(-R2(1)/R4(1)) ! normalizing constant = 1/ expected number of zero-up-crossings of X' 
         
      ELSE  ! level u separated Mm
         Nx = (Nx1-1)*(Nx1-1)
         IF ( ABS(u).LE.1D-8.AND.H(1)+H(Nx1+1).EQ.0.AND.
     &        (H(Nx1)+H(2*Nx1).EQ.0) ) THEN
            SYMMETRY=.FALSE. 
            PRINT *,' Integration region symmetric'
            ! Not implemented for DEF <= 3
            !IF (DEF.LE.3) Nx = (Nx1-1)*(Nx1-2)/2 
         ENDIF
         
         IF (DEF.GT.3) THEN
            Nstart = MAX(Nstart,3)
            Nc = 5
            NI=5; Nd=3;
         ENDIF
         CC = TWOPI*SQRT(-R0(1)/R2(1))*exp(0.5D0*u*u/R0(1)) ! normalizing constant= 1/ expected number of u-up-crossings of X
      ENDIF

      !print *,'def',def
      IF (Nx.GT.1) THEN
         IF ((DEF.EQ.0.OR.DEF.EQ.2)) THEN ! (M,m) or (M,m)v distribution wanted
            allocate(ansr(Nx1,Nx1,1),stat=status)
         ELSE                             ! (M,m,TMm), (M,m,TMm)v  (M,m,TMd)v or (M,M,Tdm)v distributions wanted 
            allocate(ansr(Nx1,Nx1,Ntime),stat=status)
         ENDIF
      ELSEIF (DEF.GT.3) THEN              ! Conditional distribution for (TMd,TMm)v or (Tdm,TMm)v given (M,m)  wanted 
         allocate(ansr(1,Ntime,Ntime),stat=status)
      ELSE                                ! Conditional distribution for  (TMm) or (TMm)v given (M,m) wanted
         allocate(ansr(1,1,Ntime),stat=status)
      ENDIF
      if (status.ne.0) print *,'can not allocate ansr'
      allocate(BIG(Ntime+Nc+1,Ntime+Nc+1),stat=status) 
      if (status.ne.0) print *,'can not allocate BIG'       
      allocate(ex(1:Ntime+Nc+1),stat=status)          
      if (status.ne.0) print *,'can not allocate ex' 
      allocate(fxind(Nx),xc(Nc,Nx))
      

! Initialization
!~~~~~~~~~~~~~~~~~
      
      BIG  = 0.d0
      ex   = 0.d0
      ansr = 0.d0
      a_up = 0.d0 
      a_lo = 0.d0 
 
      xc(:,:) = 0.d0
      !xc(:,1:Nx) = 0.d0
      !xc(2,1:Nx) = 0.d0 
      
      a_lo(1,1) = -Xtinf
      a_lo(1,2) = -XdInf
      a_up(1,3) = +XdInf 
      a_lo(1,4) = -Xtinf
      ij = 0 
      IF (DEF.LE.1) THEN     ! Max2min and period/wavelength
         do I=2,Nx1
            J = IJ+I-1
            xc(3,IJ+1:J) =  h(I)    
            xc(4,IJ+1:J) =  h(1:I-1) 
            IJ = J
         enddo
      ELSE
         ! Level u separated Max2min
         xc(Nc,:) = u
         ! H(1) = H(Nx1+1)= u => start do loop at I=2 since by definition we must have:  minimum<u-level<Maximum
         do i=2,Nx1                
            J = IJ+Nx1-1
            xc(3,IJ+1:J) =  h(i)              ! Max > u
            xc(4,IJ+1:J) =  h(Nx1+2:2*Nx1)    ! Min < u
            IJ = J
         enddo
         
         !CALL ECHO(transpose(xc(3:5,:)))
         if (DEF.GT.3) GOTO 200
      ENDIF
      do Ntd = Nstart,Ntime 
         !Ntd=tn
         Ntdc = Ntd+Nc
         Nt = Ntd-Nd;
         indI(2) = Nt;
         indI(3) = Nt+1;
         indI(4) = Ntd;
         CALL COV_INPUT(BIG(1:Ntdc,1:Ntdc),Ntd,0,R0,R1,R2,R3,R4) ! positive wave period            
         CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex,xc,indI,a_lo,a_up)
         IF (Nx.LT.2) THEN
! Density of TMm given the Max and the Min. Note that the density is not scaled to unity
            ansr(1,1,Ntd) = fxind(1)*CC 
            GOTO 100
         ENDIF
         IJ = 0
         SELECT CASE (DEF)
         CASE(:0)  
! joint density of (M,m)
!~~~~~~~~~~~~~~~~~~~~~~~~
            do  i = 2, Nx1 
               J = IJ+i-1
               ansr(1:i-1,i,1) = ansr(1:i-1,i,1)+fxind(ij+1:J)*CC*dt
               IJ=J
            enddo 
         CASE (1) 
! joint density of (M,m,TMm)
            do  i = 2, Nx1
               J = IJ+i-1
               ansr(1:i-1,i,Ntd) = fxind(ij+1:J)*CC
               IJ = J
            enddo
         CASE (2)
 ! joint density of level v separated (M,m)v
            do  i = 2,Nx1
               J = IJ+Nx1-1
               ansr(2:Nx1,i,1) = ansr(2:Nx1,i,1)+fxind(ij+1:J)*CC*dt
               IJ = J
            enddo 
         CASE (3:) 
 ! joint density of level v separated (M,m,TMm)v
            do  i = 2,Nx1 
               J = IJ+Nx1-1
               ansr(2:Nx1,i,Ntd) = ansr(2:Nx1,i,Ntd)+fxind(ij+1:J)*CC
               IJ = J
            enddo 
         END SELECT   
          
 100     if (ALLOCATED(COV)) then
            write(11,*)  COV(:) ! save coefficient of variation
         endif       
         print *,'Ready: ',Ntd,' of ',Ntime
      enddo
      
      goto 800

 200  do tn = Nstart,Ntime 
         Ntd = tn+1
         Ntdc = Ntd + Nc
         Nt   = Ntd - Nd;
         indI(2) = Nt;
         indI(3) = Nt + 1;
         indI(4) = Nt + 2;
         indI(5) = Ntd;
         !CALL COV_INPUT2(BIG(1:Ntdc,1:Ntdc),tn,-2,R0,R1,R2,R3,R4) ! positive wave period
         IF (SYMMETRY) GOTO 300
        
         do ts = 2,tn-1       
            CALL COV_INPUT(BIG(1:Ntdc,1:Ntdc),tn,ts,R0,R1,R2,R3,R4) ! positive wave period
            !print *,'Big='
            !CALL ECHO(BIG(1:Ntdc,1:MIN(Ntdc,10)))
            CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex,xc,indI,a_lo,a_up) 
           
            SELECT CASE (def)
            CASE (:4) 
               IF (Nx.EQ.1) THEN
! Joint density (TMd,TMm) given the Max and the min. Note the density is not scaled to unity
                  ansr(1,ts,tn) = fxind(1)*CC
         
               ELSE
! 4,  gives level u separated Max2min and wave period from Max to the crossing of level u (M,m,TMd).
                  ij = 0
                  do  i = 2,Nx1 
                     J = IJ+Nx1-1 
                     ansr(2:Nx1,i,ts) = ansr(2:Nx1,i,ts)+
     &                    fxind(ij+1:J)*CC*dt
                     IJ = J
                  enddo 
               ENDIF
            CASE (5:)
               IF (Nx.EQ.1) THEN
! Joint density (Tdm,TMm) given the Max and the min. Note the density is not scaled to unity
                  ansr(1,tn-ts+1,tn) = fxind(1)*CC
               ELSE   
               
! 5,  gives level u separated Max2min and wave period from the crossing of level u to the min (M,m,Tdm).
               ij = 0
               do  i = 2,Nx1 
                  J = IJ+Nx1-1
                  ansr(2:Nx1,i,tn-ts+1)=ansr(2:Nx1,i,tn-ts+1)+
     &                 fxind(ij+1:J)*CC*dt
                  IJ = J
               enddo 
               ENDIF
            END SELECT
            if (ALLOCATED(COV)) then
               write(11,*)  COV(:) ! save coefficient of variation
            endif       
         enddo
         GOTO 400
 300     do ts = 2,FLOOR(DBLE(Ntd)/2.d0)     ! Using the symmetry since U = 0 and the transformation is linear
            CALL COV_INPUT(BIG(1:Ntdc,1:Ntdc),tn,ts,R0,R1,R2,R3,R4) ! positive wave period
                                !print *,'Big='
                                !CALL ECHO(BIG(1:Ntdc,1:Ntdc))
            CALL RINDD(fxind,Big(1:Ntdc,1:Ntdc),ex,xc,indI,a_lo,a_up) 
            IF (Nx.EQ.1) THEN
! Joint density of (TMd,TMm),(Tdm,TMm) given the max and the min. Note that the density is not scaled to unity
               ansr(1,ts,tn) = fxind(1)*CC
               IF (ts.LT.tn-ts+1) THEN
                  ansr(1,tn-ts+1,tn) = fxind(1)*CC
               ENDIF
               GOTO 350
            ENDIF
            IJ = 0 
            SELECT CASE (def)
            CASE (:4) 
              
! 4,  gives level u separated Max2min and wave period from Max to the crossing of level u (M,m,TMd).
               do  i = 2,Nx1 
                  j = ij+Nx1-1  
                  ansr(2:Nx1,i,ts) = ansr(2:Nx1,i,ts)+
     &                 fxind(ij+1:J)*CC*dt
                  IF (ts.LT.tn-ts+1) THEN
                     ansr(i,2:Nx1,tn-ts+1) =  
     &                    ansr(i,2:Nx1,tn-ts+1)+fxind(ij+1:J)*CC*dt ! exploiting the symmetry
                  ENDIF
                  IJ = J
               enddo 
            CASE (5:)
! 5,   gives level u separated Max2min and wave period from the crossing of level u to min (M,m,Tdm).
               do  i = 2,Nx1 
                  J = IJ+Nx1-1
                  
                  ansr(2:Nx1,i,tn-ts+1)=ansr(2:Nx1,i,tn-ts+1)+
     &                 fxind(ij+1:J)*CC*dt
                  IF (ts.LT.tn-ts+1) THEN
                     ansr(i,2:Nx1,ts) = ansr(i,2:Nx1,ts)+
     &                    fxind(ij+1:J)*CC*dt ! exploiting the symmetry
                  ENDIF
                  IJ = J
               enddo 
            END SELECT 
 350     enddo
 400     print *,'Ready: ',tn,' of ',Ntime
      enddo
      

      
      
 800  open (unit=11, file='dens.out',  STATUS='unknown')
      !print *,'ans, IJ,def', shape(ansr),IJ,DEF
      if (Nx.GT.1) THEN
         ij = 1
         IF (DEF.GT.2.OR.DEF.EQ.1) IJ = Ntime
                                !print *,'ans, IJ,def', size(ansr),IJ,DEF
         do ts = 1,ij
            do j=1,Nx1 
               do i=1,Nx1 
                  write(11,*)  ansr(i,j,ts)
               enddo 
            enddo
         enddo
      ELSE
         ij = 1
         IF (DEF.GT.3) IJ = Ntime
                                !print *,'ans, IJ,def', size(ansr),IJ,DEF
         do ts = 1,Ntime
            do j = 1,ij 
               write(11,*)  ansr(1,j,ts)
            enddo 
         enddo
      ENDIF
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
     & (Ntime,Nstart,NIT,speed,SCIS,SEED1,Nx,dT,u,def)
      IMPLICIT NONE
      integer, intent(out):: Ntime,Nstart,NIT,speed,Nx,DEF,SCIS,SEED1
      double precision ,intent(out) :: dT,U
     

      OPEN(UNIT=14,FILE='reflev.in',STATUS= 'UNKNOWN')
      READ (14,*) Ntime
      READ (14,*) Nstart
      READ (14,*) NIT
      READ (14,*) speed 
      READ (14,*) SCIS   
      READ (14,*) seed1   
      READ (14,*) Nx
      READ (14,*) dT
      READ (14,*) U
      READ (14,*) DEF
     
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

      SUBROUTINE COV_INPUT2(BIG,tn,ts,R0,R1,R2,R3,R4)
      IMPLICIT NONE
      double precision, dimension(:,:),intent(inout) :: BIG
      double precision, dimension(:),intent(in) :: R0,R1,R2
      double precision, dimension(:),intent(in) :: R3,R4
      integer ,intent(in) :: tn,ts
      integer :: i,j,N,shft 
! the order of the variables in the covariance matrix
! are organized as follows: 
! for ts <= 1:
!    X'(t2)..X'(ts),...,X'(tn-1) X''(t1),X''(tn)  X'(t1),X'(tn),X(t1),X(tn) 
! = [          Xt               |      Xd       |          Xc             ]
!
! for ts > =2:
!    X'(t2)..X'(ts),...,X'(tn-1) X''(t1),X''(tn) X'(t1),X'(tn),X(t1),X(tn) X(ts) 
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
      
      if (ts.GT.1) THEN
         ! Assumption: a previous call to covinput has been made
         ! need only to update the last row and column of big:
         N=tn+5
           !Cov(Xt,Xc)
         do i=1,tn-2
            j=abs(i+1-ts)
            BIG(i,N)  = -sign(R1(j+1),R1(j+1)*dble(ts-i-1)) !cov(X'(ti+1),X(ts)) 
         enddo
  !Cov(Xc)
         BIG(N    ,N) =  R0(1)       ! cov(X(ts),X(ts))
         BIG(tn+3 ,N) =  R0(ts)      ! cov(X(t1),X(ts))
         BIG(tn+4 ,N) =  R0(tn-ts+1) ! cov(X(tn),X(ts))
         BIG(tn+1 ,N) = -R1(ts)      ! cov(X'(t1),X(ts))
         BIG(tn+2 ,N) =  R1(tn-ts+1) ! cov(X'(tn),X(ts))
  !Cov(Xd,Xc)
         BIG(tn-1 ,N) =  R2(ts)      !cov(X''(t1),X(ts))     
         BIG(tn   ,N) =  R2(tn-ts+1) !cov(X''(tn),X(ts))

                                ! make lower triangular part equal to upper 
         do j=1,N-1
            BIG(N,j) = BIG(j,N)
         enddo
         return
      endif
      IF (ts.LT.0) THEN
         shft = 1
         N=tn+5;
      ELSE
         shft = 0
         N=tn+4;
      ENDIF
      
      
      do i=1,tn-2
      !cov(Xt)
         do j=i,tn-2
           BIG(i,j) = -R2(j-i+1)              ! cov(X'(ti+1),X'(tj+1))
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
      BIG(tn-1  ,tn    ) = R4(tn)     !cov(X''(t1),X''(tn))
      BIG(tn    ,tn    ) = R4(1)

!cov(Xc)
      BIG(tn+3 ,tn+3) = R0(1)        ! cov(X(t1),X(t1))
      BIG(tn+3 ,tn+4) = R0(tn)       ! cov(X(t1),X(tn))
      BIG(tn+1 ,tn+3) = 0.d0         ! cov(X(t1),X'(t1))
      BIG(tn+2 ,tn+3) = R1(tn)       ! cov(X(t1),X'(tn))
      BIG(tn+4 ,tn+4) = R0(1)        ! cov(X(tn),X(tn))
      BIG(tn+1 ,tn+4) =-R1(tn)       ! cov(X(tn),X'(t1))
      BIG(tn+2 ,tn+4) = 0.d0         ! cov(X(tn),X'(tn)) 
      BIG(tn+1 ,tn+1) =-R2(1)        ! cov(X'(t1),X'(t1))
      BIG(tn+1 ,tn+2) =-R2(tn)       ! cov(X'(t1),X'(tn))
      BIG(tn+2 ,tn+2) =-R2(1)        ! cov(X'(tn),X'(tn))
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
           BIG(i,j) = BIG(j,i)
        enddo
      enddo
      RETURN
      END  SUBROUTINE COV_INPUT2

      SUBROUTINE COV_INPUT(BIG,tn,ts,R0,R1,R2,R3,R4)
      IMPLICIT NONE
      double precision, dimension(:,:),intent(inout) :: BIG
      double precision, dimension(:),intent(in) :: R0,R1,R2
      double precision, dimension(:),intent(in) :: R3,R4
      integer ,intent(in) :: tn,ts
      integer :: i,j,N,shft, tnold = 0  
! the order of the variables in the covariance matrix
! are organized as follows: 
! for  ts <= 1:
!    X'(t2)..X'(ts),...,X'(tn-1) X''(t1),X''(tn)  X'(t1),X'(tn),X(t1),X(tn) 
! = [          Xt               |      Xd       |          Xc             ]
!
! for ts > =2:
!    X'(t2)..X'(ts),...,X'(tn-1) X''(t1),X''(tn) X'(ts)  X'(t1),X'(tn),X(t1),X(tn) X(ts) 
! = [          Xt               |      Xd               |          Xc             ]
!
! where 
!
! Xt= time points in the indicator function
! Xd= derivatives
! Xc=variables to condition on
 
! Computations of all covariances follows simple rules: Cov(X(t),X(s)) = r(t,s),
! then  Cov(X'(t),X(s))=dr(t,s)/dt.  Now for stationary X(t) we have
! a function r(tau) such that Cov(X(t),X(s))=r(s-t) (or r(t-s) will give the same result).
!
! Consequently  Cov(X'(t),X(s))    = -r'(s-t)    = -sign(s-t)*r'(|s-t|)
!               Cov(X'(t),X'(s))   = -r''(s-t)   = -r''(|s-t|)
!               Cov(X''(t),X'(s))  =  r'''(s-t)  = sign(s-t)*r'''(|s-t|)
!               Cov(X''(t),X(s))   =  r''(s-t)   = r''(|s-t|)
!               Cov(X''(t),X''(s)) =  r''''(s-t) = r''''(|s-t|)
      SAVE tnold
    
      if (ts.GT.1) THEN
         shft = 1  
         N=tn+5+shft
           !Cov(Xt,Xc)
         do i=1,tn-2
            j=abs(i+1-ts)
            BIG(i,N)  = -sign(R1(j+1),R1(j+1)*dble(ts-i-1)) !cov(X'(ti+1),X(ts)) 
         enddo
  !Cov(Xc)
         BIG(N         ,N) =  R0(1)       ! cov(X(ts),X(ts))
         BIG(tn+shft+3 ,N) =  R0(ts)      ! cov(X(t1),X(ts))
         BIG(tn+shft+4 ,N) =  R0(tn-ts+1) ! cov(X(tn),X(ts))
         BIG(tn+shft+1 ,N) = -R1(ts)      ! cov(X'(t1),X(ts))
         BIG(tn+shft+2 ,N) =  R1(tn-ts+1) ! cov(X'(tn),X(ts))
  !Cov(Xd,Xc)
         BIG(tn-1 ,N) =  R2(ts)      !cov(X''(t1),X(ts))     
         BIG(tn   ,N) =  R2(tn-ts+1) !cov(X''(tn),X(ts))
 
                                !ADD a level u crossing  at ts

           !Cov(Xt,Xd)
         do i = 1,tn-2
            j = abs(i+1-ts)
            BIG(i,tn+shft)  = -R2(j+1) !cov(X'(ti+1),X'(ts)) 
         enddo
       !Cov(Xd)  
         BIG(tn+shft,tn+shft) = -R2(1)  !cov(X'(ts),X'(ts))
         BIG(tn-1   ,tn+shft) =  R3(ts) !cov(X''(t1),X'(ts))
         BIG(tn     ,tn+shft) = -R3(tn-ts+1)  !cov(X''(tn),X'(ts))

        !Cov(Xd,Xc)
         BIG(tn+shft ,N       ) =  0.d0        !cov(X'(ts),X(ts)) 
         BIG(tn+shft,tn+shft+3) =  R1(ts)      ! cov(X'(ts),X(t1))
         BIG(tn+shft,tn+shft+4) = -R1(tn-ts+1) ! cov(X'(ts),X(tn))
         BIG(tn+shft,tn+shft+1) = -R2(ts)      ! cov(X'(ts),X'(t1))
         BIG(tn+shft,tn+shft+2) = -R2(tn-ts+1) ! cov(X'(ts),X'(tn))
                 
                                 
         
         IF (tnold.EQ.tn) THEN  ! A previous call to covinput with tn==tnold has been made
                                ! need only to update  row and column N and tn+1 of big:
                                ! make lower triangular part equal to upper and then return
            do j=1,tn+shft
               BIG(N,j) = BIG(j,N)
               BIG(tn+shft,j) = BIG(j,tn+shft)   
            enddo
             do j=tn+shft+1,N-1
               BIG(N,j) = BIG(j,N)
               BIG(j,tn+shft) = BIG(tn+shft,j)   
            enddo
            return
         ENDIF
         tnold = tn
      ELSE
         N = tn+4
         shft = 0
      endif
          
      
      do i=1,tn-2
      !cov(Xt)
         do j=i,tn-2
           BIG(i,j) = -R2(j-i+1)              ! cov(X'(ti+1),X'(tj+1))
         enddo
      !cov(Xt,Xc)
         BIG(i      ,tn+shft+3) =  R1(i+1)         !cov(X'(ti+1),X(t1))  
         BIG(tn-1-i ,tn+shft+4) = -R1(i+1)         !cov(X'(ti+1),X(tn))  
         BIG(i      ,tn+shft+1) = -R2(i+1)         !cov(X'(ti+1),X'(t1))  
         BIG(tn-1-i ,tn+shft+2) = -R2(i+1)         !cov(X'(ti+1),X'(tn))  
      !Cov(Xt,Xd)
         BIG(i,tn-1)       = R3(i+1)          !cov(X'(ti+1),X''(t1))  
         BIG(tn-1-i,tn)    =-R3(i+1)          !cov(X'(ti+1),X''(tn)) 
      enddo
      
!cov(Xd)
      BIG(tn-1  ,tn-1  ) = R4(1)
      BIG(tn-1  ,tn    ) = R4(tn)     !cov(X''(t1),X''(tn))
      BIG(tn    ,tn    ) = R4(1)

!cov(Xc)
      BIG(tn+shft+3 ,tn+shft+3) = R0(1)        ! cov(X(t1),X(t1))
      BIG(tn+shft+3 ,tn+shft+4) = R0(tn)       ! cov(X(t1),X(tn))
      BIG(tn+shft+1 ,tn+shft+3) = 0.d0         ! cov(X(t1),X'(t1))
      BIG(tn+shft+2 ,tn+shft+3) = R1(tn)       ! cov(X(t1),X'(tn))
      BIG(tn+shft+4 ,tn+shft+4) = R0(1)        ! cov(X(tn),X(tn))
      BIG(tn+shft+1 ,tn+shft+4) =-R1(tn)       ! cov(X(tn),X'(t1))
      BIG(tn+shft+2 ,tn+shft+4) = 0.d0         ! cov(X(tn),X'(tn)) 
      BIG(tn+shft+1 ,tn+shft+1) =-R2(1)        ! cov(X'(t1),X'(t1))
      BIG(tn+shft+1 ,tn+shft+2) =-R2(tn)       ! cov(X'(t1),X'(tn))
      BIG(tn+shft+2 ,tn+shft+2) =-R2(1)        ! cov(X'(tn),X'(tn))
!Xc=X(t1),X(tn),X'(t1),X'(tn) 
!Xd=X''(t1),X''(tn)
!cov(Xd,Xc)
      BIG(tn-1  ,tn+shft+3) = R2(1)           !cov(X''(t1),X(t1))     
      BIG(tn-1  ,tn+shft+4) = R2(tn)          !cov(X''(t1),X(tn))     
      BIG(tn-1  ,tn+shft+1) = 0.d0            !cov(X''(t1),X'(t1))     
      BIG(tn-1  ,tn+shft+2) = R3(tn)          !cov(X''(t1),X'(tn))     
      BIG(tn    ,tn+shft+3) = R2(tn)          !cov(X''(tn),X(t1))     
      BIG(tn    ,tn+shft+4) = R2(1)           !cov(X''(tn),X(tn))     
      BIG(tn    ,tn+shft+1) =-R3(tn)          !cov(X''(tn),X'(t1))     
      BIG(tn    ,tn+shft+2) = 0.d0            !cov(X''(tn),X'(tn))
      
     
      ! make lower triangular part equal to upper 
      do j=1,N-1
        do i=j+1,N
           BIG(i,j) = BIG(j,i)
        enddo
      enddo
      RETURN
      END  SUBROUTINE COV_INPUT
      END  PROGRAM  sp2mmt
       








