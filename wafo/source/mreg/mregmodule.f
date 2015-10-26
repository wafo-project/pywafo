C     Version  July 2007
C
C  The MREG module provide 3 programs.
C
C    1) MREG
C    2) RIND
C    3) FI - normal CDF
C 
C MREG and RIND are explained in the following:
C
C
C CALL MREG(F,R,B,DB,AA,BB,A,DA,VDER,M,N,NIT,INFR)
C
C  F    = expectation
C  R    = Covariance R(i+(j-1)*N) = Cov( Delta(T(i)), Delta(T(j)),    length RDIM (in)          
C  B    = Covariance  B(i) = Cov(Delta(T(i)), XN),  B(N+1)=Var(XN)    length NMAX (in)
C  DB   = Covariance DB(i) = Cov(Delta(T(i)), Y0), DB(N+1)=Cov(XN,Y0) length NMAX (in)
C  AA   = Regression matrix coefficients    size MMAX x MMAX
C  BB   = Regression vector coefficients    length MMax + 1
C  A    = Slepian model coefficients,  length (MMax + 1) * NMAX                                                                                                                                                       
C  DA   = Slepian model coefficients,  length  MMax + 1 
C  VDER = variance of Y0, Var(Y0)
C  M    = Number of regressors       ( 0 < M < MMAX)
C  N    = dimension of the problem   (     N < NMAX)
C  NIT  = 0,1,2..., maximum # of iterations/integrations done by quadrature 
C         to calculate the indicator function 
C  INFR = 1 means all input are the same as in the previous call except BB, A and DA
C         0 indicate new input
C
C  The program MREG computes the following problem:
C
C  We consider a process X(I)=X(T(I)) at the grid of  N  points  T(1),...,T(N),
C
C          X(I) = -A(I) + Z*A(I+N) + Sum Xj*A(I+(j+1)*N) + Delta(I), j=1,...,M-1
C
C  where the sum disappears if M=1. We assume that Z,Xj are independent
C  standard Rayleigh, Gaussian distributed rv. and independent of the zero
C  mean Gaussian residual process, with covariance structure given in  R,
C
C          R(i+(j-1)N) = Cov (Delta(T(i)), Delta(T(j))).
C
C  Additionally we have a zero mean Gaussian variable XN, 
C  independent of Z,Xj with  covariance structure defined by 
C  B(i)= Cov (Delta(T(i)),XN), i=1,...,N,  B(N+1)=Var(XN). 
C  Furthermore XN and Z,Xj satisfies the following equation system
C
C     (BB + (XN,0,...,0)^T =  AA*(Z,X1,...,Xm-1)^T       (***)
C
C  where AA is (M,M) matrix, BB is M-vector. We rewrite this equation, by
C  introducing a variable Xm=XN/SQRT(Var(XN)) and construct   new matrix AA1
c  by adding the column (SQRT(Var(XN)),0,...,0) and the row with only zeros.
C  The equations (***) writtes
C
C     (BB,0)^T  =  AA1*(Z,X1,...,Xm-1,Xm)^T       (****)
C
C  where AA1 is (M+1,M+1) matrix, We assume that the rank of AA1 is M,
C  otherwise the density is singular and we give a output F=0.CC
C
C  Let Y0 be a zero-mean Gaussian variable independent of Z,Xj 
C  with  covariance structure defined by 
C  DB(i)= Cov (Delta(T(i)),Y0), i=1,...,N,  DB(N+1)=Cov(XN,Y0), Var(Y0)=VDER. 
C  Let Y be defined by
C
C     Y=-DA(1) + Z*DA(2) + Sum Xj*DA(2+j) +Y0, j=1,...,M-1.
C
C  The MREG program computes:
C
C  F = E[ Y^+ *1{ HH<X(I)<0 for all I, I=1,...,N}|Z,X1,...,Xm-1 solves (***)]
C      *f_{Z,X1,....,Xm-1}(***).
C
C  In the simplest case NIT=0 we define (Delta(1),...,Delta(N),XN)=0.0d0 
C
C  We renormalize  vectors AN and DA, the covariance fkn R, DB
C  and VDER. Then by renormalization we choose the Gaussian variable X such
C  that F is written in the form
C
C  F = E[(D0(1)+X*D0(2)+Y1)^+*(PC+X*PD)^+*1{HH <A0(I)+X*B0(I)+Delta1(I)<0}]
C
C  Observe, PC+X*PD>0 defines integration region for X.
C  In the simplest case NIT=0 we define (Delta(1),...,Delta(N),Y1)=0.0d0 
C  For NIT=1 only (Delta(1),...,Delta(N))=0, i.e. we have to compute
C  a one dimensional integral. Finally by conditioning on X the problem is
C  put in the format of RIND-problem.
C
C  INF indicates whether one
C  has already called the subroutine before and ONLY! inputs BB, DA or A
C  was changed.
C
C  Observe the limitations are :  N<NMAX = 201, 0<M< 5 = MMAX.
C
C
c                  3-IX-93
C
C  CALL  RIND(XIND,R,BU,DBUN,DB,SQ,VDER,NIT,N,INFR)
C
C  XIND = expectation/density (inout)
C  R    = Covariances Delta(ti), Delta(tj), size RDIM x 1 (in)
C         R(i+(j-1)*N) = Cov( Delta(T(i)), Delta(T(j)) 
C  BU   = expectation of y(t), i.e., E(y(t)), size NMAX x 1  (in)
C  DBUN = expectation of Y, i.e., E(Y) 
C  DB   = Covariances Delta(T(i)), Y,    size NMAX x 1  (in)
C         DB(i) = Cov(Delta(T(i)), Y)
C  SQ   = standard deviations of Delta(T(i)), size NMAX x 1
C         SQ(I) = SQRT (R(I+(I-1)*N)) 
C  VDER = variance of Y, Var(Y)
C  NIT  = 0,1,2..., maximum # of iterations/integrations done by quadrature 
C         to calculate the indicator function 
C  N    = dimension of the problem
C  INFR = 1 means R, DB and SQ are the same as in the previous
C         0 indicate new R, DB and SQ.  
C
C  The program RIND computes the following problem:
C
C  Let the process  y(t)=BU(t)+Delta(t), where  Delta(t)  is zero mean
C  Gaussian process and BU(t) be expectation of  y(t). Consider the process  x
C  at the grid T(1),...,T(N), N=0,...,50, (N=0 means the empty grid T).
C  Let  y(I) = BU(T(I)) + Delta(T(I)).  Observe we do not assume that the
C  points T(I) are ordered or from, e.g. T(I) are in R^K.
C
C  The covariance fkn of Delta at the points T(1),...,T(N), are given in
C  the vector R;  Cov( Delta(T(i)), Delta(T(j)) = R(i+(j-1)*N),
C  furter E[y(T(i))] = BU(i). Hence the dimension of R must be N*N=2500.
C  The vector SQ(1), ...., SQ(N) contains standard deviations of the residual
C  Delta(T(I)), e.g. SQ(I) = SQRT (R(I+(I-1)*N)). However the small values
C  of SQ could be corrupted by nummerical errors especially if the covariance
C  structure was computed using the FFT algorithm. IF R(I+(I-1)*N)<EPS
C  SQ(I)=0.    and is used as an indicator that one is not allowed to
C  condition on Delta(T(I)). Further when one have conditioned on the point
C  T(I) the variance is put to zero.
C
C  Consider in addition to y(t) a Gaussian variable  Y, E[Y]=DBUN, Var(Y)=VDER,
C  DB(I)=Cov(Delta(T(I)),Y).
C
C  *** XIND - is the result;  XIND=E[Y^+1{ HH<y(I)<0 for all I, I=1,...,N}] ***
C
C In the speccial case by choosing  DB(I)=0, VDER=0 and DBUN=1, IAC=0,1,
C (if IAC=0 VDER can take any positive value), the output XIND is equal to
C XIND=Prob( HH < y(I) < 0 for all I, I=1,...,N).
C
C
C  Some control variables:
C  INFR=1 means that both R, DB and SQ are the same as in the previous
C  call of RIND subroutine, INFR=0 indicates the new R, DB and SQ. The history
C  of the conditioning is saved in the vectors INF(5), INFO(5): INF(1), ...,
C  INF(5) are the times one has conditioned in the subroutines RINDT1,...,
C  RINDT5, respectively. After conditioning INFO(i)=INF(i). Now if INF=INFO
C  then the conditioning tree has not be changed and one not need to compute
C  the conditonal covariances. This is really time saving trick. We are assume
C  that the program saves all the time the matrices at the same fysical
C  location, e.g. the values of variables are saved during the execution.
C  This has all the time be checked when new compilator will be used.
C
C  The variable ISQ marks which type of conditioning will be used ISQ=0
C  means random time where the probability is minimum, ISQ=1 is the time
C  where the variance of the residual process is minimal(ISQ=1 is faster).
C
C  NIT, IAC are  described in CROSSPACK paper, EPS0 is the accuracy constant
C  used in choosing the number of nodes in numerical integrations
C  (XX1, H1 vectors). The nodes and weights and other parameters are
C  read in the subroutine INITINTEG from files Z.DAT, H.DAT and ACCUR.DAT.
C
C
C    NIT=0, IAC=1 then one uses RIND0 - subroutine, all other cases
C    goes through RIND1, ...,RIND5. NIT=0, here means explicite formula
C    approximation for XIND=E[Y^+1{ HH<BU(I)<0 for all I, I=1,...,N}], where
C    BU(I) is deterministic function.
C
C    NIT=1, leads tp call RIND1, IAC=0 is also explicit form approximation,
C    while IAC=1 leads to maximum one dimensional integral.
C    .......
C    NIT=5, leads tp call RIND5, IAC is maximally 4-dimensional integral,
C    while IAC=1 leads to maximum 5 dimensional integral.
C
!
! Revised pab August 2007
! - replaced a call to SVDCMP with DSVDC derived from Lapack
! Revised pab July 2007
!
! - fixed some bugs 
! - reimplemented as module mregmodule
! - moved the functions/subroutines in twog.f into rindmod and renamed it to MREG. -> mreg and rind publicly available
! - All commonblocks are replaced with a corresponding module

! References
! Rychlik, I and Lindgren, G (1993)
! "CROSSREG - A Technique for First Passage and Wave Density Analysis"
! Probability in the Engineering and Informational Sciences, Vol 7, pp 125--148
!
! Lindgren, G and Rychlik, I (1991)
! "Slepian Models and Regression Approximations in Crossing  and xtreme value Theory",
! International Statistical Review, Vol 59, 2, pp 195--225


      MODULE SIZEMOD
      IMPLICIT NONE
      INTEGER, PARAMETER :: MMAX = 6, NMAX = 201
      INTEGER, PARAMETER :: RDIM = NMAX*NMAX 
      END MODULE SIZEMOD
      
      MODULE EPSMOD
      IMPLICIT NONE             
                      ! Constants determining accuracy of integration
                      !-----------------------------------------------
                      !if the conditional variance are less than: 
C      DOUBLE PRECISION :: EPS2=1.d-4    !- EPS2, the variable is 
                                        !  considered deterministic 
      DOUBLE PRECISION :: EPS = 1.d-2   ! SQRT(EPS2)
C      DOUBLE PRECISION :: XCEPS2=1.d-16 ! if Var(Xc) is less return NaN
      DOUBLE PRECISION :: EPSS = 5.d-5  ! accuracy of Indicator 
C      DOUBLE PRECISION :: CEPSS=0.99995d0 ! accuracy of Indicator 
      DOUBLE PRECISION :: EPS0 = 5.d-5 ! used in GAUSSLE1 to implicitly 
                                       ! determ. # nodes  
   
C      DOUBLE PRECISION :: fxcEpss=1.d-20 ! if less do not compute E(...|Xc)
C      DOUBLE PRECISION :: xCutOff=5.d0  ! upper/lower truncation limit of the 
                                       ! normal CDF 
C      DOUBLE PRECISION :: FxCutOff  = 0.99999942669686d0 
C      DOUBLE PRECISION :: CFxCutOff = 5.733031438470704d-7       ! 1-FxCutOff, 
      
      END MODULE EPSMOD

      MODULE RINTMOD
      DOUBLE PRECISION, save :: C = 4.5d0
      DOUBLE PRECISION, save :: FC = 0.999993204653751d0 
C     COMMON /RINT/   C,FC
      END MODULE RINTMOD

      MODULE TBRMOD
      USE SIZEMOD
      IMPLICIT NONE  
      DOUBLE PRECISION, DIMENSION(NMAX) :: HH
      END MODULE TBRMOD

      MODULE EXPACCMOD
      DOUBLE PRECISION,PARAMETER:: PMAX = 40.0d0
C     COMMON /EXPACC/ PMAX
      END MODULE EXPACCMOD

      MODULE INFCMOD
      IMPLICIT NONE  
      INTEGER, save :: ISQ = 0, IAC=1
      INTEGER, DIMENSION(10) :: INF,INFO
C     DOUBLE PRECISION, DIMENSION(10):: 
C      COMMON /INFC/   ISQ,INF,INFO
      END MODULE INFCMOD
      MODULE CHECKMOD
      IMPLICIT NONE
C      III01,III11,... - variables,counts how many times one calls
C      subroutine RIND0,RIND1,..., III*1 are also modified in the
C      subroutines RIND*. This gives us statistics over the complexity of
C      numerical calculations.
      INTEGER :: III01,III11,III21,III31,III41,III51
      INTEGER :: III61,III71,III81,III91,III101
      INTEGER :: III0
      END MODULE CHECKMOD


      MODULE QUADRMOD
      IMPLICIT NONE         ! Quadratures available: Legendre
      INTEGER            :: I

C      BLOCK DATA inithermite

      INTEGER, PARAMETER :: NNW = 13
      INTEGER, DIMENSION(25) :: NN
      REAL*8  Z(126),H(126)
      
      
C      COMMON /QUADR/  Z,H,NN,NNW
c      COMMON /EXPACC/ PMAX
C      COMMON /RINT/   C,FC
      
C      DATA NNW /13/
      DATA (NN(I),I=1,NNW)/2,3,4,5,6,7,8,9,10,12,16,20,24/
C      DATA PMAX/40./
C      DATA C/4.5/
      DATA (H(I),I=1,61)/1.0d0,1.0d0,0.555555555555556d0,
     * 0.888888888888889d0,
     * 0.555555555555556d0,0.347854845137454d0,0.652145154862546d0,
     * 0.652145154862546d0,0.347854845137454d0,0.236926885056189d0,
     * 0.478628670499366d0,0.568888888888889d0,0.478628670499366d0,
     * 0.236926885056189d0,0.171324492379170d0,0.360761573048139d0,
     * 0.467913934572691d0,0.467913934572691d0,0.360761573048139d0,
     * 0.171324492379170d0,0.129484966168870d0,0.279705391489277d0,
     * 0.381830050505119d0,0.417959183673469d0,0.381830050505119d0,
     * 0.279705391489277d0,0.129484966168870d0,0.101228536290376d0,
     * 0.222381034453374d0,0.313706645877887d0,0.362683783378362d0,
     * 0.362683783378362d0,0.313706645877887d0,0.222381034453374d0,
     * 0.101228536290376d0,0.081274388361574d0,0.180648160694857d0,
     * 0.260610696402935d0,0.312347077040003d0,0.330239355001260d0,
     * 0.312347077040003d0,0.260610696402935d0,0.180648160694857d0,
     * 0.081274388361574d0,0.066671344308688d0,0.149451349150581d0,
     * 0.219086362515982d0,0.269266719309996d0,0.295524224714753d0,
     * 0.295524224714753d0,0.269266719309996d0,0.219086362515982d0,
     * 0.149451349150581d0,0.066671344308688d0,0.047175336386512d0,
     * 0.106939325995318d0,0.160078328543346d0,0.203167426723066d0,
     * 0.233492536538355d0,0.249147048513403d0,0.249147048513403d0/
      DATA (H(I),I=62,101)/0.233492536538355d0,0.203167426723066d0,
     * 0.160078328543346d0,0.106939325995318d0,
     * 0.047175336386512d0,0.027152459411754094852d0,
     * 0.062253523938647892863d0,0.095158511682492784810d0,
     * 0.124628971255533872052d0,0.149595988816576732081d0,
     * 0.169156519395002538189d0,0.182603415044923588867d0, 
     * 0.189450610455068496285d0,0.189450610455068496285d0,
     * 0.182603415044923588867d0,0.169156519395002538189d0,
     * 0.149595988816576732081d0,0.124628971255533872052d0,
     * 0.095158511682492784810d0,0.062253523938647892863d0,
     * 0.027152459411754094852d0,0.017614007139152118312d0,
     * 0.040601429800386941331d0,0.062672048334109063570d0,
     * 0.083276741576704748725d0,0.101930119817240435037d0,
     * 0.118194531961518417312d0,0.131688638449176626898d0,
     * 0.142096109318382051329d0,0.149172986472603746788d0,
     * 0.152753387130725850698d0,0.152753387130725850698d0,
     * 0.149172986472603746788d0,0.142096109318382051329d0,
     * 0.131688638449176626898d0,0.118194531961518417312d0,
     * 0.101930119817240435037d0,0.083276741576704748725d0,
     * 0.062672048334109063570d0,0.040601429800386941331d0/
      DATA (H(I),I=102,126)/0.017614007139152118312d0,
     * 0.012341229799987199547d0, 0.028531388628933663181d0,
     * 0.044277438817419806169d0, 0.059298584915436780746d0,
     * 0.073346481411080305734d0, 0.086190161531953275917d0,
     * 0.097618652104113888270d0, 0.107444270115965634783d0,
     * 0.115505668053725601353d0, 0.121670472927803391204d0,
     * 0.125837456346828296121d0, 0.127938195346752156974d0,
     * 0.127938195346752156974d0, 0.125837456346828296121d0,
     * 0.121670472927803391204d0, 0.115505668053725601353d0,
     * 0.107444270115965634783d0, 0.097618652104113888270d0,
     * 0.086190161531953275917d0, 0.073346481411080305734d0,
     * 0.059298584915436780746d0, 0.044277438817419806169d0,
     * 0.028531388628933663181d0, 0.012341229799987199547d0/ 
        
      DATA (Z(I),I=1,58)/-0.577350269189626d0,0.577350269189626d0,
     * -0.774596669241483d0,0.0d0,
     *  0.774596669241483d0, -0.861136311594053d0, -0.339981043584856d0,
     *  0.339981043584856d0,  0.861136311594053d0, -0.906179845938664d0,
     * -0.538469310105683d0,0.0d0,
     *  0.538469310105683d0,  0.906179845938664d0, -0.932469514203152d0,
     * -0.661209386466265d0, -0.238619186083197d0,  0.238619186083197d0,
     *  0.661209386466265d0,  0.932469514203152d0, -0.949107912342759d0,
     * -0.741531185599394d0, -0.405845151377397d0, 0.0d0,
     *  0.405845151377397d0,  0.741531185599394d0,  0.949107912342759d0,
     * -0.960289856497536d0, -0.796666477413627d0, -0.525532409916329d0,
     * -0.183434642495650d0,  0.183434642495650d0,  0.525532409916329d0,
     *  0.796666477413627d0,  0.960289856497536d0, -0.968160239507626d0,
     * -0.836031107326636d0, -0.613371432700590d0, -0.324253423403809d0,
     *  0.0d0, 
     *  0.324253423403809d0,  0.613371432700590d0,  0.836031107326636d0,
     *  0.968160239507626d0, -0.973906528517172d0, -0.865063366688985d0,
     * -0.679409568299024d0, -0.433395394129247d0, -0.148874338981631d0,
     *  0.148874338981631d0,  0.433395394129247d0,  0.679409568299024d0,
     *  0.865063366688985d0,  0.973906528517172d0, -0.981560634246719d0,
     * -0.904117256370475d0, -0.769902674194305d0, -0.587317954286617d0/
      DATA (Z(I),I=59,99)/-0.367831498198180d0, -0.125233408511469d0,
     *      0.125233408511469d0,  0.367831498198180d0,  
     *      0.587317954286617d0,  0.769902674194305d0,
     *      0.904117256370475d0,  0.981560634246719d0,
     *     -0.989400934991649932596d0,
     *     -0.944575023073232576078d0, -0.865631202387831743880d0,
     *     -0.755404408355003033895d0, -0.617876244402643748447d0,
     *     -0.458016777657227386342d0, -0.281603550779258913230d0,
     *     -0.095012509837637440185d0,  0.095012509837637440185d0,
     *      0.281603550779258913230d0,  0.458016777657227386342d0,
     *      0.617876244402643748447d0,  0.755404408355003033895d0,
     *      0.865631202387831743880d0,  0.944575023073232576078d0,
     *      0.989400934991649932596d0, -0.993128599185094924786d0,
     *     -0.963971927277913791268d0, -0.912234428251325905868d0,
     *     -0.839116971822218823395d0, -0.746331906460150792614d0,
     *     -0.636053680726515025453d0, -0.510867001950827098004d0,
     *     -0.373706088715419560673d0, -0.227785851141645078080d0,
     *     -0.076526521133497333755d0,  0.076526521133497333755d0,
     *      0.227785851141645078080d0,  0.373706088715419560673d0,
     *      0.510867001950827098004d0,  0.636053680726515025453d0,
     *      0.746331906460150792614d0,  0.839116971822218823395d0/
      DATA (Z(I),I=100,126)/0.912234428251325905868d0,
     *      0.963971927277913791268d0,  0.993128599185094924786d0,
     *     -0.995187219997021360180d0, -0.974728555971309498198d0,
     *     -0.938274552002732758524d0, -0.886415527004401034213d0,
     *     -0.820001985973902921954d0, -0.740124191578554364244d0,
     *     -0.648093651936975569252d0, -0.545421471388839535658d0,
     *     -0.433793507626045138487d0, -0.315042679696163374387d0,
     *     -0.191118867473616309159d0, -0.064056892862605626085d0,
     *      0.064056892862605626085d0,  0.191118867473616309159d0,
     *      0.315042679696163374387d0,  0.433793507626045138487d0,
     *      0.545421471388839535658d0,  0.648093651936975569252d0,
     *      0.740124191578554364244d0,  0.820001985973902921954d0,
     *      0.886415527004401034213d0,  0.938274552002732758524d0,
     *      0.974728555971309498198d0,  0.995187219997021360180d0/
      END MODULE QUADRMOD
      

C
      MODULE MREGMOD
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: RIND, MREG, FI

      INTERFACE RIND
      MODULE PROCEDURE RIND
      END INTERFACE

      INTERFACE MREG
      MODULE PROCEDURE MREG
      END INTERFACE

      INTERFACE FI
      MODULE PROCEDURE FI
      END INTERFACE

      INTERFACE C1_C2
      MODULE PROCEDURE C1_C2
      END INTERFACE

      INTERFACE GAUSS1
      MODULE PROCEDURE GAUSS1
      END INTERFACE

      INTERFACE GAUSINT
      MODULE PROCEDURE GAUSINT
      END INTERFACE

      INTERFACE PYTHAG
      MODULE PROCEDURE PYTHAG
      END INTERFACE


      CONTAINS


      SUBROUTINE RIND(XIND,R,BU,DBUN,DB,SQ,VDER,NIT,N,INFR)
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE SIZEMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB, SQ
      INTEGER, intent(in) :: NIT,N,INFR
      REAL*8 SDER
      INTEGER, save :: NNIT
      INTEGER I,III
C     DIMENSION R(1),BU(1),SQ(1),DB(1)
C     DIMENSION INF(10),INFO(10),HH(101)
C     COMMON /TBR/ HH
C     COMMON /INFC/   ISQ,INF,INFO
C     COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C    *,III61,III71,III81,III91,III101
C     COMMON /EPS/    EPS,EPSS,CEPSS
C
C      III01,III11,... - variables,counts how many times one calls
C      subroutine RIND0,RIND1,..., III*1 are also modified in the
C      subroutines RIND*. This gives us statistics over the complexity of
C      numerical calculations.
C
      XIND=0.0d0
      IF (N.lt.1) go to 99
      
      IF (INFR.EQ.0) THEN
         NNIT=MIN(NIT,N)
         if (NNIT.gt.10) NNIT=10
         DO I=1,10
            INF(I)=0
            INFO(I)=0
         enddo
         III=0
         DO I=1,N
          IF (SQ(I).GT.EPS) then
            III=1
            else
             IF(BU(I).GT.0.0d0) THEN
             RETURN
             END IF
             IF(BU(I).LT.HH(I)) THEN
             RETURN
             END IF
          END IF
         enddo
        END IF
        IF (III.eq.0) go to 99

!      GO TO (10,20,30,40,50,60,70,80,90,100) NNIT
      SELECT CASE (NNIT)
      CASE (1)
       CALL RIND1(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii11=iii11+1
      CASE(2)
       CALL RIND2(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii21=iii21+1
      CASE(3)
       CALL RIND3(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii31=iii31+1
      CASE(4)
       CALL RIND4(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii41=iii41+1
      CASE(5)
       CALL RIND5(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii51=iii51+1
      CASE(6)
       CALL RIND6(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii61=iii61+1
      CASE(7)
       CALL RIND7(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii71=iii71+1
      CASE(8)
       CALL RIND8(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii81=iii81+1
      CASE (9)
       CALL RIND9(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      iii91=iii91+1
      CASE (10)
       CALL RIND10(XIND,R,BU,DBUN,DB,SQ,VDER,N)
       iii101=iii101+1
      CASE DEFAULT
       CALL RIND0(XIND,BU,DBUN,VDER,N)
       iii01=iii01+1
      END SELECT
      RETURN

 99   continue   
      SDER=0.0d0
      IF(VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=PMEAN(DBUN,SDER)
      return
      END SUBROUTINE RIND

      SUBROUTINE RIND0(XIND,BU,DBUN,VDER,N)
      USE TBRMOD
      USE EPSMOD
      USE SIZEMOD
      IMPLICIT NONE
      INTEGER, intent(in) :: N
      REAL*8, intent(inout) ::  XIND,DBUN,VDER
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU
      REAL*8  SDER
      INTEGER I
!      DIMENSION BU(NMAX)
C      DIMENSION HH(101)
C      COMMON /EPS/ EPS,EPSS,CEPSS
C      COMMON /TBR/ HH
      
      IF (N.LT.1) GO TO 20
      XIND=0.0d0
      IF(DBUN.LT.0.0d0) THEN
      RETURN
      END IF
      DO I=1,N
      IF(BU(I).GT.0.0d0) THEN
      RETURN
      END IF
      IF(BU(I).LT.HH(I)) THEN
      RETURN
      END IF
      enddo
20    CONTINUE
      SDER=0.0d0
      IF(VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END SUBROUTINE RIND0

      SUBROUTINE RIND1(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(NMAX), save :: B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8 XMI,XMA,DER,SDER
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XFF,XF,X,XH, SQ0, HHB 
      INTEGER I,III,J,II0,N1
!     INTEGER IAC,N
!      real*8 XIND,R,BU,DBUN,DB,SQ,VDER
!     REAL*8 XX1,H1, B1,SQ1,XMI,XMA, SDER,DB1N , DER, SDER1
!     REAL*8 XFF,X, XH, SQ0, HHB, SS0, VDER1, XF
!     INTEGER I,J,III,II0,N1
C      DIMENSION R(1),BU(1),SQ(1),DB(1),B1(NMAX),SQ1(NMAX)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX) ,B1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
c      print *,'Topp of R1:',sq(1),sq(2),sq(3)
      XIND=0.0d0

C Choice of the time for conditioning, two methods
C
C ISQ=1; INF(1)=II0 is the point where the  SQ(I) obtaines its maximum, SQ0
C is the maximal st. deviation of the residual.
C
C ISQ=0; INF(1) is the time point when the probability P(hh<BU(I)+Delta(I)<0)
C obtains its minimum which is denoted by XFF.
C
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
C
C  If N<1 means empty grid we can not condition on any point and hence GO TO 11
C  then XFF=1.0d0 and the XIND will be approximated by E[Y^+]=PMEAN(DBUN,SDER).
C  Obs. E[Y]=DBUN and Var(Y)=VDER.
C
      SQ0=0.0d0
      DO I=1,N
      SQ1(i)=0.0d0
      IF (SQ(I).LE.eps) GO TO 1
      HHB=HH(I)-BU(I)
C
C Obs. SQ(I)<=EPS idicates that the point is not good for conditioning.
C There can be two reasons for it: 1 Variance of the residual is too small
C or the point was already used before.
C
      if (ISQ.GT.1) then
      SS0  =R(I+(I-1)*N)
      DB1N =DB(I)
      VDER1=DB1N*DB1N/SS0
      IF (VDER1.gt.SQ0) Then
      SQ0=VDER1
      II0=I
      END IF
      ELSE
      IF (SQ(I).GT.SQ0) THEN
      SQ0=SQ(I)
      II0=I
      END IF
      END IF

      X=-BU(I)/SQ(I)
      XH=HHB/SQ(I)
      XF=FI(X)-FI(XH)
      IF(XF.LT.XFF) THEN
      INF(1)=I
      XFF=XF
      END IF
1     CONTINUE
      enddo
11    CONTINUE
C
C   If the minimum probability XFF is close to 0 !!! then the indicator 1{}
C   can be bounded by EPSS leading to the approximation of XIND=0 and RETURN.
C
      IF(XFF.LT.EPSS) RETURN
C
C  We are stoping because we assume that for all sample pathes I(0,t)=1
C
C   If the minimum probability XFF is close to one then the indicator 1{}
C   can be bounded by 1 leading to the approximation of XIND by E[Y^+]=
C   PMEAN(DBUN,SDER), if IAC=1 or with E[Y]^+=MAX(0,DBUN).
C   This reduces the order of integrations.
C
      IF (XFF.GT.0.999d0*FC) THEN
      SDER=0.
      IF(VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=MAX(DBUN,0.0d0)
      IF(IAC.LT.1) RETURN
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END IF
C
C  We are conditioning on the point T(INF(1)). If ISQ=1 INF(1)=ii0.
C  Obviously, X(INF(1))=BU(INF(1))+Delta(INF(1)), where SQ0 is a standard
C  deviation of Delta(IN(1)), hence  if X(INF(1))>XMA or X(INF(1))<XM1
C  1{}=0. Hence the values of X(INF(1)) are truncated to the interval [xmi,xma].

      IF(ISQ.EQ.1) INF(1)=II0
      SQ0=SQ(INF(1))
      XMA=-BU(INF(1))/SQ0
      XMI=XMA+HH(INF(1))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
C
C   Now we are checking whether INF(I)=INFO(I), I=1,..,5, what indicates
C   that all conditional covariances and variancec are unchanged since the
C   last call of this subroutine (R,SQ,DB) as well as
C
      III=0
      DO I=1,10
      III=III + ABS(INF(I)-INFO(I))
      enddo
      IF (III.EQ.0) GO TO 99
      DO I=1,N
       B1(I) = R(I+(INF(1)-1)*N)
      enddo
      SS0 = B1(INF(1))
      DB1N = DB(INF(1))
      INFO(1) = INF(1)
      VDER1 = VDER-DB1N*DB1N/SS0
      SDER1 = 0.0d0
      IF(VDER1.GT.EPS) SDER1=SQRT(VDER1)
      DB1N = DB1N/SQ0
      DO I=1,N
       B1(I) = B1(I)/SQ0
      enddo
99    CONTINUE
C
C  Here conditioning is done
C
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,0.0d0,SQ1,N)
      IF(FI(XMA)-FI(XMI).LT.EPSS) RETURN
C *******************************************************
C
C  In this special case RIND1 if IAC=0 one can explicitly compute XIND
C  and stop.
      IF (IAC.LT.1) THEN
      XIND=GAUSINT(XMI,XMA,DBUN,DB1N,1.0d0,0.0d0)
      RETURN
      END IF
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
c      print *,XMI,XMA,EPS0,N1
c      write(11,*) XMI,XMA,EPS0,N1
      XIND=0.0d0
      DO J=1,N1
      DER=DBUN+XX1(J)*DB1N
      XIND=XIND+PMEAN(DER,SDER1)*H1(J)
      III01=III01+1
c      IF (N.eq.15) then
c      print *,'der,dbun,db1n,sder1',der,dbun,db1n,sder1
c      write(11,*) der,dbun,db1n,sder1
c      end if
10    CONTINUE
      enddo
c      IF (N.eq.15) then
c      do 999 iii=1,N
c      print *,iii,sq(Iii)
c999   continue
c      write(11,*) XIND,INF(1),INF(2),INF(3),inf(4)
c      pause
c      end if
      RETURN
      END SUBROUTINE RIND1

      SUBROUTINE RIND2(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J, II0,III, N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1,R1,B1,DB1,BU1,SQ1
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 XFF,X, XH, SQ0, SDER, XF, XMI, XMA
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1) 
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
c      COMMON/CHECK/III0,III1,III2,III3,III4
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
c      PRINT *,'Topp of 2:',sq(1),sq(2),sq(3)
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
c      CSQ=C*SQ(I)
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) GO TO 1
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) SQ1(I)=EPS1
C      IF (BU(I).GT. CSQ.OR.BU(I).LT.HH(I)-CSQ) RETURN
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
c      IF(XF.GT.CEPSS) SQ1(I)=EPS1
         IF (XF.LT.XFF) THEN
            INF(2)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
11    CONTINUE
      IF(XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
      SDER=0.0d0
      IF (VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=MAX(DBUN,0.0d0)
      IF(IAC.LT.1) RETURN
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END IF
      IF(ISQ.EQ.1) INF(2)=II0
      SQ0=SQ(INF(2))
      XMA=-BU(INF(2))/SQ0
      XMI=XMA+HH(INF(2))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
C *************************************************
C
C  We are conditioning on the point T(INF(2)) and write new model
C  BU(I)+X*B1(I)+Delta1(I), I=1,...,N  (Obs. we do not use I=1,N)
C  SQ1(I) is standard deviation of Delta1 DBUN=BU'(N), DB1N=B1'(N) and X is
C  N(0,1) independent of Delta1, SDER1 is standard deviation of Delta1'(N).
C
      III=0
      DO I=2,10
      III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(2),N)
C      III1=III1+1
      SS0=B1(INF(2))
      INFO(2)=INF(2)
      DB1N=DB(INF(2))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
C      SQ0=SQRT(SS0)
      DB1N=DB1N/SQ0
      SQ1(INF(2))=0.0d0
      DO I=1,N
C
C IF (SQ1(I).EQ.EPS1) GO TO 3 - the .EQ. can not be raplaced with .LT. without
C some general changes in the strategy of SQ and SQ1 values. More exactly
C SQ can not be changed in this subroutine when for some  I  we would like to
C put SQ1(I)=EPS1 in the first loop. This SQ1 should not be changed here and
C thus we have GO TO 3 statement. Observe that the other SQ1 values are
C
      IF (SQ(I).LE.EPS) GO TO 3
C      IF (SQ1(I).EQ.EPS1.OR.SQ(I).LE.EPS1) GO TO 3
      XR1=R1(I+(I-1)*N)
c      IF(XR1.LT.0.0d0) CALL ERROR(I,N,-1)
      SQ1(I)=0.0d0
      IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
3     CONTINUE
      ENDDO
      DO I=1,N
5     B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
C
C ***********************************************************
C
C We shall condition on the values of X, XMI<X<XMA, but for some
C X values XIND will be zero leading to reduced accuracy. Hence we try
C to exclude them and narrow the interval [XMI,XMA]
C
c      PRINT *,'2:**  ',XMI,XMA
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
c      PRINT *,'2:****',XMI,XMA
      IF(FI(XMA)-FI(XMI).LT.EPSS)  THEN
c         print *, 'Leaving R2: Exit 4', XIND,XIND1,VDER1
         RETURN
      ENDIF
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
        DO  I=1,N
20       BU1(I)=BU(I)+XX1(J)*B1(I)
        ENDDO
      DER=DBUN+XX1(J)*DB1N
c      print *,'R2: before calling R1: (SQ1):',sq1(1),sq1(2),sq1(3)
      CALL RIND1(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
      III11=III11+1
10    XIND=XIND+XIND1*H1(J)
      ENDDO
c      print *, 'Leaving R2: Exit 5', XIND,XIND1,VDER1
      RETURN
      END SUBROUTINE RIND2

      SUBROUTINE RIND3(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J, II0,III, N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C     DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON/CHECK/III0,III1,III2,III3,III4
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
c      PRINT *,'Topp of 3:',sq(1),sq(2),sq(3)
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
         IF(XF.LT.XFF) THEN
            INF(3)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
 11   CONTINUE
      IF (XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
         SDER=0.0d0
         IF(VDER.GT.EPS) SDER=SQRT(VDER)
         XIND=MAX(DBUN,0.0d0)
         IF (IAC.LT.1) RETURN
         XIND=PMEAN(DBUN,SDER)
         RETURN
      END IF
      IF(ISQ.EQ.1) INF(3)=II0
      SQ0=SQ(INF(3))
      XMA=-BU(INF(3))/SQ0
      XMI=XMA+HH(INF(3))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
      III=0
      DO I=3,10
         III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(3),N)
      SS0=B1(INF(3))
      DB1N=DB(INF(3))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
      INFO(3)=INF(3)
      DB1N=DB1N/SQ0
      SQ1(INF(3))=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) GO TO 3
         XR1=R1(I+(I-1)*N)
         SQ1(I)=0.0d0
         IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
3     CONTINUE
      ENDDO
      DO I=1,N
 5    B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
c      PRINT *,'3:**  ',XMI,XMA
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
c      PRINT *,'3:****',XMI,XMA,EPSS
      IF (FI(XMA)-FI(XMI).LT.EPSS) RETURN
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
         DO I=1,N
20        BU1(I)=BU(I)+XX1(J)*B1(I)
         ENDDO
      DER=DBUN+XX1(J)*DB1N
c      print *,'R3: before calling R2: (SQ1):',sq1(1),sq1(2),sq1(3)
      CALL RIND2(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
      III21=III21+1
10    XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND3

      SUBROUTINE RIND4(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J,II0,III, N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON/CHECK/III0,III1,III2,III3,III4
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
c      PRINT *,'Topp of 4:',SQ(1),SQ(2),SQ(3)
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
         IF (XF.LT.XFF) THEN
            INF(4)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
 11   CONTINUE
      IF (XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
         SDER=0.0d0
         IF(VDER.GT.EPS) SDER=SQRT(VDER)
         XIND=MAX(DBUN,0.0d0)
         IF (IAC.LT.1) RETURN
         XIND=PMEAN(DBUN,SDER)
         RETURN
      END IF
      IF(ISQ.EQ.1) INF(4)=II0
      SQ0=SQ(INF(4))
      XMA=-BU(INF(4))/SQ0
      XMI=XMA+HH(INF(4))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
      III=0
      DO I=4,10
         III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(4),N)
      SS0=B1(INF(4))
      DB1N=DB(INF(4))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
      INFO(4)=INF(4)
      DB1N=DB1N/SQ0
      SQ1(INF(4))=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) GO TO 3
         XR1=R1(I+(I-1)*N)
         SQ1(I)=0.0d0
         IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
3     CONTINUE
      ENDDO
      DO I=1,N
5     B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
C      PRINT *,'**',XMI,XMA
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
C      PRINT *,INF(4),XMI,XMA
      IF(FI(XMA)-FI(XMI).LT.EPSS) RETURN
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO  J=1,N1
      DO  I=1,N
20    BU1(I)=BU(I)+XX1(J)*B1(I)
      ENDDO
      DER=DBUN+XX1(J)*DB1N
      CALL RIND3(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
      III31=III31+1
10    XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND4

      SUBROUTINE RIND5(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J,III,II0,N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C     DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
      
!      DIMENSION XX1(24),H1(24)
C      DIMENSION INF(10),INFO(10),HH(101)

C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON/CHECK/III0,III1,III2,III3,III4
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
         IF (XF.LT.XFF) THEN
            INF(5)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
 11   CONTINUE
      IF (XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
         SDER=0.d0
         IF(VDER.GT.EPS) SDER=SQRT(VDER)
         XIND=MAX(DBUN,0.0d0)
         IF (IAC.LT.1) RETURN
         XIND=PMEAN(DBUN,SDER)
         RETURN
      END IF
      IF(ISQ.EQ.1) INF(5)=II0
      SQ0=SQ(INF(5))
      XMA=-BU(INF(5))/SQ0
      XMI=XMA+HH(INF(5))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
      III=0
      DO I=5,10
         III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(5),N)
      SS0=B1(INF(5))
      DB1N=DB(INF(5))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
      INFO(5)=INF(5)
      DB1N=DB1N/SQ0
      SQ1(INF(5))=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) GO TO 3
         XR1=R1(I+(I-1)*N)
         SQ1(I)=0.0d0
         IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
3     CONTINUE
      ENDDO
      DO I=1,N
5     B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
      IF(FI(XMA)-FI(XMI).LT.EPSS) RETURN
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
       DO I=1,N
        BU1(I)=BU(I)+XX1(J)*B1(I)
       ENDDO
       DER=DBUN+XX1(J)*DB1N
       CALL RIND4(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
       III41=III41+1
       XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND5
C
      SUBROUTINE RIND6(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J,III,II0,N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
c      CSQ=C*SQ(I)
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) GO TO 1
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) SQ1(I)=EPS1
C      IF (BU(I).GT. CSQ.OR.BU(I).LT.HH(I)-CSQ) RETURN
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
c      IF(XF.GT.CEPSS) SQ1(I)=EPS1
         IF (XF.LT.XFF) THEN
            INF(6)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
11    CONTINUE
      IF(XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
      SDER=0.0d0
      IF (VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=MAX(DBUN,0.0d0)
      IF(IAC.LT.1) RETURN
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END IF
      IF(ISQ.EQ.1) INF(6)=II0
      SQ0=SQ(INF(6))
      XMA=-BU(INF(6))/SQ0
      XMI=XMA+HH(INF(6))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
C *************************************************
C
C  We are conditioning on the point T(INF(2)) and write new model
C  BU(I)+X*B1(I)+Delta1(I), I=1,...,N  (Obs. we do not use I=1,N)
C  SQ1(I) is standard deviation of Delta1 DBUN=BU'(N), DB1N=B1'(N) and X is
C  N(0,1) independent of Delta1, SDER1 is standard deviation of Delta1'(N).
C
      III=0
      DO I=6,10
      III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(6),N)
C      III1=III1+1
      SS0=B1(INF(6))
      INFO(6)=INF(6)
      DB1N=DB(INF(6))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
C      SQ0=SQRT(SS0)
      DB1N=DB1N/SQ0
      SQ1(INF(6))=0.0d0
      DO I=1,N
C
C IF (SQ1(I).EQ.EPS1) GO TO 3 - the .EQ. can not be raplaced with .LT. without
C some general changes in the strategy of SQ and SQ1 values. More exactly
C SQ can not be changed in this subroutine when for some  I  we would like to
C put SQ1(I)=EPS1 in the first loop. This SQ1 should not be changed here and
C thus we have GO TO 3 statement. Observe that the other SQ1 values are
C
      IF (SQ(I).LE.EPS) GO TO 3
C      IF (SQ1(I).EQ.EPS1.OR.SQ(I).LE.EPS1) GO TO 3
      XR1=R1(I+(I-1)*N)
c      IF(XR1.LT.0.0d0) CALL ERROR(I,N,-1)
      SQ1(I)=0.0d0
      IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
3     CONTINUE
      ENDDO
      DO I=1,N
5     B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
C
C ***********************************************************
C
C We shall condition on the values of X, XMI<X<XMA, but for some
C X values XIND will be zero leading to reduced accuracy. Hence we try
C to exclude them and narrow the interval [XMI,XMA]
C
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
      IF(FI(XMA)-FI(XMI).LT.EPSS)  THEN
         RETURN
      ENDIF
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
       DO I=1,N
20       BU1(I)=BU(I)+XX1(J)*B1(I)
       ENDDO
      DER=DBUN+XX1(J)*DB1N
      CALL RIND5(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
      III51=III51+1
10    XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND6

      SUBROUTINE RIND7(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J,III,II0,N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
c      CSQ=C*SQ(I)
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) GO TO 1
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) SQ1(I)=EPS1
C      IF (BU(I).GT. CSQ.OR.BU(I).LT.HH(I)-CSQ) RETURN
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
c      IF(XF.GT.CEPSS) SQ1(I)=EPS1
         IF (XF.LT.XFF) THEN
            INF(7)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
11    CONTINUE
      IF(XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
      SDER=0.0d0
      IF (VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=MAX(DBUN,0.0d0)
      IF(IAC.LT.1) RETURN
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END IF
      IF(ISQ.EQ.1) INF(7)=II0
      SQ0=SQ(INF(7))
      XMA=-BU(INF(7))/SQ0
      XMI=XMA+HH(INF(7))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
C *************************************************
C
C  We are conditioning on the point T(INF(2)) and write new model
C  BU(I)+X*B1(I)+Delta1(I), I=1,...,N  (Obs. we do not use I=1,N)
C  SQ1(I) is standard deviation of Delta1 DBUN=BU'(N), DB1N=B1'(N) and X is
C  N(0,1) independent of Delta1, SDER1 is standard deviation of Delta1'(N).
C
      III=0
      DO I=7,10
      III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(7),N)
C      III1=III1+1
      SS0=B1(INF(7))
      INFO(7)=INF(7)
      DB1N=DB(INF(7))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
C      SQ0=SQRT(SS0)
      DB1N=DB1N/SQ0
      SQ1(INF(7))=0.0d0
      DO I=1,N
C
C IF (SQ1(I).EQ.EPS1) GO TO 3 - the .EQ. can not be raplaced with .LT. without
C some general changes in the strategy of SQ and SQ1 values. More exactly
C SQ can not be changed in this subroutine when for some  I  we would like to
C put SQ1(I)=EPS1 in the first loop. This SQ1 should not be changed here and
C thus we have GO TO 3 statement. Observe that the other SQ1 values are
C
      IF (SQ(I).LE.EPS) GO TO 3
C      IF (SQ1(I).EQ.EPS1.OR.SQ(I).LE.EPS1) GO TO 3
      XR1=R1(I+(I-1)*N)
c      IF(XR1.LT.0.0d0) CALL ERROR(I,N,-1)
      SQ1(I)=0.0d0
      IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
3     CONTINUE
      ENDDO
      DO I=1,N
       B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
C
C ***********************************************************
C
C We shall condition on the values of X, XMI<X<XMA, but for some
C X values XIND will be zero leading to reduced accuracy. Hence we try
C to exclude them and narrow the interval [XMI,XMA]
C
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
      IF(FI(XMA)-FI(XMI).LT.EPSS)  THEN
         RETURN
      ENDIF
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
       DO I=1,N
        BU1(I)=BU(I)+XX1(J)*B1(I)
       ENDDO
       DER=DBUN+XX1(J)*DB1N
       CALL RIND6(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
       III61=III61+1
       XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND7

      SUBROUTINE RIND8(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
          IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J,III,II0,N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
c      CSQ=C*SQ(I)
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) GO TO 1
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) SQ1(I)=EPS1
C      IF (BU(I).GT. CSQ.OR.BU(I).LT.HH(I)-CSQ) RETURN
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
c      IF(XF.GT.CEPSS) SQ1(I)=EPS1
         IF (XF.LT.XFF) THEN
            INF(8)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
11     CONTINUE
      IF(XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
      SDER=0.0d0
      IF (VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=MAX(DBUN,0.0d0)
      IF(IAC.LT.1) RETURN
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END IF
      IF(ISQ.EQ.1) INF(8)=II0
      SQ0=SQ(INF(8))
      XMA=-BU(INF(8))/SQ0
      XMI=XMA+HH(INF(8))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
C *************************************************
C
C  We are conditioning on the point T(INF(2)) and write new model
C  BU(I)+X*B1(I)+Delta1(I), I=1,...,N  (Obs. we do not use I=1,N)
C  SQ1(I) is standard deviation of Delta1 DBUN=BU'(N), DB1N=B1'(N) and X is
C  N(0,1) independent of Delta1, SDER1 is standard deviation of Delta1'(N).
C
      III=0
      DO I=8,10
      III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(8),N)
C      III1=III1+1
      SS0=B1(INF(8))
      INFO(8)=INF(8)
      DB1N=DB(INF(8))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
C      SQ0=SQRT(SS0)
      DB1N=DB1N/SQ0
      SQ1(INF(8))=0.0d0
      DO I=1,N
C
C IF (SQ1(I).EQ.EPS1) GO TO 3 - the .EQ. can not be raplaced with .LT. without
C some general changes in the strategy of SQ and SQ1 values. More exactly
C SQ can not be changed in this subroutine when for some  I  we would like to
C put SQ1(I)=EPS1 in the first loop. This SQ1 should not be changed here and
C thus we have GO TO 3 statement. Observe that the other SQ1 values are
C
      IF (SQ(I).LE.EPS) GO TO 3
C      IF (SQ1(I).EQ.EPS1.OR.SQ(I).LE.EPS1) GO TO 3
      XR1=R1(I+(I-1)*N)
c      IF(XR1.LT.0.0d0) CALL ERROR(I,N,-1)
      SQ1(I)=0.0d0
      IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
3     CONTINUE
      ENDDO
      DO I=1,N
       B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
C
C ***********************************************************
C
C We shall condition on the values of X, XMI<X<XMA, but for some
C X values XIND will be zero leading to reduced accuracy. Hence we try
C to exclude them and narrow the interval [XMI,XMA]
C
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
      IF(FI(XMA)-FI(XMI).LT.EPSS)  THEN
         RETURN
      ENDIF
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
       DO I=1,N
        BU1(I)=BU(I)+XX1(J)*B1(I)
       ENDDO
       DER=DBUN+XX1(J)*DB1N
       CALL RIND7(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
       III71=III71+1
       XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND8

      SUBROUTINE RIND9(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J,III,II0,N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
      
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101
      
      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
c      CSQ=C*SQ(I)
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) GO TO 1
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) SQ1(I)=EPS1
C      IF (BU(I).GT. CSQ.OR.BU(I).LT.HH(I)-CSQ) RETURN
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
c      IF(XF.GT.CEPSS) SQ1(I)=EPS1
         IF (XF.LT.XFF) THEN
            INF(9)=I
            XFF=XF
         END IF
1        CONTINUE 
      ENDDO
11    CONTINUE
      IF(XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
      SDER=0.0d0
      IF (VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=MAX(DBUN,0.0d0)
      IF(IAC.LT.1) RETURN
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END IF
      IF(ISQ.EQ.1) INF(9)=II0
      SQ0=SQ(INF(9))
      XMA=-BU(INF(9))/SQ0
      XMI=XMA+HH(INF(9))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
C *************************************************
C
C  We are conditioning on the point T(INF(2)) and write new model
C  BU(I)+X*B1(I)+Delta1(I), I=1,...,N  (Obs. we do not use I=1,N)
C  SQ1(I) is standard deviation of Delta1 DBUN=BU'(N), DB1N=B1'(N) and X is
C  N(0,1) independent of Delta1, SDER1 is standard deviation of Delta1'(N).
C
      III=0
      DO I=9,10
       III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(9),N)
C      III1=III1+1
      SS0=B1(INF(9))
      INFO(9)=INF(9)
      DB1N=DB(INF(9))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
C      SQ0=SQRT(SS0)
      DB1N=DB1N/SQ0
      SQ1(INF(9))=0.0d0
      DO I=1,N
C
C IF (SQ1(I).EQ.EPS1) GO TO 3 - the .EQ. can not be raplaced with .LT. without
C some general changes in the strategy of SQ and SQ1 values. More exactly
C SQ can not be changed in this subroutine when for some  I  we would like to
C put SQ1(I)=EPS1 in the first loop. This SQ1 should not be changed here and
C thus we have GO TO 3 statement. Observe that the other SQ1 values are
C
       IF (SQ(I)>EPS) THEN
        XR1=R1(I+(I-1)*N)
c      IF(XR1.LT.0.0d0) CALL ERROR(I,N,-1)
        SQ1(I)=0.0d0
        IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
       ENDIF
      ENDDO
      DO I=1,N
       B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
C
C ***********************************************************
C
C We shall condition on the values of X, XMI<X<XMA, but for some
C X values XIND will be zero leading to reduced accuracy. Hence we try
C to exclude them and narrow the interval [XMI,XMA]
C
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
      IF(FI(XMA)-FI(XMI).LT.EPSS)  THEN
         RETURN
      ENDIF
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
       DO I=1,N
        BU1(I)=BU(I)+XX1(J)*B1(I)
       ENDDO
       DER=DBUN+XX1(J)*DB1N
       CALL RIND8(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
       III81=III81+1
       XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND9

      SUBROUTINE RIND10(XIND,R,BU,DBUN,DB,SQ,VDER,N)
      USE SIZEMOD
      USE TBRMOD
      USE INFCMOD
      USE CHECKMOD
      USE EPSMOD
      USE RINTMOD
      IMPLICIT NONE
      REAL*8, intent(inout) :: XIND,DBUN,VDER
      REAL*8, DIMENSION(RDIM), intent(inout) :: R
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,DB,SQ
      INTEGER, intent(in) :: N
      REAL*8, DIMENSION(RDIM), save :: R1
      REAL*8, DIMENSION(NMAX), save :: BU1,DB1,B1,SQ1
      REAL*8, DIMENSION(24) :: XX1, H1
      REAL*8, save :: DB1N,SDER1,VDER1, SS0
      REAL*8 XMI,XMA,DER,SDER
      REAL*8 XIND1,XFF,XF,X,XH, XR1, SQ0 
      INTEGER I,III,J,II0,N1

!     INTEGER IAC,N
!     INTEGER I,J,III,II0,N1
!     REAL*8 XIND,R,BU,DBUN,DB,SQ,VDER
!      real*8 XX1,H1
!     REAL*8 R1,B1,DB1,BU1,SQ1
!     REAL*8 XFF, SQ0, X, XH,XF, SDER, XMI,XMA
!     REAL*8 SS0, DB1N, VDER1, XR1
!     REAL*8 SDER1, DER,XIND1
C      DIMENSION R(1),BU(1),SQ(1),DB(1)
!     DIMENSION R(RDIM),BU(NMAX),SQ(NMAX),DB(NMAX)
!      DIMENSION R1(RDIM),B1(NMAX),DB1(NMAX),BU1(NMAX),SQ1(NMAX)
!      DIMENSION XX1(24),H1(24)
C      DIMENSION HH(101),INF(10),INFO(10)
     
C      COMMON /EPS/    EPS,EPSS,CEPSS
C      COMMON /RINT/   C,FC
C      COMMON /TBR/    HH
C      COMMON /INFC/   ISQ,INF,INFO
C      COMMON /CHECK1/ III01,III11,III21,III31,III41,III51
C     *,III61,III71,III81,III91,III101

      XIND=0.0d0
      XFF=1.0d0
      IF (N.LT.1) GO TO 11
      SQ0=0.0d0
      DO I=1,N
         IF (SQ(I).LE.EPS) THEN
            SQ1(I)=SQ(I)
            GO TO 1
         END IF
c      CSQ=C*SQ(I)
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) GO TO 1
c      IF (BU(I).LT.-CSQ.AND.BU(I).GT.HH(I)+CSQ) SQ1(I)=EPS1
C      IF (BU(I).GT. CSQ.OR.BU(I).LT.HH(I)-CSQ) RETURN
         IF (SQ(I).GT.SQ0) THEN
            SQ0=SQ(I)
            II0=I
         END IF
         X=-BU(I)/SQ(I)
         XH=X+HH(I)/SQ(I)
         XF=FI(X)-FI(XH)
c      IF(XF.GT.CEPSS) SQ1(I)=EPS1
         IF (XF.LT.XFF) THEN
            INF(10)=I
            XFF=XF
         END IF
 1    CONTINUE
      ENDDO
11    CONTINUE
      IF(XFF.LT.EPSS) RETURN
      IF (XFF.GT.0.9999d0*FC) THEN
      SDER=0.0d0
      IF (VDER.GT.EPS) SDER=SQRT(VDER)
      XIND=MAX(DBUN,0.0d0)
      IF(IAC.LT.1) RETURN
      XIND=PMEAN(DBUN,SDER)
      RETURN
      END IF
      IF(ISQ.EQ.1) INF(10)=II0
      SQ0=SQ(INF(10))
      XMA=-BU(INF(10))/SQ0
      XMI=XMA+HH(INF(10))/SQ0
      XMI=MAX(-C,XMI)
      XMA=MIN(C,XMA)
      IF (XMI.GT.XMA) XMA=-C
C *************************************************
C
C  We are conditioning on the point T(INF(2)) and write new model
C  BU(I)+X*B1(I)+Delta1(I), I=1,...,N  (Obs. we do not use I=1,N)
C  SQ1(I) is standard deviation of Delta1 DBUN=BU'(N), DB1N=B1'(N) and X is
C  N(0,1) independent of Delta1, SDER1 is standard deviation of Delta1'(N).
C
      III=0
      DO I=10,10
      III=III + ABS(INF(I)-INFO(I))
      ENDDO
      IF (III.EQ.0) GO TO 99
      CALL M_COND(R1,B1,DB1,R,DB,INF(10),N)
C      III1=III1+1
      SS0=B1(INF(10))
      INFO(10)=INF(10)
      DB1N=DB(INF(10))
      VDER1=VDER-DB1N*DB1N/SS0
      SDER1=0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
C      SQ0=SQRT(SS0)
      DB1N=DB1N/SQ0
      SQ1(INF(10))=0.0d0
      DO I=1,N
C
C IF (SQ1(I).EQ.EPS1) GO TO 3 - the .EQ. can not be raplaced with .LT. without
C some general changes in the strategy of SQ and SQ1 values. More exactly
C SQ can not be changed in this subroutine when for some  I  we would like to
C put SQ1(I)=EPS1 in the first loop. This SQ1 should not be changed here and
C thus we have GO TO 3 statement. Observe that the other SQ1 values are
cc
       IF (SQ(I)>EPS) THEN

        XR1=R1(I+(I-1)*N)
c      IF(XR1.LT.0.0d0) CALL ERROR(I,N,-1)
        SQ1(I)=0.0d0
        IF (XR1.GT.EPS) SQ1(I)=SQRT(XR1)
       ENDIF
      ENDDO
      DO I=1,N
       B1(I)=B1(I)/SQ0
      ENDDO
99    CONTINUE
C
C ***********************************************************
C
C We shall condition on the values of X, XMI<X<XMA, but for some
C X values XIND will be zero leading to reduced accuracy. Hence we try
C to exclude them and narrow the interval [XMI,XMA]
C
      CALL C1_C2(XMI,XMA,BU,B1,DBUN,DB1N,SDER1,SQ1,N)
      IF(FI(XMA)-FI(XMI).LT.EPSS)  THEN
        RETURN
      ENDIF
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO J=1,N1
       DO I=1,N
        BU1(I)=BU(I)+XX1(J)*B1(I)
       ENDDO
       DER=DBUN+XX1(J)*DB1N
       CALL RIND9(XIND1,R1,BU1,DER,DB1,SQ1,VDER1,N)
       III91=III91+1
       XIND=XIND+XIND1*H1(J)
      ENDDO
      RETURN
      END SUBROUTINE RIND10
      
      SUBROUTINE C1_C2(C1,C2,BU,B1,DBUN,DB1N,SDER,SQ,N)
C
C  We assume that the process  y  is of form y(I)=BU(I)+X*B1(I)+Delta(I),
C  I=1,...,N, SQ(I) is standard deviation of Delta(I), where X is  N(0,1)
C  independent of Delta.  Let Y = DBUN + DB1N*X + Z, where Z is zero-mean
C  Gaussian with standart independent of X (it can depend on Delta(I)) with
C  standart deviation SDER. Since we are truncating all Gaussian  variables to
C  the interval [-C,C], then if for any I
C
C  a) BU(I)+x*B1(I)-C*SQ(I)>0  or
C
C  b) BU(I)+x*B1(I)+C*SQ(I)<HH  then
C
C  XIND|X=x = E[Y^+1{ HH<y(I)<0 for all I, I=1,...,N}|X=x] = 0 !!!!!!!!!
C
C  Further, see discussion in comments to the subroutine PMEAN, by first upper-
C  bounding the indicator 1{} in XIND by 1, XIND|X=x = 0 if
C
C  c) DBUN+x*DB1N+4.5*SDER<0.0d0
C
C  Consequently, for increasing the accuracy (by excluding possible discon-
C  tinuouities) we shall exclude such X=x values for which XIND|X=x = 0.0d0
C  XIND=E([XIND|X]). Hence we assume that if C1<X<C2 any of the previous
C  conditions are satisfied.
C
C  OBSERVE!!, C1, C2 has to be set to upper bounds of possible values, e.g.
C  C1=-C, C2=C before calling C1_C2 subroutine.
C

C NOTE: Check that integration limits contained in TBRMOD is used correctly
C       If order of variables are changed then also the integration limits in HH should reflect this
      USE EPSMOD
      USE RINTMOD
      USE TBRMOD
      USE SIZEMOD
      IMPLICIT NONE
      REAL*8, DIMENSION(NMAX), intent(inout) :: BU,SQ,B1
      REAL*8, intent(inout) :: C1,C2
      REAL*8, intent(in) :: DBUN,DB1N,SDER
      INTEGER, intent(in) :: N
      REAL*8 CSQ, HHB, CC1, CC2, X
      INTEGER I   
C      DIMENSION BU(1),B1(1),SQ(1)
!     DIMENSION BU(NMAX),SQ(NMAX),B1(NMAX)
C      DIMENSION HH(101)
C      COMMON /EPS/EPS,EPSS,CEPSS
C      COMMON/RINT/C,FC
C      COMMON/TBR/HH
      DO I=1,N
      CSQ=C*SQ(I)
      HHB=HH(I)-BU(I)
C
C  If ABS(B1(I)) < EPS we can have overflow and hence we consider two cases
C  1) BU(I) is so large or small so we can surely assume that the probability
C     of staying between the barriers is 0, consequently C1=C2=0
C  2) we do not change the original limits.
C
      IF (ABS(B1(I)).LT. EPS) THEN
         IF (BU(I).GT.CSQ.OR.BU(I).LT.HH(I)-CSQ) THEN
            C1=0.0d0
            C2=0.0d0
            RETURN
         END IF
C
C  In other cases this part follows from the description of the problem.
C
       ELSE
         IF (B1(I).GT.EPS) THEN
            CC1=(HHB-CSQ)/B1(I)
            CC2=(-BU(I)+CSQ)/B1(I)
            IF (C1.LT.CC1) C1=CC1
            IF (C2.GT.CC2) C2=CC2
          ELSE
            CC2=(HHB-CSQ)/B1(I)
            CC1=(-BU(I)+CSQ)/B1(I)
            IF (C1.LT.CC1) C1=CC1
            IF (C2.GT.CC2) C2=CC2
          END IF
      END IF
      ENDDO
      X=-DBUN-4.5d0*SDER
      IF(DB1N.GT.EPS.AND.C1.LT.X/DB1N) C1=X/DB1N
      IF(DB1N.LT.-EPS.AND.C2.GT.X/DB1N) C2=X/DB1N
      if(abs(db1n).lt.eps.and.x.gt.0.0d0) then
            C1=0.0d0
            C2=0.0d0
            RETURN
         END IF

c
c  In the following three rows we are cutting C1, C2 to the interval [-C,C].
c  Obs. all tree lines are neccessary.
c
      C1=MAX(-C,C1)
      C2=MIN( C,C2)
      IF (C1.GT.C2) C2=-C
C      PRINT *,2,C1,C2
      RETURN
      END SUBROUTINE C1_C2

      REAL*8 FUNCTION GAUSINT(X1,X2,A,B,C,D)
C
C Let  X  be stardized Gaussian variable, i.e. X=N(0,1).
C The function calculate the followin integral E[I(X1<X<X2)(A+BX)(C+DX)],
C where I(X1<X<X2) is an indicator function of the set {X1<X<X2}.
C
      IMPLICIT NONE
      REAL*8, intent(in) :: X1,X2,A,B,C,D
      REAL*8 Y1,Y2,Y3
      REAL*8, PARAMETER:: SP = 0.398942280401433d0 
      IF(X1.GE.X2) THEN
      GAUSINT=0.0d0
      RETURN
      END IF
      Y1=(A*D+B*C+X1*B*D)*EXP(-0.5d0*X1*X1)
      Y2=(A*D+B*C+X2*B*D)*EXP(-0.5d0*X2*X2)
      Y3=(A*C+B*D)*(FI(X2)-FI(X1))
      GAUSINT=Y3+SP*(Y1-Y2)
      RETURN
      END FUNCTION GAUSINT


      SUBROUTINE GAUSS1(N,H1,XX1,XMI,XMA,EPS0) 
      USE CHECKMOD
      USE QUADRMOD
      IMPLICIT NONE
      INTEGER, intent(out) :: N
      REAL*8, DIMENSION(24), intent(out) :: H1,XX1
      REAL*8, intent(in) :: XMI,XMA,EPS0
      REAL*8, DIMENSION(24) ::  Z1
      REAL*8 SDOT, SDOT1, DIFF1
      INTEGER NNN, J, I1
!     DIMENSION Z1(24),XX1(1),H1(1)
C      DIMENSION Z(126),H(126)
C      DIMENSION NN(25)
C      COMMON/QUADR/ Z,H,NN,NNW
C      COMMON/CHECKQ/ III0
      REAL*8, parameter::  SP= 0.398942280401433d0
      IF (XMA.LT.XMI) THEN
      PRINT *,'Error XMIN>XMAX in GAUSS1 - stop!'
C     STOP
      END IF
      NNN=0
      DO I=1,NNW
      N=NN(I)
      DO J=1,N
       XX1(J)=0.5d0*(Z(NNN+J)*(XMA-XMI)+XMA+XMI)
       Z1(J)=XX1(J)*XX1(J)
       H1(J)=0.5d0*SP*(XMA-XMI)*H(NNN+J)*EXP(-0.5d0*Z1(J))
      ENDDO
      NNN=NNN+N
      SDOT=GAUSINT(XMI,XMA,0.0d0,1.0d0,0.0d0,1.0d0)
      SDOT1=0.d0
      DO I1=1,N
        SDOT1=SDOT1+Z1(I1)*H1(I1)
      ENDDO
      DIFF1=ABS(SDOT-SDOT1)
      IF(EPS0.LT.DIFF1) GO TO 10
      III0=III0+N
C      PRINT *,'N. of nodes',III0
      RETURN
10    CONTINUE
      ENDDO
      END SUBROUTINE GAUSS1

      
      SUBROUTINE M_COND(Syy_cd,Syyii,Syx_cd,Syy,Syx,ii,N)
C
C               INPUT:
C
C   ii     IS THE INDEX OF THE TIME ON WHICH WE ARE CONDITIONING.
C   N      number of variables in covariance matrix  Syy
C
C   Covariance matrix  Syy(I+(J-1)*N)=Cov(Yi,Yj) (is unchanged)
C   Covariance vector  Syx(I)=Cov(Yi,X) (is unchanged)
C
C              OUTPUT:
C
C   Covariance matrix Syy_cd(I+(J-1)*N)=Cov(Xi,Xj|Xii)
C   Covariance vector Syyii(I)=Cov(Xi,Xii)
C   Covariance vector Syx_cd(I)=Cov(Xi,Y|Xii)
C   Variance          Q1=Var(Xii)=Syyii(ii)
c   Obs. If   Q1<EPS there is no conditioning
C
C M_COND(R1,B1,DB1,R,DB,INF(10),N)
      USE EPSMOD
      USE SIZEMOD
      IMPLICIT NONE
      INTEGER, intent(in) :: II,N
      REAL*8, DIMENSION(RDIM), intent(inout) :: Syy_cd,Syy
      REAL*8, DIMENSION(NMAX), intent(inout)  :: Syyii,Syx_cd,Syx
      REAL*8 Q1
      INTEGER I,J
!     DIMENSION Syy_cd(RDIM),Syyii(NMAX),Syx_cd(NMAX),Syy(RDIM),Syx(NMAX)
C      DIMENSION Syy_cd(1),Syyii(1),Syx_cd(1),Syy(1),Syx(1)
C      COMMON /EPS/    EPS,EPSS,CEPSS
      IF (II.LE.0.OR.II.GT.N) THEN
       PRINT *,'The conditioning time in M_COND is out of range, stop!'
       STOP
      END IF
C
C   Q1=Var(Xii)=Syyii(ii)
C
      Q1=Syy(II+(II-1)*N)
      IF(Q1.LE.eps) then
      DO I=1,N
       Syyii(I)=0.0d0
      ENDDO
      Q1=1.0d0
      else
      DO I=1,N
       Syyii(I)=Syy(I+(II-1)*N)
      ENDDO
      end if
      DO I=1,N
       DO J=1,N
        Syy_cd(I+(J-1)*N)=Syy(I+(J-1)*N)-Syy(II+(J-1)*N)*Syyii(I)/Q1
       ENDDO
       Syx_cd(I)=Syx(I)-Syx(II)*Syyii(I)/Q1
      ENDDO
      RETURN
      END SUBROUTINE M_COND
      


      REAL*8 FUNCTION PMEAN(XX,SS)
C
C   PMEAN is the positive mean of a Gaussian variable with mean  XX  and
C   standart deviation  SS,  i.e.  PMEAN=SS*FIFUNK(XX/SS), where
C   FIFUNK(x)=f(x)+x*FI(x), f  and  FI  are density and distribution
C   functions of  N(0,1)  variable, respectively. We have modified the
C   algorithm 209 from CACAM for evaluation of  FIFUNK, to avoid the operations
C   of type  SS*XX/SS, which can give numerical errors when  SS  and  XX  are
C   both small.
C
C                     **   NUMERICAL ACCURACY  **
C
C
C  Obs. that, our general assumption is that the process is normalized, i.e.
C  Var (X(t)) = Var (X'(t)) = 1.0d0  Consequently all conditional variances
C  are less than  1, and usualy close to zero, i.e.  SS<1.0d0. Now since  SS<1.0d0
C  and  FIFUNK(x)<0.0000001  for  x<-4.5  we have defined  FIFUNK(x)=0.0d0
C  if  x<-4.5  and  FIFUNK(x)=x  if  x>
C  4.5.  Under we have a table with
C  exact values of  FIFUNK.
C
C    x       FIFUNK(x)
C
C   -5.0    0.00000005233
C   -4.5    0.00000069515
C   -4.0    0.00000711075
C    4.0    4.00000700000
C    4.5    4.50000100000
C
C Obviously the tresholds  -4.5  and  4.5  can be increased.
C
C
      IMPLICIT NONE
      REAL*8, intent(in) :: XX, SS
      REAL*8 X,Y,W,Z
      REAL*8, parameter :: SP = 0.398942280401433d0
      IF(XX.LT.4.5d0*SS) GO TO 1
      PMEAN=XX
      RETURN
1     IF(XX.GT.-4.5d0*SS) GO TO 3
      PMEAN=0.0d0
      RETURN
3     continue
      if (SS .LT. 0.0000001d0) then
      PMEAN=0.0d0
      RETURN
      end if

      X=XX/SS
      
      IF(X==0) goto 8
      Y=0.5d0*ABS(X)
      IF(Y<1.0d0) then
       W=Y*Y
       Z=((((((((0.000124818987d0*W-0.001075204047d0)*W
     1   +0.005198775019d0)*W-0.019198292d0)*W+0.05905403564d0)*W 
     2   -0.15196875136d0)*W+0.3191529327d0)*W-0.5319230073d0)*W
     3   +0.7978845606d0)*Y*2.0d0
      else
       Y=Y-2.0d0
       Z=(((((((((((((-0.000045255659d0*Y+0.00015252929d0)*Y
     *  -0.000019538132d0)*Y-0.000676904986d0)*Y
     1 +0.001390604284d0)*Y-0.000794620820d0)*Y
     2 -0.002034254874d0)*Y+0.006549791214d0)*Y-0.010557625006d0)*Y+
     3 0.011630447319d0)*Y-0.009279453341d0)*Y+0.005353579108d0)*Y-
     4 0.002141268741d0)*Y+0.000535310849d0)*Y+0.9999366575d0
      endif
      IF(X.GT.0.0d0) PMEAN=SS*SP*EXP(-0.5d0*X*X)+XX*0.5d0*(Z+1.0d0)
      IF(X.LT.0.0d0) PMEAN=SS*SP*EXP(-0.5d0*X*X)+XX*0.5d0*(1.0d0-Z)
      RETURN
8     PMEAN=SS*SP
      RETURN
      END FUNCTION PMEAN


      REAL*8 FUNCTION FI(XX)
C
C   Algorithm 209 from CACAM.
C   FI(xx)  is a distribution functions of  N(0,1)  variable.
C
      IMPLICIT NONE
      REAL*8, intent(in) :: XX
      REAL*8 X, Y,Z, W
      X=XX
      IF(X==0) then
        FI=0.5d0
        RETURN
      endif
      Y=0.5d0*ABS(X)
      IF(Y>3.0d0) then
       IF(X.GT.0.0d0) FI=1.0d0
       IF(X.LT.0.0d0) FI=0.0d0
       RETURN
      endif
      IF (Y<1.0d0) then
       W=Y*Y
       Z=((((((((0.000124818987d0*W-0.001075204047d0)*W
     1   +0.005198775019d0)*W-0.019198292d0)*W+0.05905403564d0)*W 
     2   -0.15196875136d0)*W+0.3191529327d0)*W-0.5319230073d0)*W
     3   +0.7978845606d0)*Y*2.0d0
      ELSE
       Y=Y-2.0d0
       Z=(((((((((((((-0.000045255659d0*Y+0.00015252929d0)*Y
     1  -0.000019538132d0)*Y-0.000676904986d0)*Y+0.001390604284d0)*Y
     2  -0.000794620820d0)*Y-0.002034254874d0)*Y+0.006549791214d0)*Y
     3  -0.010557625006d0)*Y+0.011630447319d0)*Y-0.009279453341d0)*Y
     4  +0.005353579108d0)*Y-0.002141268741d0)*Y+0.000535310849d0)*Y
     5  +0.9999366575d0
      endif
100   IF(X.GT.0.0d0) FI=0.5d0*(Z+1.0d0)
      IF(X.LT.0.0d0) FI=0.5d0*(1.0d0-Z)
      RETURN
      END FUNCTION FI

C     Version  1991-XII-14

C  The MREG program.
C
C
C  We consider a process X(I)=X(T(I)) at the grid of  N  points  T(1),...,T(N),
C
C          X(I) = -A(I) + Z*A(I+N) + Sum Xj*A(I+(j+1)*N) + Delta(I),
C
C  the sum disapears if M=1, j=1,...,M-1. We assume that Z,Xj are independend
C  standart Rayleigh, Gaussian distributed rv. and independent of the zero
C  mean Gaussian residual process,  with covariance structure given in  R,
C
C          R(i+(j-1)N) = Cov (Delta(T(i)), Delta(T(j))).
C
C  Additionally we have a zero mean Gaussian variable XN, 
C  independent of Z,Xj with  covariance structure defined by 
C  B(i)= Cov (Delta(T(i)),XN), i=1,...,N,  B(N+1)=Var(XN). 
C  Furthermore XN and Z,Xj satisfies the following equation system
C
C     (BB + (XN,0,...,0)^T =  AA*(Z,X1,...,Xm-1)^T       (***)
C
C  where AA is (M,M) matrix, BB is M-vector. We rewrite this equation, by
C  introducing a variable X_M=XN/SQRT(XN) and construct   new matrix AA1
c  by adding the column (SQRT(Var(XN)),0,...,0) and the row with only zeros.
C  The equations (***) writtes
C
C     (BB,0)^T  =  AA1*(Z,X1,...,Xm-1,Xm)^T       (****)
C
C  where AA1 is (M+1,M+1) matrix, We assume that the rank of AA1 is M,
C  otherwise the density is singular and we give a output F=0.CC
C
C  Let Y0 be a zero-mean Gaussian variable independent of Z,Xj 
C  with  covariance structure defined by 
C  DB(i)= Cov (Delta(T(i)),Y0), i=1,...,N,  DB(N+1)=Cov(XN,Y0), Var(Y0)=VDER. 
C  Let Y be defined by
C
C     Y=-DA(1) + Z*DA(2) + Sum Xj*DA(2+j) +Y0,
C
C  j=1,...,M-1. The program computes:
C
C  F = E[ Y^+ *1{ HH<X(I)<0 for all I, I=1,...,N}|Z,X1,...,X_M-1 solves (***)]
C      *f_{Z,X1,....,XM-1}(***).
C
C  In the simplest case NIT=0 we define (Delta(1),...,Delta(N),XN)=0.0d0 
C
C  We renormalize  vectors AN and DA, the covariance fkn R, DB
C  and VDER. Then by renormalization we choose the Gaussian variable X such
C  that F is written in the form
C
C  F = E[(D0(1)+X*D0(2)+Y1)^+*(PC+X*PD)^+*1{HH <A0(I)+X*B0(I)+Delta1(I)<0}]
C
C  Observe, PC+X*PD>0 defines integration region for X.
C  In the simplest case NIT=0 we define (Delta(1),...,Delta(N),Y1)=0.0d0 
C  For NIT=1 only (Delta(1),...,Delta(N))=0, i.e. we have to compute
C  a one dimensional integral. Finally by conditioning on X the problem is
C  put in the format of RIND-problem.
C
C   INF indicates whether one
C  has already called the subroutine before and ONLY! inputs BB, DA or A
C  was changed.
C
C  Observe the limitations are :  N<=100, 0<M <= 5 = MMAX.
C
C     revised pab 2007
C     - replaced all common blocks with modules
C     - fixed some bugs 

      SUBROUTINE MREG(F,R,B,DB,AA,BB,A,DA,VDER,M,N,NIT,INFR)
      USE SIZEMOD
      USE EPSMOD
      USE RINTMOD 
      USE INFCMOD
      USE SVD
      IMPLICIT NONE
!     INTEGER, PARAMETER :: MMAX = 5, NMAX = 101, RDIM = 10201
      INTEGER, PARAMETER :: Nw = MMAX+1
      INTEGER, intent(in) :: M,N,NIT,INFR
      REAL*8, intent(in) :: VDER
      REAL*8, intent(inout) :: F
      REAL*8, intent(inout) :: R(RDIM),B(NMAX),DB(NMAX),BB(Nw)
      REAL*8, intent(inout) :: A(Nw*NMAX),DA(Nw),AA(MMAX-2,MMAX-2)
      REAL*8, DIMENSION(Nw),    save :: AO, DA0, W1, E1
      REAL*8, DIMENSION(Nw,Nw), save :: AA1, U1,V1 
      REAL*8, DIMENSION(NMAX),  save :: DB1, SQ
      REAL*8, DIMENSION(RDIM),  save :: R1
      REAL*8, DIMENSION(Nw*NMAX), save :: A1
      REAL*8, DIMENSION(NMAX) :: A0,B0,B1
      REAL*8, DIMENSION(24) :: XX1,H1
      REAL*8, DIMENSION(2)  :: D0
      REAL*8, save :: QD, DB1N,VDER1,SQD, SDER1,DET1
      REAL*8  XIND, DB0N,FR1, XR1,XMI,XMA
      REAL*8  CC, PC, PD, P, X
      INTEGER INF1, I, I1, J, N1,I2,infoID
      INTEGER, save :: IDET, NNIT         
C     DIMENSION AA1(6,6),V1(6,6)
C     DIMENSION W1(6),AO(6),DA0(6)
C      DIMENSION A0(2*Nmax),B0(2*Nmax),B1(2*Nmax),DB1(2*Nmax)
!      DIMENSION D0(2)
!      DIMENSION XX1(24),H1(24)
C      COMMON /EPS/EPS,EPSS,CEPSS
C      COMMON/RINT/ C,FC
C
C  If INFR=0 we have to initiate conditioning and renormalization transf.
C
      INF1=0
      F=0.0d0
      IF (N.LT.0) RETURN
      IF (N.EQ.0) GO TO 2
      DO I=1,N
       DO I1=1,M+1
        A1(I+(I1-1)*N)=A(I+(I1-1)*N)
       ENDDO
      ENDDO
2     CONTINUE
      DO I=1,M+1
      DA0(I)=DA(I)
      ENDDO

C
C  Renormalization
C
      IF (INFR.EQ.1) GO TO 105
      NNIT=MIN(NIT,N)

      DO i=1,n
         SQ(i)=0.0d0
      ENDDO
      
      DO I=1,M
      DO J=1,M
         AA1(I,J)=AA(I,J)
      ENDDO
      ENDDO
      NNIT=MIN(NIT,N)
      
      QD   =B(N+1)
      DB1N =DB(N+1)
      VDER1=VDER

      IF(QD.le.eps) then

         DB1N = 0.0d0
         SQD  = 0.0d0
         NNIT = 0
         DO  I=1,N
            DB1(I) = DB(I)
            A1(I+(M+1)*N)=0.0d0
            SQ(I) = 0.0d0
         ENDDO

      else
           SQD   = SQRT(QD)            
           VDER1 = VDER1-DB1N*DB1N/QD
           DB1N  = DB1N/SQD
           DO I=1,N
              DB1(I) = DB(I)-DB(N+1)*(B(I)/QD)
              A1(I+(M+1)*N) = B(I)/SQD
          ENDDO
      end if

      SDER1 = 0.0d0
      IF (VDER1.GT.EPS) SDER1=SQRT(VDER1)
C      print *,'sqd,SDER1',sqd,SDER1     
C PAB: BUG DA0 M+2 can be larger than NW = MMAX+1
      DA0(M+2) = DB1N
      BB(M+1) = 0.0d0

      
      AA1(1,M+1)=SQD
      DO I=1,M
       AA1(I+1,M+1)=0.0d0
       AA1(M+1,I)=0.0d0
      ENDDO
! New call to avoid calling Numerical recipess SVDCMP
      CALL dsvdc(AA1,M+1,M+1,W1, E1, U1, V1, 11, infoID)
!      CALL SVDCMP(AA1,M+1,M+1,NW,NW,W1,V1)


      DET1 = 1.0d0
      idet = 0
      DO I=1,M+1
       IF ( W1(I).LT.0.00001d0 ) THEN
          idet  = idet+1
          W1(I) = 0.0d0
         DO J=1,M+1
            AO(J)=V1(J,I)
         ENDDO
        GO TO 35
        END IF
        DET1=DET1*W1(I)
35    CONTINUE
      ENDDO
C      print *,'det1',det1

      IF(DET1.LT.0.001d0) NNIT=0
c
c    Obs. QD can not be small since NNIT>0
c      
      IF (NNIT.GT.1) THEN
      DO I=1,N
         XR1=R(I+(I-1)*N)-B(I)*(B(I)/QD) 
         IF(XR1.GT.EPS)  THEN
              SQ(I)=SQRT(XR1)
          ENDIF
      ENDDO
           
      DO I=1,N
       DO J=1,N
         R1(J+(I-1)*N)=R(J+(I-1)*N)-B(I)*(B(J)/QD)
       ENDDO
      ENDDO
      
      
      END IF


105   CONTINUE
      if (idet.gt.1) return
C
C  Renormalization is done
C
      CALL R_ORT(CC,PC,PD,U1,V1,W1,AO,BB,A1,A0,B0,DA0,D0,DET1,M+1,N)
      IF(CC.LT.0.0d0) RETURN
      XMI=-C
      XMA= C
      IF(ABS(PD).LE.EPS.AND.PC.LT.0.0d0) RETURN
      IF(ABS(PD).LE.EPS) GO TO 102
      X=-PC/PD
      IF(PD.GT.0.0d0.AND.XMI.LT.X) XMI=X
      IF(PD.LT.0.0d0.AND.XMA.GT.X) XMA=X
102   CONTINUE
c      PRINT *,'XMI,XMA',XMI,XMA
      IF(NNIT.eq.1.AND.IAC.LT.1.OR.NNIT.eq.0.OR.XMI.GE.XMA) THEN
      CALL C1_C2(XMI,XMA,A0,B0,D0(1),D0(2),0.0d0,SQ,N)
c         PRINT *,'XMI,XMA',XMI,XMA
         F=GAUSINT(XMI,XMA,D0(1),D0(2),PC,PD)*CC
c         print *,'return',f,cc
         RETURN
      END IF
C
C ***********************************************************
C
C We shall condition on the values of X, XMI<X<XMA, but for some
C X values XIND will be zero leading to reduced accuracy. Hence we try
C to exclude them and narrow the interval [XMI,XMA]
c
c      PRINT *,XMI,XMA

      CALL C1_C2(XMI,XMA,A0,B0,D0(1),D0(2),SDER1,SQ,N)
c      PRINT *,XMI,XMA
      IF(FI(XMA)-FI(XMI).LT.EPSS) RETURN
      CALL GAUSS1(N1,H1,XX1,XMI,XMA,EPS0)
      DO I2=1,N1
       FR1=CC*H1(I2)*(PC+XX1(I2)*PD)
        DO I=1,N
         B1(I)=A0(I)+XX1(I2)*B0(I)
        ENDDO
       DB0N=D0(1)+XX1(I2)*D0(2)

C
C  INF1=1 means that both R1 and SQ are the same as in the previous
C  call of TWOREG subroutine, INF=0 indicates the new R and SQ.
c      
c      print *,'go in rind'
       CALL RIND(XIND,R1,B1,DB0N,DB1,SQ,VDER1,NNIT-1,N,INF1)
c      if (n.gt.29) print *,XIND,DB0N,VDER1,b1(N),b1(n-1)
       INF1=1
       F=F+FR1*XIND
      ENDDO
      RETURN
      END SUBROUTINE MREG



      SUBROUTINE R_ORT(C,PC,PD,U1,V1,W1,AO,BB,A0,A,B,DA,D0,DET,M,N)
      USE EXPACCMOD
      USE SIZEMOD
      IMPLICIT NONE
      INTEGER, PARAMETER :: Nw = MMAX+1
      INTEGER, intent(in) :: M,N
      REAL*8, DIMENSION(Nw,Nw), intent(inout)  :: U1, V1
      REAL*8, DIMENSION(Nw   ), intent(inout)  :: W1,AO,BB, DA
      REAL*8, DIMENSION(Nw*NMAX), intent(inout):: A0
      REAL*8, DIMENSION(NMAX), intent(inout) :: A, B
      REAL*8, DIMENSION(2),    intent(inout) :: D0
      REAL*8, intent(inout):: C,PC,PD,DET
      REAL*8, DIMENSION(Nw   ) :: XO
      REAL*8  DER0,DER1,P
      INTEGER I,J
      
      
C      COMMON/EXPACC/PMAX
      REAL*8, parameter :: SP = 0.398942280401433d0
      C=-999.
      CALL SVBKSB(U1,W1,V1,M,M,Nw,Nw,BB,XO)
      P    = 0.0d0
      DER0 = -DA(1)
      DER1 = 0.0d0
      DO I=1,M
      P    = P + XO(I) * XO(I)
      DER0 = DER0 + XO(I) * DA(I+1)
      DER1 = DER1 + AO(I) * DA(I+1)
      ENDDO
      IF (P.GT.PMAX) RETURN
      C=(SP**(M-2))*EXP(-0.5d0*P)/ABS(DET)
c      print *,'XO',XO(1),XO(2),XO(3),XO(4)
c      print *,'AO',AO(1),AO(2),AO(3),AO(4)
      if(N.lt.1) go to 100
      DO I=1,N
       A(I) = -A0(I)
       B(I) = 0.0d0
       DO J=1,M
        B(I) = B(I)+AO(J)*A0(I+J*N)
        A(I) = A(I)+XO(J)*A0(I+J*N)
       ENDDO
      ENDDO
100   continue
      D0(1)=DER0
      D0(2)=DER1
      PC=XO(1)
      PD=AO(1)
      RETURN
      END SUBROUTINE R_ORT


      REAL*8 FUNCTION pythag(a,b)
      IMPLICIT NONE
      REAL*8, intent(in) :: a,b
      REAL*8 absa,absb
      absa=abs(a)
      absb=abs(b)
      IF (absa.GT.absb) THEN
         pythag=absa*SQRT(1.0d0+(absb/absa)**2)
      ELSE
         IF (absb.EQ.0.0d0) THEN
            pythag=0.0d0
         ELSE
            pythag=absb*SQRT(1.0d0+(absa/absb)**2)
         ENDIF
      ENDIF
      RETURN
      END FUNCTION PYTHAG


      SUBROUTINE SVBKSB(U,W,V,M,N,MP,NP,B,X)
C
C   Solves  AX=B  for a vector  X, where  A  is specified by the arrays
C   U, W, V  as returned by SVDCMP.  M  and  N  are the logical
C   dimensions of  A, and will be equal for a square matrices.  MP  and  NP
C   are the physical dimensions of  A.  B  is the input right-hand side.
C   X  is the output solution vector. No input quantities are destroyed,
C   so the routine may be called sequentialy with different  B's.
C
      USE SIZEMOD
      IMPLICIT NONE
C      INTEGER, PARAMETER :: NMAX=100
C   Maximum anticipated value of N
      INTEGER, intent(in) :: M,N,MP,NP
      INTEGER ::  J,I,JJ
      REAL*8, intent(inout) :: U,W,V,B,X
      REAL*8 TMP, S 
      DIMENSION U(MP,NP),W(NP),V(NP,NP),B(MP),X(NP),TMP(NMAX)
      DO J=1,N
C   Cumulate U^T*B
         S=0.0d0
         IF (W(J).NE.0.0d0) THEN
C   Nonzero rezult only if  wj  is nonzero
           DO I=1,M
             S=S+U(I,J)*B(I)
           ENDDO 
           S=S/W(J)
C   This is the divide by  wj
         ENDIF
         TMP(J)=S
      ENDDO
      DO J=1,N
         S=0.0d0
         DO JJ=1,N
           S=S+V(J,JJ)*TMP(JJ)
         ENDDO
         X(J)=S
      ENDDO
      RETURN
      END SUBROUTINE SVBKSB



      SUBROUTINE SVDCMP(A,M,N,MP,NP,W,V)
C
C  Given a matrix  A, with logical dimensions  M  by  N  and physical
C  dimensions  MP  by  NP, this routine computes its singular value
C  decomposition,  A=U.W.V^T, see Numerical Recipes, by Press W.,H.
C  Flannery, B. P., Teukolsky S.A. and Vetterling W., T. Cambrige
C  University Press 1986, Chapter 2.9. The matrix  U  replaces A  on
C  output. The diagonal matrix of singular values  W  is ouyput as a vector
C  W. The matrix  V (not the transpose  V^T) is output as  V.  M  must be
C  greater or equal to  N; if it is smaller, then  A  should be filled up
C  to square with zero rows.
C
      USE SIZEMOD
      IMPLICIT NONE
C      PARAMETER (NMAX=100)
      INTEGER, intent(in) :: M,N,MP,NP
      INTEGER :: I,L,K,J, ITS, NM
      REAL*8, intent(inout) :: A,W,V
      REAL*8 RV1,G, S,SCALE,ANORM, F,H, C, Y,Z,X
C  Maximum anticipated values of  N
      DIMENSION A(MP,NP),W(NP),V(NP,NP),RV1(NMAX)
      IF(M.LT.N) THEN
          print  *, 'You must augment  A  with extra zero rows. stop'
          stop
      ENDIF
C  Householder reduction to bidiagonal form
      G=0.0d0
      SCALE=0.0d0
      ANORM=0.0
      DO I=1,N
        L=I+1
        RV1(I)=SCALE*G
        G=0.0d0
        S=0.0d0
        SCALE=0.0d0
        IF (I.LE.M) THEN
          DO K=I,M
            SCALE=SCALE+ABS(A(K,I))
          ENDDO
          IF (SCALE.NE.0.0d0) THEN
            DO K=I,M
              A(K,I)=A(K,I)/SCALE
              S=S+A(K,I)*A(K,I)
            ENDDO
            F=A(I,I)
            G=-SIGN(SQRT(S),F)
            H=F*G-S
            A(I,I)=F-G
            IF (I.NE.N) THEN
              DO J=L,N
                S=0.0d0
                DO K=I,M
                  S=S+A(K,I)*A(K,J)
                ENDDO
                F=S/H
                DO K=I,M
                  A(K,J)=A(K,J)+F*A(K,I)
                ENDDO
              ENDDO
            ENDIF
            DO K=I,M
              A(K,I)=SCALE*A(K,I)
            ENDDO
          ENDIF
        ENDIF
        W(I)=SCALE*G
        G=0.0d0
        S=0.0d0
        SCALE=0.0d0
        IF ((I.LE.M).AND.(I.NE.N)) THEN
          DO K=L,N
            SCALE=SCALE+ABS(A(I,K))
          ENDDO
          IF (SCALE.NE.0.0d0) THEN
            DO K=L,N
              A(I,K)=A(I,K)/SCALE
              S=S+A(I,K)*A(I,K)
            ENDDO
            F=A(I,L)
            G=-SIGN(SQRT(S),F)
            H=F*G-S
            A(I,L)=F-G
            DO K=L,N
              RV1(K)=A(I,K)/H
            ENDDO
            IF (I.NE.M) THEN
              DO J=L,M
                S=0.0d0
                DO K=L,N
                  S=S+A(J,K)*A(I,K)
                ENDDO
                DO K=L,N
                  A(J,K)=A(J,K)+S*RV1(K)
                ENDDO
              ENDDO
            ENDIF
            DO K=L,N
              A(I,K)=SCALE*A(I,K)
            ENDDO
          ENDIF
        ENDIF
        ANORM=MAX(ANORM,(ABS(W(I))+ABS(RV1(I))))
      ENDDO
c        print *,'25'
C   Accumulation of right-hand transformations.
      DO I=N,1,-1
        IF (I.LT.N) THEN
          IF (G.NE.0.0d0) THEN
            DO J=L,N
              V(J,I)=(A(I,J)/A(I,L))/G
C   Double division to avoid posible underflow.
            ENDDO
            DO J=L,N
              S=0.0d0
              DO K=L,N
                S=S+A(I,K)*V(K,J)
              ENDDO
              DO K=L,N
                V(K,J)=V(K,J)+S*V(K,I)
              ENDDO
            ENDDO
          ENDIF
          DO J=L,N
            V(I,J)=0.0d0
            V(J,I)=0.0d0
         ENDDO
        ENDIF
        V(I,I)=1.0d0
        G=RV1(I)
        L=I
32    ENDDO
c        print *,'32'

C  Accumulation of the left-hang transformation
      DO I=N,1,-1
        L=I+1
        G=W(I)
        IF (I.LT.N) THEN
          DO J=L,N
            A(I,J)=0.0d0
          ENDDO
        ENDIF
        IF (G.NE.0.0d0) THEN
          G=1.0d0/G
          IF (I.NE.N) THEN
            DO J=L,N
              S=0.0d0
              DO K=L,M
                S=S+A(K,I)*A(K,J)
              ENDDO
              F=(S/A(I,I))*G
              DO K=I,M
                A(K,J)=A(K,J)+F*A(K,I)
              ENDDO
            ENDDO
          ENDIF
          DO J=I,M
            A(J,I)=A(J,I)*G
37        ENDDO
        ELSE
          DO J=I,M
            A(J,I)=0.0d0
38        ENDDO
        ENDIF
        A(I,I)=A(I,I)+1.0d0
39    ENDDO
c        print *,'39'

C      Diagonalization of the bidiagonal form
C      Loop over singular values
      DO K=N,1,-1
C   Loop allowed iterations
        DO ITS=1,30
C   Test for spliting
          DO L=K,1,-1
            NM=L-1
C   Note that RV1(1) is always zero
            IF((ABS(RV1(L))+ANORM).EQ.ANORM) GO TO 2
            IF((ABS(W(NM))+ANORM).EQ.ANORM) GO TO 1
          ENDDO
c          print *,'41'
1         C=0.0d0
          S=1.0d0
          DO I=L,K
            F=S*RV1(I)
            IF ((ABS(F)+ANORM).NE.ANORM) THEN
              G=W(I)
              H=SQRT(F*F+G*G)
              W(I)=H
              H=1.0d0/H
              C= (G*H)
              S=-(F*H)
              DO J=1,M
                Y=A(J,NM)
                Z=A(J,I)
                A(J,NM)=(Y*C)+(Z*S)
                A(J,I)=-(Y*S)+(Z*C)
              ENDDO
            ENDIF
          ENDDO
c          print *,'43'
2         Z=W(K)
          IF (L.EQ.K) THEN
C     Convergence
            IF (Z.LT.0.0d0) THEN
C   Singular values are made nonnegative
              W(K)=-Z
              DO J=1,N
                V(J,K)=-V(J,K)
              ENDDO
            ENDIF
            GO TO 3
          ENDIF
          IF (ITS.EQ.30) THEN
              print *, 'No convergence in 30 iterations. stop.'
              stop
          ENDIF
          X=W(L)
          NM=K-1
          Y=W(NM)
          G=RV1(NM)
          H=RV1(K)
          F=((Y-Z)*(Y+Z)+(G-H)*(G+H))/(2.0*H*Y)
          G=SQRT(F*F+1.0d0)
          F=((X-Z)*(X+Z)+H*((Y/(F+SIGN(G,F)))-H))/X
C   Next  QR  transformation
          C=1.0d0
          S=1.0d0
          DO J=L,NM
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
            DO NM=1,N
              X=V(NM,J)
              Z=V(NM,I)
              V(NM,J)= (X*C)+(Z*S)
              V(NM,I)=-(X*S)+(Z*C)
            ENDDO
c            print *,'45',F,H
            Z=pythag(F,H)
            W(J)=Z
C   Rotation can be arbitrary if  Z=0.
            IF (Z.NE.0.0d0) THEN
c            print *,1/Z
              Z=1.0d0/Z
c              print *,'*'
              C=F*Z
              S=H*Z
            ENDIF
            F= (C*G)+(S*Y)
            X=-(S*G)+(C*Y)
            DO NM=1,M
              Y=A(NM,J)
              Z=A(NM,I)
              A(NM,J)= (Y*C)+(Z*S)
              A(NM,I)=-(Y*S)+(Z*C)
            ENDDO
c          print *,'46'

          ENDDO
c          print *,'47'
          RV1(L)=0.0d0
          RV1(K)=F
          W(K)=X
        ENDDO
3     CONTINUE
      ENDDO
c        print *,'49'
       
      RETURN
      END SUBROUTINE SVDCMP

      END MODULE MREGMOD
        
      
