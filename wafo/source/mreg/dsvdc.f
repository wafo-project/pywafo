      MODULE SVD
      IMPLICIT NONE
      INTEGER, PARAMETER  :: dp = SELECTED_REAL_KIND(12, 60)

! Based upon routines from the NSWC (Naval Surface Warfare Center),
! which were based upon LAPACK routines.

! Code converted using TO_F90 by Alan Miller
! Date: 2003-11-11  Time: 17:50:44
! Revised pab 2007
! Converted to fixed form


      CONTAINS


      SUBROUTINE drotg(da, db, dc, ds)
 
!     DESIGNED BY C.L.LAWSON, JPL, 1977 SEPT 08
!
!     CONSTRUCT THE GIVENS TRANSFORMATION
!
!         ( DC  DS )
!     G = (        ) ,    DC**2 + DS**2 = 1 ,
!         (-DS  DC )
!
!     WHICH ZEROS THE SECOND ENTRY OF THE 2-VECTOR  (DA,DB)**T .
!
!     THE QUANTITY R = (+/-)SQRT(DA**2 + DB**2) OVERWRITES DA IN
!     STORAGE.  THE VALUE OF DB IS OVERWRITTEN BY A VALUE Z WHICH
!     ALLOWS DC AND DS TO BE RECOVERED BY THE FOLLOWING ALGORITHM:
!           IF Z=1  SET  DC=0.D0  AND  DS=1.D0
!           IF DABS(Z) < 1  SET  DC=SQRT(1-Z**2)  AND  DS=Z
!           IF DABS(Z) > 1  SET  DC=1/Z  AND  DS=SQRT(1-DC**2)
!
!     NORMALLY, THE SUBPROGRAM DROT(N,DX,INCX,DY,INCY,DC,DS) WILL
!     NEXT BE CALLED TO APPLY THE TRANSFORMATION TO A 2 BY N MATRIX.
!
! ------------------------------------------------------------------

      REAL (dp), INTENT(IN OUT)  :: da
      REAL (dp), INTENT(IN OUT)  :: db
      REAL (dp), INTENT(OUT)     :: dc
      REAL (dp), INTENT(OUT)     :: ds

      REAL (dp)  :: u, v, r
      IF (ABS(da) <= ABS(db)) GO TO 10

! *** HERE ABS(DA) > ABS(DB) ***

      u = da + da
      v = db / u

!     NOTE THAT U AND R HAVE THE SIGN OF DA

      r = SQRT(.25D0 + v**2) * u

!     NOTE THAT DC IS POSITIVE

      dc = da / r
      ds = v * (dc + dc)
      db = ds
      da = r
      RETURN

! *** HERE ABS(DA) <= ABS(DB) ***

   10 IF (db == 0.d0) GO TO 20
      u = db + db
      v = da / u

!     NOTE THAT U AND R HAVE THE SIGN OF DB
!     (R IS IMMEDIATELY STORED IN DA)

      da = SQRT(.25D0 + v**2) * u

!     NOTE THAT DS IS POSITIVE

      ds = db / da
      dc = v * (ds + ds)
      IF (dc == 0.d0) GO TO 15
      db = 1.d0 / dc
      RETURN
   15 db = 1.d0
      RETURN

! *** HERE DA = DB = 0.D0 ***

   20 dc = 1.d0
      ds = 0.d0
      RETURN

      END SUBROUTINE drotg


      SUBROUTINE dswap1 (n, dx, dy)
!     INTERCHANGES TWO VECTORS.
!     USES UNROLLED LOOPS FOR INCREMENTS EQUAL ONE.
!     JACK DONGARRA, LINPACK, 3/11/78.
!     This version is for increments = 1.

      INTEGER, INTENT(IN)        :: n
      REAL (dp), INTENT(IN OUT)  :: dx(*)
      REAL (dp), INTENT(IN OUT)  :: dy(*)

      REAL (dp)  :: dtemp
      INTEGER    :: i, m, mp1

      IF(n <= 0) RETURN

!       CODE FOR BOTH INCREMENTS EQUAL TO 1
!
!       CLEAN-UP LOOP

      m = MOD(n,3)
      IF( m == 0 ) GO TO 40
      DO  i = 1,m
       dtemp = dx(i)
       dx(i) = dy(i)
       dy(i) = dtemp
      END DO
      IF( n < 3 ) RETURN
  40  mp1 = m + 1
      DO  i = mp1,n,3
      dtemp = dx(i)
      dx(i) = dy(i)
      dy(i) = dtemp
      dtemp = dx(i + 1)
      dx(i + 1) = dy(i + 1)
      dy(i + 1) = dtemp
      dtemp = dx(i + 2)
      dx(i + 2) = dy(i + 2)
      dy(i + 2) = dtemp
      END DO
      RETURN
      END SUBROUTINE  dswap1


      SUBROUTINE  drot1 (n, dx, dy, c, s)
!     APPLIES A PLANE ROTATION.
!     JACK DONGARRA, LINPACK, 3/11/78.
!     This version is for increments = 1.

      INTEGER, INTENT(IN)        :: n
      REAL (dp), INTENT(IN OUT)  :: dx(*)
      REAL (dp), INTENT(IN OUT)  :: dy(*)
      REAL (dp), INTENT(IN)      :: c
      REAL (dp), INTENT(IN)      :: s

      REAL (dp)  :: dtemp
      INTEGER    :: i

      IF(n <= 0) RETURN
!       CODE FOR BOTH INCREMENTS EQUAL TO 1

      DO  i = 1,n
      dtemp = c*dx(i) + s*dy(i)
      dy(i) = c*dy(i) - s*dx(i)
      dx(i) = dtemp
      END DO
      RETURN
      END SUBROUTINE  drot1


      SUBROUTINE dsvdc(x, n, p, s, e, u, v, job, info)

      INTEGER, INTENT(IN)      :: n
      INTEGER, INTENT(IN)      :: p
      REAL (dp), INTENT(IN OUT)  :: x(:,:)
      REAL (dp), INTENT(OUT)   :: s(:)
      REAL (dp), INTENT(OUT)   :: e(:)
      REAL (dp), INTENT(OUT)   :: u(:,:)
      REAL (dp), INTENT(OUT)   :: v(:,:)
      INTEGER, INTENT(IN)      :: job
      INTEGER, INTENT(OUT)         :: info

!     DSVDC IS A SUBROUTINE TO REDUCE A DOUBLE PRECISION NXP MATRIX X
!     BY ORTHOGONAL TRANSFORMATIONS U AND V TO DIAGONAL FORM.  THE
!     DIAGONAL ELEMENTS S(I) ARE THE SINGULAR VALUES OF X.  THE
!     COLUMNS OF U ARE THE CORRESPONDING LEFT SINGULAR VECTORS,
!     AND THE COLUMNS OF V THE RIGHT SINGULAR VECTORS.
!
!     ON ENTRY
!
!         X         DOUBLE PRECISION(LDX,P), WHERE LDX.GE.N.
!                   X CONTAINS THE MATRIX WHOSE SINGULAR VALUE
!                   DECOMPOSITION IS TO BE COMPUTED.  X IS
!                   DESTROYED BY DSVDC.
!
!         LDX       INTEGER.
!                   LDX IS THE LEADING DIMENSION OF THE ARRAY X.
!
!         N         INTEGER.
!                   N IS THE NUMBER OF ROWS OF THE MATRIX X.
!
!         P         INTEGER.
!                   P IS THE NUMBER OF COLUMNS OF THE MATRIX X.
!
!         LDU       INTEGER.
!                   LDU IS THE LEADING DIMENSION OF THE ARRAY U.
!                   (SEE BELOW).
!
!         LDV       INTEGER.
!                   LDV IS THE LEADING DIMENSION OF THE ARRAY V.
!                   (SEE BELOW).
!
!         JOB       INTEGER.
!                   JOB CONTROLS THE COMPUTATION OF THE SINGULAR
!                   VECTORS.  IT HAS THE DECIMAL EXPANSION AB
!                   WITH THE FOLLOWING MEANING
!
!                        A.EQ.0    DO NOT COMPUTE THE LEFT SINGULAR VECTORS.
!                        A.EQ.1    RETURN THE N LEFT SINGULAR VECTORS IN U.
!                        A.GE.2    RETURN THE FIRST MIN(N,P) SINGULAR
!                                  VECTORS IN U.
!                        B.EQ.0    DO NOT COMPUTE THE RIGHT SINGULAR VECTORS.
!                        B.EQ.1    RETURN THE RIGHT SINGULAR VECTORS IN V.
!
!     ON RETURN
!
!         S         DOUBLE PRECISION(MM), WHERE MM=MIN(N+1,P).
!                   THE FIRST MIN(N,P) ENTRIES OF S CONTAIN THE SINGULAR
!                   VALUES OF X ARRANGED IN DESCENDING ORDER OF MAGNITUDE.
!
!         E         DOUBLE PRECISION(P).
!                   E ORDINARILY CONTAINS ZEROS.  HOWEVER SEE THE
!                   DISCUSSION OF INFO FOR EXCEPTIONS.
!
!         U         DOUBLE PRECISION(LDU,K), WHERE LDU.GE.N.  IF
!                                   JOBA.EQ.1 THEN K.EQ.N, IF JOBA.GE.2
!                                   THEN K.EQ.MIN(N,P).
!                   U CONTAINS THE MATRIX OF LEFT SINGULAR VECTORS.
!                   U IS NOT REFERENCED IF JOBA.EQ.0.  IF N.LE.P
!                   OR IF JOBA.EQ.2, THEN U MAY BE IDENTIFIED WITH X
!                   IN THE SUBROUTINE CALL.
!
!         V         DOUBLE PRECISION(LDV,P), WHERE LDV.GE.P.
!                   V CONTAINS THE MATRIX OF RIGHT SINGULAR VECTORS.
!                   V IS NOT REFERENCED IF JOB.EQ.0.  IF P.LE.N,
!                   THEN V MAY BE IDENTIFIED WITH X IN THE
!                   SUBROUTINE CALL.
!
!         INFO      INTEGER.
!                   THE SINGULAR VALUES (AND THEIR CORRESPONDING SINGULAR
!                   VECTORS) S(INFO+1),S(INFO+2),...,S(M) ARE CORRECT
!                   (HERE M=MIN(N,P)).  THUS IF INFO.EQ.0, ALL THE
!                   SINGULAR VALUES AND THEIR VECTORS ARE CORRECT.
!                   IN ANY EVENT, THE MATRIX B = TRANS(U)*X*V IS THE
!                   BIDIAGONAL MATRIX WITH THE ELEMENTS OF S ON ITS DIAGONAL
!                   AND THE ELEMENTS OF E ON ITS SUPER-DIAGONAL (TRANS(U)
!                   IS THE TRANSPOSE OF U).  THUS THE SINGULAR VALUES
!                   OF X AND B ARE THE SAME.
!
!     LINPACK. THIS VERSION DATED 03/19/79 .
!     G.W. STEWART, UNIVERSITY OF MARYLAND, ARGONNE NATIONAL LAB.
!
!     DSVDC USES THE FOLLOWING FUNCTIONS AND SUBPROGRAMS.
!
!     EXTERNAL DROT
!     BLAS DAXPY,DDOT,DSCAL,DSWAP,DNRM2,DROTG
!     FORTRAN DABS,DMAX1,MAX0,MIN0,MOD,DSQRT

!     INTERNAL VARIABLES

      INTEGER :: iter, j, jobu, k, kase, kk, l, ll, lls, lm1, lp1, ls, 
     & lu, m, maxit,mm, mm1, mp1, nct, nctp1, ncu, nrt, nrtp1
      REAL (dp) :: t, work(n)
      REAL (dp) :: b, c, cs, el, emm1, f, g, scale, shift, sl, sm, sn, 
     & smm1, t1, test, ztest
      LOGICAL :: wantu, wantv

!     SET THE MAXIMUM NUMBER OF ITERATIONS.

      maxit = 30

!     DETERMINE WHAT IS TO BE COMPUTED.

      wantu = .false.
      wantv = .false.
      jobu = MOD(job,100)/10
      ncu = n
      IF (jobu > 1) ncu = MIN(n,p)
      IF (jobu /= 0) wantu = .true.
      IF (MOD(job,10) /= 0) wantv = .true.

!     REDUCE X TO BIDIAGONAL FORM, STORING THE DIAGONAL ELEMENTS
!     IN S AND THE SUPER-DIAGONAL ELEMENTS IN E.

      info = 0
      nct = MIN(n-1, p)
      s(1:nct+1) = 0.0_dp
      nrt = MAX(0, MIN(p-2,n))
      lu = MAX(nct,nrt)
      IF (lu < 1) GO TO 170
      DO  l = 1, lu
        lp1 = l + 1
        IF (l > nct) GO TO 20
  
!           COMPUTE THE TRANSFORMATION FOR THE L-TH COLUMN AND
!           PLACE THE L-TH DIAGONAL IN S(L).
  
        s(l) = SQRT( SUM( x(l:n,l)**2 ) )
        IF (s(l) == 0.0D0) GO TO 10
        IF (x(l,l) /= 0.0D0) s(l) = SIGN(s(l), x(l,l))
        x(l:n,l) = x(l:n,l) / s(l)
        x(l,l) = 1.0D0 + x(l,l)

  10    s(l) = -s(l)

  20    IF (p < lp1) GO TO 50
        DO  j = lp1, p
          IF (l > nct) GO TO 30
          IF (s(l) == 0.0D0) GO TO 30
    
!              APPLY THE TRANSFORMATION.
    
          t = -DOT_PRODUCT(x(l:n,l), x(l:n,j)) / x(l,l)
          x(l:n,j) = x(l:n,j) + t * x(l:n,l)
    
!           PLACE THE L-TH ROW OF X INTO  E FOR THE
!           SUBSEQUENT CALCULATION OF THE ROW TRANSFORMATION.
    
   30     e(j) = x(l,j)
        END DO

   50   IF (.NOT.wantu .OR. l > nct) GO TO 70
  
! PLACE THE TRANSFORMATION IN U FOR SUBSEQUENT BACK MULTIPLICATION.
  
        u(l:n,l) = x(l:n,l)

   70   IF (l > nrt) CYCLE
  
!           COMPUTE THE L-TH ROW TRANSFORMATION AND PLACE THE
!           L-TH SUPER-DIAGONAL IN E(L).
  
        e(l) = SQRT( SUM( e(lp1:p)**2 ) )
        IF (e(l) == 0.0D0) GO TO 80
        IF (e(lp1) /= 0.0D0) e(l) = SIGN(e(l), e(lp1))
        e(lp1:lp1+p-l-1) = e(lp1:p) / e(l)
        e(lp1) = 1.0D0 + e(lp1)

   80   e(l) = -e(l)
        IF (lp1 > n .OR. e(l) == 0.0D0) GO TO 120
  
!              APPLY THE TRANSFORMATION.
  
        work(lp1:n) = 0.0D0
        DO  j = lp1, p
          work(lp1:lp1+n-l-1) = work(lp1:lp1+n-l-1) + e(j) * 
     &    x(lp1:lp1+n-l-1,j)
        END DO
        DO  j = lp1, p
          x(lp1:lp1+n-l-1,j) = x(lp1:lp1+n-l-1,j) - (e(j)/e(lp1)) *
     &     work(lp1:lp1+n-l-1)
        END DO

  120   IF (.NOT.wantv) CYCLE
  
!              PLACE THE TRANSFORMATION IN V FOR SUBSEQUENT
!              BACK MULTIPLICATION.
  
        v(lp1:p,l) = e(lp1:p)
      END DO

!     SET UP THE FINAL BIDIAGONAL MATRIX OF ORDER M.

  170 m = MIN(p,n+1)
      nctp1 = nct + 1
      nrtp1 = nrt + 1
      IF (nct < p) s(nctp1) = x(nctp1,nctp1)
      IF (n < m) s(m) = 0.0D0
      IF (nrtp1 < m) e(nrtp1) = x(nrtp1,m)
      e(m) = 0.0D0

!     IF REQUIRED, GENERATE U.

      IF (.NOT.wantu) GO TO 300
      IF (ncu < nctp1) GO TO 200
      DO  j = nctp1, ncu
        u(1:n,j) = 0.0_dp
        u(j,j) = 1.0_dp
      END DO

  200 DO  ll = 1, nct
        l = nct - ll + 1
        IF (s(l) == 0.0D0) GO TO 250
        lp1 = l + 1
        IF (ncu < lp1) GO TO 220
        DO  j = lp1, ncu
          t = -DOT_PRODUCT(u(l:n,l), u(l:n,j)) / u(l,l)
          u(l:n,j) = u(l:n,j) + t * u(l:n,l)
        END DO

  220   u(l:n,l) = -u(l:n,l)
        u(l,l) = 1.0D0 + u(l,l)
        lm1 = l - 1
        IF (lm1 < 1) CYCLE
        u(1:lm1,l) = 0.0_dp
        CYCLE

  250   u(1:n,l) = 0.0_dp
      u(l,l) = 1.0_dp
      END DO

!     IF IT IS REQUIRED, GENERATE V.

  300 IF (.NOT.wantv) GO TO 350
      DO  ll = 1, p
        l = p - ll + 1
        lp1 = l + 1
        IF (l > nrt) GO TO 320
        IF (e(l) == 0.0D0) GO TO 320
        DO  j = lp1, p
        t = -DOT_PRODUCT(v(lp1:lp1+p-l-1,l), 
     &                v(lp1:lp1+p-l-1,j)) / v(lp1,l)
        v(lp1:lp1+p-l-1,j) = v(lp1:lp1+p-l-1,j) + t * v(lp1:lp1+p-l-1,l)
        END DO

  320   v(1:p,l) = 0.0D0
        v(l,l) = 1.0D0
      END DO

!     MAIN ITERATION LOOP FOR THE SINGULAR VALUES.

  350 mm = m
      iter = 0

!        QUIT IF ALL THE SINGULAR VALUES HAVE BEEN FOUND.

!     ...EXIT
  360 IF (m == 0) GO TO 620

!        IF TOO MANY ITERATIONS HAVE BEEN PERFORMED, SET FLAG AND RETURN.

      IF (iter < maxit) GO TO 370
      info = m
!     ......EXIT
      GO TO 620

!        THIS SECTION OF THE PROGRAM INSPECTS FOR NEGLIGIBLE ELEMENTS
!        IN THE S AND E ARRAYS.  ON COMPLETION
!        THE VARIABLES KASE AND L ARE SET AS FOLLOWS.
!
!           KASE = 1     IF S(M) AND E(L-1) ARE NEGLIGIBLE AND L < M
!           KASE = 2     IF S(L) IS NEGLIGIBLE AND L < M
!           KASE = 3     IF E(L-1) IS NEGLIGIBLE, L < M, AND
!                        S(L), ..., S(M) ARE NOT NEGLIGIBLE (QR STEP).
!           KASE = 4     IF E(M-1) IS NEGLIGIBLE (CONVERGENCE).

  370 DO  ll = 1, m
          l = m - ll
!        ...EXIT
          IF (l == 0) EXIT
          test = ABS(s(l)) + ABS(s(l+1))
          ztest = test + ABS(e(l))
          IF (ztest /= test) CYCLE
          e(l) = 0.0D0
!        ......EXIT
          EXIT
      END DO

      IF (l /= m - 1) GO TO 410
      kase = 4
      GO TO 480

  410 lp1 = l + 1
      mp1 = m + 1
      DO  lls = lp1, mp1
      ls = m - lls + lp1
!           ...EXIT
      IF (ls == l) EXIT
      test = 0.0D0
      IF (ls /= m) test = test + ABS(e(ls))
      IF (ls /= l + 1) test = test + ABS(e(ls-1))
      ztest = test + ABS(s(ls))
      IF (ztest /= test) CYCLE
      s(ls) = 0.0D0
!           ......EXIT
      EXIT
      END DO

      IF (ls /= l) GO TO 450
      kase = 3
      GO TO 480

  450 IF (ls /= m) GO TO 460
      kase = 1
      GO TO 480

  460 kase = 2
      l = ls
  480 l = l + 1

!        PERFORM THE TASK INDICATED BY KASE.

      SELECT CASE ( kase )
      CASE (    1)
      GO TO 490
      CASE (    2)
      GO TO 520
      CASE (    3)
      GO TO 540
      CASE (    4)
      GO TO 570
      END SELECT

!        DEFLATE NEGLIGIBLE S(M).

  490 mm1 = m - 1
      f = e(m-1)
      e(m-1) = 0.0D0
      DO  kk = l, mm1
        k = mm1 - kk + l
        t1 = s(k)
        CALL drotg(t1, f, cs, sn)
        s(k) = t1
        IF (k == l) GO TO 500
        f = -sn*e(k-1)
        e(k-1) = cs*e(k-1)
      
  500   IF (wantv) CALL drot1(p, v(1:,k), v(1:,m), cs, sn)
      END DO
      GO TO 610

!        SPLIT AT NEGLIGIBLE S(L).

  520 f = e(l-1)
      e(l-1) = 0.0D0
      DO  k = l, m
      t1 = s(k)
      CALL drotg(t1, f, cs, sn)
      s(k) = t1
      f = -sn*e(k)
      e(k) = cs*e(k)
      IF (wantu) CALL drot1(n, u(1:,k), u(1:,l-1), cs, sn)
      END DO
      GO TO 610

!        PERFORM ONE QR STEP.
!
!           CALCULATE THE SHIFT.
 
  540 scale = MAX(ABS(s(m)),ABS(s(m-1)),ABS(e(m-1)),ABS(s(l)),ABS(e(l)))
      sm = s(m)/scale
      smm1 = s(m-1)/scale
      emm1 = e(m-1)/scale
      sl = s(l)/scale
      el = e(l)/scale
      b = ((smm1 + sm)*(smm1 - sm) + emm1**2)/2.0D0
      c = (sm*emm1)**2
      shift = 0.0D0
      IF (b == 0.0D0 .AND. c == 0.0D0) GO TO 550
      shift = SQRT(b**2+c)
      IF (b < 0.0D0) shift = -shift
      shift = c/(b + shift)

  550 f = (sl + sm)*(sl - sm) - shift
      g = sl*el

!           CHASE ZEROS.

      mm1 = m - 1
      DO  k = l, mm1
        CALL drotg(f, g, cs, sn)
        IF (k /= l) e(k-1) = f
        f = cs*s(k) + sn*e(k)
        e(k) = cs*e(k) - sn*s(k)
        g = sn*s(k+1)
        s(k+1) = cs*s(k+1)
        IF (wantv) CALL drot1(p, v(1:,k), v(1:,k+1), cs, sn)
        CALL drotg(f, g, cs, sn)
        s(k) = f
        f = cs*e(k) + sn*s(k+1)
        s(k+1) = -sn*e(k) + cs*s(k+1)
        g = sn*e(k+1)
        e(k+1) = cs*e(k+1)
        IF (wantu .AND. k < n) CALL drot1(n, u(1:,k), u(1:,k+1), cs, sn)
        END DO
        e(m-1) = f
        iter = iter + 1
       GO TO 610

!        CONVERGENCE.

!           MAKE THE SINGULAR VALUE  POSITIVE.

  570 IF (s(l) >= 0.0D0) GO TO 590
      s(l) = -s(l)
      IF (wantv) v(1:p,l) = -v(1:p,l)

!           ORDER THE SINGULAR VALUE.

  590 IF (l == mm) GO TO 600
!           ...EXIT
      IF (s(l) >= s(l+1)) GO TO 600
      t = s(l)
      s(l) = s(l+1)
      s(l+1) = t
      IF (wantv .AND. l < p) CALL dswap1(p, v(1:,l), v(1:,l+1))
      IF (wantu .AND. l < n) CALL dswap1(n, u(1:,l), u(1:,l+1))
      l = l + 1
      GO TO 590

  600 iter = 0
      m = m - 1

  610 GO TO 360

  620 RETURN
      END SUBROUTINE dsvdc

      END MODULE SVD
