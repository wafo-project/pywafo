      PROGRAM TST_RIND
      USE RIND71MOD
      USE FIMOD
      IMPLICIT NONE

*
*     Test program for rind71
*
* gfortran -W -Wall -pedantic-errors -fbounds-check -Werror -o test_rind71.exe intmodule.f  jacobmod.f fimod.f rind71.f test_rind71.f
*
      DOUBLE PRECISION ABSEPS, RELEPS, VAL
      INTEGER N, Nc,Nt,Nx, I, J, K, IJ, MAXPTS, IFT
      PARAMETER ( N = 5, Nc=0, Nt=5, Nx=1, MAXPTS = 5000*N*N*N )
      PARAMETER ( ABSEPS = 0.00005, RELEPS = 0 )
      DOUBLE PRECISION CORREL(N,N), LOW(N), UP(N), BLOW(Nx,N),BUP(Nx,N)
      DOUBLE PRECISION Ex(N), Vals(Nx),ERR(Nx), TERR(Nx),Xc(Nc,Nx)
      INTEGER INFIN(N), INDI(N+1)
*          Chen Problem
      DATA ( UP(I), I=1,N)  /.0, 1.5198, 9.7817, 9.4755, 1.5949/
      DATA (LOW(I), I=1,N)  /.0,  .0   ,-1.7817,-1.4755,-9.5949/
      DATA (INFIN(I), I=1,N)/ 1, 2     , 1     , 1     , 0     /
      INDI = (/(I,I=0,N)/)
      CORREL(:,:) = 0.0d0
      CORREL(1,2) = -0.707107
      CORREL(2,3) = 1.d0
      CORREL(2:3,4:5) = 0.5d0
      CORREL(4,5) = 0.5d0
      Ex(:) = 0.d0
      DO I = 1, N
        CORREL(I,I) = 1.0d0
        DO j = I+1, N
           CORREL(I,J) = 0.3
           CORREL(J,I) = CORREL(I,J)
        ENDDO
      ENDDO
      val = FI(releps)
      CALL initdata(1)
      BLow(1,:) = LOW
      Bup(1,:) = UP

      PRINT '(''               Test of RIND71'')'
      PRINT '(12X, ''Requested Accuracy '',F8.5)', MAX(ABSEPS,RELEPS)
      PRINT '(''           Number of Dimensions is '',I2)', N
      PRINT '(''     Maximum # of Function Values is '',I7)', MAXPTS
*
      DO K = 1, 3
         PRINT '(/'' I     Limits'')'
         PRINT '(4X,''Lower  Upper  Lower Left of Correlation Matrix'')'
         if (INFIN(1)==1) then
            BUP(1,1) = 9.d0
            BLOW(1,1) = 0.d0
         elseif (INFIN(1)==0) then
            BUP(1,1) = 0.d0
            BLOW(1,1) = -9.d0
         else
            BUP(1,1) = 9.d0
            BLOW(1,1) = -9.d0
         endif
         DO I = 1, N
            IF ( INFIN(I) .LT. 0 ) THEN
               PRINT '(I2, '' -infin  infin '', 7F9.5)',
     &              I, ( CORREL(I,J), J = 1,I )
            ELSE IF ( INFIN(I) .EQ. 0 ) THEN
               PRINT '(I2, '' -infin'', F7.4, 1X, 7F9.5)',
     &              I, UP(I), ( CORREL(I,J), J = 1,I)
            ELSE IF ( INFIN(I) .EQ. 1 ) THEN
               PRINT '(I2, F7.4, ''  infin '', 7F9.5)',
     &              I, LOW(I), ( CORREL(I,J), J = 1,I)
            ELSE
               PRINT '(I2, 2F7.4, 1X, 7F9.5)',
     &              I, LOW(I), UP(I), ( CORREL(I,J), J = 1,I )
            ENDIF
         END DO

         CALL RIND71(VALS, CORREL, Ex, Xc,Nt,indI,Blow,Bup)
!         CALL RINDD(VALS,ERR,TERR,CORREL,Ex,Xc,Nt,indI,BLow,Bup,INFIN)


         PRINT '('' Results for:  RINDD'')'
         PRINT '(''      Value      :   '', F12.6, I5 )', VALS(1), IFT
!         PRINT '(''  Error Estimate :   '', ''   ('', F8.6'')'' )', ERR
         INFIN(1) = INFIN(1) - 1

      END DO
      END
