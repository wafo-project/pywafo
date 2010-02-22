      PROGRAM TST_FIMOD
!      USE ERFCOREMOD
      USE FIMOD
*
*     Test program for fimod
*
      DOUBLE PRECISION X, XI, XI2, VAL, P, Q, CORREL
      DOUBLE PRECISION A(3), B(3), R(3), EPSI
      INTEGER I, INFIN(3)
      PRINT '('' Results for:  FI'')'
      X = -1.0
      DO I = 1,5
         VAL = FI(X)
         CALL NORMPRB(X, P, Q)
         XI = FIINV(VAL)
         XI2 = -FIINV(Q)
         PRINT *, 'X=',X, ' P=', VAL, ' XI=', XI
         PRINT *, 'X=',X, ' P2=', P, ' XI2=', XI2, ' Q=', Q
         X = X + I
      !PRINT '(''  Error Estimate :   '', ''   ('', F8.6'')'' )', ERR
      ENDDO

      PRINT '('' Results for:  BVNMVN'')'
      CORREL = -0.2D0
      DO I = 1,6
       CORREL = CORREL + 0.2D0
       A(:) = 0.0D0
       B(:) = 5.0D0
       INFIN(:) = 1
       VAL = BVNMVN( A, B, INFIN, CORREL )
       PRINT *, ' P=', VAL, ' R=', CORREL

      END DO

      PRINT '('' Results for:  TVNMVN'')'
      EPSI = 1D-10
      CORREL = -0.2D0
      DO I = 1,6
       CORREL = CORREL + 0.2D0
       A(:) = 0.0D0
       B(:) = 5.0D0
       INFIN(:) = 1
       R(:) = CORREL !(/ 0.3D0, R13, R23 /)
       VAL = TVNMVN(A, B, INFIN, R, EPSI)
       PRINT *, ' P=', VAL, ' R=', CORREL

      END DO
      END
