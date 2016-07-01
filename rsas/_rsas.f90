! -*- f90 -*-
      subroutine f_solve_RK4(J, Q, rSAS_lookup, P_list, ST_init, dt,  &
                 CS_init, C_J, alpha, k1, C_eq, C_old, ST, PQ, &
                 WaterBalance, MS, MQ, MR, C_Q, SoluteBalance, &
                 n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
      implicit none
      integer, intent(in) ::  n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list
      real(8), intent(in) :: dt
      real(8), intent(in), dimension(timeseries_length) :: J
      real(8), intent(in), dimension(timeseries_length, numflux) :: Q
      real(8), intent(in), dimension(nP_list) :: P_list
      real(8), intent(in), dimension(timeseries_length, nP_list, &
                 numflux) :: rSAS_lookup
      real(8), intent(in), dimension(max_age) :: ST_init
      real(8), intent(in), dimension(max_age,numsol) :: CS_init
      real(8), intent(in), dimension(timeseries_length,numsol) :: C_J
      real(8), intent(in), dimension(timeseries_length,numflux,numsol) &
                 :: alpha
      real(8), intent(in), dimension(timeseries_length,numsol) :: k1
      real(8), intent(in), dimension(timeseries_length,numsol) :: C_eq
      real(8), intent(in), dimension(numsol) :: C_old
      real(8), intent(out), dimension(timeseries_length, numflux, &
                 numsol) :: C_Q
      real(8), intent(out), dimension(max_age + 1, timeseries_length + &
                 1) :: ST
      real(8), intent(out), dimension(max_age, timeseries_length) :: &
                 WaterBalance
      real(8), intent(out), dimension(max_age + 1, timeseries_length + &
                 1, numflux) :: PQ
      real(8), intent(out), dimension(max_age + 1, timeseries_length + &
                 1, numsol) :: MS
      real(8), intent(out), dimension(max_age + 1, timeseries_length + &
                 1, numflux, numsol) :: MQ
      real(8), intent(out), dimension(max_age + 1, timeseries_length + &
                 1, numsol) :: MR
      real(8), intent(out), dimension(max_age, timeseries_length, &
                 numsol) :: SoluteBalance
      integer :: k, i
      real(8) :: h
      real(8), dimension(max_age*n_substeps+1) :: ST0_cum
      real(8), dimension(max_age+1, numsol) :: MS_init
      real(8), dimension(max_age*n_substeps+1, numflux) :: PQ0_cum
      real(8), dimension(max_age*n_substeps+1, numflux) :: PQt_cum
      real(8), dimension(max_age*n_substeps+1) :: STt_cum
      real(8), dimension(max_age*n_substeps) :: sTp
      real(8), dimension(max_age*n_substeps) :: sTt
      real(8), dimension(max_age*n_substeps) :: sTn
      real(8), dimension(max_age*n_substeps+1) :: STn_cum
      real(8), dimension(max_age*n_substeps, numflux) :: pQ1
      real(8), dimension(max_age*n_substeps, numflux) :: pQ2
      real(8), dimension(max_age*n_substeps, numflux) :: pQ3
      real(8), dimension(max_age*n_substeps, numflux) :: pQ4
      real(8), dimension(max_age*n_substeps, numflux) :: pQn
      real(8), dimension(max_age, numflux) :: pQs
      real(8), dimension(max_age*n_substeps, numflux, numsol) :: mQ1
      real(8), dimension(max_age*n_substeps, numflux, numsol) :: mQ2
      real(8), dimension(max_age*n_substeps, numflux, numsol) :: mQ3
      real(8), dimension(max_age*n_substeps, numflux, numsol) :: mQ4
      real(8), dimension(max_age*n_substeps, numflux, numsol) :: mQn
      real(8), dimension(max_age, numflux, numsol) :: mQs
      real(8), dimension(max_age*n_substeps, numsol) :: mR1
      real(8), dimension(max_age*n_substeps, numsol) :: mR2
      real(8), dimension(max_age*n_substeps, numsol) :: mR3
      real(8), dimension(max_age*n_substeps, numsol) :: mR4
      real(8), dimension(max_age*n_substeps, numsol) :: mRn
      real(8), dimension(max_age, numsol) :: mRs
      real(8), dimension(max_age*n_substeps, numsol) :: mSp
      real(8), dimension(max_age*n_substeps, numsol) :: mSn
      real(8), dimension(max_age*n_substeps+1, numsol) :: MSn_cum
      real(8), dimension(max_age*n_substeps, numsol) :: mSt
      integer is, ie, iq, s, M
      M = max_age * n_substeps
      h = dt / n_substeps
      do s=1,numsol
        MS_init(1:max_age+1,s) = CS_init(1:max_age+1,s) * ST_init
      enddo
      do i=1,max_age
        is = (i-1)*n_substeps+1
        ie = i*n_substeps
        sTn(is:ie) = (ST_init(i+1) - ST_init(i))/n_substeps
        do s=1,numsol
          mSn(is:ie,s) = (MS_init(i+1,s) - MS_init(i,s))/n_substeps
        enddo
      enddo
      ST0_cum = cumsum(sTn, M)
      do iq=1,numflux
        call lookup(rSAS_lookup(1:nP_list,1,iq), P_list, ST0_cum, &
                 PQ0_cum(1:M+1,iq), nP_list, M+1)
      enddo
      ST(1:M+1,1) = ST0_cum(1:M+1:n_substeps)
      do iq=1,numflux
        PQ(1:M+1,1,iq) = PQ0_cum(1:M+1:n_substeps, iq)
      enddo
      do s=1,numsol
        MSn_cum(1:M+1,s) = cumsum(mSn(1:M,s), M)
        MS(1:M+1,1,s) = MSn_cum(1:M+1:n_substeps, s)
      enddo
      do i=1,timeseries_length
        do k=1,n_substeps
          sTp(2:M) = sTn(1:M-1)
          sTp(1) = 0.
          mSp(2:M,:) = mSn(1:M-1,:)
          mSp(1,:) = 0.
          call get_flux(i, sTp, STt_cum, pQ1, PQt_cum, J, Q, rSAS_lookup, &
                 P_list, mSp, mQ1, mR1, alpha, k1, C_eq, n_substeps, &
                 numflux, numsol, max_age, timeseries_length, nP_list)
          call new_state(i, h/2, sTp, sTt, pQ1, J, Q, mSp, mSt, mQ1, &
                 mR1, C_J, n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
          call get_flux(i, sTt, STt_cum, pQ2, PQt_cum, J, Q, rSAS_lookup, &
                 P_list, mSt, mQ2, mR2, alpha, k1, C_eq, n_substeps, &
                 numflux, numsol, max_age, timeseries_length, nP_list)
          call new_state(i, h/2, sTp, sTt, pQ2, J, Q, mSp, mSt, mQ2, &
                 mR2, C_J, n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
          call get_flux(i, sTt, STt_cum, pQ3, PQt_cum, J, Q, rSAS_lookup, &
                 P_list, mSt, mQ3, mR3, alpha, k1, C_eq, n_substeps, &
                 numflux, numsol, max_age, timeseries_length, nP_list)
          call new_state(i, h, sTp, sTt, pQ3, J, Q, mSp, mSt, mQ3, mR3&
                 , C_J, n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
          call get_flux(i, sTt, STt_cum, pQ4, PQt_cum, J, Q, rSAS_lookup, &
                 P_list, mSt, mQ4, mR4, alpha, k1, C_eq, n_substeps, &
                 numflux, numsol, max_age, timeseries_length, nP_list)
          pQn = (pQ1 + 2*pQ2 + 2*pQ3 + pQ4) / 6.
          mQn = (mQ1 + 2*mQ2 + 2*mQ3 + mQ4) / 6.
          mRn = (mR1 + 2*mR2 + 2*mR3 + mR4) / 6.
          do iq=1,numflux
            if (Q(i,iq)==0) then
              pQn(1:M,iq) = 0.
            endif
          enddo
          call new_state(i, h, sTp, sTt, pQn, J, Q, mSp, mSt, mQn, mRn&
                 , C_J, n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
          do s=1,numsol
            do iq=1,numflux
              if (Q(i,iq)>0) then
                C_Q(i,iq,s) = C_Q(i,iq,s) + sum(mQn(1:M,iq,s)) / &
                 Q(i,iq) / n_substeps
              endif
            enddo
          enddo
          do iq=1,numflux
            pQs(1, iq) = pQs(1, iq) + sum(pQn(1:k+1, iq))/ &
                 n_substeps
            pQs(2:max_age, iq) = pQs(2:max_age, iq) + sum( &
                 reshape(pQn(k:M-(n_substeps-k), iq),[max_age-1,&
                 n_substeps]), 1)/ n_substeps
            do s=1,numsol
              mQs(1, iq, s) = mQs(1, iq, s) + sum(mQn(1:k+1, iq&
                 , s), 1)/n_substeps
              mQs(2:max_age, iq, s) = mQs(2:max_age, iq, s)&
                 + sum(reshape(mQn(k:M-(n_substeps-k), iq, s), &
                 [max_age-1, n_substeps]), 1)/n_substeps
            enddo
          enddo
          do s=1,numsol
            mRs(1, s) = mRs(1, s) + sum(mRn(1:k+1, s), 1)/ &
                 n_substeps
            mRs(2:max_age, s) = mRs(2:max_age, s) + sum( &
                 reshape(mRn(k:M-(n_substeps-k), s), [max_age-1,&
                 n_substeps]), 1)/ n_substeps
          enddo
        enddo
        STn_cum = cumsum(sTn, M)
        ST(1:max_age+1, i+1) = STn_cum(n_substeps:M:n_substeps)
        WaterBalance(2:max_age, i) = diff(ST(1:max_age, i), M+1) - &
                 diff(ST(1:max_age+1, i+1), M+1)
        WaterBalance(1, i) = J(i) * dt - ST(2, i+1)
        do iq=1,numflux
          PQ(1:M+1,i+1,iq) = cumsum(pQs(1:M,iq), M)
          WaterBalance(2:max_age, i) = WaterBalance(2:max_age, i) - dt &
                 * (Q(i,iq) * diff(PQ(2:M+1,i+1,iq), M+1))
          WaterBalance(1, i) = WaterBalance(1, i) - dt * Q(i,iq) * &
                 (PQ(1,i+1,iq) - PQ(1,i+1,iq))
        enddo
        do s=1,numsol
          MSn_cum(1:M+1,s) = cumsum(mSn(1:M,s), M)
          MS(1:max_age+1,i+1,s)=MSn_cum(1:M+1:n_substeps,s)
          MR(1:M+1,i+1,s) = cumsum(mRs(1:M,s), M)
          do iq=1,numflux
            MQ(1:M+1,i+1,iq,s) = cumsum(mQs(1:M,iq,s), M)
            C_Q(i,iq,s) = C_Q(i,iq,s) + alpha(i,iq,s) * C_old(s) * (1 &
                 - PQ(max_age, i+1, iq))
          enddo
        enddo
        do s=1,numsol
          SoluteBalance(1:max_age,i,s) = (diff(MS(1:max_age,i,s), M+1)-&
                 diff(MS(1:max_age+1,i+1,s), M+1) &
                 + dt * diff(MR(1:,i+1,s), M+1))
          SoluteBalance(1,i,s) = (C_J(i,s) * J(i) * dt - MS(1,i+1,s) + &
                 dt * (MR(1,i+1,s) - MR(1,i+1,s)))
          do iq=1,numflux
            SoluteBalance(1:max_age,i,s) = (SoluteBalance(1:max_age,i,s&
                 ) - dt * diff(MQ(1:,i+1,iq,s), M+1))
            SoluteBalance(1,i,s) = (SoluteBalance(1,i,s) - dt * (MQ(1,&
                 i +1,iq,s) - MQ(1,i+1,iq,s)))
          enddo
        enddo
      enddo
      contains

      subroutine lookup(xa, ya, x, y, na, n)
      implicit none
      integer, intent(in) ::  na, n
      real(8), intent(in), dimension(na) :: xa
      real(8), intent(in), dimension(na) :: ya
      real(8), intent(in), dimension(n) :: x
      real(8), intent(out), dimension(n) :: y
      integer :: i, j, i0
      real(8) :: dif, grad
      do j=1,n
        i0 = 0
        do i=1,na
            if (x(j).ge.xa(i)) then
                i0 = i
                exit
            endif
        enddo
        if (i0.eq.0) then
            y(j) = ya(na)
        else
            dif = x(j) - xa(i0)
            grad = (ya(i0+1)-ya(i0))/(xa(i0+1)-xa(i0))
            y(j) = y(i0) + dif * grad
        endif
      enddo
      end subroutine

      function diff(arr, n)
      implicit none
      integer, intent(in) ::  n
      real(8), intent(in), dimension(n) :: arr
      real(8), dimension(n-1) :: diff
      integer :: i
      do i=1,n-1
        diff(i) = arr(i+1) - arr(i)
      enddo
      end function diff

      function cumsum(arr, n)
      implicit none
      integer, intent(in) ::  n
      real(8), intent(in), dimension(n) :: arr
      real(8), dimension(n+1) :: cumsum
      integer :: i
      cumsum(1) = 0
      do i=1,n
        cumsum(i+1) = cumsum(i) + arr(i)
      enddo
      end function cumsum

      subroutine get_flux(i, sTt, STt_cum, pQt, PQt_cum, J, Q, rSAS_lookup&
                 , P_list, mSt, mQt, mRt, alpha, k1, C_eq, n_substeps, &
                 numflux, numsol, max_age, timeseries_length, nP_list)
      implicit none
      integer, intent(in) :: i
      integer, intent(in) :: n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list
      real(8), intent(in), dimension(timeseries_length) :: J
      real(8), intent(in), dimension(timeseries_length,numflux) :: Q
      real(8), intent(in), dimension(nP_list) :: P_list
      real(8), intent(in), dimension(timeseries_length, nP_list, &
                 numflux) :: rSAS_lookup
      real(8), intent(in), dimension(timeseries_length,numflux,numsol) &
                 :: alpha
      real(8), intent(in), dimension(timeseries_length,numsol) :: k1
      real(8), intent(in), dimension(timeseries_length,numsol) :: C_eq
      real(8), intent(inout), dimension(max_age*n_substeps+1, numflux) &
                 :: PQt_cum
      real(8), intent(inout), dimension(max_age*n_substeps+1) :: STt_cum
      real(8), intent(inout), dimension(max_age*n_substeps, numflux) &
                 :: pQt
      real(8), intent(in), dimension(max_age*n_substeps) :: sTt
      real(8), intent(inout), dimension(max_age*n_substeps, numflux, &
                 numsol) :: mQt
      real(8), intent(inout), dimension(max_age*n_substeps, numsol) :: &
                 mRt
      real(8), intent(in), dimension(max_age*n_substeps, numsol) :: mSt
      integer M, iq, s 
      M = max_age * n_substeps
      STt_cum = cumsum(sTt, M)
      do iq=1,numflux
        call lookup(rSAS_lookup(1:nP_list,i,iq), P_list, STt_cum, &
                 PQt_cum, nP_list, M+1)
      enddo
      do iq=1,numflux
        pQt(1:M,iq) = diff(PQt_cum(1:M+1,iq), M+1)
        do s=1,numsol
          where (sTt(1:M)>0)
            mQt(1:M,iq,s) = mSt(1:M,s) / sTt(1:M) * alpha(i,iq,s) &
                 * Q(i,iq) * pQt(1:M,iq)
          elsewhere
           mQt(1:M,iq,s) = 0.
          end where
        enddo
      enddo
      do s=1,numsol
         mRt(1:M,s) = k1(i,s) * (C_eq(i,s) * sTt - mSt(1:M,s))
      enddo
      end subroutine get_flux

      subroutine new_state(i, hr, sTp, sTt, pQt, J, Q, mSp, mSt, mQt, &
                 mRt, C_J, n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
      integer, intent(in) :: n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list
      integer, intent(in) :: i
      real(8), intent(in) :: hr
      real(8), intent(in), dimension(timeseries_length) :: J
      real(8), intent(in), dimension(timeseries_length, numflux) :: Q
      real(8), intent(in), dimension(max_age*n_substeps, numflux) :: pQt
      real(8), intent(inout), dimension(max_age*n_substeps) :: sTt
      real(8), intent(in), dimension(max_age*n_substeps) :: sTp
      real(8), intent(in), dimension(max_age*n_substeps, numflux, &
                 numsol) :: mQt
      real(8), intent(inout), dimension(max_age*n_substeps, numsol) :: &
                 mSt
      real(8), intent(in), dimension(max_age*n_substeps, numsol) :: mSp
      real(8), intent(in), dimension(max_age*n_substeps, numsol) :: mRt
      real(8), intent(in), dimension(timeseries_length, numsol) :: C_J
      M = max_age * n_substeps
      sTt = sTp
      do iq=1,numflux
        sTt = sTt - Q(i,iq) * pQt(1:M,iq) * hr
      enddo
      sTt(1) = sTt(1) + J(i) * hr
      do s=1,numsol
        mSt(1:M,s) = mSp(1:M,s) + mRt(1:M,s) * hr
        do iq=1,numflux
          mSt(1:M,s) = mSt(1:M,s) - mQt(1:M,iq,s) * hr
          mSt(1,s) = mSt(1,s) + J(i) * C_J(i,s) * hr
        enddo
      enddo
      end subroutine new_state

      end subroutine f_solve_RK4
