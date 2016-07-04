! -*- f90 -*-
      subroutine f_solve_RK4(J, Q, rSAS_lookup, P_list, ST_init, dt,  &
                 verbose, debug, full_outputs,&
                 CS_init, C_J, alpha, k1, C_eq, C_old, ST, PQ, &
                 WaterBalance, MS, MQ, MR, C_Q, SoluteBalance, &
                 n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
      implicit none
      logical, intent(in) :: verbose, debug, full_outputs
      integer, intent(in) ::  n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list
      real(8), intent(in) :: dt
      real(8), intent(in), dimension(0:timeseries_length-1) :: J
      real(8), intent(in), dimension(0:timeseries_length-1, 0:numflux-1) :: Q
      real(8), intent(in), dimension(0:nP_list-1) :: P_list
      real(8), intent(in), dimension(0:nP_list-1, 0:timeseries_length-1, 0:numflux-1) :: rSAS_lookup
      real(8), intent(in), dimension(0:max_age) :: ST_init
      real(8), intent(in), dimension(0:max_age-1,0:numsol-1) :: CS_init
      real(8), intent(in), dimension(0:timeseries_length-1,0:numsol-1) :: C_J
      real(8), intent(in), dimension(0:timeseries_length-1,0:numflux-1,0:numsol-1) &
                 :: alpha
      real(8), intent(in), dimension(0:timeseries_length-1,0:numsol-1) :: k1
      real(8), intent(in), dimension(0:timeseries_length-1,0:numsol-1) :: C_eq
      real(8), intent(in), dimension(0:numsol-1) :: C_old
      real(8), intent(out), dimension(0:timeseries_length-1, 0:numflux-1, 0:numsol-1) :: C_Q
      real(8), intent(out), dimension(0:max_age, 0:timeseries_length) :: ST
      real(8), intent(out), dimension(0:max_age-1, 0:timeseries_length-1) :: &
                 WaterBalance
      real(8), intent(out), dimension(0:max_age, 0:timeseries_length, 0:numflux-1) :: PQ
      real(8), intent(out), dimension(0:max_age, 0:timeseries_length, 0:numsol-1) :: MS
      real(8), intent(out), dimension(0:max_age, 0:timeseries_length, 0:numflux-1, 0:numsol-1) :: MQ
      real(8), intent(out), dimension(0:max_age, 0:timeseries_length, 0:numsol-1) :: MR
      real(8), intent(out), dimension(0:max_age-1, 0:timeseries_length-1, 0:numsol-1) :: SoluteBalance
      integer :: k, i
      real(8) :: h
      real(8), dimension(0:max_age*n_substeps) :: ST0_cum
      real(8), dimension(0:max_age*n_substeps, 0:numflux-1) :: PQ0_cum
      real(8), dimension(0:max_age*n_substeps) :: PQt_cum
      real(8), dimension(0:max_age*n_substeps) :: STt_cum
      real(8), dimension(0:max_age*n_substeps-1) :: sTp
      real(8), dimension(0:max_age*n_substeps-1) :: sTt
      real(8), dimension(0:max_age*n_substeps-1) :: sTn
      real(8), dimension(0:max_age*n_substeps) :: STn_cum
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1) :: pQ1
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1) :: pQ2
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1) :: pQ3
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1) :: pQ4
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1) :: pQn
      real(8), dimension(0:max_age-1, 0:numflux-1) :: pQs
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1, 0:numsol-1) :: mQ1
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1, 0:numsol-1) :: mQ2
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1, 0:numsol-1) :: mQ3
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1, 0:numsol-1) :: mQ4
      real(8), dimension(0:max_age*n_substeps-1, 0:numflux-1, 0:numsol-1) :: mQn
      real(8), dimension(0:max_age-1, 0:numflux-1, 0:numsol-1) :: mQs
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mR1
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mR2
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mR3
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mR4
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mRn
      real(8), dimension(0:max_age-1, 0:numsol-1) :: mRs
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mSp
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mSn
      real(8), dimension(0:max_age*n_substeps, 0:numsol-1) :: MSn_cum
      real(8), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mSt
      character(len=128) :: debugstring
      integer is, ie, iq, s, M
      call f_verbose(verbose,'...Initializing arrays...')
      C_Q(:, :, :) = 0.
      ST(:, :) = 0.
      WaterBalance(:, :) = 0.
      PQ(:, :, :) = 0.
      MS(:, :, :) = 0.
      MQ(:, :, :, :) = 0.
      MR(:, :, :) = 0.
      SoluteBalance(:, :, :) = 0.
      ST0_cum(:) = 0.
      PQ0_cum(:, :) = 0.
      PQt_cum(:) = 0.
      STt_cum(:) = 0.
      sTp(:) = 0.
      sTt(:) = 0.
      sTn(:) = 0.
      STn_cum(:) = 0.
      pQ1(:, :) = 0.
      pQ2(:, :) = 0.
      pQ3(:, :) = 0.
      pQ4(:, :) = 0.
      pQn(:, :) = 0.
      pQs(:, :) = 0.
      mQ1(:, :, :) = 0.
      mQ2(:, :, :) = 0.
      mQ3(:, :, :) = 0.
      mQ4(:, :, :) = 0.
      mQn(:, :, :) = 0.
      mQs(:, :, :) = 0.
      mR1(:, :) = 0.
      mR2(:, :) = 0.
      mR3(:, :) = 0.
      mR4(:, :) = 0.
      mRn(:, :) = 0.
      mRs(:, :) = 0.
      mSp(:, :) = 0.
      mSn(:, :) = 0.
      MSn_cum(:, :) = 0.
      mSt(:, :) = 0.
      M = max_age * n_substeps
      h = dt / n_substeps
      call f_verbose(verbose,'...Setting initial conditions...')
      do i=0,max_age-1
        is = i*n_substeps
        ie = (i+1)*n_substeps-1
        sTn(is:ie) = (ST_init(i+1) - ST_init(i))/n_substeps
        do s=0,numsol-1
          mSn(is:ie,s) = (CS_init(i,s) * (ST_init(i+1) - ST_init(i)))&
                          /n_substeps
        enddo
      enddo
      ST0_cum = cumsum(sTn, M)
      do iq=0,numflux-1
        call lookup(rSAS_lookup(:,0,iq), P_list, ST0_cum, &
                 PQ0_cum(:,iq), nP_list, M+1)
      enddo
      ST(1:max_age,0) = ST0_cum(n_substeps:M:n_substeps)
      do iq=0,numflux-1
        PQ(:,0,iq) = PQ0_cum(0:M:n_substeps, iq)
      enddo
      do s=0,numsol-1
        MSn_cum(:,s) = cumsum(mSn(:,s), M)
        MS(:,0,s) = MSn_cum(0:M:n_substeps, s)
      enddo
      call f_verbose(verbose,'...Starting main loop...')
      do i=0,timeseries_length-1
        pQs(:, :) = 0.
        mQs(:, :, :) = 0.
        mRs(:, :) = 0.
        do k=0,n_substeps-1
          if (debug) then
            write (debugstring,*) i, ' ', k
            print*, debugstring
          endif
          sTp(1:M-1) = sTn(0:M-2)
          sTp(0) = 0.
          mSp(1:M-1,:) = mSn(0:M-2,:)
          mSp(0,:) = 0.
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
          if (debug) then
            print*, 'pQn', pQn(:,0)
            print*, 'mQn', mQn(:,0,0)
            print*, 'mRn', mRn(:,0)
          endif
          do iq=0,numflux-1
            if (Q(i,iq)==0) then
              pQn(:,iq) = 0.
            endif
          enddo
          call new_state(i, h, sTp, sTn, pQn, J, Q, mSp, mSn, mQn, mRn&
                 , C_J, n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
          if (debug) then
            print*, 'mSn', mSn
          endif
          do s=0,numsol-1
            do iq=0,numflux-1
              if (Q(i,iq)>0) then
                C_Q(i,iq,s) = C_Q(i,iq,s) + sum(mQn(:,iq,s)) / &
                 Q(i,iq) / n_substeps
              endif
            enddo
          enddo
          if (debug) then
            print*, 'C_Q', C_Q(i,0,0), Q(i,0)
          endif
          if (full_outputs) then
            do iq=0,numflux-1
              pQs(0, iq) = pQs(0, iq) &
                           + sum(pQn(0:k, iq))/n_substeps
              pQs(1:max_age-1, iq) = pQs(1:max_age-1, iq) &
                   + sum(reshape(pQn(k+1:M-n_substeps+k, iq),&
                                 [n_substeps,max_age-1]),1)/n_substeps
              do s=0,numsol-1
                mQs(0, iq, s) = mQs(0, iq, s) &
                           + sum(mQn(0:k, iq, s))/n_substeps
                mQs(1:max_age-1, iq, s) = mQs(1:max_age-1, iq, s)&
                           + sum(reshape(mQn(k+1:M-n_substeps+k, iq, s), &
                                 [n_substeps,max_age-1]),1)/n_substeps
              enddo
            enddo
            do s=0,numsol-1
              mRs(0, s) = mRs(0, s) &
                           + sum(mRn(0:k, s))/n_substeps
              mRs(1:max_age-1, s) = mRs(1:max_age-1, s) &
                           + sum(reshape(mRn(k+1:M-n_substeps+k, s), &
                                 [n_substeps,max_age-1]),1)/n_substeps
            enddo
          endif
        enddo
        if (full_outputs) then
          STn_cum = cumsum(sTn, M)
          ST(1:max_age, i+1) = STn_cum(n_substeps:M:n_substeps)
          WaterBalance(1:max_age-1, i) = diff(ST(1:max_age-1, i), max_age) - &
                   diff(ST(1:max_age, i+1), max_age)
          WaterBalance(0, i) = J(i) * dt - ST(1, i+1)
          do iq=0,numflux-1
            PQ(0:max_age,i+1,iq) = cumsum(pQs(0:max_age-1,iq), max_age)
            WaterBalance(1:max_age-1, i) = WaterBalance(1:max_age-1, i) - dt &
                   * (Q(i,iq) * diff(PQ(1:max_age,i+1,iq), max_age))
            WaterBalance(0, i) = WaterBalance(0, i) - dt * Q(i,iq) * &
                   (PQ(0,i+1,iq) - PQ(0,i+1,iq))
          enddo
          do s=0,numsol-1
            MSn_cum(:,s) = cumsum(mSn(:,s), M)
            MS(:,i+1,s) = MSn_cum(0:M:n_substeps,s)
            MR(:,i+1,s) = cumsum(mRs(:,s), max_age)
            do iq=0,numflux-1
              MQ(:,i+1,iq,s) = cumsum(mQs(:,iq,s),&
                   max_age)
              C_Q(i,iq,s) = C_Q(i,iq,s) + alpha(i,iq,s) * C_old(s) * (1 &
                   - PQ(max_age, i+1, iq))
            enddo
          enddo
          do s=0,numsol-1
            SoluteBalance(1:max_age-1,i,s)=(&
                   diff(MS(0:max_age-1,i,s), max_age) - &
                   diff(MS(1:max_age,i+1,s), max_age) &
                   + dt * diff(MR(1:max_age-1,i+1,s), max_age))
            SoluteBalance(0,i,s) = (C_J(i,s) * J(i) * dt - MS(0,i+1,s) + &
                   dt * (MR(0,i+1,s) - MR(0,i+1,s)))
            do iq=0,numflux-1
              SoluteBalance(0:max_age-1,i,s) = (SoluteBalance(0:max_age-1,i,s&
                   ) - dt * diff(MQ(0:max_age-1,i+1,iq,s), max_age+1))
              SoluteBalance(0,i,s) = (SoluteBalance(0,i,s) - dt * (MQ(0,&
                   i +1,iq,s) - MQ(0,i+1,iq,s)))
            enddo
          enddo
        endif
        if (mod(i,1000).eq.1000) then
            write (debugstring,*) '...Done ', char(i), &
                                  'of', char(timeseries_length)
            call f_verbose(verbose, debugstring)
        endif
      enddo
      call f_verbose(verbose,'...Finished...')
      contains

      subroutine f_debug(debug, debugstring)
      implicit none
      logical, intent(in) :: debug
      character(len=*), intent(in) :: debugstring
      if (debug) then
          print *, debugstring
      endif
      end subroutine f_debug

      subroutine f_verbose(verbose, debugstring)
      implicit none
      logical, intent(in) :: verbose
      character(len=*), intent(in) :: debugstring
      if (verbose) then
          print *, debugstring
      endif
      end subroutine f_verbose

      subroutine lookup(xa, ya, x, y, na, n)
      implicit none
      integer, intent(in) ::  na, n
      real(8), intent(in), dimension(0:na-1) :: xa
      real(8), intent(in), dimension(0:na-1) :: ya
      real(8), intent(in), dimension(0:n-1) :: x
      real(8), intent(out), dimension(0:n-1) :: y
      integer :: i, j, i0
      real(8) :: dif, grad
      logical :: foundit
      i0 = 0
      do j=0,n-1
        if (x(j).le.xa(0)) then
            y(j) = ya(0)
        else
            foundit = .FALSE.
            do i=i0,na-1
                if (x(j).lt.xa(i)) then
                    i0 = i-1
                    foundit = .TRUE.
                    exit
                endif
            enddo
            if (.not. foundit) then
                y(j) = ya(na-1)
            else
                dif = x(j) - xa(i0)
                grad = (ya(i0+1)-ya(i0))/(xa(i0+1)-xa(i0))
                y(j) = ya(i0) + dif * grad
            endif
        endif
      enddo
      end subroutine

      function diff(arr, n)
      implicit none
      integer, intent(in) ::  n
      real(8), intent(in), dimension(0:n-1) :: arr
      real(8), dimension(0:n-1-1) :: diff
      integer :: i
      do i=0,n-1-1
        diff(i) = arr(i+1) - arr(i)
      enddo
      end function diff

      function cumsum(arr, n)
      implicit none
      integer, intent(in) ::  n
      real(8), intent(in), dimension(0:n-1) :: arr
      real(8), dimension(0:n) :: cumsum
      integer :: i
      cumsum(1) = 0
      do i=0,n-1
        cumsum(i+1) = cumsum(i) + arr(i)
      enddo
      end function cumsum

      subroutine get_flux(i, sTt, STt_cum, pQt, PQt_cum, J, Q, rSAS_lookup&
                 , P_list, mSt, mQt, mRt, alpha, k1, C_eq, n_substeps, &
                 numflux, numsol, max_age, timeseries_length, nP_list)
      implicit none
      integer, intent(in) :: i
      integer, intent(in) :: n_substeps, numflux, numsol, max_age, timeseries_length, nP_list
      real(8), intent(in), dimension(0:timeseries_length-1) :: J
      real(8), intent(in), dimension(0:timeseries_length-1,0:numflux-1) :: Q
      real(8), intent(in), dimension(0:nP_list-1) :: P_list
      real(8), intent(in), dimension(0:nP_list-1, 0:timeseries_length-1, 0:numflux-1) :: rSAS_lookup
      real(8), intent(in), dimension(0:timeseries_length-1,0:numflux-1,0:numsol-1) :: alpha
      real(8), intent(in), dimension(0:timeseries_length-1,0:numsol-1) :: k1
      real(8), intent(in), dimension(0:timeseries_length-1,0:numsol-1) :: C_eq
      real(8), intent(inout), dimension(0:max_age*n_substeps) :: PQt_cum
      real(8), intent(inout), dimension(0:max_age*n_substeps) :: STt_cum
      real(8), intent(out), dimension(0:max_age*n_substeps-1, 0:numflux-1) :: pQt
      real(8), intent(in), dimension(0:max_age*n_substeps-1) :: sTt
      real(8), intent(out), dimension(0:max_age*n_substeps-1, 0:numflux-1, 0:numsol-1) :: mQt
      real(8), intent(out), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mRt
      real(8), intent(in), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mSt
      integer M, iq, s 
      M = max_age * n_substeps
      STt_cum = cumsum(sTt, M)
      if (debug) then
        print*, 'get_flux'
        print*, '  sTt', sTt
        print*, '  STt_cum', STt_cum
      endif
      do iq=0,numflux-1
        call lookup(rSAS_lookup(:,i,iq), P_list, STt_cum, &
                 PQt_cum, nP_list, M+1)
        pQt(:,iq) = diff(PQt_cum, M+1)
      if (debug) then
        print*, '  PQt_cum', PQt_cum
        print*, '  pQt', pQt(:,0)
        print*, '  mSt', mSt(:,0)
      endif
        do s=0,numsol-1
          where (sTt(:)>0)
            mQt(:,iq,s) = mSt(:,s) / sTt(:) * alpha(i,iq,s) &
                 * Q(i,iq) * pQt(:,iq)
          elsewhere
            mQt(:,iq,s) = 0.
          end where
        enddo
      enddo
      do s=0,numsol-1
         mRt(:,s) = k1(i,s) * (C_eq(i,s) * sTt - mSt(:,s))
      enddo
      if (debug) then
        print*, '  mEt', mQt(:,1,0)
        print*, '  mQt', mQt(:,0,0)
        print*, '  mRt', mRt(:,0)
      endif
      end subroutine get_flux

      subroutine new_state(i, hr, sTp, sTt, pQt, J, Q, mSp, mSt, mQt, &
                 mRt, C_J, n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list)
      integer, intent(in) :: n_substeps, numflux, numsol, max_age, &
                 timeseries_length, nP_list
      integer, intent(in) :: i
      real(8), intent(in) :: hr
      real(8), intent(in), dimension(0:timeseries_length-1) :: J
      real(8), intent(in), dimension(0:timeseries_length-1, 0:numflux-1) :: Q
      real(8), intent(inout), dimension(0:max_age*n_substeps-1, 0:numflux-1) :: pQt
      real(8), intent(inout), dimension(0:max_age*n_substeps-1) :: sTt
      real(8), intent(inout), dimension(0:max_age*n_substeps-1) :: sTp
      real(8), intent(inout), dimension(0:max_age*n_substeps-1, 0:numflux-1, 0:numsol-1) :: mQt
      real(8), intent(inout), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mSt
      real(8), intent(inout), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mSp
      real(8), intent(inout), dimension(0:max_age*n_substeps-1, 0:numsol-1) :: mRt
      real(8), intent(in), dimension(0:timeseries_length-1, 0:numsol-1) :: C_J
      M = max_age * n_substeps
      sTt = sTp
      do iq=0,numflux-1
        sTt = sTt - Q(i,iq) * pQt(:,iq) * hr
      enddo
      sTt(0) = sTt(0) + J(i) * hr
      do s=0,numsol-1
        mSt(:,s) = mSp(:,s) + mRt(:,s) * hr
        do iq=0,numflux-1
          mSt(:,s) = mSt(:,s) - mQt(:,iq,s) * hr
          mSt(0,s) = mSt(0,s) + J(i) * C_J(i,s) * hr
        enddo
      enddo
      end subroutine new_state

      end subroutine f_solve_RK4
