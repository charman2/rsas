! -*- f90 -*-
      subroutine f_convolve(PQ, C_J, C_old,&
                 max_age, timeseries_length,C_Q, C_Q_raw, observed_fraction)
      implicit none
      integer, intent(in) ::  max_age, timeseries_length
      real(8), intent(in) ::  C_old
      real(8), intent(in), dimension(0:max_age,0:timeseries_length) :: PQ
      real(8), intent(in), dimension(0:timeseries_length-1) :: C_J
      real(8), intent(out), dimension(0:timeseries_length-1) :: C_Q
      real(8), intent(out), dimension(0:timeseries_length-1) :: C_Q_raw
      real(8), intent(out), dimension(0:timeseries_length-1) :: observed_fraction
      real(8), dimension(0:timeseries_length-1) :: pQe
      integer :: maxT, t, a
    C_Q(:) = 0.
    C_Q_raw(:) = 0.
    observed_fraction(:) = 0.
    do t=0,timeseries_length-1
      pQe = diff(PQ(:,t+1),max_age+1)
      maxT = min(t, max_age)
      do a=0,maxT
        C_Q_raw(t) = C_Q_raw(t) + C_J(t-a) * pQe(a)
      enddo
      observed_fraction(t) = PQ(maxT,t)
      C_Q(t) = C_Q_raw(t) + (1-observed_fraction(t)) * C_old
    enddo
    contains

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

      end subroutine f_convolve
