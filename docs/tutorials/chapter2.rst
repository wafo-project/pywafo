.. _cha:rand-loads-stoch:

Random waves and loads
======================

[cha:2]

In this chapter we present some tools for analysis of random functions
with respect to their correlation, spectral, and distributional
properties. We first give a brief introduction to the theory of Gaussian
processes and then we present programs in Wafo, which can be used to
analyse random functions. For deeper insight in the theory we refer to
(Lindgren 2013; Lindgren, Rootzén, and Sandsten 2014).

The presentation will be organized in three examples:
Example `[Ex_sea_statistics] <#Ex_sea_statistics>`__ is devoted to
estimation of different parameters in the model,
Example `[Ex_sea_spectra] <#Ex_sea_spectra>`__ deals with spectral
densities and Example `[Ex_sea_simulation] <#Ex_sea_simulation>`__
presents the use of Wafo to simulate samples of a Gaussian process. The
commands, collected in ``Chapter2.m``, run in less than 5 seconds on a
3.60 GHz 64 bit PC with Windows 10; add another two minutes for the
display of simulated wave fields.

.. _sec:intr-prel-analys:

Introduction and preliminary analysis
-------------------------------------

[sect2.1] The functions we shall analyse can be measured stresses or
strains, which we call loads, or other measurements, where waves on the
sea surface is one of the most important examples. We assume that the
measured data are given in one of the following forms:

#. In the time domain, as measurements of a response function denoted by
   :math:`x(t)`, :math:`0\le t\le T`, where :math:`t` is time and
   :math:`T` is the duration of the measurements. The
   :math:`x(t)`-function is usually sampled with a fixed sampling
   frequency and a given resolution, i.e. the values of :math:`x(t)` are
   also discretised. The effects of sampling can not always be neglected
   in estimation of parameters or distributions. We assume that measured
   functions are saved as a two column ASCII or ``mat`` file.

   Some general properties of measured functions can be summarized by
   using a few simple characteristics. Those are the *mean* :math:`m`,
   defined as the average of all values, the *standard deviation*
   :math:`\sigma`, and the *variance* :math:`\sigma^2`, which measure
   the variability around the mean in linear and quadratic scale. These
   quantities are estimated by

   .. math::

      \begin{aligned}
      m &=&1/T\,\int_0^T x(t)\, \mathrm{d}t, \\
      \sigma^2 &=& 1/T\,\int_0^T (x(t)-m)^2\, \mathrm{d}t,\end{aligned}

   for a continuous recording or by corresponding sums for a sampled
   series.

#. In the frequency domain, as a power spectrum, which is an important
   mode in systems analysis. This means that the signal is represented
   by a Fourier series,

   .. math::

      x(t)\approx
      m + \sum_{i=1}^N a_i\cos(\omega_i\,t)+b_i
      \sin(\omega_i\,t),
      \label{eqn:fourier_series}

   where :math:`\omega_i=i\cdot 2\pi/T` are angular frequencies,
   :math:`m` is the mean of the signal and :math:`a_i,b_i` are Fourier
   coefficients.

#. Another important way to represent a load sequence is by means of the
   *crossing spectrum* or *crossing intensity*, :math:`\mu(u)` = the
   intensity of upcrossings = average number of upcrossings per time
   unit, of a level :math:`u` by :math:`x(t)` as a function of
   :math:`u`, see further in
   Section `3.2.3 <#subsec:crossing_intensity>`__. The *mean frequency*
   :math:`f_0` is usually defined as the number of times :math:`x(t)`
   crosses upwards (upcrosses) the mean level :math:`m` normalized by
   the length of the observation interval :math:`T`,
   i.e. :math:`f_0=\mu(m)`. An alternative definition, [5]_ which we
   prefer to use is that :math:`f_0=\max \mu(u))`, i.e. it is equal to
   the maximum of :math:`\mu(u)`. The *irregularity factor*
   :math:`\alpha` is defined as the mean frequency :math:`f_0` divided
   by the intensity of local maxima (“intensity of cycles”, i.e. the
   average number of local maxima per time unit) in :math:`x(t)`. Note,
   a small :math:`\alpha` means an irregular process,
   :math:`0 < \alpha \leq 1`.

**. *(Sea data)*\ 1em [Ex_sea_statistics][pageseadat] In this example we
use a series with wave data ``sea.dat`` with time argument in the first
column and function values in the second column. The data used in the
examples are wave measurements at shallow water location, sampled with a
sampling frequency of 4 Hz, and the units of measurement are seconds and
meters, respectively. The file ``sea.dat`` is loaded into Matlab and
after the mean value has been subtracted the data are saved in the two
column matrix ``xx``.**

::

         xx = load('sea.dat');
         me = mean(xx(:,2))
         sa = std(xx(:,2))
         xx(:,2) = xx(:,2) - me;
         lc = dat2lc(xx);
         plotflag = 2;
         lcplot(lc,plotflag,0,sa)

Here ``me`` and ``sa`` are the mean and standard deviation of the
signal, respectively. The variable ``lc`` is a two column matrix with
levels in the first column and the number of upcrossing of the level in
the second. In Figure `3.1 <#fig1_cr>`__ the number of upcrossings of
``xx`` is plotted and compared with an estimation based on the
assumption that ``xx`` is a realization of a Gaussian sea.

Next, we compute the mean frequency as the average number of upcrossings
per time unit of the mean level (= 0); this may require interpolation in
the crossing intensity curve, as follows.

::

         T = max(xx(:,1))-min(xx(:,1))
         f0 = interp1(lc(:,1),lc(:,2),0)/T
            % zero up-crossing frequency

.. figure:: fig1_cr
   :alt:  The observed crossings intensity compared with the
   theoretically expected for Gaussian signals, see
   (`[eq:rice] <#eq:rice>`__).
   :name: fig1_cr
   :width: 80mm

   The observed crossings intensity compared with the theoretically
   expected for Gaussian signals, see (`[eq:rice] <#eq:rice>`__).

The process of fatigue damage accumulation depends only on the values
and the order of the local extremes in the load. The sequence of local
extremes is called the *sequence of turning points*. It is a two column
matrix with time for the extremes in the first column and the values of
the extremes in the second.

::

         tp = dat2tp(xx);
         fm = length(tp)/(2*T)            % frequency of maxima
         alfa = f0/fm

Here ``alfa`` is the irregularity factor. Note that ``length(tp)`` is
equal to the number of local maxima and minima and hence we have a
factor 2 in the expression for ``fm``. 1em :math:`\Box`

We finish this section with some remarks about the quality of the
measured data. Especially sea surface measurements can be of poor
quality. It is always good practice to visually examine the data before
the analysis to get an impression of the quality, non-linearities and
narrow-bandedness of the data.

1ex1em [page:spurious] First we shall plot the data ``xx`` and zoom in
on a specific region. A part of the sea data is presented in
Figure `3.2 <#fig2-1>`__ obtained by the following commands.

::

         waveplot(xx,tp,'k-','*',1,1)
         axis([0 2 -inf inf])

.. figure:: figure_Ch2_1
   :alt: A part of the sea data with turning points marked as stars.
   :name: fig2-1
   :width: 80mm

   A part of the sea data with turning points marked as stars.

However, if the amount of data is too large for visual examination, or
if one wants a more objective measure of the quality of the data, one
could use the following empirical criteria:

-  :math:`x'(t) < 5` [m/s], since the raising speed of Gaussian waves
   rarely exceeds 5 [m/s],

-  :math:`x''(t) < 9.81/2`, :math:`[m/s^2]` which is the limiting
   maximum acceleration of Stokes waves,

-  if the signal is constant in some intervals, then this will add high
   frequencies to the estimated spectral density; constant data may
   occur if the measuring device is blocked during some period of time.

To find possible spurious points of the dataset use the following
commands.

::

         dt = diff(xx(1:2,1));
         dcrit = 5*dt;
         ddcrit = 9.81/2*dt*dt;
         zcrit = 0;
         [inds indg] = findoutliers(xx,zcrit,dcrit,ddcrit);

The program will give the following list when used on the sea data.

::

   Found 0 missing points
   Found 0 spurious positive jumps of Dx
   Found 0 spurious negative jumps of Dx
   Found 37 spurious positive jumps of D^2x
   Found 200 spurious negative jumps of D^2x
   Found 244 consecutive equal values
   Found the total of 1152 spurious points

The values for ``zcrit``, ``dcrit`` and ``ddcrit`` can be chosen more
carefully. One must be careful using the criteria for extreme value
analysis, because it might remove extreme waves that belong to the data
and are not spurious. However, small changes of the constants are
usually not so crucial. As seen from the transcripts from the program a
total of 1152 points are found to be spurious which is approximately
12 % of the data. Based on this one may classify the datasets into good,
reasonable, poor, and useless. Obviously, uncritical use of data may
lead to unsatisfactory results. We return to this problem when
discussing methods to reconstruct the data. 1em :math:`\Box`

.. _sec:freq-model-load:

Frequency modelling of load histories
-------------------------------------

.. _powerspectrum:

Power spectrum, periodogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important characteristic of signals of the form
(`[eqn:fourier_series] <#eqn:fourier_series>`__) in frequency domain is
their power spectrum

.. math:: \hat{s}_i=(a_i^2+b_i^2)/(2\Delta\omega),

where :math:`\Delta\omega` is the sampling interval in frequency domain,
i.e. :math:`\omega_i=i\cdot \Delta\omega`. The two-column matrix
:math:`\hat{s}(\omega_i)=(\omega_i,\hat{s}_i)` will be called the *power
spectrum* of :math:`x(t)`. The alternative term *periodogram* was
introduced as early as 1898 by A. Schuster, (Schuster 1898).

The sequence :math:`\theta_i` such that
:math:`\cos \theta_i = a_i/\sqrt{2\, \hat{s}_i\, \Delta\omega}` and
:math:`\sin \theta_i = - b_i/\sqrt{2\, \hat{s}_i\, \Delta\omega}`, is
called a sequence of phases and the Fourier series can be written as
follows:

.. math::

   x(t)\approx m + \sum_{i=1}^N \sqrt{2\, \hat{s}_i\Delta\omega}
   \cos(\omega_i\,t+\theta_i).

If the sampled signal contains exactly :math:`2N+1` points, then
:math:`x(t)` is equal to its Fourier series at the sampled points. In
the special case when :math:`N=2^k`, the so-called FFT (Fast Fourier
Transform) can be used to compute the Fourier coefficients (and the
spectrum) from the measured signal and in reverse the signal from the
Fourier coefficients.

The Fourier coefficient to the zero frequency is just the mean of the
signal, while the variance is given by
:math:`\sigma^2=\Delta\omega\sum \hat{s}(\omega_i)
\approx \int_0^\infty \hat{s}(\omega)\, \mathrm{d}\omega`. The last
integral is called the zero-order spectral moment :math:`m_0`.
Similarly, higher-order spectral moments are defined by

.. math:: m_n=\int_0^\infty \omega^n \, \hat{s}(\omega)\, \mathrm{d}\omega.

.. figure:: fig1_spc
   :alt:  The observed, unsmoothed, spectrum in the data set
   ``sea.dat``.
   :name: fig1_spc
   :width: 80mm

   The observed, unsmoothed, spectrum in the data set ``sea.dat``.

1ex1em We now calculate the spectrum :math:`\widehat{s}(\omega)` for the
sea data signal ``xx``.

::

         Lmax = 9500;
         SS = dat2spec(xx,Lmax);
         plotspec(SS); axis([0 5 0 0.7])

The produced plot, not reproduced in this tutorial, shows the spectrum
:math:`\widehat{s}(\omega)` in blue surrounded by 95% confidence limits,
in red. These limits can be removed by the command

::

         SS = rmfield(SS,'CI')
         plotspec(SS); axis([0 5 0 0.7])

giving Figure `3.3 <#fig1_spc>`__ where we can see that the spectrum is
extremely irregular with sharp peaks at many distinct frequencies. In
fact, if we had analysed another section of the sea data we had found a
similar general pattern, but the sharp peaks had been found at some
other frequencies. It must be understood, that the observed
irregularities are random and vary between measurements of the sea even
under almost identical conditions. This will be further discussed in the
following section, where we introduce smoothing techniques to get a
stable spectrum that represents the “average randomness” of the sea
state.

Next, the spectral moments will be computed.

::

         [mom text] = spec2mom(SS,4)
         [sa sqrt(mom(1))]

The vector ``mom`` now contains spectral moments :math:`m_0, m_2, m_4`,
which are the variances of the signal and its first and second
derivative. We can speculate that the variance of the derivatives is too
high because of spurious points. For example, if there are several
points with the same value, the Gibb’s phenomenon leads to high
frequencies in the spectrum. 1em :math:`\Box`

.. _gaussian-processes-in-spectral-domain-1:

Gaussian processes in spectral domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous section we studied the properties of one specific signal
in frequency domain. Assume now that we get a new series of measurements
of a signal, which we are willing to consider as equivalent to the first
one. However, the two series are seldom identical and differ in some
respect that it is natural to regard as purely random. Obviously it will
have a different spectrum :math:`\hat{s}(\omega)` and the phases will be
changed.

A useful mathematical model for such a situation is the random function
(stochastic process) model which will be denoted by :math:`X(t)`. Then
:math:`x(t)` is seen as particular randomly chosen function. The
simplest model for a stationary signal with a fixed spectrum
:math:`\hat{s}(\omega)` is

.. math::

   X(t)= m + \sum_{i=1}^N \sqrt{\hat{s}_i\, \Delta\omega} \,
   \sqrt{2}\cos(\omega_i\,t+\Theta_i),
   \label{eqn:sum_with_random_phases}

where the phases :math:`\Theta_i` are random variables, independent and
uniformly distributed between :math:`0` and :math:`2 \pi`. However, this
is not a very realistic model either, since in practice one often
observes a variability in the spectrum amplitudes
:math:`\hat{s}(\omega)` between measured functions. Hence,
:math:`\hat{s}_i` should also be modelled to include a certain
randomness.

The best way to accomplish this realistic variability is to assume that
there exists a deterministic function :math:`S(\omega)` such that the
*average value* of :math:`\widehat{s}(\omega_i)\Delta\omega` over many
observed series can be approximated by :math:`S(\omega_i)\Delta\omega`.
In fact, in many cases one can model the variability of
:math:`\hat{s}_i` as

.. math:: \hat{s}_i=R_i^2\cdot S(\omega_i)/2,

where :math:`R_i` are independent random factors, all with a Rayleigh
distribution, with probability density function
:math:`f_R(r)=r \exp (-r^2/2), r > 0`. (Observe that the average value
of :math:`R_i^2` is 2.) This gives the following random function as a
model for the series,

.. math::

   X(t)= m +
   \sum_{i=1}^N \sqrt{S(\omega_i)\, \Delta\omega}\,
   R_i\cos(\omega_i\,t+\Theta_i). \label{discretespectrumprocess}

The process :math:`X(t)` has many useful properties that can be used for
analysis. In particular, for any fixed :math:`t`, :math:`X(t)` is
normally (Gaussian) distributed. Then, the probability of any event
defined for :math:`X(t)` can, in principle, be computed when the mean
:math:`m` and the spectral density :math:`S` are known.

In sea modelling, the components in the sum defining :math:`X(t)` can be
interpreted as individual waves. The assumption that :math:`R_i` and
:math:`\Theta_i` are independent random variables implies that the
individual waves are independent stationary Gaussian processes [6]_ with
mean zero and covariance function given by

.. math:: r_i(\tau) = \Delta\omega \, S(\omega_i) \cos(\omega_i\,\tau).

Consequently, the covariance between :math:`X(t)` and :math:`X(t+\tau)`
is given by

.. math::

   r_X(\tau) = \mbox{\sf E}[(X(t)-m)(X(t+\tau)-m)]=\Delta\omega \,\sum_{i=1}^N  S(\omega_i)
   \cos(\omega_i\,\tau).

A process like `[discretespectrumprocess] <#discretespectrumprocess>`__,
which is the sum of discrete terms, is said to have discrete spectrum.
Its total energy is concentrated to a set of distinct frequencies.

More generally, for a stationary stochastic process with spectral
density :math:`S(\omega)`, the covariance function :math:`r(\tau)` of
the process is defined by its spectral density function
:math:`S(\omega)`, also called power spectrum,

.. math::

   r(\tau) = \mbox{\sf C}[X(t), X(t+\tau)] = \int_0^\infty \cos(\omega\tau)
   \,S(\omega) \, \mathrm{d}\omega. \label{spec2cov}

Since
:math:`\mbox{\sf V}[X(t)] = r_X(0) = \int_0^\infty S(\omega)\, \mathrm{d}\omega`,
the spectral density represents a continuous distribution of the wave
energy over a continuum of frequencies. The spectrum is continuous.

The covariance function and the spectral density form a *Fourier
transform pair*. The spectral density can be computed from the
covariance function by the Fourier inversion formula, which for
continuous-time signals reads,

.. math::

   S(\omega) = \frac{2}{\pi} \int_0^\infty \cos (\omega  \tau) r(\tau)\, \mathrm{d}\tau.
   \label{cov2spec}

The Gaussian process model is particularly useful in connection with
linear filters. If :math:`Y(t)` is the output of a linear filter with
the Gaussian process :math:`X(t)` as input, then :math:`Y(t)` is also
normally distributed. Further, the spectrum of :math:`Y(t)` is related
to that of :math:`X(t)` in a simple way. If the filter has *transfer
function* :math:`H(\omega)`, also called *frequency function*, then the
spectrum of :math:`Y(t)`, denoted by :math:`S_Y`, is given by

.. math:: S_Y(\omega)=|H(\omega)|^2S_X(\omega).

For example, the derivative :math:`X'(t)` is a Gaussian process with
mean zero and spectrum :math:`S_{X'}(\omega)=\omega^2S_X(\omega)`. The
variance of the derivative is equal to the second spectral moment,

.. math::

   \sigma_{X'} ^2=\int S_{X'}(\omega)\, \mathrm{d}\omega =
   \int \omega ^2 S_X(\omega)\, \mathrm{d}\omega = m_2.

.. figure:: fig2_spc
   :alt:  Estimated spectra ``SS1, SS2`` for the data set ``sea.dat``
   with varying degree of smoothing; dash-dotted: ``Lmax1 = 200``,
   solid: ``Lmax2 = 50``.
   :name: fig2_spc
   :width: 80mm

   Estimated spectra ``SS1, SS2`` for the data set ``sea.dat`` with
   varying degree of smoothing; dash-dotted: ``Lmax1 = 200``, solid:
   ``Lmax2 = 50``.

1ex1em In order to estimate the spectrum of a Gaussian process one needs
several realizations of the process. Then, one spectrum estimate can be
made for each realization, which are then averaged. However, in many
cases only one realization of the process is available. In such a case
one is often assuming that the spectrum is a smooth function of
:math:`\omega` and one can use this information to improve the estimate.
In practice, it means that one has to use some smoothing techniques. For
the ``sea.dat`` we shall estimate the spectrum by means of the Wafo
function ``dat2spec`` with a second parameter defining the degree of
smoothing.[page:SS1]

::

         Lmax1 = 200; 
         Lmax2 = 50;
         SS1 = dat2spec(xx,Lmax1);
         SS2 = dat2spec(xx,Lmax2);
         plotspec(SS1,[],'-.'), hold on
         plotspec(SS2), hold off

In Figure `3.4 <#fig2_spc>`__ we see that with decreasing second input
the spectrum estimate becomes smoother, and that in the end it becomes
unimodal.

An obvious question after this exercise is the following: Which of the
two estimates in Figure `3.4 <#fig2_spc>`__ is more correct, in the
sense that it best reflects important wave characteristics? Since the
correlation structure is a very important characteristic, we check which
spectrum can best reproduce the covariance function of the data ``xx``.

The following code in Wafo will compute the covariance for the bimodal
spectral density ``S1`` and compare it with the estimated covariance in
the signal ``xx``.

::

         Lmax = 80;
         R1 = spec2cov(SS1,1);
         Rest = dat2cov(xx,Lmax);
         covplot(R1,Lmax,[],'.'), hold on
         covplot(Rest), hold off

With the unimodal spectrum ``SS2`` instead of ``SS1`` we also compute
the covariance function ``R2 = spec2cov(SS2,1)`` and plot it together
with ``Rest``.

We can see in Figure `[fig4_s2c] <#fig4_s2c>`__\ (a) that the covariance
function corresponding to the spectral density ``SS2`` differs
significantly from the one estimated directly from data. The covariance
corresponding to ``SS1`` agrees much better with the estimated
covariance function, as seen in Figure `[fig4_s2c] <#fig4_s2c>`__\ (b).
Our conclusion is that the bimodal spectrum in
Figure `3.4 <#fig2_spc>`__ is a better model for the data ``sea.dat``
than the unimodal one.   1em :math:`\Box`

Observe that the Wafo function ``spec2cov`` can be used to compute a
covariance structure which can contain covariances both in time and in
space as well as that of the derivatives. The input can be any spectrum
structure, e.g. wavenumber spectrum, directional spectrum or encountered
directional spectrum; type ``help spec2cov`` for detailed information.

.. _subsec:crossing_intensity:

Crossing intensity – Rice’s formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gaussian process is a sum (or in fact an integral) of cosine terms
with amplitudes defined by the spectrum, and the instantaneous value
:math:`X(t)` has a normal distribution with mean :math:`0` and variance
:math:`\sigma^2 = \int S(\omega)\, \mathrm{d}\omega`. The spectral
density :math:`S(\omega)` determines the relative importance of
components with different frequencies.

In wave analysis and fatigue applications there is another quantity that
plays a central role, namely the *upcrossing intensity* :math:`\mu(u)`,
which yields the average number, per time or space unit, of upcrossings
of the level :math:`u`. It contains important information on the fatigue
properties of a load signal and also of the wave character of a random
wave. [7]_

For a Gaussian process the crossing intensity is given by the celebrated
*Rice’s formula*,

.. math:: \mu(u)=f_0\exp\left\{-\frac{(u-m)^2}{2\sigma^2}\right\}.\label{eq:rice}

Using spectral moments we have that :math:`\sigma^2=m_0` while
:math:`f_0=\frac{1}{2\pi}\sqrt{\frac{m_2}{m_0}}` is the mean frequency.

.. _ss:transformedGaussianmodels:

Transformed Gaussian models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard assumptions for a sea state under stationary conditions are
that :math:`X(t)` is a stationary and ergodic stochastic process with
mean :math:`\mbox{\sf E}[X(t)]` assumed to be zero, and with a spectral
density :math:`S(\omega)`. The knowledge of which kind of spectral
densities :math:`S(\omega)` are suitable to describe different sea state
data is well established from experimental studies.

Real data :math:`x(t)` seldom perfectly support the Gaussian assumption
for the process :math:`X(t)`. But since the Gaussian case is well
understood and there are approximative methods to obtain wave
characteristics from the spectral density :math:`S(\omega)` for Gaussian
processes, one often looks for a model of the sea state in the class of
Gaussian processes. Furthermore, in previous work, (Rychlik,
Johannesson, and Leadbetter 1997), we have found that for many sea wave
data, even such that are clearly non-Gaussian, the wavelength and
amplitude densities can be very accurately approximated using the
Gaussian process model.

However, the Gaussian model can lead to less satisfactory results when
it comes to the distribution of crest heights or joint densities of
troughs and crests. In that case we found in (Rychlik, Johannesson, and
Leadbetter 1997) that a simple transformed Gaussian process used to
model :math:`x(t)` gave good approximations for those densities.

Consequently, in Wafo we shall model :math:`x(t)` by a process
:math:`X(t)` which is a function of a single Gaussian process
:math:`\widetilde X (t)`, i.e. 

.. math::

   \label{eq:trprocess}
   X(t)=G(\widetilde X(t)),

where :math:`G(\cdot)` is a continuously differentiable function with
positive derivative. We shall denote the spectrum of :math:`X` by
:math:`S`, and the spectrum of :math:`\widetilde X (t)` by
:math:`\widetilde S`. The transformation :math:`G` performs the
appropriate non-linear translation and scaling so that
:math:`\widetilde X(t)` is always normalized to have mean zero and
variance one, i.e. the first spectral moment of :math:`\widetilde S` is
one.

Note that once the distributions of crests, troughs, amplitudes or
wavelengths in a Gaussian process :math:`\widetilde X (t)` are computed,
then the corresponding wave distributions in :math:`X(t)` are obtained
by a simple variable transformation involving only the inverse of
:math:`G`, which we shall denote by :math:`g`. Actually we shall use the
function :math:`g` to define the transformation instead of :math:`G`,
and use the relation :math:`\widetilde x
(t) = g(x(t))` between the real sea data :math:`x(t)` and the
transformed data :math:`\widetilde x (t)`. If the model in
Eq. (`[eq:trprocess] <#eq:trprocess>`__) is correct, then
:math:`\widetilde x(t)` should be a sample function of a process with
Gaussian marginal distributions.

There are several different ways to proceed when selecting a
transformation. The simplest alternative is to estimate the function
:math:`g` directly from data by some parametric or non-parametric
techniques. A more physically motivated procedure is to use some of the
parametric functions proposed in the literature, based on approximations
of non-linear wave theory. The following options are programmed in the
toolbox:

::

         dat2tr    - non-parametric transformation g proposed by Rychlik,
         hermitetr - transformation g proposed by Winterstein,
         ochitr    - transformation g proposed by Ochi et al.

The transformation proposed by by Ochi et al., (Ochi and Ahn 1994), is a
monotonic exponential function, while Winterstein’s model, (Winterstein
1988), is a monotonic cubic Hermite polynomial. Both transformations use
higher moments of :math:`X(t)` to compute :math:`g`. Information about
the moments of the process can be obtained by site specific data,
laboratory measurements or from physical considerations. Rychlik’s
non-parametric method is based on the crossing intensity :math:`\mu(u)`;
see (Rychlik, Johannesson, and Leadbetter 1997). Martinsen and
Winterstein, (Marthinsen and Winterstein 1992), derived an expression
for the skewness and kurtosis for narrow banded Stokes waves to the
leading order and used these to define the transformation. The skewness
and kurtosis (excess) of this model can also be estimated from data by
the Wafo functions ``skew`` and ``kurt``.

1ex1em We begin with computations of skewness and kurtosis for the data
set ``xx``. The commands

::

         rho3 = skew(xx(:,2))
         rho4 = kurt(xx(:,2))

give the values ``rho3 = 0.25`` and ``rho4 = 3.17``, respectively,
compared to ``rho3 = 0`` and ``rho4 = 3`` for Gaussian waves. We can
compute the same model for the spectrum :math:`\tilde S` using the
second order wave approximation proposed by Winterstein. His
approximation gives suitable values for skewness and kurtosis

::

         [sk, ku] = spec2skew(S1);

Here we shall use Winterstein’s Hermite transformation and denote it by
``gh``, and compare it with the linear transformation, denoted by ``g``,
that only has the effect to standardize the signal, assuming it is
already Gaussian,

::

         gh = hermitetr([],[sa sk ku me]);
         g  = gh; g(:,2)=g(:,1)/sa;
         trplot(g)

These commands will result in two two-column matrices, ``g, gh``, with
equally spaced :math:`y`-values in the first column and the values of
:math:`g(y)` in the second column.

Since we have data we may estimate the transformation directly by the
method proposed by Rychlik et al., in (Rychlik, Johannesson, and
Leadbetter 1997):

::

         [glc test0 cmax irr gemp] = dat2tr(xx,[],'plotflag',1);
         hold on
         plot(glc(:,1),glc(:,2),'b-')
         plot(gh(:,1),gh(:,2),'b-.'), hold off

The same transformation can be obtained from data and the crossing
intensity by use of the Wafo functions ``dat2lc`` followed by ``lc2tr``.

.. figure:: fig4_tr
   :alt:  Comparisons of the three transformations :math:`g`, straight
   line is the Gaussian model, dash dotted line the Hermite
   transformation ``gh`` and solid line the Rychlik method ``glc``.
   :name: fig4_tr
   :width: 80mm

   Comparisons of the three transformations :math:`g`, straight line is
   the Gaussian model, dash dotted line the Hermite transformation
   ``gh`` and solid line the Rychlik method ``glc``.

In Figure `3.5 <#fig4_tr>`__ we compare the three transformations, the
straight line is the Gaussian linear model, the dash dotted line is the
Hermite transformation based on higher moments of the response computed
from the spectrum and the solid line is the direct transformation
estimated from crossing intensity. (The unsmoothed line shows the
estimation of the direct transformation from unsmoothed crossing
intensity). We can see that the transformation derived from crossings
will give the highest crest heights. It can be proved that
asymptotically the transformation based on crossings intensity gives the
correct density of crest heights.

The transformations indicate that data ``xx`` has a light lower tail and
heavy upper tail compared to a Gaussian model. This is also consistent
with second order wave theory, where the crests are higher and the
troughs shallower compared to Gaussian waves. Now the question is
whether this difference is significant compared to the natural
statistical variability due to finite length of the time series.

.. figure:: figure_Ch2_8
   :alt:  The simulated 50 values of the ``test`` variable for the
   Gaussian process with spectrum ``S1`` compared with the observed
   value (dashed line).
   :name: fig2-3
   :width: 80mm

   The simulated 50 values of the ``test`` variable for the Gaussian
   process with spectrum ``S1`` compared with the observed value (dashed
   line).

To determine the degree of departure from Gaussianity, we can compare an
indicator of non-Gaussianity ``test0`` obtained from Monte Carlo
simulation. The value of ``test0`` is a measure of how munch the
transformation ``g`` deviates from a straight line.

The significance test is done by simulating 50 independent samples of
``test0`` from a true Gaussian process with the same spectral density
and length as the original data. This is accomplished by the Wafo
program ``testgaussian``. The output from the program is a plot of the
ratio ``test1`` between the simulated (Gaussian) ``test0``-values and
the sample ``test0``, that was calculated in the previous call to
``dat2tr``:

::

         N = length(xx);
         test1 = testgaussian(S1,[N,50],test0);

The program gives a plot of simulated ``test`` values, see
Figure `3.6 <#fig2-3>`__. As we see from the figure none of the
simulated values of ``test1`` is above 1.00. Thus the data significantly
departs from a Gaussian distribution; see (Rychlik, Johannesson, and
Leadbetter 1997) for more detailed discussion of the testing procedure
and the estimation of the transformation ``g`` from the crossing
intensity.

We finish the tests for Gaussianity of the data by a more classical
approach and simply plot the data on normal probability paper. Then
:math:`N` independent observations of identically distributed Gaussian
variables form a straight line in a normalplot. Now, for a time series
the data is clearly not independent. However, if the process is ergodic
then the data forms a straight line as :math:`N` tends to infinity.

The command

::

         plotnorm(xx(:,2))

produces Figure `3.7 <#fig2-4>`__. As we can see the normal probability
plot is slightly curved indicating that the underlying distribution has
a heavy upper tail and a light lower tail. 1em :math:`\Box`

.. figure:: figure_Ch2_2
   :alt: The data ``sea.dat`` on normal probability plot.
   :name: fig2-4
   :width: 80mm

   The data ``sea.dat`` on normal probability plot.

.. _spectral-densities-of-sea-data-1:

Spectral densities of sea data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The knowledge of which kind of spectral density :math:`S(\omega)` is
suitable to describe sea state data is well established from
experimental studies. One often uses some parametric form of spectral
density functions, e.g. a Jonswap-spectrum. This formula is programmed
in a Wafo function ``jonswap``, which evaluates the spectral density
:math:`S(\omega)` with specified wave characteristics. There are several
other programmed spectral densities in Wafo to allow for bimodal and
finite water depth spectra. The list includes the following spectra:

::

         jonswap       - JONSWAP spectral density
         wallop        - Wallop spectral density
         ochihubble    - Bimodal Ochi-Hubble spectral density
         torsethaugen  - Bimodal (swell + wind) spectral density
         bretschneider - Bretschneider (Pierson-Moskowitz)
                         spectral density
         mccormick     - McCormick spectral density
         tmaspec       - JONSWAP spectral density
                         for finite water depth

Wafo also contains some different spreading functions; use the help
function on ``spec`` and ``spreading`` for more detailed information.

The spectrum of the sea can be given in many different formats, that are
interconnected by the dispersion relation [8]_. The spectrum can be
given using frequencies, angular frequencies or wavenumbers, and it can
also be directional.

A related spectrum is the encountered spectrum for a moving vessel. The
transformations between the different types of spectra are defined by
means of integrals and variable change defined by the dispersion
relation and the Doppler shift of individual waves. The function
``spec2spec`` makes all these transformations easily accessible for the
user. (Actually many programs perform the appropriate transformations
internally whenever it is necessary and for example one can compute the
density of wave-length, which is a quantity in space domain, from an
input that is the directional frequency spectrum, which is related to
the time domain.)

.. figure:: figure_spec_enc
   :alt:  Directional spectrum ``SJd`` of Jonswap sea (dashed line)
   compared with the encountered directional spectrum ``SJe`` for
   heading sea, speed 10 [m/s] (solid line).
   :name: fig4-5
   :width: 80mm

   Directional spectrum ``SJd`` of Jonswap sea (dashed line) compared
   with the encountered directional spectrum ``SJe`` for heading sea,
   speed 10 [m/s] (solid line).

**. *(Different forms of spectra)*\ 1em [Ex_sea_spectra] In this example
we have chosen a Jonswap spectrum with parameters defined by significant
wave height ``Hm0 = 7[m]`` and peak period ``Tp = 11[s]``. This spectrum
describes the measurements of sea level at a fixed point (buoy).**

::

         Hm0 = 7; Tp = 11;
         SJ = jonswap([],[Hm0 Tp]);
         SJ.note

In order to include the space dimension, i.e. the direction in which the
waves propagate, we compute a directional spectrum by adding spreading;
see dashed curves in Figure `3.8 <#fig4-5>`__.

::

         D = spreading(101,'cos2s',0,[],SJ.w,1)
         SJd = mkdspec(SJ,D)  % Directional spectrum

Next, we consider a vessel moving with speed ``10[m/s]`` against the
waves. The sea measured from the vessel will have a different
directional spectrum, called the encountered directional spectrum. The
following code will compute the encountered directional spectrum and
plot it on top of the original spectrum. The result is shown as the
solid curves in Figure `3.8 <#fig4-5>`__.

::

         SJe = spec2spec(SJd,'encdir',0,10);  % Encountered dir spectrum
         plotspec(SJe), hold on
         plotspec(SJd,1,'--'), hold off

Obviously, the periods of waves in the directional sea are defined by
the Jonswap spectrum (in a linear wave model spreading does not affect
the sea level at a fixed point), but the encountered periods will be
shorter with heading seas. This can be seen by comparing the original
Jonswap spectrum ``SJ`` with the following two point spectra.

::

         SJd1 = spec2spec(SJd,'freq'); % Point spectrum from dir spectrum
         SJd2 = spec2spec(SJe,'enc');  % Point encountered spectrum at ship
         plotspec(SJ), hold on
         plotspec(SJd1,1,'.'),
         plotspec(SJd2), hold off

We can see in Figure `[fig4-4] <#fig4-4>`__\ (a) that the spectra ``SJ``
and ``SJd1`` are identical (in numerical sense), while spectrum ``SJd2``
contains more energy at higher frequencies.

A similar question is how much the wave length differs between a
longcrested Jonswap sea and a Jonswap sea with spreading. The answer
lies in the *wavenumber spectrum* that measures how the wave enegy is
distributed over different wavenumbers, i.e. radians (or cycles) per
unit length along a specified direction, usually the main wave
direction.

The wavenumber spectra along the main wave direction can be computed by
the following code, the result of which is shown in
Figure `[fig4-4] <#fig4-4>`__\ (b).

::

         SJk = spec2spec(SJ,'k1d')  % Unidirectional waves
         SJkd = spec2spec(SJd,'k1d')  % Waves with directional spreading 
         plotspec(SJk), hold on
         plotspec(SJkd,1,'--'), hold off

We see how the spreading of energy away from the main direction makes
observed waves longer. The total energy is of course the same but
shifted to lower wavenumbers.

Finally, we shall show how the Jonswap spectrum can be corrected for a
finite depth, see (Buows et al. 1985) for a theoretical and empirical
study. The Wafo function ``phi1`` computes the frequency dependent
reduction of the spectrum for waters of finite depth, here 20 meters.

::

         plotspec(SJ,1,'--'), hold on
         SJ20 = SJ;
         SJ20.S = SJ20.S.*phi1(SJ20.w,20); % Finite depth correction
         SJ20.h = 20;
         plotspec(SJ20),  hold off

The resulting spectra are shown in Figure `3.9 <#fig4-1>`__. 1em
:math:`\Box`

.. figure:: figure_spec
   :alt:  Standard Jonswap spectrum ``SJ`` (dashed line) compared with
   the spectrum ``SJ20`` on finite depth of 20 [m] (solid line)
   :name: fig4-1
   :width: 80mm

   Standard Jonswap spectrum ``SJ`` (dashed line) compared with the
   spectrum ``SJ20`` on finite depth of 20 [m] (solid line)

.. _sec:simulationofGaussian:

Simulation of transformed Gaussian process
------------------------------------------

In this section we shall present some of the programs that can be used
to simulate the transformed Gaussian model for sea
:math:`X(t)=G(\widetilde X(t))`. In Wafo there are several other
programs to simulate random functions or surfaces, both Gaussian and
non-Gaussian; use ``help simtools``. We give examples of some of these
functions in Section `3.4 <#s:2.4>`__.

The first important case is when we wish to reproduce random versions of
the measured signal :math:`x(t)`. Using ``dat2tr`` one first estimates
the transformation ``g``. Next, using a function ``dat2gaus`` one can
compute :math:`\tilde x(t)=g(x(t))`, which we assume is a realization of
a Gaussian process. From :math:`\tilde x` we can then estimate the
spectrum :math:`\tilde S(\omega)` by means of the function ``dat2spec``.
The spectrum :math:`\tilde S(\omega)` and the transformation :math:`g`
will uniquely define the transformed Gaussian model. A random function
that models the measured signal can then be obtained using the
simulation program ``spec2sdat``, which includes the desired
transformation. In the following example we shall illustrate this
approach on the data set ``sea.dat``.

Before we can start to simulate we need to put the transformation into
the spectrum data structure, which is a Matlab structure variable; see
Section `[sec:datastructures] <#sec:datastructures>`__, page . Since
Wafo is based on transformed Gaussian processes the entire process
structure is defined by the spectrum and the transformation together.
Therefore the transformation has been incorporated, as part of a model,
into the spectrum structure, and is passed to other Wafo programs via
the spectrum. If no transformation is given then the process is
Gaussian.

Observe that the possibly nonzero mean ``m`` for the model is included
in the transformation. The change of mean by for example 0.5 [m] is
simply accomplished by modifying the transformation by the command
``g(:,2) = g(:,2)+0.5;`` Consequently the spectrum structure completely
defines the model.

**Important note 1:** [ImpNote_1] When the simulation routine
``spec2sdat`` is called with a spectrum argument that contains a scale
changing transformation ``spectrum.tr``, then it is assumed that the
input spectrum is standardized with spectral moment :math:`m_0=1`,
i.e. unit variance. The correct standard deviation for the output should
normally be obtained via the transformation ``spectrum.tr``. If you
happen to use a transformation *together with* an input spectrum that
does not have unit variance, then you get the double scale effect, both
from the transformation and via the standard deviation from the
spectrum. It is only the routine ``spec2sdat`` that works in this way.
All other routines, in particular those which calculate cycle
distributions, perform an internal normalization of the spectrum before
the calculation, and then transforms back to the original scale at the
end.

**Important note 2:** When you run the simulation examples with
different time horizon and time step you may experience a warning

``’Spectrum matrix is very large’``.

This is a warning from the Wafo routine ``specinterp``, which is used
internally to adapt the spectrum to the correct Nyquist frequency and
frequency resolution. You can turn off the warning by commenting out
three lines in ``specinterp`` in the ``spec`` module.

**. *(Simulation of a random sea)*\ 1em [Ex_sea_simulation] In
Example **\ `[Ex_sea_statistics] <#Ex_sea_statistics>`__\ **on page  we
have shown that the data set ``xx = sea.dat`` contains a considerable
amount of spurious points that we would like to omit or censor.**

The program ``reconstruct`` replaces the spurious data by synthetic
data. The new data points are obtained by simulation of a conditional
Gaussian vector, based on the remaining data, taking the fitted
transformed Gaussian process model into account; see (Brodtkorb,
Myrhaug, and Rue 1999, 2001) for more details. The reconstruction is
performed as

::

         [y grec] = reconstruct(xx,inds);

where ``y`` is the reconstructed data and ``grec`` is the transformation
estimated from the signal ``y``. In Figure `3.10 <#fig4-11>`__ we can
see the transformation (solid line) compared with the empirical smoothed
transformation, ``glc``, which is obtained from the original sequence
``xx`` without removing the spurious data (dash-dotted line). We can see
that the new transformation has slightly smaller crests. Actually, it is
almost identical with the transformation ``gh`` computed from the
spectrum of the signal, however, it may be only a coincident (due to
random fluctuations) and hence we do not draw any conclusions from this
fact.

.. figure:: fig4grec
   :alt:  The transformation computed from the reconstructed signal
   ``y`` (solid line) compared with the transformation computed from the
   original signal ``xx`` (dashed dotted line).
   :name: fig4-11
   :width: 80mm

   The transformation computed from the reconstructed signal ``y``
   (solid line) compared with the transformation computed from the
   original signal ``xx`` (dashed dotted line).

The value of the ``test`` variable for the transformation ``grec`` is
0.84 and, as expected, it is smaller than the value of ``test0`` = 1.00
computed for the transformation ``glc``. However, it is still
significantly larger then the values shown in Figure `3.6 <#fig2-3>`__,
i.e. the signal ``y`` is not a Gaussian signal.

We turn now to estimation of the spectrum in the model from the
simulated data but first we transform data to obtain a sample
:math:`\tilde x(t)`:

::

         L = 200
         x = dat2gaus(y,grec); %Gaussian process from reconstructed data
         SSx = dat2spec(x,L);  %Spectrum from Gaussian reconstructed process

The important remark here is that the smoothing of the spectrum defined
by the parameter ``L``, see ``help dat2spec``, removes almost all
differences between the spectra in the three signals ``xx``, ``y``, and
``x``. (The spectrum ``SSx`` is normalized to have first spectral moment
one and has to be scaled down to have the same energy as the spectrum
``SS1`` on page .)

Next, we shall simulate a random function equivalent to the
reconstructed measurements ``y``. The Nyquist frequency gives us the
time sampling of the simulated signal,

::

         dt = spec2dt(Sx)

and is equal to 0.25 seconds, since the data has been sampled with a
sampling frequency of 4 Hz. We then simulate 2 minutes
(:math:`2\times 60\times 4` points) of the signal, to obtain a synthetic
wave equivalent to the reconstructed non-Gaussian sea data.

::

         Ny = fix(2*60/dt)  % = two minutes
         SSx.tr = grec;
         ysim = spec2sdat(SSx,Ny);
         waveplot(ysim,'-')

The result is shown in Figure `3.11 <#fig_2minutes>`__. 1em :math:`\Box`

.. figure:: fig_2minutes
   :alt:  Two minutes of simulated sea data, equivalent to the
   reconstructed data.
   :name: fig_2minutes
   :width: 80mm

   Two minutes of simulated sea data, equivalent to the reconstructed
   data.

In the next example we consider a signal with a given theoretical
spectrum. Here we have a problem whether the theoretical spectrum is
valid for the transformed Gaussian model, i.e. it is a spectrum
:math:`S(\omega)` or is it the spectrum of the linear sea
:math:`\widetilde S`. In the previous example the spectrum of the
transformed process was almost identical with the normalized spectrum of
the original signal. In (Rychlik, Johannesson, and Leadbetter 1997) it
was observed that for sea data the spectrum estimated from the original
signal and that for the transformed one do not differ significantly.
Although more experiments should be done in order to recommend using the
same spectrum in the two cases, here, if we wish to work with
non-Gaussian models with a specified transformation, we shall derive the
:math:`\widetilde S` spectrum by dividing the theoretical spectrum by
the square root of the first spectral moment of :math:`S`.

1ex1em Since the spectrum ``SS1`` in Figure `3.4 <#fig2_spc>`__ is
clearly two-peaked with peak frequency :math:`T_p = 1.1` [Hz] we choose
to use the Torsethaugen spectrum. (This spectrum is derived for a
specific location and we should not expect that it will work well for
our case.) The inputs to the programs are :math:`T_p` and :math:`H_s`,
which we now compute.

::

         Tp = 1.1;
         H0 = 4*sqrt(spec2mom(S1,1))
         ST = torsethaugen([0:0.01:5],[H0  2*pi/Tp]);
         plotspec(SS1), hold on
         plotspec(ST,[],'-.')

In Figure `3.12 <#fig4-12>`__, we can see that the Torsethaugen spectrum
has too little energy on the swell peak. Despite this fact we shall use
this spectrum in the rest of the example.

.. figure:: fig4spec
   :alt:  Comparison between the estimated spectrum ``SS1`` in the
   signal ``sea.dat`` (solid line) and the theoretical spectrum ``ST``
   of the Torsethaugen type (dash-dotted line).
   :name: fig4-12
   :width: 80mm

   Comparison between the estimated spectrum ``SS1`` in the signal
   ``sea.dat`` (solid line) and the theoretical spectrum ``ST`` of the
   Torsethaugen type (dash-dotted line).

We shall now create the spectrum :math:`\tilde S(\omega)` (=
``STnorm``), i.e. the spectrum for the standardized Gaussian process
:math:`{\widetilde X}(t)` with standard deviation equal to one.

::

         STnorm = ST;
         STnorm.S = STnorm.S/sa^2;
         dt = spec2dt(STnorm)

The sampling interval ``dt`` = 0.63 [s] (= :math:`\pi / 5`), is a
consequence of our choice of cut off frequency in the definition of the
``ST`` spectrum. This will however not affect our simulation, where any
sampling interval ``dt`` can be used.

Next, we recompute the theoretical transformation ``gh``.

::

         [Sk Su] = spec2skew(ST);
         sa = sqrt(spec2mom(ST,1));
         gh = hermitetr([],[sa sk ku me]);
         STnorm.tr = gh;

The transformation is actually almost identical to ``gh`` for the
spectrum ``SS1``, which can be seen in Figure `3.5 <#fig4_tr>`__, where
it is compared to the Gaussian model ``g``, given by a straight line. We
can see from the diagram that the waves in a transformed Gaussian
process :math:`X(t)=G({\widetilde X}(t))`, will have an excess of high
crests and shallow troughs compared to waves in the Gaussian process
:math:`\widetilde X(t)`. The difference is largest for extreme waves
with crests above 1.5 meters, where the excess is 10 cm, ca 7 %. Such
waves, which have crests above three standard deviations, are quite rare
and for moderate waves the difference is negligible.

In order to illustrate the difference in distribution for extreme waves
we will simulate a sample of 4 minutes of :math:`X(t)` with sampling
frequency 2 Hz. The result is put into ``ysim_t``. In order to obtain
the corresponding sample path of the process :math:`\widetilde X` we use
the transformation ``gh``, stored in ``STnorm.tr``, and put the result
in ``xsim_t``.

::

         dt = 0.5;
         ysim_t = spec2sdat(STnorm,240,dt);
         xsim_t = dat2gaus(ysim_t,STnorm.tr);

Since the process :math:`\tilde X(t)` always has variance one, in order
to compare the Gaussian and non-Gaussian models we scale ``xsim_t`` to
have the same second spectral moment as ``ysim_t``, which will be done
by the following commands:

::

         xsim_t(:,2) = sa*xsim_t(:,2);
         waveplot(xsim_t,ysim_t,5,1,sa,4.5,'r.','b')

.. figure:: figure_sim1
   :alt:  Simulated :math:`X(t)=G(\widetilde X(t))` (solid line)
   compared with :math:`\widetilde X(t)` (dots) scaled to have the same
   :math:`H_s` as :math:`X(t)` for a theoretical spectrum given by
   Torsethaugen spectrum ``St``.
   :name: fig4-2
   :width: 80mm

   Simulated :math:`X(t)=G(\widetilde X(t))` (solid line) compared with
   :math:`\widetilde X(t)` (dots) scaled to have the same :math:`H_s` as
   :math:`X(t)` for a theoretical spectrum given by Torsethaugen
   spectrum ``St``.

In Figure `3.13 <#fig4-2>`__ we have waves that are not extremely high
and hence the difference between the two models is hardly noticeable in
this scale. Only in the second subplot we can see that Gaussian waves
(dots) have troughs deeper and crests lower than the transformed
Gaussian model (solid line). This also indicates that the amplitude
estimated from the transformed Gaussian and Gaussian models are
practically identical. Using the empirical transformation ``glc``
instead of the Hermite transformation ``gh`` would give errors of ca
11%, which for waves with higher significant wave height would give
considerable underestimation of the crest height of more extreme waves.
Even if the probability for observing an extreme wave during the period
of 20 minutes is small, it is not negligible for safety analysis and
therefore the choice of transformation is one of the most important
questions in wave modelling.

Since the difference between Gaussian and non-Gaussian model is not so
big we may ask whether 20 minutes of observation of a transformed
Gaussian process presented in this example is long enough to reject the
Gaussian model. Using the function ``testgaussian`` we can see that
rejection of Gaussian model would occur very seldom. Observe that the
``sea.dat`` is 40 minutes long and that we clearly rejected the Gaussian
model. 1em :math:`\Box`

.. _s:2.4:

More on simulation with Wafo
----------------------------

The Wafo toolbox contains additional routines for simulation of random
loads and random waves. One important class used in fatigue analysis and
in modelling the long term variability of sea state are the Markov
models, in which the model parameters are allowed to switch to new
values according to a Markov chain. Another group of simulation routines
generate non-linear waves and loads, like second order Stokes waves with
interaction between frequency components in the Gaussian wave model, and
Lagrange waves, with a horizontal deformation of space. This group
includes a program to simulate the output of second order oscillators
with nonlinear spring, when external force is white noise. The nonlinear
oscillators can be used to model nonlinear responses of sea structures.

.. _markov-models-1:

Markov models
~~~~~~~~~~~~~

The following routines from the ``simtools`` module can be used to
generate realistic load sequencies for for reliability and fatigue
analysis; see more in Chapter `[cha:5] <#cha:5>`__.

``lc2sdat``
   Simulates a load process with specified crossing spectrum and
   irregularity

``mcsim``
   Simulates a finite Markov chain from its probability transition
   matrix

``mctpsim``
   Simulates a Markov chain of turning points from transition matrix

``sarmasim``
   Simulates an ARMA time series with Markov switching parameters

``smcsim``
   Simulates a Markov chain with Markov switching regime

**. *(Switching ARMA model)*\ 1em [SARMA] In many applications, the
standard stationary time series Auto-regressive/Moving average
ARMA-model, (Lindgren, Rootzén, and Sandsten 2014 Ch. 7), can be made
more realistic if the parameters are allowed to change with time,
according to abrupt changes in environment conditions. Examples are
found in econometrics (Hamilton 1989), climate and environmental
research (Ailliot and Monbet 2012), automotive safety (Johannesson
1998), and many other areas. A special case are the *Hidden Markov
models* where the changes occur according to an (unobserved) Markov
chain**

The Wafo routine ``sarmasim`` was developed by Johannesson (Johannesson
1998) in order to find a good model for stress loads on a truck serving
on an irregular scheme, loading and unloading gravel. The following
example from ``sarmasim`` illutrates the principles on an
ARMA(4,2)-process swhitching by a Markov *regime process* between two
states.

::

       p1 = 0.005; p2=0.003;  % Switching probabilities
       P = [1-p1 p1; p2 1-p2];  % Markov transition matrix
       C = [1.00 1.63 0.65; 1.00 0.05 -0.88];  % MA-parameters 
       A = [1.00 -0.55 0.07 -0.26 -0.02; ... 
       		   1.00 -2.06 1.64 -0.98 0.41];  % AR-parameters
       m = [46.6; 7.4]*1e-3;  % Mean values for sub-processes
       s2 = [0.5; 2.2]*1e-3;  % Innovation variances
       [x,z] = sarmasim(C,A,m,s2,P,2000);  % Simulate 2000 steps
       plothmm(x,z)  % Ploting
       		   %  x = Switching ARMA, z = Markov regime process

Figure `3.14 <#fig:sarmasim>`__ shows the hidden Markov states and the,
seemingly non-stationary, ARMA-process. In fact, the Markov chain is
simulated from its stationary distribution and each ARMA-section starts
with the stationary distribution for that particular ARMA-process, so
the process is stationary! 1em :math:`\Box`

.. figure:: sarmasim
   :alt: Switching ARMA-process with Markov regime process.
   :name: fig:sarmasim
   :width: 80mm

   Switching ARMA-process with Markov regime process.

.. _space-time-and-non-linear-waves-and-loads-1:

Space-time and non-linear waves and loads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following routines in modules ``simtools`` and ``lagrange`` generate
continuous time series and continuous space-time realizations.

``duffsim``
   Generates a sample path of a linear or non-linear random oscillator

``spec2wave`` and ``spec2field``
   Generate samples of space-time (transformed) 2D and 3D Gaussian waves

``seamovie``
   Makes a movie in time of 2D or 3D random waves and optionally exports
   movie in ``avi`` format

``seasim``
   Old routine that generates a space-time Gaussian process and movie
   with 1D or 2D space parameter

``spec2ldat`` and ``spec2ldat3D``
   Routines in module ``lagrange`` that generate front-back and
   crest-trough asymmetric modifications of Gaussian wave fields

``spec2nlsdat``
   Simulates a random 2nd order non-linear wave from a spectral density

**. *(How to export a sea movie)*\ 1em [Seamovieforexport] In
Section **\ `[ss:dirspec] <#ss:dirspec>`__\ **we gave an example of how
to use ``spec2field`` to generate and plot a Gaussian sea surface with
directional spreading. We now show how to use ``seamovie`` to generate
and save a movie of the time evolution of the sea.**

We define and plot the ``demospec`` spectrum, with frequency independent
spreading

::

       SD = demospec('dir')
       plotspec(SD)

to get Figure `3.15 <#fig:demospec>`__ and the structure

::

   SD = 
     struct with fields:
             S: [101 x 257 double]
           w: [257 x 1 double]
       theta: [101 x 1 double]
          tr: []
           h: Inf
        type: 'dir'
         phi: 0
        norm: 0
        note: 'Demospec: JONSWAP, Hm0 = 7, Tp = 11, gamma = 2.3853; Sprea...'

.. figure:: fig_demospec
   :alt: Directional spectrum ``demospec``
   :name: fig:demospec
   :width: 50.0%

   Directional spectrum ``demospec``

To generate two minutes of a wave field of size 500[m] by 250[m] we
define parameters by the function ``simoptset``, generate the field with
``spec2field``, and make three different types of movies with
``seamovie``.

::

       opt = simoptset('Nt',600,'dt',0.2,'Nu',501','du',1,'Nv',251,'dv',1)
       rng('default')
       W = spec2field(SD,opt);  % structure with fields .Z, .x, .y, .t
       figure(2); Mv2 = seamovie(W,2); pause
       figure(3); Mv3 = seamovie(W,3); pause
       figure(1); Mv1 = seamovie(W,1,'sea.avi')

The last movie command saves the movie as ``sea.avi`` in the working
directory. If ``sea.avi`` already exists, the new movie file is given a
random name. 1em :math:`\Box`

.. figure:: Mv1
   :alt: Three aspects of a simulated Gaussian wave field with
   directional spectrum ``demospec(’dir’)``; last frames in ``Mv1``,
   ``Mv1``, and ``Mv3``.
   :name: Fig:threefields
   :width: 130mm

   Three aspects of a simulated Gaussian wave field with directional
   spectrum ``demospec(’dir’)``; last frames in ``Mv1``, ``Mv1``, and
   ``Mv3``.

.. figure:: Mv2
   :alt: Three aspects of a simulated Gaussian wave field with
   directional spectrum ``demospec(’dir’)``; last frames in ``Mv1``,
   ``Mv1``, and ``Mv3``.
   :name: Fig:threefields
   :width: 130mm

   Three aspects of a simulated Gaussian wave field with directional
   spectrum ``demospec(’dir’)``; last frames in ``Mv1``, ``Mv1``, and
   ``Mv3``.

.. figure:: Mv3
   :alt: Three aspects of a simulated Gaussian wave field with
   directional spectrum ``demospec(’dir’)``; last frames in ``Mv1``,
   ``Mv1``, and ``Mv3``.
   :name: Fig:threefields
   :width: 130mm

   Three aspects of a simulated Gaussian wave field with directional
   spectrum ``demospec(’dir’)``; last frames in ``Mv1``, ``Mv1``, and
   ``Mv3``.

**..5em The wave fields generated by ``spec2field`` differ from animated
wave fields used for example in feature films. They are intended to be
used in applications where the fine structure, so important for the
visual impression, can be neglected. Examples involve extreme value
analysis, fatigue loads, and ship dynamics. 1em :math:`\Box`**

.. _structure-of-spectral-simulation-1:

Structure of spectral simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table explaines the sequencing of spectral simulation
routines in modules ``simtools`` and ``lagrange``. A (freq) or (dir)
indicates the type of input spectrum.

90

+-------------+-------------+-------------+-------------+-------------+
|             |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
|             |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Routine     | ``          | ``          | ``s         | ``sp        |
|             | spec2sdat`` | spec2wave`` | pec2field`` | ec2nlsdat`` |
+-------------+-------------+-------------+-------------+-------------+
| Type        | 1\ :m       | 1\ :m       | 1\ :m       | 2\ :m       |
|             | ath:`^{st}` | ath:`^{st}` | ath:`^{st}` | ath:`^{nd}` |
|             | order       | order       | order       | order       |
|             | Gaussian    | Gaussian    | Gaussian    | Gauss-Euler |
+-------------+-------------+-------------+-------------+-------------+
|             | time or     | 2D          | 3D          | 2D time     |
|             | space       | space-time  | space-time  | wave        |
|             | series      | wave        | field       |             |
+-------------+-------------+-------------+-------------+-------------+
|             | :math:`     | :math:`     | :math:`     | :math:`     |
|             | \downarrow` | \downarrow` | \downarrow` | \downarrow` |
+-------------+-------------+-------------+-------------+-------------+
| Output      | `           | ``W.        | ``W.Z, .    | ``          |
|             | `[x,xder]`` | Z, .x, .t`` | x, .y, .t`` | [xs2,xs1]`` |
+-------------+-------------+-------------+-------------+-------------+
|             |             | :math:`     | :math:`     |             |
|             |             | \downarrow` | \downarrow` |             |
+-------------+-------------+-------------+-------------+-------------+
| Movie       |             | ``          | ``          |             |
|             |             | M = seamovi | M = seamovi |             |
|             |             | e(W,type)`` | e(W,type)`` |             |
+-------------+-------------+-------------+-------------+-------------+
|             |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
|             |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Routine     | ``          | ``spe       | ``sp        | ``spec2     |
|             | spec2ldat`` | c2lseries`` | ec2ldat3D`` | ldat3DM/P`` |
|             | (freq)      | (dir)       | (dir)       |             |
+-------------+-------------+-------------+-------------+-------------+
| Type        | 1\ :m       | 1\ :m       | 1\ :m       | 2\ :m       |
|             | ath:`^{st}` | ath:`^{st}` | ath:`^{st}` | ath:`^{nd}` |
|             | order       | order       | order       | order       |
|             | Lagrange 2D | Lagrange 2D | Lagrange 3D | Lagrange 3D |
+-------------+-------------+-------------+-------------+-------------+
|             | space/time  | time series | space/time  | space/time  |
|             | components  |             | components  | components  |
+-------------+-------------+-------------+-------------+-------------+
|             | :math:`     | :math:`     | :math:`     | :math:`     |
|             | \downarrow` | \downarrow` | \downarrow` | \downarrow` |
+-------------+-------------+-------------+-------------+-------------+
| Output      | ``[w,x]``   | ``L.Z, .t   | ``[w,x,y]`` | ``[w,x,y,   |
|             |             | , .points`` |             | w2,x2,y2]`` |
+-------------+-------------+-------------+-------------+-------------+
|             | :math:`     | :math:`     | :math:`     | :math:`     |
|             | \downarrow` | \downarrow` | \downarrow` | \downarrow` |
+-------------+-------------+-------------+-------------+-------------+
| S           | ``L =       | ---         | ``L = ld    | ``L = ld    |
| eries/field | ldat2lwav`` | ----------- | at2lwav3D`` | at2lwav3D`` |
+-------------+-------------+-------------+-------------+-------------+
|             | :math:`     | :math:`     | :math:`     | :math:`     |
|             | \downarrow` | \downarrow` | \downarrow` | \downarrow` |
+-------------+-------------+-------------+-------------+-------------+
| Movie       | ``          | ``          | ``          | ``          |
|             | M = seamovi | M = seamovi | M = seamovi | M = seamovi |
|             | e(L,type)`` | e(L,type)`` | e(L,type)`` | e(L,type)`` |
+-------------+-------------+-------------+-------------+-------------+

.. container:: references hanging-indent
   :name: refs

   .. container::
      :name: ref-AilliotMonbet2012

      Ailliot, P., and V. Monbet. 2012. “Markov-Swithching
      Autoregressive Models for Wind Time Series.” *Eniviron. Modell.
      Sorftw.* 30: 92–101.

   .. container::
      :name: ref-BrodtkorbEtal1999Joint

      Brodtkorb, P.A., D. Myrhaug, and H. Rue. 1999. “Joint Distribution
      of Wave Height and Wave Crest Velocity from Reconstructed Data.”
      In *Proc. 9’th Int. Offshore and Polar Eng. Conf., Isope, Brest,
      France*, III:66–73.

   .. container::
      :name: ref-BrodtkorbEtal2001Joint

      ———. 2001. “Joint Distribution of Wave Height and Wave Crest
      Velocity from Reconstructed Data with Application to Ringing.”
      *Int. J. Offshore and Polar Eng.* 11 (1): 23–32.

   .. container::
      :name: ref-BuowsEtal1985Similarity

      Buows, E., H. Gunther, W. Rosenthal, and C. L. Vincent. 1985.
      “Similarity of the Wind Wave Spectrum in Finite Depth Water: 1
      Spectral Form.” *J. Geophys. Res.* 90 (C1): 975–86.

   .. container::
      :name: ref-Hamilton1989

      Hamilton, J. D. 1989. “A New Approach to the Economic Analysis
      Analysis of Nonstationary Time Series and the Business Cycle.”
      *Econometrica* 57: 357–84.

   .. container::
      :name: ref-Johannesson1998Rainflow

      Johannesson, P. 1998. “Rainflow Cycles for Switching Processes
      with Markov Structure.” *Prob. Eng. Inform. Sci.* 12 (2): 143–75.

   .. container::
      :name: ref-Lindgren2013

      Lindgren, G. 2013. *Stationary Stochastic Processes – Theory and
      Applications*. Boca Raton: CRC Press.

   .. container::
      :name: ref-LindgrenRootzenSandsten2014

      Lindgren, G., H. Rootzén, and M. Sandsten. 2014. *Stationary
      Stochastic Processes for Scientists and Engineers*. Boca Raton:
      CRC Press.

   .. container::
      :name: ref-MarthinsenAndWinterstein1992Skewness

      Marthinsen, T., and S. R Winterstein. 1992. “On the Skewness of
      Random Surface Waves.” In *Proc. 2’nd Int. Offshore and Polar Eng.
      Conf., Isope, San Francisco, Usa*, III:472–78.

   .. container::
      :name: ref-OchiAndAhn1994Probability

      Ochi, M. K., and K. Ahn. 1994. “Probability Distribution
      Applicable to Non-Gaussian Random Processes.” *Prob. Eng. Mech.*
      9: 255–64.

   .. container::
      :name: ref-RychlikEtal1997Modelling

      Rychlik, I., P. Johannesson, and M. R. Leadbetter. 1997.
      “Modelling and Statistical Analysis of Ocean-Wave Data Using
      Transformed Gaussian Process.” *Marine Structures, Design,
      Construction and Safety* 10: 13–47.

   .. container::
      :name: ref-Schuster

      Schuster, A. 1898. “On the Investigation of Hidden Periodicities
      with Application to a Supposed 26 Day Period of Meteorological
      Phenomena.” *Terrestrial Magnetism and Atmospheric Electricity* 3:
      13–41.

   .. container::
      :name: ref-Winterstein1988Nonlinear

      Winterstein, S. R. 1988. “Nonlinear Vibration Models for Extremes
      and Fatigue.” *J. Eng. Mech., ASCE* 114 (10): 1772–90.

.. [1]
   Still another definition, to be used in Chapter `[cha:5] <#cha:5>`__,
   is that :math:`f_0` is the average number of completed load cycles
   per time unit.

.. [2]
   A *Gaussian* stochastic process :math:`X(t)` is any process such that
   all linear combinations :math:`\sum a_k X(t_k)` have a Gaussian
   distribution; also derivatives :math:`X'(t)` and integrals
   :math:`\int_a^b X(t)\, \mathrm{d}t` are Gaussian.

.. [3]
   The general expression for the upcrossing intensity for a stationary
   process is
   :math:`\mu(u)=\int_{z=0}^\infty z\, f_{X(0),X'(0)}(u,z)\, \mathrm{d}z`,
   where :math:`f_{X(0),X'(0)}(u,z)` is a joint probability density
   function.

.. [4]
   The dispersion relation between frequency :math:`\omega` and
   wavenumber :math:`\kappa` on finite water depth :math:`h`, reads
   :math:`\omega^2=g \kappa \tanh h\kappa`, where :math:`g` is the
   acceleration of gravity.

.. [5]
   Still another definition, to be used in Chapter `[cha:5] <#cha:5>`__,
   is that :math:`f_0` is the average number of completed load cycles
   per time unit.

.. [6]
   A *Gaussian* stochastic process :math:`X(t)` is any process such that
   all linear combinations :math:`\sum a_k X(t_k)` have a Gaussian
   distribution; also derivatives :math:`X'(t)` and integrals
   :math:`\int_a^b X(t)\, \mathrm{d}t` are Gaussian.

.. [7]
   The general expression for the upcrossing intensity for a stationary
   process is
   :math:`\mu(u)=\int_{z=0}^\infty z\, f_{X(0),X'(0)}(u,z)\, \mathrm{d}z`,
   where :math:`f_{X(0),X'(0)}(u,z)` is a joint probability density
   function.

.. [8]
   The dispersion relation between frequency :math:`\omega` and
   wavenumber :math:`\kappa` on finite water depth :math:`h`, reads
   :math:`\omega^2=g \kappa \tanh h\kappa`, where :math:`g` is the
   acceleration of gravity.
