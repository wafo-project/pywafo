.. _cha:distr-appar-wave-data:

 Empirical wave characteristics
==============================

[cha:3]

One of the unique capabilities of Wafo is the treatment of the
statistical properties of wave characteristic. This, and the next
chapter, describe how to extract information on distributions of
observables like wave period, wave length, crest height, etc, either
directly from data, or from empirically fitted approximative models, or,
in the next chapter, by means of exact statistical distributions,
numerically computed from a spectral model.

We first define the different wave characteristics commonly used in
oceanographic engineering and science, and present the Wafo routines for
handling them. Then we compare the empirical findings with some
approximative representations of the statistical distributions, based on
empirical parameters from observed sea states. The code for the examples
are found in the m-file ``Chapter3.m``, and it takes a few seconds to
run.

.. _introduction-1:

Introduction
------------

.. _ss:gaussianparadigm:

The Gaussian paradigm - linear wave theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous chapter we discussed modelling of random functions by
means of Fourier methods. The signal was represented as a sum of
independent random cosine functions with random amplitudes and phases.
In linear wave theory those cosine functions are waves travelling in
water. Waves with different frequencies have different speeds, defined
by the dispersion relation. This property causes the characteristic
irregularity of the sea surface. Even if it were possible to arrange a
very particular combination of phases and amplitudes, so that the signal
looks, for example, like a saw blade, it will, after a while, change
shape totally. The phases will be almost independent and the sea would
again look like a Gaussian random process. On the other hand an observer
clearly can identify moving sea waves. The shape of those waves, which
are often called the *apparent waves*, since theoretically, those are
not mathematical waves, but are constantly changing up to the moment
when they disappear.

The wave action on marine structures is often modelled using linear
filters. Then the sea spectrum, together with the filter frequency
function, gives a complete characterization of the response of the
structure. However, often such models are too simplistic and
non-linearities have to be considered to allow more complex responses.
Then one may not wish to perform a complicated numerical analysis to
derive the complete response but is willing to accept the simplification
that the response is proportional to the waves. One may also wish to
identify some properties of waves that are dangerous in some way for the
particular ocean operation. Also the apparent waves themselves can be
the reason for non-linear response. For example, for waves with crests
higher than some threshold, water may fill a structure and change its
dynamical properties. The combined effect of apparent waves, often
described by their height and wave period, is therefore important in
ocean engineering. These aspects are discussed in more detail in the
textbook (Ochi 1998).

The apparent waves will be described by some geometric properties,
called wave characteristics, while frequencies of occurrences of waves
with specified characteristics will be treated in the statistical sense
and described by a probability distribution. Such distributions can then
be used to estimate the frequency of occurrences of some events
important in the design of floating marine systems, e.g. wave breaking,
slamming, ringing, etc.

.. _wave-characteristics-in-time-and-space-1:

Wave characteristics in time and space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wave surface is clearly a two-dimensional phenomenon that changes
with time and its study naturally deals with moving two-dimensional
objects (surfaces). Theoretical studies of random surfaces are the
subject of ongoing research, for example, (Azaïs and Wschebor 2009;
Mercardier 2006), for general studies of Gaussian random surfaces,
(Åberg 2007; Åberg, Rychlik, and Leadbetter 2008; Baxevani, Podgórski,
and Rychlik 2003; Baxevani and Rychlik 2006; Podgórski, Rychlik, and Sjö
2000; Sjö 2000, 2001), for space-time related wave results, and
(Podgórski and Rychlik 2016) for wave geometry. Related results for
Lagrange models are found in (Åberg and Lindgren 2008; Lindgren 2006,
2009, 2010; Lindgren and Åberg 2009; Lindgren, Bolin, and Lindgren
2010).

At present, there are only few programs in Wafo that handle the
space-time relations of waves, and hence in this tutorial we limit the
presentation to simpler cases of waves in one-dimensional records. [3]_
By this we mean the apparent waves extracted from functions (measured
signals) with one-dimensional parameter, either in time or in space.
These functions can be extracted from a photograph of the sea surface
as, for example, the *instantaneous profile* along a line in some fixed
horizontal direction on the sea, or they can be obtained directly as a
*record taken in time at a fixed position in space* as by means of a
wave pole or distance meter. The *encountered sea*, another important
one-dimensional record, can be collected by means of a ship-borne wave
recorder moving across the random sea.

To analyze collected wave data we need natural and operational
definitions of an individual wave, its period, height, steepness, and
possibly some other meaningful characteristics. There are several
possible definitions of apparent wave, and here we shall concentrate
mostly on zero down-crossing waves. Namely, the *apparent individual
wave* at a fixed time or position is defined as the part of the record
that falls between two consecutive down-crossings of the zero seaway
level (the latter often more descriptively referred to as the still
water level). For individual waves one can consider various natural
characteristics, among them *apparent periods* and *apparent heights
(amplitudes)*. The pictorial definitions of these characteristics are
given in Figure `3.1 <#fig:wavpar>`__.

.. figure:: waveparamNew
   :alt: Definition of wave parameters. The notation for the parameters
   used in our examples are given in Table `3.1 <#tab3_1>`__ at the end
   of this chapter.
   :name: fig:wavpar
   :width: 70.0%

   Definition of wave parameters. The notation for the parameters used
   in our examples are given in Table `3.1 <#tab3_1>`__ at the end of
   this chapter.

The definitions of the most common wave characteristics are given in
Table `3.1 <#tab3_1>`__. In the Wafo toolbox, the most important can be
retrieved by the help commands for ``wavedef``, ``perioddef``,
``ampdef``, and ``crossdef``, producing the output in
Section `3.4 <#sec:WAFOcharacteristics>`__.

Having precisely defined the characteristics of interest, one can
extract their frequency (empirical) distributions from a typical
sufficiently long record. For example, measurements of the apparent
period and height of waves could be taken over a long observation
interval to form an empirical two-dimensional distribution. This
distribution will represent some aspects of a given sea surface.
Clearly, because of the irregularity of the sea, empirical frequencies
will vary from record to record. However if the sea is in “steady”
condition, which corresponds mathematically to the assumption that the
observed random field is stationary and ergodic, their variability will
be insignificant for sufficiently large records. Such limiting
distributions (limiting with respect to observation time, for records
measured in time, increasing without bound) are termed the *long-run
distributions*. Obviously, in a real sea we seldom have a so long period
of "steady" conditions that the limiting distribution will be reached.
On average, one may observe 400-500 waves per hour of measurements,
while the stationary conditions may last from 20 minutes to only a few
hours.

Despite of this, a fact that makes these long-run distributions
particularly attractive is that they give probabilities of occurrence of
waves that may not be observed in the short records but still are
possible. Hence, one can estimate the intensity of occurrence of waves
with special properties and then extrapolate beyond the observed types
of waves. What we shall be concerned with next is how to compute such
distributional properties.

In the following we shall consider three different ways to obtain the
wave characteristic probability densities (or distributions):

-  To fit an empirical distribution to observed (or simulated) data in
   some parametric family of densities, and then relate the estimated
   parameters to some observed wave climate described by means of
   significant wave heigh and wave period. Algorithms to extract waves,
   estimate the densities and compute some simple statistics will be
   presented here in Chapter `[cha:3] <#cha:3>`__

-  To simplify the model for the sea surface to such a degree that
   explicit computation of wave characteristic densities (in the
   simplified model) is possible. Some examples of proposed models from
   the literature will also be given in this chapter.

-  To exactly compute the statistical distribution from the mathematical
   form of a random seaway. This requires computation of infinite
   dimensional integrals and expectations that have to be computed
   numerically. Wafo contains efficient numerical algorithms to compute
   these integrals, algorithms which do not require any particular form
   of the sea surface spectrum. The method are illustrated in
   Chapter `[cha:4] <#cha:4>`__ on period, wavelength, and amplitude
   distributions, for many standard types of wave spectra.

.. _sec:estim-wave-char:

Estimation of wave characteristics from data
--------------------------------------------

In this section we shall extract the wave characteristics from a
measured signal and then use non-parametric statistical methods to
describe the data, i.e. empirical distributions, histograms, and kernel
estimators. (In the last chapter of this tutorial we present some
statistical tools to fit parametric models; for kernel estimators, see
Appendix `[cha:KDE] <#cha:KDE>`__.)

It is generally to be advised that, before analyzing sea wave
characteristics, one should check the quality of the data by inspection
and by the routine ``findoutliers`` used in
Section `[sect2.1] <#sect2.1>`__. Then, one usually should remove any
present trend from the data. Trends could be due to tides or atmospheric
pressure variations that affect the mean level. De-trending can be done
using the Wafo functions ``detrend`` or ``detrendma``.

.. _wave-period-1:

Wave period
~~~~~~~~~~~

1ex1em We now continue the analysis of the shallow water waves in
``sea.dat`` that we started on page . We begin with extracting the
apparent waves and record their period. The signal ``sea.dat`` is
recorded at 4 Hz sampling frequency. One of the possible definitions of
period is the time between the consecutive wave crests. For this
particular variable it may be convenient to have a higher resolution
than 4 Hz and hence we shall interpolate the signal to a denser grid.
This will be obtained by giving an appropriate value to the variable
``rate`` which can be used as input to the Wafo routine ``dat2wa``. The
following code will return crest2crest wave periods :math:`T_{cc}` in
the variable ``Tcrcr`` and return the crest period :math:`T_c` in
``Tc``, i.e. the time from up-crossings to the following down-crossing.

::

         xx = load('sea.dat');
         xx(:,2) = detrend(xx(:,2));
         rate = 8;
         Tcrcr = dat2wa(xx,0,'c2c','tw',rate);
         Tc = dat2wa(xx,0,'u2d','tw',rate);

Next we shall use a kernel density estimator (KDE) to estimate the
probability density function (pdf) of the crest period and compare the
resulting pdf with a histogram of the observed periods stored in ``Tc``.
In order to define a suitable scale for the density we first compute the
mean and maximum of the observed crest periods.

::

         mean(Tc)
         max(Tc)
         t = linspace(0.01,8,200);
         kopt = kdeoptset('L2',0);
         ftc1 = kde(Tc,kopt,t);
         pdfplot(ftc1), hold on
         histgrm(Tc,[],[],1)
         axis([0 8 0 0.5])

(The parameter ``L2=0`` is used internally in ``kde``, and causes a
logarithmic transformation of the data to ensure that the density is
zero for negative values. Run ``help kdeoptset`` to see the definition.)

.. figure:: fig7_kde_tc
   :alt:  Kernel estimate of crest period density observed in
   ``sea.dat``; solid line: full KDE, dash dotted line: binned KDE,
   compared with histogram of the data.
   :name: fig7_kde_tc
   :width: 80mm

   Kernel estimate of crest period density observed in ``sea.dat``;
   solid line: full KDE, dash dotted line: binned KDE, compared with
   histogram of the data.

In Figure `3.2 <#fig7_kde_tc>`__ we can see that many short waves have
been recorded (due to relatively high sampling frequency). The kernel
estimate will be compared with the theoretically computed density in
Figure `[fig73] <#fig73>`__ in Chapter `[cha:4] <#cha:4>`__, page . 1em
:math:`\Box`

**..5em Note that the program ``kde`` can be quite slow for large data
sets. If a faster estimate of the density for the observations is
preferred one can use ``kdebin``, which is an approximation to the true
kernel density estimator. An important input parameter in the program,
that defines the degree of approximation, is ``inc`` which should be
given a value between 100 and 500. (A value of ``inc`` below 50 gives
fast execution times but can lead to inaccurate results.)**

::

         kopt.inc = 128;
         ftc2 = kdebin(Tc,kopt); pdfplot(ftc2,'-.')
         title('Kernel Density Estimates'), hold off

The result is in Figure `3.2 <#fig7_kde_tc>`__ 1em :math:`\Box`

.. _extreme-waves-model-check-1:

Extreme waves – model check
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We turn now to joint wave characteristics, e.g. the joint density of
half period and crest height ``(Tc,Ac)``, or waveheight and steepness
``(Ac,S)``. The program ``dat2steep`` identifies apparent waves and for
each wave gives several wave characteristics (use the help function on
``dat2steep`` for a list of computed variables). We begin by examining
profiles of waves having some special property, e.g. with high crests,
or that are extremely steep.

1ex1em The following code finds a sequence of waves in ``sea.dat`` and
extracts their characteristics:

::

         method = 0; rate = 8;
         [S, H, Ac, At, Tcf, Tcb, z_ind, yn] = ...
                dat2steep(xx,rate,method);

The first preliminary analysis of the data is to find the individual
waves which are extreme by some specified criterion, e.g. the steepest
or the highest waves, etc. To do such an analysis one can use the
function ``spwaveplot(xx,ind)``, which plots waves in ``xx`` that are
selected by the index variable ``ind``. For example, let us look at the
highest and the steepest waves.

::

         [Smax indS] = max(S)
         [Amax indA] = max(Ac)
         spwaveplot(yn,[indA indS],'k.')

The two waves are shown in Figure `[fig_c_wave] <#fig_c_wave>`__\ (a).
The shape of the biggest wave reminds of the so called "extreme" waves.
In the following we shall examine whether this particular shape
contradicts the assumption of a transformed Gaussian model for the sea.

This is done as follows. First we find the wave with the highest crest.
Then we mark all positive values in that wave as missing. Next we
reconstruct the signal, assuming the Gaussian model is valid, and
compare the profile of the reconstructed wave with the actual
measurements. Confidence bands for the reconstruction will also be
plotted. In the previous chapter we have already used the program
``reconstruct``, and here we shall need some additional output from that
function, to be used to compute and plot the confidence bands.

::

         inds1 = (5965:5974)'; Nsim = 10;
         [y1, grec1, g2, test, tobs, mu1o, mu1oStd] = ...
                reconstruct(xx,inds1,Nsim);
         spwaveplot(y1,indA-10), hold on
         plot(xx(inds1,1),xx(inds1,2),'+')
         lamb = 2.;
         muLstd = tranproc(mu1o-lamb*mu1oStd,fliplr(grec1));
         muUstd = tranproc(mu1o+lamb*mu1oStd,fliplr(grec1));
         plot (y1(inds1,1), [muLstd muUstd],'b-')
         axis([1482 1498 -1 3]), hold off

(Note that we have used the function ``tranproc`` instead of
``gaus2dat``, since the last function requires a two column matrix.
Furthermore we have to use the index ``indA-10`` to identify the highest
wave in ``y1``. This is caused by the fact that the interpolated signal
``yn`` has a few additional small waves that are not in ``xx``.)

In Figure `[fig_c_wave] <#fig_c_wave>`__\ (b) the crosses are the
removed values from the wave. The reconstructed wave, plotted by a solid
line, is close to the measured. (Observe that this is a simulated wave,
using the transformed Gaussian model, and hence each time we execute the
command the shape will change.) The confidence bands gives limits
containing 95% of the simulated values, pointwise. From the figure we
can deduce that the highest wave could have been even higher and that
the height is determined by the particularly high values of the
derivatives at the zero crossings which define the wave. The observed
wave looks more asymmetric in time than the reconstructed one. Such
asymmetry is unusual for the transformed Gaussian waves but not
impossible. By executing the following commands we can see that actually
the observed wave is close to the expected in a transformed Gaussian
model.

::

         plot(xx(inds1,1),xx(inds1,2),'+'), hold on
         mu = tranproc(mu1o,fliplr(grec1));
         plot(y1(inds1,1), mu), hold off

We shall not investigate this question further in this tutorial. 1em
:math:`\Box`

.. _crest-height-1:

Crest height
~~~~~~~~~~~~

We turn now to the kernel estimators of the crest height density. It is
well known that for Gaussian sea the tail of the density is well
approximated by the Rayleigh distribution. Wand and Jones (1995, Chap.
2.9) show that Gaussian distribution is one of the easiest distributions
to obtain a good Kernel Density Estimate from. It is more difficult to
find good estimates for distributions with skewness, kurtosis, and
multi-modality. Here, one can get help by transforming data. This can be
done choosing different values of input ``L2`` into the program ``kde``.

1ex1em We shall continue with the analysis of the crest height
distribution. By letting ``L2 = 0.6`` we see that the normalplot of the
transformed data is approximately linear. (Note: One should try out
several different values for ``L2``. It is also always good practise to
try out several different values of the smoothing parameter; see the
help text of ``kde`` and ``kdebin`` for further explanation.)

::

         L2 = 0.6;
         plotnorm(Ac.^L2)
         fac = kde(Ac,{'L2',L2},linspace(0.01,3,200));
         pdfplot(fac)
         simpson(fac.x{1},fac.f)

The integral of the estimated density ``fac`` is 0.9675 but it should be
one. Therefore, when we use the estimated density to compute different
probabilities concerning the crest height the uncertainty of the
computed probability is at least 0.03. We suspect that this is due to
the estimated density being non-zero for negative values. In order to
check this we compute the cumulative distribution using the formula,

.. math:: \mbox{\sf P}(Ac\le h)=1-\int_h^{+\infty} f_{Ac}(x)\, \mathrm{d}x,

where :math:`f_{Ac}(x)` is the estimated probability density of
:math:`Ac`. For the pdf saved in ``fac`` the following code gives an
estimate of the cumulative distribution function (cdf) for crest height
and compares it with the empirical distribution computed from data by
means of function ``edf`` or ``plotedf``.

::

         Fac = flipud(cumtrapz(fac.x{1},flipud(fac.f)));
         Fac = [fac.x{1} 1-Fac];
         Femp = plotedf(Ac,Fac);
         axis([0 2 0 1]), hold off

Since a kernel density estimator KDE in essence is a smoothed histogram
it is not very well suited for extrapolation of the density to the
region where no data are available, e.g. for high crests. In such a case
a parametric model should be used. In Wafo there is a function
``trraylpdf`` that combines the non-parametric approach of KDE with a
Rayleigh density. Simply, if the Rayleigh variable can be used to
described the crests of Gaussian waves then a transformed Rayleigh
variable should be used for the crests of the transformed Gaussian
waves. The method has several nice properties and will be described more
in Section `3.3.3 <#ss:Rayleighappr>`__. Here we just use it in order to
compare with the non-parametric KDE method.

::

         facr = trraylpdf(fac.x{1},'Ac',grec1);
         Facr = cumtrapz(facr.x{1},facr.f); hold on
         plot(facr.x{1},Facr,'.')
         axis([1.25 2.25 0.95 1]), hold off

Figure `[fig_Ac1] <#fig_Ac1>`__\ (a) shows that our hypothesis that the
pdf ``fac`` is slightly too low for small crests seems to be correct.
Next from Figure `[fig_Ac1] <#fig_Ac1>`__\ (b) we can see that also the
tail is reasonably well modelled even if it is lighter than, i.e. gives
smaller probabilities of high waves than, the one derived from the
transformed Gaussian model. 1em :math:`\Box`

.. _joint-crest-period-and-crest-height-distribution-1:

Joint crest period and crest height distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We shall use the kernel density estimator to find a good estimator of
the central part of the joint density of crest period and crest height.
Usually, kernel density estimators give poor estimates of the tail of
the distribution, unless large amounts of data is available. However, a
KDE gives qualitatively good estimates in regions with sufficient data,
i.e.  in the main part of the distribution. This is good for
visualization (``pdfplot``) and detecting modes, symmetries
(anti-symmetry) of distributions.

1ex1em The following command examines and plots the joint distribution
of crest period ``Tc = Tcf+Tcb`` and crest height ``Ac`` in ``sea.dat``.

::

         kopt2 = kdeoptset('L2',0.5,'inc',256);
         Tc = Tcf+Tcb;
         fTcAc = kdebin([Tc Ac],kopt2);
         fTcAc.labx={'Tc [s]'  'Ac [m]'} % make labels for the plot
         pdfplot(fTcAc), hold on
         plot(Tc,Ac,'k.'), hold off

.. figure:: fig_TcAc
   :alt:  Kernel estimate of joint density of crest period ``Tc`` and
   crest height ``Ac`` in ``sea.dat`` compared with the observed data
   (dots). The contour lines are drawn in such a way that they contain
   specified (estimated) proportions of data.
   :name: fig_TcAc
   :width: 80mm

   Kernel estimate of joint density of crest period ``Tc`` and crest
   height ``Ac`` in ``sea.dat`` compared with the observed data (dots).
   The contour lines are drawn in such a way that they contain specified
   (estimated) proportions of data.

In Figure `3.3 <#fig_TcAc>`__ are plotted 544 pairs of crest period and
height. We can see that the kernel estimate describes the distribution
of data quite well. It is also obvious that it can not be used to
extrapolate outside the observation range. In the following chapter we
shall compute the theoretical joint density of crest period and height
from the transformed Gaussian model and compare with the KDE estimate.
1em :math:`\Box`

.. _sec:explicitresults_wavemodels:

Explicit results - parametric wave models
-----------------------------------------

In this section we shall consider the Gaussian sea with well-defined
spectrum. We assume that the reference level is zero. We will present
some explicit results that are known and studied in the literature about
wave characteristics. Some of them are exact, others are derived by
simplification of the random functions describing the sea surface.

.. _the-average-wave-1:

The average wave
~~~~~~~~~~~~~~~~

For Gaussian waves the spectrum and the spectral moments contain exact
information about the average behaviour of many wave characteristics.
The Wafo routines ``spec2char`` and ``spec2bw`` compute a long list of
wave characteristic parameters.

**. *(Simple wave characteristics obtained from spectral density)*\ 1em
[Ex_wave_parameters] We start by defining a Jonswap spectrum, describing
a sea state with :math:`T_p = 10` [s], :math:`H_{m_0} = 5` [m]. Type
``spec2mom`` to see what spectral moments are computed.**

::

        SJ = jonswap([],[5 10]);
        [m mt]= spec2mom(SJ,4,[],0);

The most basic information about waves is contained in the spectral
moments. The variable ``mt`` now contains information about what kind of
moments have been computed, in this case spectral moments up to order
four (:math:`m_0, \ldots , m_4`). Next, the irregularity factor
:math:`\alpha`, significant wave height, zero crossing wave period, and
peak period can be computed.

::

         spec2bw(SJ)
         [ch Sa2] = spec2char(SJ,[1  3])

The interesting feature of the program ``spec2char`` is that it also
computes an estimate of the variance of the characteristics, given the
length of observations (assuming the Gaussian sea); see (Krogstad et al.
1999), (Tucker 1993), and (Young 1999) for more detailed discussion. For
example, for the Jonswap Gaussian sea, the standard deviation of
significant wave height estimated from 20 minutes of observations is
approximately 0.25 meter. 1em :math:`\Box`

.. _sec:explicit_approximations:

Explicit approximations of wave distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the module ``wavemodels``, we have implemented some of the
approximative models that have been suggested in the literature. To get
an overview of the routines in the module, use the help function on
``wavemodels``.

We will investigate two suggested approximations for the joint pdf of
``(Tc,Ac)``; for the nomenclature, see the routines ``perioddef`` and
``ampdef`` in the module ``docs``. Both functions need spectral moments
as inputs. One should bear in mind that the models only depend on a few
spectral moments and not on the full wave spectrum.

.. _model-by-longuet-higgins-1:

Model by Longuet-Higgins
^^^^^^^^^^^^^^^^^^^^^^^^

Longuet-Higgins, (Longuet-Higgins 1975, 1983), derived his approximative
distribution by considering the joint distribution of the envelope
amplitude and the time derivative of the envelope phase. The model is
valid for narrow-band processes. It seams to give relatively accurate
results for big waves, e.g. for waves with significant amplitudes.

The Longuet-Higgins density depends, besides the significant wave height
:math:`H_s` and peak period :math:`T_p`, on the spectral width parameter
:math:`\nu = \frac{m_0m_2}{m_1^2}-1`, which can be calculated by the
command ``spec2bw(S,’eps2’)``, (for a narrow-band process,
:math:`\nu \approx 0`). The explicit density is given by

.. math::

   f^{\mbox{\scriptsize{LH}}}_{T_c,A_c}(t,x)=c_{\mbox{\scriptsize{LH}}}\,
   \left(\frac{x}{t}\right)^2 \exp\left\{
       -\frac{x^2}{8}\big[1+\nu^{-2}(1-t^{-1})^2\big] \right\} ,

where

.. math::

   c_{\mbox{\scriptsize{LH}}}=\frac{1}{8}(2\pi)^{-1/2}\nu^{-1}[
     1+(1+\nu^2)^{-1/2}]^{-1}.

The density is calculated by the function ``lh83pdf``.

1ex1em For the Longuet-Higgins approximation of the :math:`T_c,A_c`
distribution for Jonswap waves we use the spectral moments just
calculated.

::

         t = linspace(0,15,100);
         h = linspace(0,6,100);
         flh = lh83pdf(t,h,[m(1),m(2),m(3)]);

In Wafo we have modified the Longuet-Higgins density to be applicable
also for transformed Gaussian models. Following the examples from the
previous chapter we compute the transformation proposed by Winterstein
and combine it with the Longuet-Higgins model.

::

         [sk, ku] = spec2skew(SJ);
         sa = sqrt(m(1));
         gh = hermitetr([],[sa sk ku 0]);
         flhg = lh83pdf(t,h,[m(1),m(2),m(3)],gh);

In Figure `[fig:lhdens] <#fig:lhdens>`__ the densities ``flh`` and
``flhg`` are compared. The contour lines are drawn in such a way that
they contain predefined proportions of the total probability mass inside
the contours. We can see that including some nonlinear effects gives
somewhat higher waves for the Jonswap spectrum. 1em :math:`\Box`

.. _model-by-cavanié-et-al.-1:

Model by Cavanié et al.
^^^^^^^^^^^^^^^^^^^^^^^

Another explicit density for the crest height was proposed by Cavanié et
al., (Cavanié, Arhan, and Ezraty 1976). Here any positive local maximum
is considered as a crest of a wave, and then the second derivative
(curvature) at the local maximum defines the wave period as if the
entire wave was a cosine function with the same height and the same
crest curvature.

The model uses the parameter :math:`\nu` and a higher order bandwidth
parameter [4]_ :math:`\epsilon`, defined by

.. math::

   \begin{aligned}
   \epsilon &= \sqrt{1-\frac{m_2^2}{m_0m_4}};\end{aligned}

where, for a narrow-band process, :math:`\epsilon \approx 0`. The
Cavanié distribution is given by

.. math::

   \begin{aligned}
   f^{\mbox{\scriptsize{CA}}}_{T_c,A_c}(t,x) &=
   c_{\mbox{\scriptsize{CA}}}
   \frac{x^2}{t^5}\exp \left \{-\frac{x^2}{8\varepsilon^2 t^4}\left[
   \left( t^2-\left(\frac{1-\varepsilon^2}{1+\nu^2}\right)\right)^2+
   \beta^2\left(\frac{1-\varepsilon^2}{1+\nu^2}\right)\right]\right\},
   \intertext{where}
   c_{\mbox{\scriptsize{CA}}} &=
     \frac{1}{4}(1-\epsilon^2)(2\pi)^{-1/2}{\epsilon}^{-1}
                {\alpha_2}^{-1}(1+\nu^2)^{-2}, \\
     {\alpha}_2 &= \frac{1}{2}[1+(1-{\epsilon}^2)^{1/2}], \\
     \beta &= {\epsilon}^2/(1-{\epsilon}^2) .
     \end{aligned}

The density is computed by

::

         t = linspace(0,10,100);
         h = linspace(0,7,100);
         fcav = cav76pdf(t,h,[m(1) m(2) m(3) m(5)],[]);

and a contour plot of the pdf is obtained by ``pdfplot(fcav)``;
see Figure `[fig:cavdens] <#fig:cavdens>`__.

.. _ss:Rayleighappr:

Rayleigh approximation for wave crest height
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[ss:Rayleighapproximation] There are several densities proposed in the
literature to approximate the height of a wave crest or its amplitude.
Some of them are programmed in Wafo; execute ``help wavemodels`` for a
list. For Gaussian sea the most simple and most frequently used model is
the Rayleigh density. The standardized Rayleigh variable :math:`R` has
probability density :math:`f(r)=r\exp(-r^2/2)`, :math:`x > 0`. It is
well known that for Gaussian sea the Rayleigh approximation works very
well for high waves, and actually it is a conservative approximation
since we have

.. math:: \mbox{\sf P}(A_c > h) \leq \mbox{\sf P}(R> 4 h/H_s) = e^{-8h^2/H_s^2},

see (Rychlik 1997). In that paper it is also shown that for any sea wave
model with crossing intensity :math:`\mu(u)`, one has
:math:`\mbox{\sf P}(A_c>h) \leq \mu(u)/\mu(0)`. The approximation
becomes more accurate as the level :math:`h` increases.

The crossing intensity :math:`\mu(u)` is given by Rice’s formula, Rice
(1944), and it can be computed when the joint density of sea level
:math:`X(t)` and its derivative :math:`X'(t)` is known, see
Section `[subsec:crossing_intensity] <#subsec:crossing_intensity>`__,

.. math:: \mu(u) = \int_0^{\infty} zf_{X(t), X'(t)}(u,z)\,\mathrm{d}z.

For a Gaussian sea it can be computed explicitly,

.. math:: \mu(u) = \frac{1}{T_z} e^{-8u^2/H_s^2}.

For non-linear wave models with random Stokes waves the crossing
intensity has to be computed using numerical integration; see the work
by Machado and Rychlik, (Machado and Rychlik 2003).

Knowing the crossing intensity :math:`\mu(u)` one can compute the
transformation :math:`g`, by using the routine ``lc2tr``, such that the
transformed Gaussian model has crossing intensity equal to
:math:`\mu(u)`. Consequently, we have that
:math:`\mbox{\sf P}(A_c>h) \leq \mbox{\sf P}(R> g(h)) = 1- \mbox{\sf P}(G(R)\leq h).`
The function ``trraylpdf`` computes the pdf of :math:`G(R)`. (Obviously
the function works for any transformation :math:`g`.)

In previous examples we used the estimated crossing intensity to compute
the transformation and then approximated the crest height density using
the transformed Rayleigh variable. The accuracy of the approximation for
the high crests in the data set ``xx = sea.dat`` was checked, see
Figure `[fig_Ac1] <#fig_Ac1>`__\ (b). A more extensive study of the
applicability of this approximation is done in (Rychlik 1997).

**. *(Rayleigh approximation of crest height from spectral
density)*\ 1em [rayleighapproximation] In this example we shall use a
transformed Rayleigh approximation for crest height derived from a sea
spectrum. In order to check the accuracy of the approximations we shall
use the estimated spectrum from the record ``sea.dat``.**

::

         xx = load('sea.dat');
         x = xx;
         x(:,2) = detrend(x(:,2));
         SS = dat2spec2(x);
         [sk, ku, me, si] = spec2skew(SS);
         gh = hermitetr([],[si sk ku me]);
         Hs = 4*si;
         r = (0:0.05:1.1*Hs)';
         fac_h = trraylpdf(r,'Ac',gh);
         fat_h = trraylpdf(r,'At',gh);
         h = (0:0.05:1.7*Hs)';
         facat_h = trraylpdf(h,'AcAt',gh);
         pdfplot(fac_h), hold on
         pdfplot(fat_h), hold off

Next, we shall compare the derived approximation with the observed crest
heights in ``x``. As before, we could use the function ``dat2steep`` to
find the crests. Here, for illustration only, we shall use ``dat2tc`` to
find the crest heights ``Ac`` and trough depth ``At``.

::

         TC = dat2tc(xx, me);
         tc = tp2mm(TC);
         Ac = tc(:,2); At = -tc(:,1);
         AcAt = Ac+At;

Finally, the following commands will give the cumulative distributions
for the computed densities.

::

         Fac_h = [fac_h.x{1} cumtrapz(fac_h.x{1},fac_h.f)];
         subplot(3,1,1)
         Fac = plotedf(Ac,Fac_h); hold on
         plot(r,1-exp(-8*r.^2/Hs^2),'.')
         axis([1. 2. 0.9 1])
         Fat_h = [fat_h.x{1} cumtrapz(fat_h.x{1},fat_h.f)];
         subplot(3,1,2)
         Fat = plotedf(At,Fat_h); hold on
         plot(r,1-exp(-8*r.^2/Hs^2),'.')
         axis([1. 2. 0.9 1])
         Facat_h = [facat_h.x{1} cumtrapz(facat_h.x{1},facat_h.f)];
         subplot(3,1,3)
         Facat = plotedf(AcAt,Facat_h); hold on
         r2 = (0:05:2.1*Hs)';
         plot(r2,1-exp(-2*r2.^2/Hs^2),'.')
         axis([1.5 3.5 0.9 1]), hold off

In Figure `[fig:cavdens] <#fig:cavdens>`__\ (b) we can see some
differences between the observed crest and trough distributions and
those obtained from the transformation ``gh``. However, it still gives a
much better approximation than the standard Rayleigh approximation
(dots). As it was shown before, using the transformation computed from
the crossing intensity, the transformed Rayleigh approach is giving a
perfect fit. Finally, one can see that the Rayleigh and transformed
Rayleigh variables give too conservative approximations to the
distribution of wave amplitude. 1em :math:`\Box`

.. _sec:WAFOcharacteristics:

Wafo wave characteristics
-------------------------

.. _ss:spectralcharacteristics:

spec2char
~~~~~~~~~

::

   help spec2char

    SPEC2CHAR Evaluates spectral characteristics and their variance

    CALL: [ch r chtext] = spec2char(S,fact,T)

          ch = vector of spectral characteristics
          r  = vector of the corresponding variances given T
      chtext = a cellvector of strings describing the elements of ch
          S  = spectral struct with angular frequency
        fact = vector with factor integers, see below.
               (default [1])
          T  = recording time (sec) (default 1200 sec = 20 min)

    If input spectrum is of wave number type, output are factors
    for corresponding 'k1D', else output are factors for 'freq'.
    Input vector 'factors' correspondence:
       1 Hm0   = 4*sqrt(m0)           Significant wave height
       2 Tm01  = 2*pi*m0/m1           Mean wave period
       3 Tm02  = 2*pi*sqrt(m0/m2)     Mean zero-crossing period
       4 Tm24  = 2*pi*sqrt(m2/m4)     Mean period between maxima
       5 Tm_10 = 2*pi*m_1/m0          Energy period
       6 Tp    = 2*pi/{w | max(S(w))} Peak period
       7 Ss    = 2*pi*Hm0/(g*Tm02^2)  Significant wave steepness
       8 Sp    = 2*pi*Hm0/(g*Tp^2)    Average wave steepness
       9 Ka    = abs(int S(w) exp(i*w*Tm02) dw) / m0
                                      Groupiness parameter
      10 Rs    = se help spec2char    Quality control parameter
      11 Tp    = 2*pi*int S(w)^4 dw   Peak Period
                 ------------------   (more robust estimate)
                 int w*S(w)^4 dw

      12 alpha = m2/sqrt(m0*m4)       Irregularity factor
      13 eps2  = sqrt(m0*m2/m1^2-1)   Narrowness factor
      14 eps4  = sqrt(1-m2^2/(m0*m4)) = sqrt(1-alpha^2)  Broadness factor
      15 Qp    = (2/m0^2)int_0^inf w*S(w)^2 dw           Peakedness factor

    Order of output is same as order in 'factors'
    The variances are computed with a Taylor expansion technique
    and is currently only available for factors 1,2 and 3.

.. _spec2bw-1:

spec2bw
~~~~~~~

::

   help spec2bw

    SPEC2BW Evaluates some spectral bandwidth and irregularity factors

     CALL:  bw = spec2bw(S,factors)

            bw = vector of factors
            S  = spectrum struct
       factors = vector with integers, see below. (default [1])

     If input spectrum is of wave-number type, output are factors for
     corresponding 'k1D', else output are factors for 'freq'.
     Input vector 'factors' correspondence:
        1 alpha=m2/sqrt(m0*m4)                        (irregularity factor)
        2 eps2 = sqrt(m0*m2/m1^2-1)                   (narrowness factor)
        3 eps4 = sqrt(1-m2^2/(m0*m4))=sqrt(1-alpha^2) (broadness factor)
        4 Qp=(2/m0^2)int_0^inf f*S(f)^2 df            (peakedness factor)
     Order of output is the same as order in 'factors'

     Example:
       S=demospec;
       bw=spec2bw(S,[1 2 3 4]);

.. _wavedef-1:

wavedef
~~~~~~~

::

   help wavedef

     WAVEDEF wave definitions and nomenclature

     Definition of trough and crest:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     A trough (t) is defined as the global minimum between a
     level v down-crossing (d) and the next up-crossing (u)
     and a crest (c) is defined as the global maximum between
     a level v up-crossing and the following down-crossing.

     Definition of down- and up-crossing waves:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     A level v-down-crossing wave (dw) is a wave from a
     down-crossing to the following down-crossing.
     Similarly a level v-up-crossing wave (uw) is a wave from
     an up-crossing to the next up-crossing.

     Definition of trough and crest waves:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     A trough to trough wave (tw) is a wave from a trough (t)
     to the following trough.
     The crest to crest wave (cw) is defined similarly.

     Definition of min2min and Max2Max wave:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     A min2min wave (mw) is defined starting from a minimum (m)
     and ending in the following minimum.
     A Max2Max wave (Mw) is a wave from a maximum (M) to
     the next maximum (all waves optionally rainflow filtered).

               <----- Direction of wave propagation
       <------Mw-----> <----mw---->
       M             : :  c       :
      / \            M : / \_     :     c_            c
     F   \          / \m/    \    :    /: \          /:\   level v
    ------d--------u----------d-------u----d--------u---d--------
           \      /:           \  :  /: :  :\_    _/  : :\_   L
            \_   / :            \_t_/ : :  :  \t_/    : :  \m/
              \t/  <-------uw---------> :  <-----dw----->
               :                  :     :             :
               <--------tw-------->     <------cw----->
     (F= first value and L=last value).
    See also: tpdef, crossdef, dat2tc, dat2wa, dat2crossind

.. _perioddef-1:

perioddef
~~~~~~~~~

::

   help perioddef

     PERIODDEF wave periods (lengths) definitions and
     nomenclature

     Definition of wave periods (lengths):
    ---------------------------------------

               <----- Direction of wave propagation

                   <-------Tu-------->
                   :                 :
                   <---Tc----->      :
                   :          :      : <------Tcc---->
       M           :      c   :      : :             :
      / \          : M   / \_ :      : c_            c
     F   \         :/ \m/    \:      :/  \          / \   level v
    ------d--------u----------d------u----d--------u---d--------
           \      /            \    /     :\_    _/:   :\_   L
            \_   /              \t_/      :  \t_/  :   :  \m/
              \t/                :        :        :   :
               :<-------Ttt----->:        <---Tt--->   :
                                          :<----Td---->:
      Tu   = Wave up-crossing period
      Td   = Wave down-crossing period
      Tc   = Crest period, i.e., period between up-crossing and
             the next down-crossing
      Tt   = Trough period, i.e., period between down-crossing and
             the next up-crossing
      Ttt  = Trough2trough period
      Tcc  = Crest2crest period

::

               <----- Direction of wave propagation

                    <--Tcf->                           Tuc
                    :      :               <-Tcb->     <->
       M            :      c               :     :     : :
      / \           : M   / \_             c_    :     : c
     F   \          :/ \m/    \           /  \___:     :/ \ level v
    ------d---------u----------d---------u-------d-----u---d-------
          :\_      /            \     __/:        \   /     \_   L
          :  \_   /              \_t_/   :         \t/        \m/
          :    \t/                 :     :
          :     :                  :     :
          <-Ttf->                  <-Ttb->

      Tcf  = Crest front period, i.e., period between up-crossing
             and crest
      Tcb  = Crest back period, i.e., period between crest and
             down-crossing
      Ttf  = Trough front period, i.e., period between
             down-crossing and trough
      Ttb  = Trough back period, i.e., period between trough and
             up-crossing

     Also note that Tcf and Ttf can also be abbreviated by their
     crossing marker, e.g. Tuc (u2c) and Tdt (d2t), respectively.
     Similar rules apply to all the other wave periods and wave
     lengths. (The nomenclature for wave length is similar, just
     substitute T and period with L and length, respectively)

                 <----- Direction of wave propagation
                          <--TMm-->
               <-TmM->    :       :
       M       :     :    M       :
      / \      :     M   /:\_     :     M_            M
     F   \     :    / \m/ :  \    :    /: \          / \
          \    :   /      :   \   :   / :  \        /   \
           \   :  /       :    \  :  /  :   \_    _/     \_   L
            \_ : /        :     \_m_/   :     \m_/         \m/
              \m/         :             :      :            :
                          <-----TMM----->      <----Tmm----->

      TmM = Period between minimum and the following Maximum
      TMm = Period between Maximum and the following minimum
      TMM = Period between Maximum and the following Maximum
      Tmm = Period between minimum and the following minimum
   See also: wavedef, ampdef, crossdef, tpdef

.. _ampdef-1:

ampdef
~~~~~~

::

   help ampdef

     AMPDEF wave heights and amplitude definitions and
     nomenclature

     Definition of wave amplitude and wave heights:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                <----- Direction of wave propagation

               ...............c_..........
               |             /| \         |
           Hd  |           _/ |  \        |  Hu
       M       |          /   |   \       |
      / \      |     M   / Ac |    \_     |     c_
     F   \     |    / \m/     |      \    |    /  \  level v
    ------d----|---u------------------d---|---u----d------
           \   |  /|                   \  |  /      \L
            \_ | / | At                 \_|_/
              \|/..|                      t
               t

      Ac   = crest amplitude
      At   = trough amplitude
      Hd   = wave height as defined for down-crossing waves
      Hu   = wave height as defined for up-crossing waves

     See also: wavedef, ampdef, crossdef, tpdef

.. _crossdef-1:

crossdef
~~~~~~~~

::

   help crossdef

     CROSSDEF level v crossing definitions and nomenclature

     Definition of level v crossing:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Let the letters 'm', 'M', 'F', 'L','d' and 'u' in the
     figure below denote local minimum, maximum, first value, last
     value, down- and up-crossing, respectively. The remaining
     sampled values are indicated with a '.'. Values that are
     identical with v, but do not cross the level is indicated
     with the letter 'o'.

     We have a level up-crossing at index, k, if

              x(k) <  v and v < x(k+1)
     or if
              x(k) == v and v < x(k+1) and x(r) < v for some
                        di < r <= k-1

     where di is  the index to the previous down-crossing.
     Similarly there is a level down-crossing at index, k, if

              x(k) >  v and v > x(k+1)
      or if
              x(k) == v and v > x(k+1) and x(r) > v  for some
                        ui < r <= k-1

     where ui is  the index to the previous up-crossing.

     The first (F) value is a up-crossing if x(1) = v and x(2) > v.
     Similarly, it is a down-crossing if     x(1) = v and x(2) < v.

          M
        .   .                  M                   M
      .      . .             .                   .   .
    F            d               .             .       L   level v
     ----------------------u-------d-------o---------------------
                   .     .           .   .   u
                     .                 m
                      m

     See also: perioddef, wavedef, tpdef, findcross, dat2tp

.. container::
   :name: tab3_1

   .. table:: Wave characteristic definitions

      +----------------------+----------------------+----------------------+
      | upcrossing wave      |                      | wave between two     |
      |                      |                      | successive mean      |
      |                      |                      | level upcrossings    |
      +----------------------+----------------------+----------------------+
      | downcrossing wave    |                      | wave between two     |
      |                      |                      | successive mean      |
      |                      |                      | level downcrossings  |
      +----------------------+----------------------+----------------------+
      | wave crest           |                      | the maximum value    |
      |                      |                      | between a mean level |
      |                      |                      | upcrossing and the   |
      |                      |                      | next downcrossing =  |
      |                      |                      | the highest point of |
      |                      |                      | a wave               |
      +----------------------+----------------------+----------------------+
      | wave trough          |                      | the minimum value    |
      |                      |                      | between a mean level |
      |                      |                      | downcrossing and the |
      |                      |                      | next upcrossing =    |
      |                      |                      | the lowest point of  |
      |                      |                      | a wave               |
      +----------------------+----------------------+----------------------+
      | crest front wave     | :math:`T_{cf}`       | time span from       |
      | period               |                      | upcrossing to wave   |
      |                      |                      | crest                |
      +----------------------+----------------------+----------------------+
      | crest back (rear)    | :ma                  | time from wave crest |
      | wave period          | th:`T_{cb} (T_{cr})` | to downcrossing      |
      +----------------------+----------------------+----------------------+
      | crest period         | :math:`T_c`          | time from mean level |
      |                      |                      | up- to downcrossing  |
      +----------------------+----------------------+----------------------+
      | trough period        | :math:`T_t`          | time from mean level |
      |                      |                      | down- to upcrossing  |
      +----------------------+----------------------+----------------------+
      | upcrossing period    | :math:`T_u`          | time between mean    |
      |                      |                      | level upcrossings    |
      +----------------------+----------------------+----------------------+
      | downcrossing period  | :math:`T_d`          | time between mean    |
      |                      |                      | level downcrossings  |
      +----------------------+----------------------+----------------------+
      | crest-to-crest wave  | :math:`T_{cc}`       | time between         |
      | period               |                      | successive wave      |
      |                      |                      | crests               |
      +----------------------+----------------------+----------------------+
      | crest amplitude      | :math:`A_c`          | crest height above   |
      |                      |                      | mean level           |
      +----------------------+----------------------+----------------------+
      | trough depth         | :math:`A_t`          | through depth below  |
      |                      |                      | mean level           |
      +----------------------+----------------------+----------------------+
      |                      |                      | (:math:`A_t > 0`)    |
      +----------------------+----------------------+----------------------+
      | upcrossing wave      | :math:`H_u`          | crest-to-trough      |
      | amplitude            |                      | vertical distance    |
      +----------------------+----------------------+----------------------+
      | downcrossing wave    | :math:`H_d`          | trough-to-crest      |
      | amplitude            |                      | vertical distance    |
      +----------------------+----------------------+----------------------+
      | wave steepness       | :math:`S`            | Generic symbol for   |
      |                      |                      | wave steepness       |
      +----------------------+----------------------+----------------------+
      |                      |                      | Symbol also used for |
      |                      |                      | spectral density     |
      +----------------------+----------------------+----------------------+
      | min-to-max period    |                      | time from local      |
      |                      |                      | minimum to next      |
      |                      |                      | local maximum        |
      +----------------------+----------------------+----------------------+
      | min-to-max amplitude |                      | height between local |
      |                      |                      | minimum and the next |
      |                      |                      | local maximum        |
      +----------------------+----------------------+----------------------+
      | max-to-min           |                      | similar to           |
      | period/amplitude     |                      | min-to-max           |
      |                      |                      | definitions          |
      +----------------------+----------------------+----------------------+

.. container:: references hanging-indent
   :name: refs

   .. container::
      :name: ref-AzaisWschebor2009

      Azaïs, J.-M., and M. Wschebor. 2009. *Level Sets and Extrema of
      Random Processes and Fields*. Hoboken: John Wiler; Sons.

   .. container::
      :name: ref-Aberg2007diss

      Åberg, S. 2007. “Applications of Rice’s Formula in Oceanographic
      and Environmental Problems.” PhD thesis, Math. Stat., Center for
      Math. Sci., Lund Univ., Sweden.

   .. container::
      :name: ref-AbergLindgren2008

      Åberg, Sofia, and Georg Lindgren. 2008. “Height distribution of
      stochastic Lagrange ocean waves.” *Probabilistic Engineering
      Mechanics* 23 (4): 359–63.
      http://dx.doi.org/10.1016/j.probengmech.2007.08.006.

   .. container::
      :name: ref-AbergRychlikLeadbetter2008

      Åberg, S., I. Rychlik, and M. R. Leadbetter. 2008. “Palm
      Distributions of Wave Characteristics in Encountering Seas.” *Ann.
      Appl. Probab.* 18: 1059–84.

   .. container::
      :name: ref-BaxevaniEtal2003Velocities

      Baxevani, A., K. Podgórski, and I. Rychlik. 2003. “Velocities for
      Moving Random Surfaces.” *Prob. Eng. Mech.* 18 (3): 251–71.

   .. container::
      :name: ref-BaxevaniRychlik2006

      Baxevani, A., and I. Rychlik. 2006. “Maxima for Gaussian Seas.”
      *Ocean Eng.* 33: 895–911.

   .. container::
      :name: ref-CavanieEtal1976Statistical

      Cavanié, A., M. Arhan, and R. Ezraty. 1976. “A Statistical
      Relationship Between Individual Heights and Periods of Storm
      Waves.” In *Proc. 1’st Int. Conf. On Behaviour of Offshore
      Structures, Boss, Trondheim, Norway*, 354–60.

   .. container::
      :name: ref-KrogstadEtal1999Methods

      Krogstad, H. E., J. Wolf, S. P. Thompson, and L. R. Wyatt. 1999.
      “Methods for Intercomparison of Wave Measurements.” *Coastal Eng.*
      37: 235–57.

   .. container::
      :name: ref-Lindgren2006

      Lindgren, Georg. 2006. “Slepian models for the stochastic shape of
      individual Lagrange sea waves.” *Advances in Applied Probability*
      38 (2): 430–50. http://dx.doi.org/10.1239/aap/1151337078.

   .. container::
      :name: ref-Lindgren2009

      ———. 2009. “Exact asymmetric slope distributions in stochastic
      Gauss–Lagrange ocean waves.” *Applied Ocean Research* 31 (1):
      65–73. http://dx.doi.org/10.1016/j.apor.2009.06.002.

   .. container::
      :name: ref-Lindgren2010a

      ———. 2010. “Slope distribution in front-back asymmetric stochastic
      Lagrange time waves.” *Advances in Applied Probability* 42 (2):
      489–508.
      http://dx.doi.org/http://dx.doi.org/10.1239/aap/1275055239.

   .. container::
      :name: ref-LindgrenAberg2009

      Lindgren, Georg, and Sofia Åberg. 2009. “First Order Stochastic
      Lagrange Model for Asymmetric Ocean Waves.” *Journal of Offshore
      Mechanics and Arctic Engineering* 131 (3): 031602–1–031602–8.
      http://dx.doi.org/10.1115/1.3124134.

   .. container::
      :name: ref-Lindgrenetal2010

      Lindgren, Georg, David Bolin, and Finn Lindgren. 2010.
      “Non-traditional stochastic models for ocean waves.” *The European
      Physical Journal Special Topics* 185: 209–24.
      http://dx.doi.org/10.1140/epjst/e2010-01250-y.

   .. container::
      :name: ref-Longuet-Higgins1975Joint

      Longuet-Higgins, M. S. 1975. “On the Joint Distribution Wave
      Periods and Amplitudes of Sea Waves.” *J. Geophys. Res.* 80:
      2688–94.

   .. container::
      :name: ref-Longuet-Higgins1983Joint

      ———. 1983. “On the Joint Distribution Wave Periods and Amplitudes
      in a Random Wave Field.” In *Proc. R. Soc.*, A389:24–258.

   .. container::
      :name: ref-MachadoAndRychlik2003Wave

      Machado, U, and I Rychlik. 2003. “Wave Statistics in Nonlinear
      Random Sea.” *Extremes* 6 (6): 125–46.

   .. container::
      :name: ref-Mercardier2006

      Mercardier, C. 2006. “Numerical Bounds for the Distribution of
      Maxima of Some One- and Two-Parameter Gaussian Processes.” *Adv.
      Appl. Probab.* 38: 149–70.

   .. container::
      :name: ref-Ochi1998Ocean

      Ochi, Michel K. 1998. *Ocean Waves, the Stochastic Approach*.
      Ocean Tech. Series 6. Campridge University Press.

   .. container::
      :name: ref-PodgorskiRychlik2016SizeOfWaves

      Podgórski, K., and I. Rychlik. 2016. “Spatial Size of Waves.”
      *Marine Structures* 50: 55–71.

   .. container::
      :name: ref-PodgorskiEtal2000Statistics

      Podgórski, K., I. Rychlik, and E. Sjö. 2000. “Statistics for
      Velocities of Gaussian Waves.” *Int. J. Offshore and Polar Eng.*
      10 (2): 91–98.

   .. container::
      :name: ref-RyclikAndLeadbettter1997Analysis

      Rychlik, M. R., I And Leadbetter. 1997. “Analysis of Ocean Waves
      by Crossing- and Oscillation-Intensities.” In *Proc. 7’th Int.
      Offshore and Polar Eng. Conf., Isope, Honolulu, Usa*.

   .. container::
      :name: ref-Sjo2001Simultaneous

      Sjö, E. 2001. “Simultaneous Distributions of Space-Time Wave
      Characteristics in a Gaussian Sea.” *Extremes* 3: 263–88.

   .. container::
      :name: ref-Sjo2000Crossing

      Sjö, Eva. 2000. “Crossings and Maxima in Gaussian Fields and
      Seas.” PhD thesis, Math. Stat., Center for Math. Sci., Lund Univ.,
      Sweden.

   .. container::
      :name: ref-Tucker1993recommended

      Tucker, M. J. 1993. “Recommended Standard for Wave Data Sampling
      and Near-Real-Time Processing.” *Ocean Eng.* 20 (5): 459–74.

   .. container::
      :name: ref-Young1999Wind

      Young, I. R. 1999. “Wind Generated Ocean Waves.” *Elsevier Ocean
      Engineering Book Series* 2: 239.

.. [1]
   The Lagrange module in Wafo contains special space-time routines for
   (Gaussian and) non-Gaussian waves; a tutorial is included in that
   module.

.. [2]
   The value of :math:`\epsilon` may be calculated by
   ``spec2bw(S,’eps4’)``

.. [3]
   The Lagrange module in Wafo contains special space-time routines for
   (Gaussian and) non-Gaussian waves; a tutorial is included in that
   module.

.. [4]
   The value of :math:`\epsilon` may be calculated by
   ``spec2bw(S,’eps4’)``
