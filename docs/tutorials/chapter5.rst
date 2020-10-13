.. _cha:5:

Fatigue load analysis and rain-flow cycles
==========================================

This chapter contains some elementary facts about random fatigue and how
to compute expected fatigue damage from a stochastic, stationary load
process. The commands can be found in ``Chapter5.m``, taking a few
seconds to run.

.. _random-fatigue-1:

Random fatigue
--------------

[sec:randomfatigue]

.. _sec:loadmodels:

Random load models
~~~~~~~~~~~~~~~~~~

This chapter presents some tools from Wafo for analysis of random loads
in order to assess random fatigue damage. A complete list of fatigue
routines can be obtained from the help function on ``fatigue``.

We shall assume that the load is given by one of three possible forms:

#. As measurements of the stress or strain function with some given
   sampling frequency in Hz. Such loads will be called measured loads
   and denoted by :math:`x(t)`, :math:`0\le t\le T`, where :math:`t` is
   time and :math:`T` is the duration of the measurements.

#. In the frequency domain (that is important in system analysis) as a
   power spectrum. This means that the signal is represented by a
   Fourier series

   .. math::

      x(t)\approx
      m + \sum_{i=1}^{[T/2]} a_i\cos(\omega_i\,t)+b_i
      \sin(\omega_i\,t)

   where :math:`\omega_i=i\cdot 2\pi/T` are angular frequencies,
   :math:`m` is the mean of the signal and :math:`a_i,b_i` are Fourier
   coefficients. The properties are summarized in a spectral density as
   in described in
   Section `[sec:freq-model-load] <#sec:freq-model-load>`__.

#. In the rainflow domain, i.e. the measured load is given in the form
   of a rainflow matrix.

We shall now review some simple means to characterize and analyze loads
which are given in any of the forms (1)–(3), and show how to derive
characteristics, important for fatigue evaluation and testing.

We assume that the reader has some knowledge about the concept of cycle
counting, in particular rainflow cycles, and damage accumulation using
Palmgren-Miners linear damage accumulation hypotheses. The basic
definitions are given in the end of this introduction. Another important
property is the crossing spectrum :math:`\mu(u)`, introduced in
Section `[sect2.1] <#sect2.1>`__, defined as the intensity of
upcrossings of a level :math:`u` by :math:`x(t)` as a function of
:math:`u`.

The process of damage accumulation depends only on the values and the
order of the local extremes (maxima and minima), in the load. The
sequence of local extremes is called the *sequence of turning points*.
The irregularity factor :math:`\alpha` measures how dense the local
extremes are relatively to the mean frequency :math:`f_0`. For a
completely regular function there would be only one local maximum
between upcrossings of the mean level, giving irregularity factor equal
to one. In the other extreme case, there are infinitely many local
extremes giving irregularity factor zero. However, if the crossing
intensity :math:`\mu(u)` is finite, most of those local extremes are
irrelevant for the fatigue and should be disregarded by means of some
smoothing device.

A particularly useful filter is the so-called *rainflow filter* that
removes all local extremes that build rainflow cycles with amplitude
smaller than a given threshold. We shall always assume that the signals
are rainflow filtered; see Section `3.2.1 <#sec:rainflowfilter>`__.

If more accurate predictions of fatigue life are needed, then more
detailed models are required for the sequence of turning points. Here
the Markov chain theory has shown to be particularly useful. There are
two reasons for this:

-  the Markov models constitute a broad class of processes that can
   accurately model many real loads,

-  for Markov models, the fatigue damage prediction using rainflow
   method is particularly simple, (Rychlik 1988) and (Johannesson 1999).

In the simplest case, the necessary information is the intensity of
pairs of local maxima and the following minima, summarized in the
so-called Markov matrix or min-max matrix. The dependence between other
extremes is modelled using Markov chains, see (Igor Rychlik, Lindgren,
and Lin 1995) and (Frendahl and Rychlik 1993).

.. _sec:fatigueprediction:

Damage accumulation in irregular loads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In laboratory experiments, one often subjects a specimen of a material
to a constant amplitude load, e.g. :math:`L(t)= s \sin(\omega t)`, where
:math:`s` and :math:`\omega` are the constant amplitude and frequency,
and one counts the number of cycles (periods) until the specimen breaks.
The number of load cycles until failure, :math:`N(s)`, as well as the
amplitudes :math:`s` are recorded. For small amplitudes,
:math:`s<s_{\infty}`, the fatigue life is often very large, and is set
to infinity, :math:`N(s)\approx\infty`, i.e. no damage will be observed
even during an extended experiment. The amplitude :math:`s_{\infty}` is
called *the fatigue limit* or *the endurance limit*. In practice, one
often uses a simple model for the S-N curve, also called the Wöhler
curve, i.e. the relation between the amplitude :math:`s` and
:math:`N(s)`,

.. math::

   \label{eq:SNmodel}
      N(s)=\left\{ \begin{array}{c@{\quad}l}
           K^{-1} s^{-\beta}, & s> s_{\infty},\\
           \infty, & s\le s_{\infty},\end{array}\right.

where :math:`K` and :math:`\beta` are material dependent parameters.
Often :math:`K` is considered as a random variable, usually lognormally
distributed, i.e. with :math:`K^{-1}=E\epsilon^{-1}` where
:math:`\ln E \in\mbox{N}(0,\sigma_E^2)`, and :math:`\epsilon`,
:math:`\beta` are fixed constants.

For irregular loads, also called variable amplitude loads, one often
combines the S-N curve with a cycle counting method by means of the
*Palmgren-Miner linear damage accumulation theory*, to predict fatigue
failure time. A cycle counting procedure is used to form equivalent load
cycles, which are used in the life prediction.

If the :math:`k`:th cycle in an irregular load has amplitude :math:`s_k`
then it is assumed that it causes a damage equal to :math:`1/N(s_k)`.
The total damage at time :math:`t` is then

.. math::

   \label{eq:Damage}\index[xentr]{damage}
     D(t)=\sum_{t_k\le t}\frac{1}{N(s_k)}=K\sum_{t_k\le
     t}s_k^\beta=K D_\beta(t),

where the sum contains all cycles that have been completed up to time
:math:`t`. Then, the fatigue life time :math:`T^f`, say, is shorter than
:math:`t` if the total damage at time :math:`t` exceeds 1, i.e. if
:math:`D(t)>1`. In other words, :math:`T^f` is defined as the time when
:math:`D(t)` crosses level 1 for the first time.

A very simple predictor of :math:`T^f` is obtained by replacing
:math:`K = E^{-1}\epsilon` in Eq. (`[eq:Damage] <#eq:Damage>`__) by a
constant, for example the median value of :math:`K`, which is equal to
:math:`\epsilon`, under the lognormal assumption. For high cycle
fatigue, the time to failure is long, more than :math:`10^5/f_0`, and
then for stationary (and ergodic and some other mild assumptions) loads,
the damage :math:`D_\beta(t)` can be approximated by its mean
:math:`E(D_\beta(t))=d_\beta\cdot t`. Here :math:`d_\beta` is the
*damage intensity*, i.e. how much damage is accumulated per unit time.
This leads to a very simple predictor of fatigue life time,

.. math::

   \widehat T^f=\frac{1}{\epsilon d_\beta}.
   \label{eq:fatiguelifetime}

.. _sec:CCRainflow:

Rainflow cycles and hysteresis loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The now commonly used cycle counting method is the rainflow counting,
which was introduced 1968 by Matsuishi and Endo in (Matsuishi and Endo
1968). It was designed to catch both slow and rapid variations of the
load by forming cycles by pairing high maxima with low minima even if
they are separated by intermediate extremes. Each local maximum is used
as the maximum of a *hysteresis loop* with an amplitude that is computed
by the rainflow algorithm. A new definition of the rainflow cycle,
equivalent to the original definition, was given 1987 by Rychlik,
(Rychlik 1987). The formal definition is also illustrated in
Figure `3.1 <#FigRFCdef>`__.

[textRFCdef] From each local maximum :math:`M_k` one shall try to reach
above the same level, in the backward (left) and forward (right)
directions, with an as small downward excursion as possible. The minima,
:math:`m_k^-` and :math:`m_k^+`, on each side are identified. The
minimum that represents the smallest deviation from the maximum
:math:`M_k` is defined as the corresponding rainflow minimum
:math:`m_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}`. The
:math:`k`:th rainflow cycle is defined as
:math:`(m_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}},M_k)`.

.. figure:: FigRFCdef_introNew
   :alt: Definition of the rainflow cycle as given by (Rychlik 1987).
   :name: FigRFCdef
   :width: 110mm

   Definition of the rainflow cycle as given by (Rychlik 1987).

If :math:`t_k` is the time of the :math:`k`:th local maximum and the
corresponding rainflow amplitude is
:math:`s_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}} = M_k - m_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}`,
i.e. the amplitude of the attached hysteresis loop, then the total
damage at time :math:`t` is

.. math::

   \label{eq:rainflowDamage}\index[xentr]{damage!rainflow}
     D(t)=\sum_{t_k\le t}\frac{1}{N(s_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}})}=K\sum_{t_k\le
     t}(s_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}})^\beta=K D_\beta(t),

where the sum contains all rainflow cycles that have been completed up
to time :math:`t`.

To use Eq. (`[eq:fatiguelifetime] <#eq:fatiguelifetime>`__) to predict
the fatigue life we need the damage intensity :math:`d_\beta`, i.e. the
damage per time unit caused by the rainflow cycles. If there are on the
average :math:`f_0` maxima [2]_ per time unit, after rainflow filtering,
and equally many rainflow cycles, and each rainflow cycle causes an
expected damage
:math:`\epsilon E(1/N_{S^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}})`
it is clear that the damage intensity is equal to

.. math:: d_\beta = {f_0}\, E\left((S^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}})^\beta \right).

Thus, an important parameter for prediction of fatigue life is the
distribution of the rainflow amplitudes and in particular the expected
value of the rainflow amplitudes raised to the material dependent power
parameter :math:`\beta`. Wafo contains a number of routines for handling
the rainflow cycles in observed load data and in theoretical load
models.

.. _sec:loadcycle:

Load cycle characteristics
--------------------------

.. _sec:rainflowfilter:

Rainflow filtered load data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In previous chapters we have presented models for sea wave data, treated
as functions of time. The models can be used in response analysis for
marine structures to wave forces or to compute wave characteristics for
specified random wave models, e.g. those defined by their power
spectrum.

Measured wave or load signals are often very noisy and need to be
smoothed before further analysis. A common practice is to use a bandpass
filters to exclude high frequencies from the power spectrum and to
filter out slow trends. If the function is modelled by a transformed
Gaussian process ``xx``, as described in
Section `[ss:transformedGaussianmodels] <#ss:transformedGaussianmodels>`__,
such a filtration is performed on the inverse transformed signal
``yy = g(xx)``. Obviously, one should not over-smooth data since that
will affect the height of extreme waves or cycles. Consequently, if the
signal is still too irregular even after smoothing, this is an
indication that one should use the trough-to-crest wave concept, defined
as in Table `[tab3_1] <#tab3_1>`__, instead of the simpler min-to-max
cycles. Chapter `[cha:4] <#cha:4>`__ of this tutorial was aimed at
showing how one can compute the crest-to-trough wave characteristics
from a Gaussian or transformed Gaussian model.

The trough-to-crest cycle concept is a nonlinear means to remove small
irregularities from a load series. Another nonlinear method to remove
small cycles from data is the rainflow filtering, introduced in (I.
Rychlik 1995), and included in the Wafo toolbox. For completeness, we
describe the algorithm of the rainflow filter.

In this tutorial we have used a simple definition of rainflow cycles
that is convenient for functions with finitely many local maxima and
minima. However, rainflow filters and rainflow cycles can be defined for
very irregular functions, like a sample function of Brownian motion,
where there are infinitely many local extremes in any finite interval,
regardless how small. This is accomplished by defining the rainflow
minimum
:math:`m^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}(t)` for
all time points :math:`t` of a function :math:`x(t)` in such a way that
the rainflow amplitude
:math:`x(t)-m^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}(t)`
is zero if the point :math:`x(t)` is not a strict local maximum of the
function; see (I. Rychlik 1995) for more detailed discussion. Now, a
*rainflow filter with threshold :math:`h`*, extracts all rainflow cycles
:math:`(m^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}(t), x(t))`
such that
:math:`x(t)-m^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}(t)>h`.
Consequently, if :math:`h<0` then the signal is unchanged by the filter,
if :math:`h=0` we obtain a sequence of turning points, and, finally, if
:math:`h>0`, all small oscillations are removed, see
Figure `3.7 <#fig6-1>`__ for an example.

.. _sec:oscillationcount:

Oscillation count and the rainflow matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rainflow count is a generalization of the crossing count. The
crossing spectrum counts the number of times a signal upcrosses any
level :math:`u`. More important for fatigue damage is the *oscillation
count*,
:math:`N^{{\protect\mbox{\protect\footnotesize\protect\sc osc}}}(u,v)`
that counts the number of times a signal upcrosses an interval
:math:`[u,v]`. The oscillation count is thus a function of two
variables, :math:`u` and :math:`v`, and is plotted as a bivariate count.
The oscillation count is a counting distribution for the rainflow
cycles. Consequently, if the matrix ``Nosc`` with elements
:math:`N^{{\protect\mbox{\protect\footnotesize\protect\sc osc}}}(u_j,u_i)`
is known, for a discrete set of levels,
:math:`u_1 \leq u_2 \leq \ldots \leq u_n`, we can compute the frequency
(or rather histogram) matrix of the rainflow count by means of the
Wafo-function ``nt2fr`` and obtain the matrix ``Frfc = nt2fr(Nosc)``, in
fatigue practice called the *rainflow matrix*. Knowing the rainflow
matrix of a signal one can compute the oscillation count by means of the
inverse function ``fr2nt``.

The rainflow matrix will play an important role in the analysis of a
rainflow filtered signal. Let :math:`x(t)` be a measured signal and
denote by :math:`x_h(t)` the rainflow filtered version, filtered with
threshold :math:`h`. Now, if we know a rainflow matrix ``Frfc``, say, of
:math:`x`, then the rainflow matrix of :math:`x_h` is obtained by
setting some sub-diagonals of ``Frfc`` to zero, since there are no
cycles in :math:`x_h` with amplitudes smaller than :math:`h`. Thus, the
oscillation count of :math:`x_h` can be derived from the oscillation
count of :math:`x`.

Note that extracting a sequence of troughs and crests
:math:`(m_i^{{\protect\mbox{\protect\footnotesize\protect\sc tc}}},M_i^{{\protect\mbox{\protect\footnotesize\protect\sc tc}}})`
from the signal is closely related to rainflow filtering. Given a
reference level
:math:`u^{{\protect\mbox{\protect\footnotesize\protect\sc tc}}}`, the
sequence
:math:`(m_i^{{\protect\mbox{\protect\footnotesize\protect\sc tc}}},M_i^{{\protect\mbox{\protect\footnotesize\protect\sc tc}}})`
can be obtained by first removing all rainflow cycles
:math:`(m_j^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}},M_j)`
such that
:math:`M_j<u^{{\protect\mbox{\protect\footnotesize\protect\sc tc}}}` or
:math:`m_j^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}>u^{{\protect\mbox{\protect\footnotesize\protect\sc tc}}}`
and then finding the min-to-max pairs in the filtered signal.

Clearly, the oscillation count is an important characteristic of
irregularity of a sea level function, and similarly, the expected
oscillation count, also called an *oscillation intensity matrix*, is an
important characteristic of the random processes used as a model for the
data. Consequently we face two problems: how to compute the oscillation
intensity for a specified model, and if knowing the oscillation
intensity, how can one find an explicit and easy way to handle random
processes with this intensity. Note that by solving these two problems
one increases the applicability of rainflow filters considerably. Since
then, given a random process, one can find its oscillation intensity,
and next one can compute the oscillation intensity of the rainflow
filtered random process, and finally, find a random process model for
the filtered signal.

.. _subsec:markov_chain:

Markov chain of turning points, Markov matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An upcrossing of an interval :math:`[u, v]` occurs if the process, after
an upcrossing of the level :math:`u`, passes the higher level :math:`v`
before it returns below :math:`u`. Therefore, the oscillation intensity
is closely related to a special first passage problem, and it can be
practically handled if some Markov structure of the process is assumed.
While Gaussian processes are an important class of models for linear
filtering, Markov processes are the appropriate models as far as
rainflow filtering is concerned. In this section a class of models, the
so called Markov chain of turnings points will be introduced.

For any load sequence we shall denote by ``TP`` the sequence of turning
points. The sequence ``TP`` will be called a *Markov chain of turning
points* if it forms a Markov chain, i.e. if the distribution of a local
extremum, given all previous extrema, depends only on the value and type
(minimum or maximum) of the most recent previous extremum. The elements
in the histogram matrix of min-to-max cycles and max-to-min cycles are
equal to the observed number of transitions from a minimum (maximum) to
a maximum (minimum) of specified height. Consequently, the probabilistic
structure of the Markov chain of turning points is fully defined by the
expected histogram matrix of min-to-max and max-to-min cycles; sometimes
called *Markov matrices*. Note that for a transformed Gaussian process,
a Markov matrix for min-to-max cycles was computed in
Section `[sect3_5] <#sect3_5>`__ by means of the Wafo function
``spec2mmtpdf``. In Wafo there is also an older version of that program,
called ``spec2cmat``, which we shall use in this chapter. The max-to-min
matrix is obtained by symmetry.

Next, the function ``mctp2tc`` (= Markov Chain of Turning Points to
Trough Crests), computes the trough2crest intensity, using a Markov
matrix to approximate the sequence of turning points by a Markov chain.
This approximation method is called the *Markov method*. Be aware that
the Markov matrix is not the transition matrix of the Markov chain of
turning points, but the intensity of different pairs of turning points.

Figure `3.2 <#fig:TP_Matrix>`__ shows the general principle of a Markov
transition count between turning points of local maxima and minima. The
values have been discretized to levels labeled ``1, ..., n``, from
smallest to largest.

.. figure:: FigTP_MatrixNew
   :alt:  Part of a discrete load process where turning points are
   marked with :math:`\bullet`. The scale to the left is the discrete
   levels. The transitions from minimum to maximum and from maximum to
   minimum are collected in the min-max matrix,
   :math:`\mbox{\boldmath $F$}` and max-min matrix,
   :math:`\mbox{\boldmath $\widehat F$}`, respectively. The rainflow
   cycles are collected in the rainflow matrix,
   :math:`\mbox{\boldmath $F$}^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}`.
   The numbers in the squares are the number of observed cycles and the
   grey areas are by definition always zero.
   :name: fig:TP_Matrix
   :width: 110mm

   Part of a discrete load process where turning points are marked with
   :math:`\bullet`. The scale to the left is the discrete levels. The
   transitions from minimum to maximum and from maximum to minimum are
   collected in the min-max matrix, :math:`\mbox{\boldmath $F$}` and
   max-min matrix, :math:`\mbox{\boldmath $\widehat F$}`, respectively.
   The rainflow cycles are collected in the rainflow matrix,
   :math:`\mbox{\boldmath $F$}^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}`.
   The numbers in the squares are the number of observed cycles and the
   grey areas are by definition always zero.

Finding the expected rainflow matrix is a difficult problem and explicit
results are known only for special classes of processes, e.g. if ``x``
is a stationary diffusion, a Markov chain or a function of a vector
valued Markov chain. Markov chains are very useful in wave analysis
since they form a broad class of processes and for several sea level
data, as well as for transformed Gaussian processes, one can observe a
very good agreement between the observed or simulated rainflow matrix
and that computed by means of the Markov method. Furthermore, Markov
chains can be simulated in a very efficient way. However, the most
important property is that, given a rainflow matrix or oscillation count
of a Markov chain of turning points one can find its Markov matrix. This
means that a Markov chain of turning points can be defined by either a
Markov matrix ``FmM`` or by its rainflow matrix ``Frfc``, and these are
connected by the following nonlinear equation

.. math::

   \mbox{\tt Frfc} = \mbox{\tt FmM} + {\cal F}(\mbox{\tt FmM}),
   \label{eq:rfc_mM_transformation}

where :math:`{\cal F}` is a matrix valued function, defined in (I.
Rychlik 1995), where also an algorithm to compute the inverse
:math:`({\cal I} + {\cal F})^{-1}` is given. The Wafo functions for
computing ``Frfc`` from ``FmM`` are ``mctp2rfm`` and ``mctp2rfc``, while
the inverse, i.e. ``FmM`` as a function of ``Frfc``, is computed by
``arfm2mctp``. It might be a good idea to check the modules ``cycles``
and ``trgauss`` in Wafo for different routines for handling these
matrices.

.. _sec:cycleanalysiswithWAFO:

Cycle analysis with Wafo
------------------------

In this section we shall demonstrate how Wafo can be used to extract
rainflow cycles from a load sequence, and how the corresponding fatigue
life can be estimated. The Markov method is used for simulation and
approximation of real load sequences. We shall use three load examples,
the deep water sea load, a simulated transformed Gaussian model, and a
load sequence generated from a special Markov structure.

.. _sec:crossingintensity:

Crossing intensity
~~~~~~~~~~~~~~~~~~

Basic to the analysis is the crossing intensity function :math:`\mu(u)`,
i.e. the number of times per time unit that the load up-crosses the
level :math:`u`, considered as a function of :math:`u`. We illustrate
the computations on the deep water sea waves data.

::

         xx_sea = load('sea.dat');
         tp_sea = dat2tp(xx_sea);
         lc_sea = tp2lc(tp_sea);
         T_sea = xx_sea(end,1)-xx_sea(1,1);
         lc_sea(:,2) = lc_sea(:,2)/T_sea;
         subplot(221), plot(lc_sea(:,1),lc_sea(:,2))
         title('Crossing intensity, (u, \mu(u))')
         subplot(222), semilogx(lc_sea(:,2),lc_sea(:,1))
         title('Crossing intensity, (log \mu(u), u)')

The routines ``dat2tp`` and ``tp2lc`` take a load sequence and extracts
the turning points, and from this calculates the number of up-crossings
as a function of level. The plots produced,
Figure `3.3 <#fig_wafo_6.12>`__, show the crossing intensity plotted in
two common modes, lin-lin of :math:`(u, \mu(u))` and log-lin of
:math:`(\log \mu (u), u)`.

.. figure:: fatigue_3
   :alt: Level crossing intensity for ``sea.dat``
   :name: fig_wafo_6.12
   :width: 110mm

   Level crossing intensity for ``sea.dat``

We shall also have use for the *mean frequency* :math:`f_0`, i.e. the
number of mean level upcrossings per time unit, and the irregularity
factor, :math:`\alpha`, which is the mean frequency divided by the mean
number of local maxima per time unit. Thus :math:`1/\alpha` is the
average number of local maxima that occur between the mean level
upcrossings.

To compute :math:`f_0` we use the Matlab function ``interp1``, (make
help ``interp1``), to find the crossing intensity of the mean level.

::

         m_sea = mean(xx_sea(:,2));
         f0_sea = interp1(lc_sea(:,1),lc_sea(:,2),m_sea,'linear')
         extr_sea = length(tp_sea)/(2*T_sea);
         alfa_sea = f0_sea/extr_sea

.. _sec:rainflowextraction:

Extraction of rainflow cycles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We start by a study of rainflow cycles in the deep water sea data.
Recall the definition of rainflow and min-max cycle counts. The demo
program ``democc`` illustrates these definitions. To use it to identify
the first few rainflow and min-max cycles, just use,

::

         proc = xx_sea(1:500,:);
         democc

Two windows will appear. In Demonstration Window 1, first mark the
turning points by the button TP. Then choose a local maximum (with the
buttons marked :math:`+1,-1,+5,-5`) and find the corresponding cycle
counts, using the buttons RFC, PT. The cycles are visualized in the
other window.

We shall now examine cycle counts in the load ``xx_ sea``. From the
sequence of turning points ``tp`` we find the rainflow and min-max
cycles in the data set,

::

         RFC_sea = tp2rfc(tp_sea);
         mM_sea = tp2mm(tp_sea);

Since each cycle is a pair of a local maximum and a local minimum in the
load, a cycle count can be visualized as a set of pairs in the
:math:`\mathbb{R}^2`-plane. This is done by the routine ``ccplot``.
Compare the rainflow and min-max counts in the load in
Figure `3.4 <#fig_wafo_6.4>`__ obtained by the following commands.

::

         subplot(121), ccplot(mM_sea)
         title('min-max cycle count')
         subplot(122), ccplot(RFC_sea)
         title('Rainflow cycle count')

.. figure:: fatigue_4_2017
   :alt: Rainflow and min-max cycle plots for ``sea.dat``.
   :name: fig_wafo_6.4
   :width: 110mm

   Rainflow and min-max cycle plots for ``sea.dat``.

Observe that ``RFC`` contains more cycles with high amplitudes, compared
to ``mM``. This becomes more evident in an amplitude histogram as seen
in Figure `3.5 <#fig_wafo_6.13>`__.

::

         ampmM_sea = cc2amp(mM_sea);
         ampRFC_sea = cc2amp(RFC_sea);
         subplot(221), hist(ampmM_sea,25);
         title('min-max amplitude distribution')
         subplot(222), hist(ampRFC_sea,25);
         title('Rainflow amplitude distribution')

.. figure:: fatigue_5
   :alt: min-max and rainflow cycle distributions for ``sea.dat``.
   :name: fig_wafo_6.13
   :width: 110mm

   min-max and rainflow cycle distributions for ``sea.dat``.

.. _sec:simulationcycles:

Simulation of rainflow cycles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _sec:simulationmarkov:

Simulation of cycles in a Markov model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most simple cycle model assumes that the sequence of turning points
forms a Markov chain. Then the model is completely defined by the
min-max matrix, ``G``. The matrix has dimension :math:`n \times n`,
where :math:`n` is the number of discrete levels (e.g. :math:`32` or
:math:`64`). In this example the discrete levels ``u`` are chosen in the
range from :math:`-1` to :math:`1`. The matrix ``G`` will contain the
probabilities of transitions between the different levels in ``u``; see
the help function for ``mktestmat`` for the generation of ``G``.

::

         n = 41; param_m = [-1 1 n]; 
         u_markov = levels(param_m);
         G_markov = mktestmat(param_m,[-0.2 0.2],0.15,1);

The model is easy to simulate and this is performed by the simulation
routine ``mctpsim``. This routine simulates only the sequence of turning
points and not the intermediate load values.

::

         T_markov = 5000;
         xxD_markov = mctpsim({G_markov []},T_markov);
         xx_markov = [(1:T_markov)' u_markov(xxD_markov)'];

Here ``xxD_markov`` takes values :math:`1,\ldots,n`, and by changing the
scale, as in the third command line, we get the load ``xx_markov``, with
TP-number in first column load values between :math:`-1` and 1 in second
column. The first 50 samples of the simulation is plotted in
Figure `3.6 <#fig_wafo_6.2>`__ by

::

         plot(xx_markov(1:50,1),xx_markov(1:50,2))

.. figure:: fatigue_6
   :alt: Simulated Markov sequence of turning points.
   :name: fig_wafo_6.2
   :width: 80mm

   Simulated Markov sequence of turning points.

We shall later use the matrix ``G_markov`` to calculate the theoretical
rainflow matrix, but first we construct a similar sequence of turning
points from a transformed Gaussian model.

.. _sec:RFC_filtered:

Rainflow cycles in a transformed Gaussian model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example we shall consider a sea-data-like series obtained as a
transformed Gaussian model with Jonswap spectrum. Since that spectrum
contains also rather high frequencies a Jonswap load will contain many
cycles with small amplitude. These are often uninteresting and can be
removed by a rainflow filter as follows.

Let ``g`` be the Hermite transformation proposed by Winterstein, which
we used in Chapter `[cha:2] <#cha:2>`__. Suppose the spectrum is of the
Jonswap type. To get the transform we need as input the approximative
higher moments, skewness and kurtosis, which are automatically
calculated from the spectrum by the routine ``spec2skew``. We define the
spectrum structure, including the transformation, and simulate the
transformed Gaussian load ``xx_herm``. The routine ``dat2dtp`` extracts
the turning points discretized to the levels specified by the parameter
vector ``param``.

Note that when calling the simulation routine ``spec2sdat`` with a
spectrum structure including a transformation, the input spectrum must
be normalized to have standard deviation 1, i.e. one must divide the
spectral values by the variance ``sa^2``.

::

         me = mean(xx_sea(:,2));
         sa = std(xx_sea(:,2));
         Hm0_sea = 4*sa;
         Tp_sea = 1/max(lc_sea(:,2));
         SJ = jonswap([],[Hm0_sea Tp_sea]);

         [sk, ku] = spec2skew(SJ);
         SJ.tr = hermitetr([],[sa sk ku me]);
         param_h = [-1.5 2 51];
         SJnorm = SJ;
         SJnorm.S = SJnorm.S/sa^2;
         xx_herm = spec2sdat(SJnorm,[2^15 1],0.1);
         h = 0.2;
         [dtp,u_herm,xx_herm_1] = dat2dtp(param_h,xx_herm,h);
         plot(xx_herm(:,1),xx_herm(:,2),'k','LineWidth',2);
         hold on;
         plot(xx_herm_1(:,1),xx_herm_1(:,2),'k--','Linewidth',2);
         axis([0 50 -1 1]), hold off;
         title('Rainflow filtered wave data')

The rainflow filtered data ``xx_herm_1`` contains the turning points of
``xx_herm`` with rainflow cycles less than ``h=0.2`` removed. In
Figure `3.7 <#fig6-1>`__ the dashed curve connects the remaining turning
points after filtration.

.. figure:: fatigue_7
   :alt:  Hermite transformed wave data together with rainflow filtered
   turning points, ``h = 0.2``.
   :name: fig6-1
   :width: 80mm

   Hermite transformed wave data together with rainflow filtered turning
   points, ``h = 0.2``.

Try different degree of filtering on the Ochi transformed sequence and
see how it affects the min-max cycle distribution. You can use the
following sequence of commands, with different ``h`` -values; see
Figure `3.8 <#fig_wafo_6.16>`__ for the results. Note that the rainflow
cycles have their original values in the left figure but that they have
been discretized to the discrete level defined by ``param_h`` in the
right figure.

::

         tp_herm=dat2tp(xx_herm);
         RFC_herm=tp2rfc(tp_herm);
         mM_herm=tp2mm(tp_herm);
         h=0.2;
         [dtp,u,tp_herm_1]=dat2dtp(param_h,xx_herm,h);
         RFC_herm_1 = tp2rfc(tp_herm_1);
         subplot(121), ccplot(RFC_herm)
         title('h=0')
         subplot(122), ccplot(RFC_herm_1)
         title('h=0.2')

.. figure:: fatigue_8
   :alt:  Rainflow cycles and rainflow filtered rainflow cycles in the
   transformed Gaussian process.
   :name: fig_wafo_6.16
   :width: 110mm

   Rainflow cycles and rainflow filtered rainflow cycles in the
   transformed Gaussian process.

.. _sec:calrainflowmatrix:

Calculating the Rainflow Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have now shown how to extract rainflow cycles from a load sequence
and to perform rainflow filtering in measured or simulated load
sequences. Next we shall demonstrate how the expected (theoretical)
rainflow matrix can be calculated in any random load or wave model,
defined either as a Markov chain of turning points, or as a stationary
random process with some spectral density. We do this by means of the
Markov method based on the max-min transition matrix for the sequence of
turning points. This matrix can either be directly estimated from or
assigned to a load sequence, or it can be calculated from the
correlation or spectrum structure of a transformed Gaussian model by the
methods described in Section `[sect3_5] <#sect3_5>`__.

.. _sec:calrfcmatrixinmarkovmodel:

Calculation of rainflow matrix in the Markov model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The theoretical rainflow matrix ``Grfc`` for the Markov model is
calculated in Wafo by the routine ``mctp2rfm``. Let ``G_markov`` be as
in Section `3.3.3.1 <#sec:simulationmarkov>`__ and calculate the
theoretical rainflow matrix by

::

         Grfc_markov=mctp2rfm({G_markov []});

A cycle matrix, e.g. a min-max or rainflow matrix, can be plotted by
``cmatplot``. Now we will compare the min-max and the rainflow matrices.

::

         subplot(121),cmatplot(u_markov,u_markov,G_markov),...
                 axis('square')
         subplot(122),cmatplot(u_markov,u_markov,Grfc_markov),...
                 axis('square')

Both 2D- and 3D-plots can be drawn; see the help on ``cmatplot``. It is
also possible to plot many matrices in one call.

::

         cmatplot(u_markov,u_markov,{G_markov Grfc_markov},3)

A plot with ``method = 4`` gives contour lines; see
Figure `3.9 <#fig_wafo_6.1>`__. Note that for high maxima and low
minima, the rainflow matrix has a pointed shape while the min-max matrix
has a more rounded shape.

::

         cmatplot(u_markov,u_markov,{G_markov Grfc_markov},4)
         subplot(121), axis('square'),...
                       title('min2max transition matrix')
         subplot(122), axis('square'), title('Rainflow matrix')

.. figure:: fatigue_9
   :alt: min-max-matrix and theoretical rainflow matrix for test Markov
   sequence.
   :name: fig_wafo_6.1
   :width: 110mm

   min-max-matrix and theoretical rainflow matrix for test Markov
   sequence.

We now compare the theoretical rainflow matrix with an observed rainflow
matrix obtained in the simulation. In this case we have simulated a
discrete Markov chain of turning points with states ``1,...,n`` and put
them in the variable ``xxD_markov``. It is turned into a rainflow matrix
by the routine ``dtp2rfm``. The comparison in
Figure `3.10 <#fig_wafo_6.3>`__ between the observed rainflow matrix and
the theoretical one is produced as follows.

::

         n = length(u_markov);
         Frfc_markov = dtp2rfm(xxD_markov,n);
         cmatplot(u_markov,u_markov,...
                  {Frfc_markov Grfc_markov*T/2},3)
         subplot(121), axis('square')
                       title('Observed rainflow matrix')
         subplot(122), axis('square')
                       title('Theoretical rainflow matrix')

Note that in order to compare the observed matrix ``Frfc_markov`` with
the theoretical matrix ``Grfc_markov`` we have to multiply the latter by
the number of cycles in the simulation which is equal to ``T/2``.

.. figure:: fatigue_10
   :alt: Observed and theoretical rainflow matrix for test Markov
   sequence.
   :name: fig_wafo_6.3
   :width: 110mm

   Observed and theoretical rainflow matrix for test Markov sequence.

We end this section by an illustration of the rainflow smoothing
operation. The observed rainflow matrix is rather irregular, due to the
statistical variation in the finite sample. To facilitate comparison
with the theoretical rainflow matrix we smooth it by the built in
smoothing facility in the routine ``cc2cmat``. To see how it works for
different degrees of smoothing we calculate the rainflow cycles by
``tp2rfc``.

::

         tp_markov = dat2tp(xx_markov);
         RFC_markov = tp2rfc(tp_markov);
         h = 0.2;
         Frfc_markov_smooth = cc2cmat(param_m,RFC_markov,[],1,h);
         cmatplot(u_markov,u_markov,...
                  {Frfc_markov_smooth Grfc_markov*T/2},4)
         subplot(121), axis('square')
                       title('Smoothed observed rainflow matrix')
         subplot(122), axis('square')
                       title('Theoretical rainflow matrix')

Here, the smoothing is done as a kernel smoother with a bandwidth
parameter ``h = 1``. The effect of the smoothing is shown in
Figure `3.11 <#fig_wafo_6.7>`__.

.. figure:: fatigue_11
   :alt: Smoothed observed and calculated rainflow matrix for test
   Markov sequence.
   :name: fig_wafo_6.7
   :width: 110mm

   Smoothed observed and calculated rainflow matrix for test Markov
   sequence.

.. _sec:rainflowfromspectrum:

Rainflow matrix from spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are now ready to demonstrate how the rainflow matrix can be
calculated in a load or wave model defined by its correlation or
spectrum structure. We chose the transformed Gaussian model with the
Hermite transform ``xx_herm`` which was studied in
Section `3.3.3.2 <#sec:RFC_filtered>`__. This model was defined by its
Jonswap spectrum and the standard Hermite transform for asymmetry.

We first need to find the structure of the turning points, which is
defined by the min-to-max density by the methods in
Section `[sect3_5] <#sect3_5>`__. We start by computing an
approximation, ``GmM3_herm``, of the min-max density by means of the
cycle routine ``spec2cmat`` (as an alternative one can use
``spec2mmtpdf``). The type of cycle is specified by a cycle parameter,
in this case ``’Mm’``.

::

         GmM3_herm = spec2cmat(SJ,[],'Mm',[],param_h,2);

The result is seen in Figure `3.12 <#fig_wafo_6.5>`__.

Then, we approximate the distribution of the turning points by a Markov
chain with transitions between extrema calculated from ``GmM3_herm``,
and compute the rainflow matrix by
Eq. (`[eq:rfc_mM_transformation] <#eq:rfc_mM_transformation>`__).

::

         Grfc_herm = mctp2drfm({GmM3_herm.f,[]});

In Wafo, the rainflow matrix can be calculated directly from the
spectrum by the cycle distribution routine ``spec2cmat`` by specifying
the cycle parameter to ``’rfc’``.

::

         Grfc_direct_herm = spec2cmat(SJ,[],'rfc',[],[],2);

The output is a structure array which contains the rainflow matrix in
the cell ``.f``.

The min-max matrix ``GmM3_herm`` and the rainflow matrix ``Grfc_herm``
are shown together in Figure `3.12 <#fig_wafo_6.5>`__, obtained using
the following commands.

::

         u_herm = levels(param_h);
         cmatplot(u_herm,u_herm,{GmM3_herm.f Grfc_herm},4)
         subplot(121), axis('square'),...
                       title('min-max matrix')
         subplot(122), axis('square'),...
                       title('Theoretical rainflow matrix')

.. figure:: fatigue_12
   :alt: min-max matrix and theoretical rainflow matrix for Hermite
   transformed Gaussian waves.
   :name: fig_wafo_6.5
   :width: 110mm

   min-max matrix and theoretical rainflow matrix for Hermite
   transformed Gaussian waves.

We can also compare the theoretical min-max matrix with the observed
cycle count and the theoretical rainflow matrix with the observed one.
In both comparisons we smooth the observed matrix to get a more regular
structure. We also illustrate the multi-plotting capacity of the routine
``cmatplot``.

::

         tp_herm = dat2tp(xx_herm);
         RFC_herm = tp2rfc(tp_herm);
         mM_herm = tp2mm(tp_herm);
         h = 0.2;
         FmM_herm_smooth = cc2cmat(param_o,mM_herm,[],1,h);
         Frfc_herm_smooth = cc2cmat(param_o,RFC_herm,[],1,h);
         T_herm=xx_herm(end,1)-xx_herm(1,1);
         cmatplot(u_herm,u_herm,{FmM_herm_smooth ...
                  GmM3_herm.f*T_herm/2;...
                  Frfc_herm_smooth Grfc_herm*T_herm/2},4)
         subplot(221), axis('square')
                       title('Observed smoothed min-max matrix')
         subplot(222), axis('square')
                       title('Theoretical min-max matrix')
         subplot(223), axis('square')
                       title('Observed smoothed rainflow matrix')
         subplot(224), axis('square')
                       title('Theoretical rainflow matrix')

.. figure:: fatigue_13
   :alt: Observed smoothed and theoretical min-max matrix, and observed
   smoothed and theoretical rainflow matrix for Hermite transformed
   Gaussian waves.
   :name: fig_wafo_6.8
   :width: 110mm

   Observed smoothed and theoretical min-max matrix, and observed
   smoothed and theoretical rainflow matrix for Hermite transformed
   Gaussian waves.

.. _sec:crossingrainflowsimulation:

Simulation from crossings structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In fatigue experiments it is important to generate load sequences with a
prescribed rainflow or other crossing property. Besides the previously
used simulation routines for Markov loads and spectrum loads,
Wafo contains algorithms for generation of random load sequences that
have a specified average rainflow distribution or a specified
irregularity and crossing spectrum. We illustrate the crossing structure
simulation by means of the routine ``lc2sdat``. Simulation from a
rainflow distribution can be achieved by first calculating the
corresponding Markov matrix and then simulate by means of ``mctpsim``.

The routine ``lc2sdat`` simulates a load with specified irregularity
factor and crossing spectrum. We first estimate these quantities in the
simulated Hermite transformed Gaussian load, and then simulate series
with the same crossing spectrum but with varying irregularity factor.
The sampling variability increases with decreasing irregularity factor,
as is seen in Figure `3.15 <#fig_wafo_6.9>`__. The figures were
generated by the following commands.

::

         cross_herm = dat2lc(xx_herm);
         alpha1 = 0.25;
         alpha2 = 0.75;
         xx_herm_sim1 = lc2sdat(cross_herm,500,alpha1);
         cross_herm_sim1 = dat2lc(xx_herm_sim1);
         subplot(211)
         plot(cross_herm(:,1),cross_herm(:,2)/max(cross_herm(:,2)))
         hold on
         stairs(cross_herm_sim1(:,1),...
             cross_herm_sim1(:,2)/max(cross_herm_sim1(:,2)))
         hold off
         title('Crossing intensity, \alpha = 0.25')
         subplot(212)
         plot(xx_herm_sim1(:,1),xx_herm_sim1(:,2))
         title('Simulated load, \alpha = 0.25')

         xx_herm_sim2 = lc2sdat(500,alpha2,cross_herm);
         cross_herm_sim2 = dat2lc(xx_herm_sim2);
         subplot(211)
         plot(cross_herm(:,1),cross_herm(:,2)/max(cross_herm(:,2)))
         hold on
         stairs(cross_herm_sim2(:,1),...
             cross_herm_sim2(:,2)/max(cross_herm_sim2(:,2)))
         hold off
         title('Crossing intensity, \alpha = 0.75')
         subplot(212)
         plot(xx_herm_sim2(:,1),xx_herm_sim2(:,2))
         title('Simulated load, \alpha = 0.75')

|Upper figures show target crossing spectrum (smooth curve) and obtained
spectrum (wiggled curve) for simulated process shown in lower figures.
Irregularity factor: left :math:`\alpha=0.25`, right
:math:`\alpha=0.75`.| |Upper figures show target crossing spectrum
(smooth curve) and obtained spectrum (wiggled curve) for simulated
process shown in lower figures. Irregularity factor: left
:math:`\alpha=0.25`, right :math:`\alpha=0.75`.|

.. _sec:damageintensity:

Fatigue damage and fatigue life distribution
--------------------------------------------

.. _sec:fatigueintroduction:

Introduction
~~~~~~~~~~~~

We shall now give a more detailed account of how Wafo can be used to
estimate and bound the fatigue life distribution under random loading.
The basic assumptions are the Wöhler curve
Eq. (`[eq:SNmodel] <#eq:SNmodel>`__) and the Palmgren-Miner damage
accumulation rule Eq. (`[eq:Damage] <#eq:Damage>`__),

.. math::

   \begin{aligned}
    %\label{SNmodel}
   N(s)&=&\left\{ \begin{array}{c@{\quad}l}
   K^{-1} s^{-\beta}, & s> s_{\infty},\\
   \infty, & s\le s_{\infty},\end{array}\right.\label{eq:W} \\[0.6em]
     D(t)&=&\sum_{t_k\le t}\frac{1}{N(s_k)}=K\sum_{t_k\le
     t}s_k^\beta=K D_\beta(t). \label{eq:PM}\end{aligned}

Here :math:`N(s)` is the expected fatigue life from constant amplitude
test with amplitude :math:`s`, and :math:`D(t)` is the total damage at
time :math:`t` caused by variable amplitude cycles :math:`s_k`,
completed before time :math:`t`. The damage intensity
:math:`d_\beta = D(t)/t` for large :math:`t` is the amount of damage per
time unit.

Most information is contained in the cycle amplitude distribution, in
particular in the rainflow cycles, in which case (`[eq:PM] <#eq:PM>`__)
becomes,

.. math::

   D(t) = \sum_{t_k\le t} \frac{1}{N_{s_k}}
     = K \sum_{t_k\le t} \left(S_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}\right)^{\beta}, \qquad
     S_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}} = \left(M_k-m_k^{{\protect\mbox{\protect\footnotesize\protect\sc rfc}}}\right)/2.

The rainflow cycle count ``RFC`` can be directly used for prediction of
expected fatigue life. The expression
Eq. (`[eq:fatiguelifetime] <#eq:fatiguelifetime>`__) gives the expected
time to fatigue failure in terms of the material constant
:math:`\epsilon` and the expected damage :math:`d_\beta` per time unit.
The parameters :math:`\epsilon` and :math:`\beta` can be estimated from
an S-N curve. In the examples here we will use
:math:`\epsilon=5.5\cdot10^{-10}`, :math:`\beta=3.2`; see
Section `3.4.4 <#sec:estimationofSNcurve>`__. For our sea load
``xx_sea``, the computations go directly from the rainflow cycles as
follows.

::

         beta=3.2; gam=5.5E-10; T_sea=xx_sea(end,1)-xx_sea(1,1);
         d_beta=cc2dam(RFC_sea,beta)/T_sea;
         time_fail=1/gam/d_beta/3600

giving the time to failure ``5.9693e+006`` when time to failure is
counted in hours (= 3600 sec). Obviously, this load causes little damage
to the material with the specified properties, since the failure time is
almost 700 years – of course, the sea wave data is not a fatigue load
sequence, so the example is meaningless from a fatigue point of view.

.. _sec:levelcrossings:

Level Crossings
~~~~~~~~~~~~~~~

We have in Section `3.3.5 <#sec:crossingrainflowsimulation>`__ seen how
the crossing intensity contains information about the load sequence and
how it can be used for simulation. We shall now investigate the relation
between the crossing intensity, the rainflow cycles, and the expected
fatigue life.

We use the Markov model from Section `3.3.3.1 <#sec:simulationmarkov>`__
for the sequence of turning points as an example. First we go from the
rainflow matrix to the crossing intensity.

::

         mu_markov = cmat2lc(param_m,Grfc_markov);
         muObs_markov = cmat2lc(param_m,Frfc_markov/(T_markov/2));
         plot(mu_markov(:,1),mu_markov(:,2),...
            muObs_markov(:,1),muObs_markov(:,2),'--')
         title('Theoretical and observed crossing intensity ')

The plot in Figure `3.16 <#fig_wafo_6.10>`__ compares the theoretical
upcrossing intensity ``mu_markov`` with the observed upcrossing
intensity ``muObs_markov``, as calculated from the theoretical and
observed rainflow matrices, respectively.

.. figure:: fatigue_15
   :alt: Crossing intensity as calculated from the Markov observed
   rainflow matrix (solid curve) and from the observed rainflow matrix
   (dashed curve).
   :name: fig_wafo_6.10
   :width: 80mm

   Crossing intensity as calculated from the Markov observed rainflow
   matrix (solid curve) and from the observed rainflow matrix (dashed
   curve).

.. _damage-1:

Damage
~~~~~~

[sec:damage] The Wafo toolbox contains a number of routines to compute
and bound the damage, as defined by (`[eq:PM] <#eq:PM>`__), inflicted by
a load sequence. The most important routines are ``cc2dam`` and
``cmat2dam``, which give the total damage from a cycle count and from a
cycle matrix, respectively. More detailed information is given by
``cmat2dmat``, which gives a damage matrix, separated for each cycle,
from a cycle matrix. An upper bound for total damage from level
crossings is given by ``lc2dplus``.

We first calculate the damage by the routines ``cc2dam`` for a cycle
count (e.g. rainflow cycles) and ``cmat2dam`` for a cycle matrix
(e.g. rainflow matrix).

::

         beta = 4;
         Dam_markov = cmat2dam(param_m,Grfc_markov,beta)
         DamObs1_markov = ...
            cc2dam(u_markov(RFC_markov),beta)/(T_markov/2)
         DamObs2_markov = ...
            cmat2dam(param_m,Frfc_markov,beta)/(T_markov/2)

Here, ``Dam_markov`` is the theoretical damage per cycle in the assumed
Markov chain of turning points, while ``DamObs1`` and ``DamObs2`` give
the observed damage per cycle, calculated from the cycle count and from
the rainflow matrix, respectively. For this model the result should be
``Dam_markov = 0.0073`` for the theoretical damage and very close to
this value for the simulated series.

The damage matrix is calculated by ``cmat2dmat``. It shows how the
damage is distributed among the different cycles as illustrated in
Figure `3.17 <#fig_wafo_6.11>`__. The sum of all the elements in the
damage matrix gives the total damage.

::

         Dmat_markov = cmat2dmat(param_m,Grfc_markov,beta);
         DmatObs_markov = cmat2dmat(param_m,...
                                   Frfc_markov,beta)/(T_markov/2);}
         subplot(121), cmatplot(u_markov,u_markov,Dmat_markov,4)
         title('Theoretical damage matrix')
         subplot(122), cmatplot(u_markov,u_markov,DmatObs_markov,4)
         title('Observed damage matrix')
         sum(sum(Dmat_markov))
         sum(sum(DmatObs_markov))

.. figure:: fatigue_16
   :alt: Distribution of damage from different RFC cycles, from
   calculated theoretical and from observed rainflow matrix.
   :name: fig_wafo_6.11
   :width: 110mm

   Distribution of damage from different RFC cycles, from calculated
   theoretical and from observed rainflow matrix.

It is possible to calculate an upper bound on the damage intensity from
the crossing intensity only, without using the rainflow cycles. This is
done by the routine ``lc2dplus``, which works on any theoretical or
observed crossing intensity function.

::

         Damplus_markov = lc2dplus(mu_markov,beta)

.. _sec:estimationofSNcurve:

Estimation of S-N curve
~~~~~~~~~~~~~~~~~~~~~~~

Wafo contains routines for computation of parameters in the basic S-N
curve (`[eq:SNmodel] <#eq:SNmodel>`__), for the relation between the
load cycle amplitude :math:`s` and the fatigue life :math:`N(s)` in
fixed amplitude tests, defined by (`[eq:W] <#eq:W>`__). The variation of
the material dependent variable :math:`K` is often taken to be random
with a lognormal distribution,

.. math:: K = E \epsilon ^{-1},

where :math:`\epsilon` is a fixed parameter, depending on material, and
:math:`\ln E` has a normal distribution with mean :math:`0` and standard
deviation :math:`\sigma _E`. Thus, there are three parameters,
:math:`\epsilon`, :math:`\beta`, :math:`\sigma _E`, to be estimated from
an S-N experiment. Taking logarithms in (`[eq:SNmodel] <#eq:SNmodel>`__)
the problem turns into a standard regression problem,

.. math:: \ln N(s) = - \ln E - \ln \epsilon - \beta \ln s,

in which the parameters can easily be estimated.

The Wafo toolbox contains a data set ``sn.dat`` with fatigue lives from
40 experiments with :math:`s` = 10, 15, 20, 25, and 30 MPa, stored in a
variable ``N``, in groups of five. The estimation routine is called
``snplot``, which performs both estimation and plotting; see
``help snplot``.

First load SN-data and plot in log-log scale.

::

         SN = load('sn.dat');
         s = SN(:,1); N = SN(:,2);
         loglog(N,s,'o'), axis([0 14e5 10 30])

To further check the assumptions of the S-N-model we plot the results
for each :math:`s`-level separately on normal probability paper. As seen
from Figure `3.18 <#fig_wafo_6.14>`__ the assumptions seem acceptable
since the data fall on almost parallel straight lines.

::

         plotnorm(reshape(log(N),8,5))

.. figure:: fatigue_17_2017
   :alt: Check of S-N-model on normal probability paper.
   :name: fig_wafo_6.14
   :width: 80mm

   Check of S-N-model on normal probability paper.

The estimation is performed and fitted lines plotted in
Figure `3.20 <#fig_wafo_6.15>`__, with linear and log-log plotting
scales:

::

         [e0,beta0,s20] = snplot(s,N,12);
         title('S-N-data with estimated N(s)')

gives linear scale and

::

         [e0,beta0,s20] = snplot(s,N,14);
         title('S-N-data with estimated N(s)')

gives log-log scales.

|Estimation of S-N-model on linear and log-log scale.| |Estimation of
S-N-model on linear and log-log scale.|

.. _sec:fatiguelifedistribution:

From S-N-curve to fatigue life distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Palmgren-Miner hypothesis states that fatigue failure occurs when
the accumulated damage exceeds one, :math:`D(t) > 1`. Thus, if the
fatigue failure time is denoted by :math:`T_f`, then

.. math:: \mbox{\sf P}(T_f \leq t) = \mbox{\sf P}(D(t) \geq 1) = \mbox{\sf P}(K \leq \epsilon D_\beta (t)).

Here :math:`K=E^{-1}\epsilon` takes care of the uncertainty in the
material. In the previous section we used and estimated a lognormal
distribution for the variation of :math:`K` around :math:`\epsilon`,
when we assumed that :math:`\ln K = \ln \epsilon - \ln E` is normal with
mean :math:`\ln \epsilon` and standard deviation :math:`\sigma _E`.

The cycle sum :math:`D_\beta(t)` is the sum of a large number of damage
terms, only dependent on the cycles. For loads with short memory one can
assume that :math:`D_\beta(t)` is approximately normal,

.. math:: D_\beta(t) \approx N(d_\beta t,\, \sigma_\beta ^2 \,t),

where

.. math::

   d_\beta = \lim _{t \to \infty} \frac{D_\beta (t)}{t} \qquad \mbox{and} \qquad
   \sigma_\beta ^2 =  \lim _{t \to \infty} \frac{V(D_\beta (t))}{t}.

Thus the fatigue life distribution can be computed by combining the
lognormal distribution for :math:`K` with the normal distribution for
:math:`D_\beta (t)`. Denoting the standard normal density and
distribution functions by :math:`\phi(x)` and :math:`\Phi(x)`,
respectively, an approximate explicit expression for the failure
probability within time :math:`t` is

.. math::

   \mbox{\sf P}(T^f \leq t) \approx \int_{-\infty}^\infty
   \Phi \left(
   \frac{\ln \epsilon + \ln d_\beta t +
   \ln (1 + \frac{\sigma_\beta}{d_\beta \sqrt{t}}z)}{\sigma_E}
   \right) \phi (z) \,dz.
   \label{eq:failuretimedistribution}

We have already estimated the material dependent parameters
:math:`\epsilon` ``= e0``, :math:`\beta` ``= beta0``, and
:math:`\sigma _E ^2` ``= s20``, in the S-N data, so we need the damage
intensity :math:`d_\beta` and its variability :math:`\sigma _\beta` for
the acting load.

We first investigate the effect of uncertainty in the
:math:`\beta`-estimate.

::

         beta = 3:0.1:8;
         DRFC = cc2dam(RFC_sea,beta);
         dRFC = DRFC/T_sea;
         plot(beta,dRFC), axis([3 8 0 0.25])
         title('Damage intensity as function of \beta')

The plot in Figure `3.21 <#fig_wafo_6.17>`__ shows the increase in
damage with increasing :math:`\beta`.

.. figure:: fatigue_19
   :alt: Increasing damage intensity from sea-load with increasing
   :math:`\beta`.
   :name: fig_wafo_6.17
   :width: 80mm

   Increasing damage intensity from sea-load with increasing
   :math:`\beta`.

Next, we shall see how the load variability affects the fatigue life. We
use three different values for :math:`\sigma _\beta ^2`, namely
:math:`0`, :math:`0.5`, and :math:`5`. With ``beta0``, ``e0``, ``s20``
estimated in Section `3.4.4 <#sec:estimationofSNcurve>`__, we compute
and plot the following three possible fatigue life distributions.

::

         dam0 = cc2dam(RFC_sea,beta0)/T_sea;
         [t0,F0] = ftf(e0,dam0,s20,0.5,1);
         [t1,F1] = ftf(e0,dam0,s20,0,1);
         [t2,F2] = ftf(e0,dam0,s20,5,1);
         plot(t0,F0,t1,F1,t2,F2)

Here, the fourth parameter is the value of :math:`\sigma _\beta^2` used
in the computation; see ``help ftf``.

.. figure:: fatigue_20
   :alt: Fatigue life distribution with sea load.
   :name: fatigue_20
   :width: 80mm

   Fatigue life distribution with sea load.

The resulting fatigue life distribution function is shown in
Figure `3.22 <#fatigue_20>`__. As seen, the curves are identical,
indicating that the correct value of :math:`\sigma _\beta ^2` is not
important for such small :math:`\epsilon`-values as are at hand here.
Hence, one can use :math:`\sigma _\beta ^2 = 0`, and assume that the
damage accumulation process is proportional to time.

.. _sec:complexloads:

Fatigue analysis of complex loads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loads which cause fatigue are rarely of the homogeneous and stationary
character as the loads used in the previous sections. On the contrary,
typical load characteristics often change their value during the life
time of a structure, for example, load spectra on an airplane part have
very different fatigue properties during the different stages of an air
mission. Marin loads on a ship are quite different during loading and
unloading, compared to a loaded ocean voyage, and the same holds for any
road vehicle.

The Wafo toolbox can be used to analyse also loads of complex structure
and we shall illustrate some of these capabilities in this section. To
be eligible for Wafo-analysis, the loads have to have a piecewise
stationary character, for example the mean level or the standard
deviation may take two distinct levels and change abruptly, or the
frequency content can alternate between two modes, one irregular and one
more regular. Such processes are called *switching processes*. A
flexible family of switching loads are those where the change between
the different stationary states is governed by a Markov chain. Wafo
contains a special package of routines for analysis of such switching
Markov loads, based on methods by Johannesson, (Johannesson 1998, 1999).

In the following example the load alternates between two different mean
levels, corresponding to one heavy-load state (1) and one light-load
state (2). In Figure `3.23 <#fatigue_21>`__ the observed load is shown
in the upper part. The alternating curve in the lower part shows the
switches between the two states.

.. figure:: FigEx2SamplePath
   :alt: Simulated switching load with two states. Upper graph shows the
   load, and the states are indicated in the lower graph.
   :name: fatigue_21
   :width: 80mm

   Simulated switching load with two states. Upper graph shows the load,
   and the states are indicated in the lower graph.

As long as the load is in one of the states, the rainflow cycles are
made up of alternations between turning points belonging only to that
part of the load. When the state changes there is introduced extra
rainflow cycles with larger amplitudes. These extra cycles can be seen
in the total rainflow matrix, shown in
Figure `[fatigue_22] <#fatigue_22>`__. The two large groups of cycles
around (min,max) = (0.5, 0.75) and (min,max) = (0, 0) come from states
(1) and (2), respectively. The contribution from the switching is seen
in the small assembly of cycles around (min,max) = (-0.5, 1).

More details on how to analyse and model switching loads can be found in
(Johannesson 1997).

.. container:: references hanging-indent
   :name: refs

   .. container::
      :name: ref-FrendahlAndRychlik1993Rainflow

      Frendahl, M., and I. Rychlik. 1993. “Rainflow Analysis: Markov
      Method.” *Int. J. Fatigue* 15: 265–72.

   .. container::
      :name: ref-Johannesson1997Matlab

      Johannesson, P. 1997. *Matlab Toolbox: Rainflow Cycles for
      Switching Processes, V. 1.0*. Department of Mathematical
      Statistics, Lund Institute of Technology.

   .. container::
      :name: ref-Johannesson1998Rainflow

      ———. 1998. “Rainflow Cycles for Switching Processes with Markov
      Structure.” *Prob. Eng. Inform. Sci.* 12 (2): 143–75.

   .. container::
      :name: ref-Johannesson1999Rainflow

      ———. 1999. “Rainflow Analysis of Switching Markov Loads.”
      PhD thesis, Math. Stat., Center for Math. Sci., Lund Univ.,
      Sweden.

   .. container::
      :name: ref-MatsuishiAndEndo1968Fatigue

      Matsuishi, M., and T.: Endo. 1968. “Fatigue of Metals Subject to
      Varying Stress.” Paper presented to Japan Soc. Mech. Engrs,
      Jukvoka, Japan.

   .. container::
      :name: ref-Rychlik1987New

      Rychlik, I. 1987. “A New Definition of the Rainflow Cycle Counting
      Method.” *Int. J. Fatigue* 9: 119–21.

   .. container::
      :name: ref-Rychlik1988Rainflow

      ———. 1988. “Rain-Flow-Cycle Distribution for Ergodic Load
      Processes.” *SIAM J. Appl. Math.* 48: 662–79.

   .. container::
      :name: ref-Rychlik1995Simulation

      ———. 1995. “Simulation of Load Sequences from Rainflow Matrices:
      Markov Method.” Stat. Research Report 29. Dept. of Mathematical
      Statistics, Lund, Sweden.

   .. container::
      :name: ref-RychlikLindgrenLin1995

      Rychlik, Igor, Georg Lindgren, and Y. K. Lin. 1995. “Markov based
      correlations of damage cycles in Gaussian and non-Gaussian loads.”
      *Prob. Eng. Mech.* 10 (2): 103–15.
      http://dx.doi.org/10.1016/0266-8920(95)00001-F.

.. [1]
   We have defined :math:`f_0` as the mean level upcrossing frequency,
   i.e. the mean number of times per time unit that the load upcrosses
   the mean level. Thus there are in fact at least :math:`f_0` local
   maxima per time unit. Since the rainflow filter reduces the number of
   cycles, we let :math:`f_0` here be *defined as* the average number of
   rainflow cycles per time unit.

.. [2]
   We have defined :math:`f_0` as the mean level upcrossing frequency,
   i.e. the mean number of times per time unit that the load upcrosses
   the mean level. Thus there are in fact at least :math:`f_0` local
   maxima per time unit. Since the rainflow filter reduces the number of
   cycles, we let :math:`f_0` here be *defined as* the average number of
   rainflow cycles per time unit.

.. |Upper figures show target crossing spectrum (smooth curve) and obtained spectrum (wiggled curve) for simulated process shown in lower figures. Irregularity factor: left :math:`\alpha=0.25`, right :math:`\alpha=0.75`.| image:: fatigue_14_25
   :name: fig_wafo_6.9
.. |Upper figures show target crossing spectrum (smooth curve) and obtained spectrum (wiggled curve) for simulated process shown in lower figures. Irregularity factor: left :math:`\alpha=0.25`, right :math:`\alpha=0.75`.| image:: fatigue_14_75
   :name: fig_wafo_6.9
.. |Estimation of S-N-model on linear and log-log scale.| image:: fatigue_18a
   :name: fig_wafo_6.15
.. |Estimation of S-N-model on linear and log-log scale.| image:: fatigue_18b
   :name: fig_wafo_6.15
.. |Upper figures show target crossing spectrum (smooth curve) and obtained spectrum (wiggled curve) for simulated process shown in lower figures. Irregularity factor: left :math:`\alpha=0.25`, right :math:`\alpha=0.75`.| image:: fatigue_14_25
   :name: fig_wafo_6.9
   :width: 70mm
.. |Upper figures show target crossing spectrum (smooth curve) and obtained spectrum (wiggled curve) for simulated process shown in lower figures. Irregularity factor: left :math:`\alpha=0.25`, right :math:`\alpha=0.75`.| image:: fatigue_14_75
   :name: fig_wafo_6.9
   :width: 70mm
.. |Estimation of S-N-model on linear and log-log scale.| image:: fatigue_18a
   :name: fig_wafo_6.15
   :width: 70mm
.. |Estimation of S-N-model on linear and log-log scale.| image:: fatigue_18b
   :name: fig_wafo_6.15
   :width: 70mm
