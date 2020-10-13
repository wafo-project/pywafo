
.. _cha:KDE:

Kernel density estimation
=========================

Histograms are among the most popular ways to visually present data.
They are particular examples of density estimates and their appearance
depends on both the choice of origin and the width of the intervals
(bins) used. In order for the histogram to give useful information about
the true underlying distribution, a sufficient amount of data is needed.
This is even more important for histograms in two dimensions or higher.
Also the discontinuity of the histograms may cause problems, e.g., if
derivatives of the estimate are required.

An effective alternative to the histogram is the kernel density estimate
(KDE), which may be considered as a “smoothed histogram”, only depending
on the bin-width and not depending on the origin, see (Silverman 1986).

The univariate kernel density estimator
---------------------------------------

The univariate KDE is defined by

.. math::

   \label{eq:kdeformula}
     \hat{f}_{X}(x;h_{s}) = \frac{1}{n\,h_{s}}
                            \sum_{j=1}^{n} K_{d}\left( \frac{x-X_{j}}{h_{s}}\right),

where :math:`n` is the number of datapoints,
:math:`X_{1},X_{2},\ldots,X_{n}`, is the data set, and :math:`h_{s}` is
the smoothing parameter or window width. The kernel function
:math:`K_{d}` is usually a unimodal, symmetric probability density
function. This ensures that the KDE itself is also a density. However,
kernels that are not densities are also sometimes used (see (Wand and
Jones 1995)), but these are not implemented in the Wafo toolbox.

To illustrate the method, consider the kernel estimator as a sum of
“bumps” placed at the observations. The shape of the bumps are given by
the kernel function while the width is given by the smoothing parameter,
:math:`h_{s}`. Fig. `8.3 <#fig:kdedemo1>`__ shows a KDE constructed
using 7 observations from a standard Gaussian distribution with a
Gaussian kernel function. One should note that the 7 points used here,
is purely for clarity in illustrating how the kernel method works.
Practical density estimation usually involves much higher number of
observations.

Fig. `8.3 <#fig:kdedemo1>`__ also demonstrates the effect of varying the
smoothing parameter, :math:`h_{s}`. A too small value for :math:`h_{s}`
may introduce spurious bumps in the resulting KDE (top), while a too
large value may obscure the details of the underlying distribution
(bottom). Thus the choice of value for the smoothing parameter,
:math:`h_{s}`, is very important. How to select one will be elaborated
further in the next section.

.. figure:: kdedemo1f1
   :alt: Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).
   :name: fig:kdedemo1
   :height: 2.5in

   Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).

.. figure:: kdedemo1f2
   :alt: Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).
   :name: fig:kdedemo1
   :height: 2.5in

   Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).

.. figure:: kdedemo1f3
   :alt: Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).
   :name: fig:kdedemo1
   :height: 2.5in

   Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).

The particular choice of kernel function, on the other hand, is not very
important since suboptimal kernels are not suboptimal by very much, (see
pp. 31 in (Wand and Jones 1995)). However, the kernel that minimizes the
mean integrated square error is the Epanechnikov kernel, and is thus
chosen as the default kernel in the software, see
Eq. (`[eq:multivariateEpan] <#eq:multivariateEpan>`__). For a discussion
of other kernel functions and their properties, see (Wand and Jones
1995).

Smoothing parameter selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The choice of smoothing parameter, :math:`h_{s}`, is very important, as
exemplified in Fig.\ `8.3 <#fig:kdedemo1>`__. In many situations it is
satisfactory to select the smoothing parameter subjectively by eye,
i.e., look at several density estimates over a range of bandwidths and
selecting the density that is the most “pleasing” in some sense.
However, there are also many circumstances where it is beneficial to use
an automatic bandwidth selection from the data. One reason is that it is
very time consuming to select the bandwidth by eye. Another reason, is
that, in many cases, the user has no prior knowledge of the structure of
the data, and does not have any feeling for which bandwidth gives a good
estimate. One simple, quick and commonly used automatic bandwidth
selector, is the bandwidth that minimizes the mean integrated square
error (MISE) asymptotically. As shown in (Wand and Jones 1995 Section
2.5 and 3.2.1), the one dimensional AMISE [2]_-optimal normal scale rule
assuming that the underlying density is Gaussian, is given by

.. math::

   \label{eq:Hamise}
     h_{AMISE} =
     %  \left[\frac{8\,\sqrt{\pi}\,R(K_{d})}{3\,\mu_{2}^{2}(K_{d})
     %\,n}\right]^{1/5}\,\widehat{\sigma} =
   \left[\frac{4}{3\,n}\right]^{1/5}\,\widehat{\sigma} ,

where :math:`\widehat{\sigma}` is some estimate of the standard
deviation of the underlying distribution. Common choices of
:math:`\widehat{\sigma}` are the sample standard deviation,
:math:`\widehat{\sigma}_{s}`, and the standardized interquartile range
(denoted IQR):

.. math::

   \label{eq:Oiqr}
     \widehat{\sigma}_{IQR} = (\text{sample IQR})/
     (\Phi^{-1}(3/4)-\Phi^{-1}(1/4)) \approx  (\text{sample IQR})/1.349 ,

where :math:`\Phi^{-1}` is the standard normal quantile function. The
use of :math:`\widehat{\sigma}_{IQR}` guards against outliers if the
distribution has heavy tails. A reasonable approach is to use the
smaller of :math:`\widehat{\sigma}_{s}` and
:math:`\widehat{\sigma}_{IQR}` in order to lessen the chance of
oversmoothing, (see Silverman 1986, 47).

Various other automatic methods for selecting :math:`h_{s}` are
available and are discussed in (Silverman 1986) and in more detail in
(Wand and Jones 1995).

Transformation kernel denstity estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Densities close to normality appear to be the easiest for the kernel
estimator to estimate. The estimation difficulty increases with
skewness, kurtosis and multimodality (Chap. 2.9 in (Wand and Jones
1995)).

Thus, in the cases where the random sample
:math:`X_{1},X_{2},\ldots,X_{n}`, has a density, :math:`f`, which is
difficult to estimate, a transformation, :math:`t`, might give a good
KDE, i.e., applying a transformation to the data to obtain a new sample
:math:`Y_{1},Y_{2},\ldots,Y_{n}`, with a density :math:`g` that more
easily can be estimated using the basic KDE. One would then
backtransform the estimate of :math:`g` to obtain the estimate for
:math:`f`.

Suppose that :math:`Y_{i} = t(X_{i})`, where :math:`t` is an increasing
differentiable function defined on the support of :math:`f`. Then a
standard result from statistical distribution theory is that

.. math::

   \label{eq:transform}
     f(x) = g(t(x))\,t'(x),

where :math:`t'(x)` is the derivative. Backtransformation of the KDE of
:math:`g` based on :math:`Y_{1},Y_{2},\ldots,Y_{n}`, leads to the
explicit formula

.. math::

   \label{eq:transformkdeformula}
     \hat{f}_{X}(x;h_{s},t) = \frac{1}{n\,h_{s}} \sum_{j=1}^{n}
   K_{d}\left( \frac{t(x)-t(X_{j})}{h_{s}}\right)\,t'(x)

A simple illustrative example comes from the problem of estimating the
Rayleigh density. This density is very difficult to estimate by direct
kernel methods. However, if we apply the transformation
:math:`Y_{i} = \sqrt{X_{i}}` to the data, then the normal plot of the
transformed data, :math:`Y_{i}`, becomes approximately linear.
Fig. `8.5 <#fig:transformkde>`__ shows that the transformation KDE is a
better estimate around 0 than the ordinary KDE.

.. figure:: rayleightransformedkde
   :alt: True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.
   :name: fig:transformkde
   :width: 2.5in

   True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.

.. figure:: rayleighkde
   :alt: True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.
   :name: fig:transformkde
   :width: 2.5in

   True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.

.. _sec:multivariateKDE:

The multivariate kernel density estimator
-----------------------------------------

The multivariate kernel density estimator is defined in its most general
form by

.. math::

   \label{eq:multivariateKDE}
     \widehat{f}_{\ensuremath{\mathbf{X} }}(\ensuremath{\mathbf{x} };\ensuremath{\mathbf{H} }) = \frac{|\ensuremath{\mathbf{H} }|^{-1/2}}{n}
    \sum_{j=1}^{n} K_{d}\left(\ensuremath{\mathbf{H} }^{-1/2}(\ensuremath{\mathbf{x} }-\ensuremath{\mathbf{X} }_{j})\right),

where :math:`\ensuremath{\mathbf{H} }` is a symmetric positive definite
:math:`d \times d` matrix called the *bandwidth matrix*. A
simplification of Eq. (`[eq:multivariateKDE] <#eq:multivariateKDE>`__)
can be obtained by imposing the restriction
:math:`\ensuremath{\mathbf{H} } = \text{diag}(h_{1}^{2}, h_{2}^{2}, \ldots ,
h_{d}^{2})`. Then Eq. (`[eq:multivariateKDE] <#eq:multivariateKDE>`__)
reduces to

.. math::

   \label{eq:multivariateKDE2}
     \widehat{f}_{\ensuremath{\mathbf{X} }}(\ensuremath{\mathbf{x} };\ensuremath{\mathbf{h} }) =
     \frac{1}{n\,\prod_{i=1}^{n}\,h_{i}}
     \sum_{j=1}^{n} K_{d}\left(\frac{x-X_{j\,1}}{h_{1}},
       \frac{x-X_{j\,2}}{h_{2}},\ldots,
       \frac{x-X_{j\,d}}{h_{d}} \right),

and is, in combination with a transformation, a reasonable solution to
visualize multivariate densities.

The multivariate Epanechnikov kernel also forms the basis for the
optimal spherically symmetric multivariate kernel and is given by

.. math::

   \label{eq:multivariateEpan}
     K_{d}(\ensuremath{\mathbf{x} }) = \frac{d+2}{2\,v_{d}}
     \left(1-\ensuremath{\mathbf{x} }^{T}\ensuremath{\mathbf{x} } \right)\ensuremath{\mathbf{1} }_{\ensuremath{\mathbf{x} }^{T}\ensuremath{\mathbf{x} }\le 1},

where :math:`v_{d}=2\,\pi^{d/2}/( \Gamma(d/2) \, d )` is the volume of
the unit :math:`d`-dimensional sphere.

In this tutorial we use the KDE to find a good estimator of the central
part of the joint densities of wave parameters extracted from time
series. Clearly, such data are dependent, so it is assumed that the time
series are ergodic and short range dependent to justify the use of
KDE:s, (see Chap. 6 in (Wand and Jones 1995)). Usually, KDE gives poor
estimates of the tail of the distribution, unless large amounts of data
is available. However, a KDE gives qualitatively good estimates in the
regions of sufficient data, i.e., in the main parts of the distribution.
This is good for visualization, e.g. detecting modes, symmetries of
distributions.

The kernel density estimation software is based on ``KDETOOL``, which is
a Matlab toolbox produced by Christian Beardah, (Beardah and Baxter
1996), which has been totally rewritten and extended to include the
transformation kernel estimator and generalized to cover any dimension
for the data. The computational speed has also been improved.

.. _cha:waveSpectra:

Standardized wave spectra
=========================

Knowledge of which kind of spectral density is suitable to describe sea
state data are well established from experimental studies. Qualitative
considerations of wave measurements indicate that the spectra may be
divided into 3 parts, (see Fig. `9.1 <#fig:qspecvar20>`__):

#. Sea states dominated by wind sea but significantly influenced by
   swell components.

#. More or less pure wind seas or, possibly, swell component located
   well inside the wind frequency band.

#. Sea states more or less dominated by swell but significantly
   influenced by wind sea.

.. figure:: qspecvar1
   :alt: Qualitative indication of spectral variability.
   :name: fig:qspecvar20
   :width: 4in

   Qualitative indication of spectral variability.

One often uses some parametric form of the spectral density. The three
most important parametric spectral densities implemented in Wafo will be
described in the following sections.

.. _sec:jonswap:

Jonswap spectrum
----------------

The Jonswap (JOint North Sea WAve Project) spectrum of (Hasselmann et
al. 1973) is a result of a multinational project to characterize
standardized wave spectra for the Southeast part of the North Sea. The
spectrum is valid for not fully developed sea states. However, it is
also used to represent fully developed sea states. It is particularly
well suited to characterize wind sea when
:math:`3.6 \sqrt{H_{m0}} < T_{p} < 5 \sqrt{H_{m0}}`. The Jonswap
spectrum is given in the form:

.. math::

   \begin{aligned}
      S^{+}(\ensuremath{\omega }) &= \frac{\alpha \, g^{2}}{\ensuremath{\omega }^{M}}
      \exp \Big( -\frac{M}{N} \, \big(   \frac{\ensuremath{\omega }_{p}}{\ensuremath{\omega }} \big)^{N} \Big) \,
      \gamma^{\exp \Big( \frac{-(\ensuremath{\omega }/ \ensuremath{\omega }_{p}-1)^{2}}{2 \, \sigma^{2}} \Big)},  \label{eq:jonswap} \\
   \intertext{where}
   \sigma &=  \begin{cases}
        0.07 & \text{if} \; \ensuremath{\omega }< \ensuremath{\omega }_{p}, \\
        0.09 & \text{if} \; \ensuremath{\omega }\ge \ensuremath{\omega }_{p},
       \end{cases} \nonumber \\
     M &= 5, \quad  N = 4, \nonumber \\
     \alpha &\approx 5.061 \frac{H_{m0}^{2}}{T_{p}^{4}} \,\Big\{ 1-0.287\, \ln(\gamma)  \Big\}. \nonumber
    \end{aligned}

A standard value for the peakedness parameter, :math:`\gamma`, is
:math:`3.3`. However, a more correct approach is to relate
:math:`\gamma` to :math:`H_{m0}` and :math:`T_{p}`, and use

.. math::

   \begin{gathered}
     \gamma = \exp \Big\{3.484 \,\big(1-0.1975\,(0.036-0.0056\,T_{p}/\sqrt{H_{m0}})
   \,T_{p}^{4}/H_{m0}^2\big) \Big\} .
   %\intertext{where}
   % D = 0.036-0.0056\,T_{p}/\sqrt{H_{m0}}\end{gathered}

Here :math:`\gamma` is limited by :math:`1 \le \gamma \le 7`. This
parameterization is based on qualitative considerations of deep water
wave data from the North Sea; see (Torsethaugen and others 1984) and
(Haver and Nyhus 1986).

The relation between the peak period and mean zero-upcrossing period may
be approximated by

.. math:: T_{m02} \approx T_{p}/\left(1.30301-0.01698\,\gamma+0.12102/\gamma \right)

The Jonswap spectrum is identical with the two-parameter
Pierson-Moskowitz, Bretschneider, ITTC (International Towing Tank
Conference) or ISSC (International Ship and Offshore Structures
Congress) wave spectrum, given :math:`H_{m0}` and :math:`T_{p}`, when
:math:`\gamma=1`. (For more properties of this spectrum, see the Wafo
function ``jonswap.m``.

.. _sec:torsethaugen:

Torsethaugen spectrum
---------------------

Torsethaugen, (Torsethaugen 1993, 1994, 1996), proposed to describe
bimodal spectra by

.. math::

   \begin{gathered}
    S^{+}(\ensuremath{\omega }) = \sum_{i=1}^{2} S_{J}^{+}(\ensuremath{\omega };H_{m0,i},\ensuremath{\omega }_{p,i},\gamma_{i},N_{i},M_{i},\alpha_{i})\end{gathered}

where :math:`S_{J}^{+}` is the Jonswap spectrum defined by
Eq. (`[eq:jonswap] <#eq:jonswap>`__). The parameters :math:`H_{m0,\,i}`,
:math:`\ensuremath{\omega }_{p,\,i}`, :math:`N_{i}`, :math:`M_{i}`, and
:math:`\alpha_{i}` for :math:`i=1,2`, are the significant wave height,
angular peak frequency, spectral shape and normalization parameters for
the primary and secondary peak, respectively.

These parameters are fitted to 20 000 spectra divided into 146 different
classes of :math:`H_{m0}` and :math:`T_{p}` obtained at the Statfjord
field in the North Sea in the period from 1980 to 1989. The measured
:math:`H_{m0}` and :math:`T_{p}` values for the data range from
:math:`0.5` to :math:`11` meters and from :math:`3.5` to :math:`19`
seconds, respectively.

Given :math:`H_{m0}` and :math:`T_{p}` these parameters are found by the
following steps. The borderline between wind dominated and swell
dominated sea states is defined by the fully developed sea, for which

.. math:: T_{p} = T_{f} = 6.6 \, H_{m0}^{1/3},

while for :math:`T_{p} < T_{f}`, the local wind sea dominates the
spectral peak, and if :math:`T_{p} > T_{f}`, the swell peak is
dominating.

For each of the three types a non-dimensional period scale is introduced
by

.. math::

   \begin{aligned}
   \epsilon_{l\,u} &= \frac{T_{f}-T_{p}}{T_{f}-T_{lu}},
   \intertext{where}
   T_{lu} &=
   \begin{cases}
        2 \, \sqrt{H_{m0}} & \text{if} \; T_{p} \le T_{f} \quad
        \text{(Lower limit)},\\
        25 & \text{if} \; T_{p} > T_{f} \quad
        \text{(Upper limit)},
       \end{cases}
   \intertext{defines the lower or upper value for $T_{p}$.
   The significant wave height for each peak is given as}
   H_{m0,1} &= R_{pp} \, H_{m0} \quad  H_{m0,2} = \sqrt{1-R_{pp}^{2}} \, H_{m0},
   \intertext{where}
   R_{pp} &= \big(1-A_{10}\big) \,\exp\Big(-\big(\frac{\epsilon_{l\,u}}{A_{1}}
   \big)^{2} \Big) +A_{10}, \\
   %\label{eq:A-values}
   A_{1} &= \begin{cases}
      0.5 & \text{if} \; T_{p} \le T_{f}, \\
      0.3 & \text{if} \; T_{p} > T_{f},
       \end{cases} \quad
   A_{10} = \begin{cases}
      0.7 & \text{if} \; T_{p} \le T_{f}, \\
      0.6 & \text{if} \; T_{p} > T_{f}.
       \end{cases}\end{aligned}

The primary and secondary peak periods are defined as

.. math::

   \begin{aligned}
   T_{p,\,1} &= T_{p}, \\
   T_{p,\,2} &=
   \begin{cases}
       T_{f} + 2  & \text{if} \; T_{p} \le T_{f}, \\[0.3em]
       \left( \frac{M_{2}\,(N_{2}/M_{2})^{(N_{2}-1)/M_{2}}/\Gamma((N_{2}-1)/M_{2} )}
       {1.28\,(0.4)^{N_{2}} \{1-\exp (-H_{m0,\,2}/3)
         \} } \right)^{1/(N_{2}-1)} & \text{if} \; T_{p} > T_{f},
   \end{cases}
   \intertext{where the spectral shape parameters are given as}
   N_{1} &= N_{2} = 0.5\, \sqrt{H_{m0}}+3.2, \\
   M_{i} &=  \begin{cases}
   4\, \Big( 1-0.7\exp
   \big(\frac{-H_{m0}}{3}\big)\Big) & \text{if} \; T_{p} > T_{f} \;
   \text{and} \; i=2, \\
   4 & \text{otherwise}.
     \end{cases}\end{aligned}

The peakedness parameters are defined as

.. math::

   \begin{aligned}
   \gamma_{1} &= 35 \,\Big(1+3.5\, \exp \big( -H_{m0} \big)\Big) \gamma_{T}, \qquad \gamma_{2} = 1,
   \intertext{where}
   \gamma_{T} &= \begin{cases}
   \Big( \frac{2 \, \pi \, H_{m0,\,1}}{g \, T_{p}^{2}}\Big)^{0.857} & \text{if} \; T_{p} \le T_{f}, \\
   \big(1+6\,\epsilon_{lu}\big) \Big( \frac{2 \,  \pi \, H_{m0}}{g \, T_{f}^{2}}\Big)^{0.857} & \text{if} \; T_{p} > T_{f}.
       \end{cases}\end{aligned}

Finally the normalization parameters :math:`\alpha_{i}` (:math:`i=1,2`)
are found by numerical integration so that

.. math::

   \begin{aligned}
   \int_{0}^{\infty}
   S_{J}^{+}(\ensuremath{\omega };H_{m0,i},\ensuremath{\omega }_{p,i},\gamma_{i},N_{i},M_{i},\alpha_{i})\,d\ensuremath{\omega }
   = H_{m0,\,i}^{2}/16.\end{aligned}

Preliminary comparisons with spectra from other areas indicate that the
empirical parameters in the Torsethaugen spectrum can be dependent on
geographical location. This spectrum is implemented as a matlab function
``torsethaugen.m`` in the Wafo toolbox.

.. _sec:ochi-hubble:

Ochi-Hubble spectrum
--------------------

Ochi and Hubble (Ochi and Hubble 1976), suggested to describe bimodal
spectra by a superposition of two modified Bretschneider
(Pierson-Moskovitz) spectra:

.. math::

   \begin{aligned}
      \label{eq:ohspec1} %
      S^{+}(\ensuremath{\omega }) &=\frac{1}{4} \sum_{i=1}^{2}
      \frac{\big( \big( \lambda_{i}+1/4 \big) \, \ensuremath{\omega }_{p,\,i}^{4}
   \big)^{\lambda_{i}}}{\Gamma(\lambda_{i})}
      \frac{H_{m0,\,i}^{2}}{\ensuremath{\omega }^{4 \, \lambda_{i}}+1}
      \exp \Big( \frac{-\big( \lambda_{i}+1/4 \big) \, \ensuremath{\omega }_{p,\,i}^{4}}{\ensuremath{\omega }^{4}}\Big),
    \end{aligned}

where :math:`H_{m0,\,i}`, :math:`\ensuremath{\omega }_{p,\,i}`, and
:math:`\lambda_{i}` for :math:`i=1,2`, are significant wave height,
angular peak frequency, and spectral shape parameter for the low and
high frequency components, respectively.

The values of these parameters are determined from an analysis of data
obtained in the North Atlantic. The source of the data is the same as
that for the development of the Pierson-Moskowitz spectrum, but analysis
is carried out on over :math:`800` spectra including those in partially
developed seas and those having a bimodal shape. In contrast to the
Jonswap and Torsethaugen spectra, which are parameterized as function of
:math:`H_{m0}` and :math:`T_{p}`, Ochi and Hubble, (Ochi and Hubble
1976) gave, from a statistical analysis of the data, a family of wave
spectra consisting of 11 members generated for a desired sea severity
(:math:`H_{m0}`) with the coefficient of :math:`0.95`.

The values of the six parameters as functions of :math:`H_{m0}` are
given as:

.. math::

   \begin{aligned}
    H_{m0,1} &= R_{p,1} \, H_{m0}, \\
    H_{m0,2} &= \sqrt{1-R_{p,1}^{2}} \, H_{m0}, \\
    \ensuremath{\omega }_{p,i} &= a_{i}\,\exp \big( -b_{i}\, H_{m0} \big), \\
    \lambda_{i} &= c_{i}\,\exp \big( -d_{i}\, H_{m0} \big),\end{aligned}

where :math:`d_{1}=0` and the remaining empirical constants
:math:`a_{i}`, :math:`b_{i}` (:math:`i=1,2`), and :math:`d_{2}`, are
given in Table `9.1 <#tab:oh-parameters>`__. (See also the function
``ochihubble.m`` in the Wafo toolbox.)

.. container::
   :name: tab:oh-parameters

   .. table:: Empirical parameter values for the Ochi-Hubble spectral
   model.

      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | M     | :math | :ma   | :ma   | :ma   | :ma   | :ma   | :ma   | :ma   |
      | ember | :`R_{ | th:`a | th:`a | th:`b | th:`b | th:`c | th:`c | th:`d |
      | no.   | p,1}` | _{1}` | _{2}` | _{1}` | _{2}` | _{1}` | _{2}` | _{2}` |
      +=======+=======+=======+=======+=======+=======+=======+=======+=======+
      | 1     | 0.84  | 0.70  | 1.15  | 0.046 | 0.039 | 3.00  | 1.54  | 0.062 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 2     | 0.84  | 0.93  | 1.50  | 0.056 | 0.046 | 3.00  | 2.77  | 0.112 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 3     | 0.84  | 0.41  | 0.88  | 0.016 | 0.026 | 2.55  | 1.82  | 0.089 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 4     | 0.84  | 0.74  | 1.30  | 0.052 | 0.039 | 2.65  | 3.90  | 0.085 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 5     | 0.84  | 0.62  | 1.03  | 0.039 | 0.030 | 2.60  | 0.53  | 0.069 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 6     | 0.95  | 0.70  | 1.50  | 0.046 | 0.046 | 1.35  | 2.48  | 0.102 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 7     | 0.65  | 0.61  | 0.94  | 0.039 | 0.036 | 4.95  | 2.48  | 0.102 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 8     | 0.90  | 0.81  | 1.60  | 0.052 | 0.033 | 1.80  | 2.95  | 0.105 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 9     | 0.77  | 0.54  | 0.61  | 0.039 | 0.000 | 4.50  | 1.95  | 0.082 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 10    | 0.73  | 0.70  | 0.99  | 0.046 | 0.039 | 6.40  | 1.78  | 0.069 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 11    | 0.92  | 0.70  | 1.37  | 0.046 | 0.039 | 0.70  | 1.78  | 0.069 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+

Member no. 1 given in Table `9.1 <#tab:oh-parameters>`__ defines the
most probable spectrum, while member no. 2 to 11 define the :math:`0.95`
percent confidence spectra.

A significant advantage of using a family of spectra for design of
marine systems is that one of the family members yields the largest
response such as motions or wave induced forces for a specified sea
severity, while anothers yield the smallest response with confidence
coefficient of :math:`0.95`.

Rodrigues and Soares (Rodriguez and Guedes Soares 2000), used the
Ochi-Hubble spectrum with 9 different parameterizations representing 3
types of sea state categories: swell dominated (a), wind sea dominated
(b) and mixed wind sea and swell system with comparable energy (c). Each
category is represented by 3 different inter-modal distances between the
swell and the wind sea spectral components. These three subgroups are
denoted by I, II and III, respectively. The exact values for the six
parameters are given in Table `9.2 <#tab:soares-param>`__. (See the
function ``ohspec3.m`` in the Wafo toolbox.)

.. container::
   :name: tab:soares-param

   .. table:: Target spectra parameters for mixed sea states.

      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       |       |       |       |       |       |       |     |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | type  |       |       |       |       |       |       |     |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | group | :ma   | :ma   | :     | :     | :mat  | :mat  |     |
      |       | th:`H | th:`H | math: | math: | h:`\l | h:`\l |     |
      |       | _{m0, | _{m0, | `\ens | `\ens | ambda | ambda |     |
      |       | \,1}` | \,2}` | urema | urema | _{1}` | _{2}` |     |
      |       |       |       | th{\o | th{\o |       |       |     |
      |       |       |       | mega  | mega  |       |       |     |
      |       |       |       | }_{p, | }_{p, |       |       |     |
      |       |       |       | \,1}` | \,2}` |       |       |     |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | I     | 5.5   | 3.5   | 0.440 | 0.691 | 3.0   | 6.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | a     | II    | 6.5   | 2.0   | 0.440 | 0.942 | 3.5   | 4.0 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | III   | 5.5   | 3.5   | 0.283 | 0.974 | 3.0   | 6.0 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | I     | 2.0   | 6.5   | 0.440 | 0.691 | 3.0   | 6.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | b     | II    | 2.0   | 6.5   | 0.440 | 0.942 | 4.0   | 3.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | III   | 2.0   | 6.5   | 0.283 | 0.974 | 2.0   | 7.0 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | I     | 4.1   | 5.0   | 0.440 | 0.691 | 2.1   | 2.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | c     | II    | 4.1   | 5.0   | 0.440 | 0.942 | 2.1   | 2.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | III   | 4.1   | 5.0   | 0.283 | 0.974 | 2.1   | 2.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+

.. _cha:wave-models:

Wave models
===========

Generally the wind generated sea waves is a non-linear random process.
Non-linearities are important in the wave-zone, i.e., from the crest to
1-2 wave amplitudes below the trough. Below this zone linear theory is
acceptable. However, there are unsolved physical problems associated
with the modelling of breaking waves. In practice, linear theory is used
to simulate irregular sea and to obtain statistical estimates. Often a
transformation of the linear model is used to emulate the non-linear
behavior of the sea surface, but as faster computers are becoming
available also higher order approximations will become common in ocean
engineering practice. In the following sections we will outline these
wave models. Only long-crested sea is used here either recorded at a
fixed spatial location or at a fixed point in time.

.. _sec:linear-gaussian-wave:

The linear Gaussian wave model
------------------------------

Gaussian random surface can be obtained as a first order approximation
of the solutions to differential equations based on linear hydrodynamic
theory of gravity waves. The first order component is given by the
following Fourier series

.. math::

   \label{eq:linearcomponent}
    \eta_{l}(x,t) = \sum_{n=-N}^{N} \frac{A_{n}}{2} e^{i\psi_{n}}

where the phase functions are

.. math::

   \label{eq:phasefunction}
     \psi_{n} = \omega_{n}\,t-k_{n}\,x  %- \epsilon_{n}

If :math:`\eta_{l}` is assumed to be stationary and Gaussian then the
complex amplitudes :math:`A_{j}` are also Gaussian distributed. The mean
square amplitudes are related to the one-sided wave spectrum
:math:`S_{\eta\eta}^{+}(\omega)` by

.. math::

   \label{eq:304}
     E[|A_{n }|^{2}] = 2\,S_{\eta\eta}^{+}(|\omega_{n}|) \Delta \omega

The individual frequencies, :math:`\ensuremath{\omega }_{n}` and
wavenumbers, :math:`k_{n}` are related through the linear dispersion
relation

.. math::

   \label{eq:dispersionrelation}
     \ensuremath{\omega }^{2} = g \,k\, \tanh(k\,d)

where :math:`g` and :math:`d` are the acceleration of gravity and water
depth, respectively. For deep water
Eq. (`[eq:dispersionrelation] <#eq:dispersionrelation>`__) simplifies to

.. math::

   \label{eq:29}
     \ensuremath{\omega }^{2} = g\,k

It implies the following relation between the wave frequency spectrum
and the wave number spectrum

.. math::

   \label{eq:33}
     S_{\eta\eta}^{+}(\ensuremath{\omega }) = \frac{2\ensuremath{\omega }}{g} S_{\eta\eta}^{+}(k)

Without loss of generality it is assumed that :math:`\eta_{l}` has zero
expectation. It is also assumed that :math:`\eta` is ergodic, i.e., any
ensemble average may be replaced by the corresponding time-space
average. This means that one single realization of :math:`\eta` is
representative of the random field. Here it is also assumed
:math:`\ensuremath{\omega }_{-j} = -\ensuremath{\omega }_{j}`,
:math:`k_{-j} = -k_{j}` and :math:`A_{-j} = \bar{A}_{j}` where
:math:`\bar{A}_{j}` is the complex conjugate of :math:`A_{j}`. The
matlab program ``spec2sdat.m`` in WAFO use the Fast Fourier Transform
(FFT) to evaluate Eq. (`[eq:linearcomponent] <#eq:linearcomponent>`__).

.. _sec:second-order-non:

The Second order non-linear wave model
--------------------------------------

Real wave data seldom follow the linear Gaussian model perfectly. The
model can be corrected by including quadratic terms. Following (Langley
1987) the quadratic correction :math:`\eta_{q}` is given by

.. math::

   \label{eq:nonlinearcomponent}
     \eta_{q}(x,t) = \sum_{n=-N}^{N} \sum_{m=-N}^{N} \frac{A_{n}A_{m}}{4} E(\ensuremath{\omega }_{n},\ensuremath{\omega }_{m})\,e^{i\,(\psi_{n}+\psi_{m})}

where the quadratic transferfunction (QTF),
:math:`E(\ensuremath{\omega }_{n},\ensuremath{\omega }_{m})` is given by

.. math::

   \label{eq:QTF}
   E(\ensuremath{\omega }_{i},\ensuremath{\omega }_{j}) = \frac{\frac{gk_{i}k_{j}}{\ensuremath{\omega }_{i}\ensuremath{\omega }_{j}} -
     \frac{1}{2g}(\ensuremath{\omega }_{i}^{2}+\ensuremath{\omega }_{j}^{2}+\ensuremath{\omega }_{i}\ensuremath{\omega }_{j})+\frac{g}{2}\frac{\ensuremath{\omega }_{i}k_{j}^{2}+\ensuremath{\omega }_{j}k_{i}^{2}}{\ensuremath{\omega }_{i}\,\ensuremath{\omega }_{j}(\ensuremath{\omega }_{i}+\ensuremath{\omega }_{j})}}{1-g\frac{k_{i}+k_{j}}{(\ensuremath{\omega }_{i}+\ensuremath{\omega }_{j})^{2}}\tanh\bigl((k_{i}+k_{j})d\bigr)}
   -\frac{gk_{i}k_{j}}{2\ensuremath{\omega }_{i}\ensuremath{\omega }_{j}}+\frac{1}{2g}(\ensuremath{\omega }_{i}^{2}+\ensuremath{\omega }_{j}^{2}+\ensuremath{\omega }_{i}\ensuremath{\omega }_{j})

For deep water waves the QTF simplifies to

.. math::

   \label{eq:EsumAndEdiff}
     E(\ensuremath{\omega }_{i},\ensuremath{\omega }_{j}) = \frac{1}{2\,g}(\ensuremath{\omega }_{i}^{2}+\ensuremath{\omega }_{j}^{2}),
   \quad
     E(\ensuremath{\omega }_{i},-\ensuremath{\omega }_{j}) = -\frac{1}{2\,g}|\ensuremath{\omega }_{i}^{2}-\ensuremath{\omega }_{j}^{2}|

where :math:`\ensuremath{\omega }_{i}` and
:math:`\ensuremath{\omega }_{j}` are positive and satisfies the same
relation as in the linear model.

However, if the spectrum does not decay rapidly enough towards zero, the
contribution from the 2nd order wave components at the upper tail can be
very large and unphysical. The predicted non-linearities are sensitive
to how the input spectrum is treated (cut-off) as shown by (Stansberg
1994).

One method to ensure convergence of the perturbation series is to
truncate the upper tail of the spectrum at
:math:`\ensuremath{\omega }_{max}` in the calculation of the 1st and 2nd
order wave components. The (Nestegård and Stokka 1995) program *WAVSIM*
set :math:`\ensuremath{\omega }_{max}=\sqrt{2.0\,g/(0.95\, H_{m0})}`.
(Brodtkorb, Myrhaug, and Rue 2000) showed that this will have the side
effect of giving the medium to low wave-heights a too low steepness
(which may not be too serious in real application). However, using the
truncation method the spectrum for the simulated series will deviate
from the target spectrum in 2 ways: (1) no energy or wave components
exist above the upper frequency limit
:math:`\ensuremath{\omega }_{max}`, (2) the energy in the spectrum below
:math:`\ensuremath{\omega }_{max}` will be higher than the target
spectrum. In order to retain energy above
:math:`\ensuremath{\omega }_{max}` in the spectrum, one may only
truncate the upper tail of the spectrum for the calculation of the 2nd
order components. However, in a real application one usually wants the
simulated data to have a prescribed target spectrum. Thus a more correct
approach is to eliminate the second order effects from the spectrum
before using it in the non-linear simulation. One way to do this is to
extract the linear components from the spectrum by a fix-point iteration
on the spectral density using the non-linear simulation program so that
the simulated data will have approximately the prescribed target
spectrum. This method is implemented as matlab function
``spec2linspec.m`` available in the WAFO toolbox. To accomplish
convergence, the same seed is used in each call of the non-linear
simulation program.

.. figure:: spec6comparisonNew
   :alt: Target spectrum, :math:`S_{T}(\ensuremath{\omega })`, (solid)
   and its linear component, :math:`S_{L}(\ensuremath{\omega })`
   (dash-dot) compared with :math:`S_{T}^{NLS}` (dash) and
   :math:`S_{L}^{NLS}` (dot), i.e., spectra of non-linearly simulated
   data using input spectrum :math:`S_{T}(\ensuremath{\omega })` (method
   1) and :math:`S_{L}(\ensuremath{\omega })` (method 2), respectively.
   :name: fig:spec6comparison
   :width: 3in

   Target spectrum, :math:`S_{T}(\ensuremath{\omega })`, (solid) and its
   linear component, :math:`S_{L}(\ensuremath{\omega })` (dash-dot)
   compared with :math:`S_{T}^{NLS}` (dash) and :math:`S_{L}^{NLS}`
   (dot), i.e., spectra of non-linearly simulated data using input
   spectrum :math:`S_{T}(\ensuremath{\omega })` (method 1) and
   :math:`S_{L}(\ensuremath{\omega })` (method 2), respectively.

Fig.\ `10.1 <#fig:spec6comparison>`__ demonstrates the differences in
the spectra obtained from data simulated using these methods. The solid
line is the target spectrum, :math:`S_{T}(\ensuremath{\omega })`, and
the dash-dotted line is its linear component,
:math:`S_{L}(\ensuremath{\omega })`, obtained using method 2. The
spectra :math:`S_{T}^{NLS}` (dashed) and :math:`S_{L}^{NLS}` (dotted)
are estimated from non-linearly simulated data using the
:math:`S_{T}(\ensuremath{\omega })` and
:math:`S_{L}(\ensuremath{\omega })` spectra, respectively. As expected
:math:`S_{T}^{NLS}` is higher than :math:`S_{T}`, while
:math:`S_{L}^{NLS}` is indistinguishable from :math:`S_{T}`. It is also
worth noting that the difference between the spectra is small, but have
some impact on the higher order moments. For instance, the
:math:`\epsilon_{2}` and :math:`\epsilon_{4}` parameters calculated from
:math:`S_{T}^{NLS}` increase with :math:`6.1\%` and :math:`2.5\%`,
respectively. The corresponding values calculated from
:math:`S_{L}^{NLS}` increase with only :math:`0.5\%` and :math:`0.2\%`,
respectively.

The small difference between :math:`S_{T}(\ensuremath{\omega })` and
:math:`S_{L}(\ensuremath{\omega })` also lends some support to the view
noted earlier, that the difference frequency effect can not fully
explain the high values of the spectrum in the lower frequency part as
found by (Wist 2003).

The effects these methods have are discussed further in (Brodtkorb 2004)
and especially on wave steepness parameters. The second order non-linear
model explained here is implemented in WAFO as ``spec2nlsdat.m``. This
is a very efficient implementation that calculate
Eqs. (`[eq:nonlinearcomponent] <#eq:nonlinearcomponent>`__) to
(`[eq:EsumAndEdiff] <#eq:EsumAndEdiff>`__) in the bi-frequency domain
using a one-dimensional *FFT*. This is similar to the *WAVSIM* program
of (Nestegård and Stokka 1995), but is made even more efficient by
summing over non-zero spectral values only and by eliminating the need
for saving the computed results to the hard drive. *WAVSIM* use
:math:`40\, s` to evaluate a transform with :math:`32000` time
steps/frequencies compared with :math:`2\, s` for ``spec2nlsdat.m`` on a
Pentium M :math:`1700` *MHz* with :math:`1` *GB* of RAM. Thus the use of
second order random waves should now become common in ocean engineering
practice.

``spec2nlsdat.m`` also allows finite water depth and any spectrum as
input, in contrast to *WAVSIM*, which only uses infinite water depth and
the JONSWAP spectrum.

.. _sec:transf-line-gauss:

Transformed linear Gaussian model
---------------------------------

An alternative and faster method than including the quadratic terms to
the linear model, is to use a transformation. The non-Gaussian process,
:math:`\eta(x,t)`, is then a function of a single Gaussian process,
:math:`\eta_{l}(x,t)`

.. math::

   \label{eq:tran1}
     \eta(x,t)=G(\eta_{l}(x,t))

where :math:`G(\cdot)` is a continuously differentiable function with
positive derivative.

There are several ways to proceed when selecting the transformation. The
simplest alternative is to estimate :math:`G(\cdot)` by some parametric
or non-parametric means (see e.g. (Winterstein 1988; Ochi and Ahn 1994;
Rychlik, Johannesson, and Leadbetter 1997)).

The parametric formulas proposed by (Ochi and Ahn 1994) as well as
(Winterstein 1988) use the moments of :math:`\eta_{l}(x,t)` to compute
:math:`G(\cdot)`. Information about the moments can be obtained directly
from data or by using theoretical models. (Marthinsen and Winterstein
1992) derived an expression for the skewness and kurtosis of narrow
banded Stokes waves to the leading order and used these to define the
transformation. (Winterstein and Jha 1995) fitted a parametric model to
skewness and kurtosis of a second order model with a JONSWAP spectrum.

(Machado 2003) studied the performance of 6 transformation methods
including those mentioned above and concluded that the Hermite method in
general produces very good results.

Hermite model
~~~~~~~~~~~~~

The Hermite transformation model proposed by (Winterstein 1985)
approximates the true process by the following transformation of a
standard normal process :math:`Y(t)`:

.. math::

   \begin{gathered}
     G(y) = \mu + K \,\sigma \,[ y + c_{3}(y^2-1) + c_{4} \,(y^3-3\,y)] \\
   %\intertext{where}
        K  = 1/\sqrt{1+2\,c_{3}^2+6\,c_{4}^2}\end{gathered}

where :math:`\mu` and :math:`\sigma` are the mean and standard
deviation, respectively, of the true process. The unitless coefficients
:math:`c_{3}` and :math:`c_{4}` are chosen so that the transformed model
match the skewness, :math:`\rho_{3}`, and excess, :math:`\rho_{4}`, of
the true process. (Winterstein, Ude, and Kleiven 1994) improved the
parameterizations by minimizing lack-of-fit errors on :math:`\rho_{3}`
and :math:`\rho_{4}`, giving

.. math::

   \begin{aligned}
        c_{3}  &= \frac{\rho_{3}}{6} \,
   \frac{1-0.015\,|\rho_{3}|+ 0.3\, \rho_{3}^2}{1+0.2\,\rho_{4}} \\
        c_{4}  &= 0.1\,\left( \left( 1+1.25\,\rho_{4} \right)^{1/3}-1 \right)\,c_{41}                 \\
        c_{41} &= \left(1-\frac{1.43\,\rho_{3}^2}{\rho_{4}} \right)^{1-0.1\,(\rho_{4}+3)^{0.8}}\end{aligned}

These results apply for :math:`0\le 3/2\,\rho_{3}^{2}<\rho_{4}<12`,
which include most cases of practical interest. One may then estimate
:math:`c_{3}` and :math:`c_{4}` using the sample skewness,
:math:`\hat{\rho}_{3}`, but restrict :math:`\rho_{4}` so that
:math:`\rho_{4} =
\min(\max(3\,\hat{\rho}_{3}^{2}/2,\hat{\rho}_{4}),\min(4\,(4\hat{\rho}_{3}/3)^{2},12))`.
:math:`\hat{\rho}_{4}` is the sample excess and
:math:`(4\hat{\rho}_{3}/3)^{2}` is the leading excess contribution for
narrow banded Stokes waves as found by (Marthinsen and Winterstein
1992).

| Mathematical Statistics
| Lund University
| Box 118
| SE-221 00 Lund
| Sweden
| http://www.maths.lth.se/

in the Hope that it is Useful

.. _foreword-1:

Foreword
========

.. _foreword-to-2017-edition-1:

Foreword to 2017 edition
------------------------

This Wafo tutorial 2017 has been successfully tested with Matlab 2017a
on Windows 10.

The tutorial for Wafo 2.5 appeared 2011, with routines tested on Matlab
2010b. Since then, many users have commented on the toolbox, suggesting
clarifications and corrections to the routines and to the tutorial text.
We are grateful for all suggestions, which have helped to keep the Wafo
project alive.

Major updates and additions have also been made duringing the years,
many of them caused by new Matlab versions. The new graphics system
introduced with Matlab2014b motivated updates to all plotting routines.
Syntax changes and warnings for deprecated functions have required other
updates.

Several additions have also been made. In 2016, a new module, handling
non-linear Lagrange waves, was introduced. A special tutorial for the
Lagrange routines is included in the module ``lagrange``; (Wafo Lagrange
– a Wafo Module for Analysis of Random Lagrange Waves 2017). Two sets of
file- and string-utility routines were also added 2016.

During 2015 the Wafo-project moved from
``http://code.google.com/p/wafo/`` to to
``https://github.com/wafo-project/``, where it can now be found under
the generic name Wafo – no version number needed.

In order to facilitate the use of Wafo outside the Matlab environment,
most of the Wafo routines have been checked for use with Octave. On
``github`` one can also find a start of a Python-version, called pywafo.

Recurring changes in the Matlab language may continue to cause the
command window flood with warnings for deprecated functions. The
routines in this version of Wafo have been updated to work well with
Matlab2017a. We will continue to update the toolbox in the future,
keeping compatibility with older versions.

.. _foreword-to-2011-edition-1:

Foreword to 2011 edition
------------------------

This is a tutorial for how to use the Matlab toolbox Wafo for analysis
and simulation of random waves and random fatigue. The toolbox consists
of a number of Matlab m-files together with executable routines from
Fortran or C++ source, and it requires only a standard Matlab setup,
with no additional toolboxes.

A main and unique feature of Wafo is the module of routines for
computation of the exact statistical distributions of wave and cycle
characteristics in a Gaussian wave or load process. The routines are
described in a series of examples on wave data from sea surface
measurements and other load sequences. There are also sections for
fatigue analysis and for general extreme value analysis. Although the
main applications at hand are from marine and reliability engineering,
the routines are useful for many other applications of Gaussian and
related stochastic processes.

The routines are based on algorithms for extreme value and crossing
analysis, developed over many years by the authors as well as many
results available in the literature. References are given to the source
of the algorithms whenever it is possible. These references are given in
the Matlab-code for all the routines and they are also listed in the
Bibliography section of this tutorial. If the references are not used
explicitly in the tutorial; it means that it is referred to in one of
the Matlab m-files.

Besides the dedicated wave and fatigue analysis routines the toolbox
contains many statistical simulation and estimation routines for general
use, and it can therefore be used as a toolbox for statistical work.
These routines are listed, but not explicitly explained in this
tutorial.

The present toolbox represents a considerable development of two earlier
toolboxes, the Fat and Wat toolboxes, for fatigue and wave analysis,
respectively. These toolboxes were both Version 1; therefore Wafo has
been named Version 2. The routines in the tutorial are tested on
Wafo-version 2.5, which was made available in beta-version in January
2009 and in a stable version in February 2011.

The persons that take actively part in creating this tutorial are (in
alphabetical order): *Per Andreas Brodtkorb*\  [3]_, *Pär Johannesson*,
*Georg Lindgren*

, *Igor Rychlik*.

Many other people have contributed to our understanding of the problems
dealt with in this text, first of all Professor Ross Leadbetter at the
University of North Carolina at Chapel Hill and Professor Krzysztof
Podgórski, Mathematical Statistics, Lund University. We would also like
to particularly thank Michel Olagnon and Marc Provosto, at Institut
Français de Recherches pour l’Exploitation de la Mer (IFREMER), Brest,
who have contributed with many enlightening and fruitful discussions.

Other persons who have put a great deal of effort into Wafo and its
predecessors FAT and WAT are Mats Frendahl, Sylvie van Iseghem, Finn
Lindgren, Ulla Machado, Jesper Ryén, Eva Sjö, Martin Sköld, Sofia Åberg.

This tutorial was first made available for the beta version of Wafo
Version 2.5 in November 2009. In the present version some misprints have
been corrected and some more examples added. All examples in the
tutorial have been run with success on MATLAB up to 2010b.

.. _technical-information-1:

Technical information
=====================

-  Wafo was released in a stable version in February 2011. The most
   recent stable updated and expanded version of Wafo can be downloaded
   from

   ``https://github.com/wafo-project/``

   Older versions can also be downloaded from the Wafo homepage
   (WAFO-group 2000)

   ``http://www.maths.lth.se/matstat/wafo/``

-  To get access to the Wafo toolbox, unzip the downloaded file,
   identify the wafo package and save it in a folder of your choise.
   Take a look at the routines ``install.m``, ``startup.m``,
   ``initwafo.m`` in the ``WAFO`` and ``WAFO/docs`` folders to learn how
   Matlab can find Wafo.

-  To let Matlab start Wafo automatically, edit ``startup.m`` and save
   it in the starting folder for Matlab.

-  To start Wafo manually in Matlab, add the ``WAFO`` folder manually to
   the Matlab-path and run ``initwafo``.

-  In this tutorial, the word ``WAFO``, when used in path
   specifications, means the full name of the Wafo main catalogue, for
   instance ``C:/wafo/``

-  The Matlab code used for the examples in this tutorial can be found
   in the Wafo catalogue ``WAFO/papers/tutorcom/``

   The total time to run the examples in fast mode is less than fifteen
   minutes on a PC from 2017, running Windows 10 pro with Intel(R)
   Core(TM) i7-7700 CPU, 3.6 GHz, 32 GB RAM. All details on execution
   times given in this tutorial relates to that configuration.

-  Wafo is built of modules of platform independent Matlab m-files and a
   set of executable files from ``C++`` and ``Fortran`` source files.
   These executables are platform and Matlab-version dependent, and they
   have been tested with recent Matlab and Windows installations.

-  If you have many Matlab-toolboxes installed, name-conflicts may
   occur. Solution: arrange the Matlab-path with ``WAFO`` first.

-  For help on the toolbox, write ``help wafo``.

-  Comments and suggestions are solicited — send to
   ``wafo@maths.lth.se``

.. _nomenclature-1:

Nomenclature
============

.. _roman-letters-1:

Roman letters
-------------

+----------------------------------+----------------------------------+
| :math:`A_{c}`, :math:`A_{t}`     | Zero-crossing wave crest height  |
|                                  | and trough excursion.            |
+----------------------------------+----------------------------------+
| :math:`a_{i}`                    | Lower integration limit.         |
+----------------------------------+----------------------------------+
| :math:`b_{i}`                    | Upper integration limit.         |
+----------------------------------+----------------------------------+
| :math:`c_{0}`                    | Truncation parameter of          |
|                                  | truncated Weibull distribution.  |
+----------------------------------+----------------------------------+
| :math:`\mbox{\sf C}[X,Y]`        | Covariance between random        |
|                                  | variables :math:`X` and          |
|                                  | :math:`Y`.                       |
+----------------------------------+----------------------------------+
| :math:`                          | Directional spreading function.  |
| D(\ensuremath{\omega },\theta),` |                                  |
| :math:`D(\theta)`                |                                  |
+----------------------------------+----------------------------------+
| :math:`dd_{crit}\, d_{crit}\, %  | Critical distances used for      |
|   z_{crit}`                      | removing outliers and spurious   |
|                                  | points.                          |
+----------------------------------+----------------------------------+
| :math:`\mbox{\sf E}[X]`          | Expectation of random variable   |
|                                  | :math:`X`.                       |
+----------------------------------+----------------------------------+
| :math:`E(\ensuremath{\omega      | Quadratic transfer function.     |
| }_{i},\ensuremath{\omega }_{j})` |                                  |
+----------------------------------+----------------------------------+
| :math:`f`                        | Wave frequency                   |
|                                  | :math:`[\textit{Hz}]`.           |
+----------------------------------+----------------------------------+
| :math:`f_{p}`                    | Spectral peak frequency.         |
+----------------------------------+----------------------------------+
| :math:`F_{X}(\cdot)`,            | Cumulative distribution function |
| :math:`f_{X}(\cdot)`             | and                              |
+----------------------------------+----------------------------------+
|                                  | probability density function of  |
|                                  | variable :math:`X`.              |
+----------------------------------+----------------------------------+
| :math:`G(\cdot),\,g(\cdot)`      | The transformation and its       |
|                                  | inverse.                         |
+----------------------------------+----------------------------------+
| :math:`g`                        | Acceleration of gravity.         |
+----------------------------------+----------------------------------+
| :math:`H`, :math:`h`             | Dimensional and dimensionless    |
|                                  | wave height.                     |
+----------------------------------+----------------------------------+
| :math:`H_{m0}`, :math:`H_s`      | Significant wave height,         |
|                                  | :math:`4\sqrt{m_{0}}`.           |
+----------------------------------+----------------------------------+
| :math:`H_{c}`                    | Critical wave height.            |
+----------------------------------+----------------------------------+
| :math:`H_{d}`, :math:`H_{u}`     | Zero-downcrossing and            |
|                                  | -upcrossing wave height.         |
+----------------------------------+----------------------------------+
| :math:`h`                        | Water depth.                     |
+----------------------------------+----------------------------------+
| :math:`h_{\max}`                 | Maximum interval width for       |
|                                  | Simpson method.                  |
+----------------------------------+----------------------------------+
| :math:`H_{rms}`                  | Root mean square value for wave  |
|                                  | height defined as                |
|                                  | :math:`H_{m0}/\sqrt{2}`.         |
+----------------------------------+----------------------------------+
| :math:`K_{d}(\cdot)`             | Kernel function.                 |
+----------------------------------+----------------------------------+
| :math:`k`                        | Wave number                      |
|                                  | :math:`[\textit{rad/m}]` or      |
|                                  | index.                           |
+----------------------------------+----------------------------------+
| :math:`L_{p}`                    | Average wave length.             |
+----------------------------------+----------------------------------+
| :math:`L_{\max}`                 | Maximum lag beyond which the     |
|                                  | autocovariance is set to zero.   |
+----------------------------------+----------------------------------+
| :math:`M,\,M_{k}`                | Local maximum.                   |
+----------------------------------+----------------------------------+
| :math:`M_{k}^{tc}`               | Crest maximum for wave no.       |
|                                  | :math:`k`.                       |
+----------------------------------+----------------------------------+

+----------------------------------+----------------------------------+
| :math:`m,\,m_{k}`                | Local minimum.                   |
+----------------------------------+----------------------------------+
| :math                            | Rainflow minimum no. :math:`k`.  |
| :`m_{k}^{{\protect\mbox{\protect |                                  |
| \footnotesize\protect\sc rfc}}}` |                                  |
+----------------------------------+----------------------------------+
| :math:`m_{k}^{tc}`               | Trough minimum for wave no.      |
|                                  | :math:`k`.                       |
+----------------------------------+----------------------------------+
| :math:`m_{n}`                    | n’th spectral moment,            |
|                                  | :math:`\int                      |
|                                  | _{0}^{\infty}\omega^{n}  S_{\eta |
|                                  |           \eta}^                 |
|                                  | {+}(\omega) \,\mathrm{d}\omega`. |
+----------------------------------+----------------------------------+
| :math:`N`                        | Number of variables or waves.    |
+----------------------------------+----------------------------------+
| :math:`N_{c1c2}`                 | Number of times to apply         |
|                                  | regression equation.             |
+----------------------------------+----------------------------------+
| ``NIT``                          | Order in the integration of wave |
|                                  | characteristic distributions.    |
+----------------------------------+----------------------------------+
| :math:`n_{i},` :math:`n`         | Sample size.                     |
+----------------------------------+----------------------------------+
| :math:`\mbox{\sf P}(A)`          | Probability of event :math:`A`.  |
+----------------------------------+----------------------------------+
| :math:`O(\cdot)`                 | Order of magnitude.              |
+----------------------------------+----------------------------------+
| :math:`Q_p`                      | Peakedness factor.               |
+----------------------------------+----------------------------------+
| :math:`R_{\eta}(\tau)`           | Auto covariance function of      |
|                                  | :math:`\eta(t)`.                 |
+----------------------------------+----------------------------------+
| :math:`S_{p}`                    | Average wave steepness.          |
+----------------------------------+----------------------------------+
| :math:`S_s`                      | Significant wave steepness.      |
+----------------------------------+----------------------------------+
| :math:`S_{\eta \eta}^{+}(f),  %  | One sided spectral density of    |
|   S_{\eta \eta}^{+}(\omega)`     | the surface elevation            |
|                                  | :math:`\eta`.                    |
+----------------------------------+----------------------------------+
| :math:                           | Directional wave spectrum.       |
| `S(\ensuremath{\omega },\theta)` |                                  |
+----------------------------------+----------------------------------+
| :math:`s`                        | Normalized crest front           |
|                                  | steepness.                       |
+----------------------------------+----------------------------------+
| :math:`s_{c}`                    | Critical crest front steepness.  |
+----------------------------------+----------------------------------+
| :math:`s_{cf}`                   | Crest front steepness.           |
+----------------------------------+----------------------------------+
| :math:`s_N`                      | Return level for return period   |
|                                  | :math:`N`.                       |
+----------------------------------+----------------------------------+
| :math:`s_{rms}`                  | Root mean square value for crest |
|                                  | front steepness,                 |
+----------------------------------+----------------------------------+
|                                  | i.e.,                            |
|                                  | :math:`5/4\,H_{m0}/T_{m02}^{2}`. |
+----------------------------------+----------------------------------+
| :math:`T_{c}`, :math:`T_{cf}`,   | Crest, crest front, and crest    |
| :math:`T_{cr}`                   | rear period.                     |
+----------------------------------+----------------------------------+
| :math:`T_{m(-1)0}`               | Energy period.                   |
+----------------------------------+----------------------------------+
| :math:`T_{m01}`                  | Mean wave period.                |
+----------------------------------+----------------------------------+
| :math:`T_{m02}`                  | Mean zero-crossing wave period   |
|                                  | calculated as                    |
|                                  | :math:`2\pi\sqrt{m_{0}/m_{2}}`.  |
+----------------------------------+----------------------------------+
| :math:`T_{m24}`                  | Mean wave period between maxima  |
|                                  | calculated as                    |
|                                  | :math:`2\pi\sqrt{m_{2}/m_{4}}`.  |
+----------------------------------+----------------------------------+
| :math:`T_{Md}`                   | Wave period between maximum and  |
|                                  | downcrossing.                    |
+----------------------------------+----------------------------------+
| :math:`T_{Mm}`                   | Wave period between maximum and  |
|                                  | minimum.                         |
+----------------------------------+----------------------------------+
| :math:`T_{p}`                    | Spectral peak period.            |
+----------------------------------+----------------------------------+
| :math:`T_{z}`                    | Mean zero-crossing wave period   |
|                                  | estimated directly from time     |
|                                  | series.                          |
+----------------------------------+----------------------------------+
| :math:`T`                        | Wave period.                     |
+----------------------------------+----------------------------------+
| :math:`U_{10}`                   | 10 min average of windspeed      |
|                                  | :math:`10 [m]` above the         |
|                                  | watersurface.                    |
+----------------------------------+----------------------------------+
| :math:`U_{i}`                    | Uniformly distributed number     |
|                                  | between zero and one.            |
+----------------------------------+----------------------------------+
| :math:`V`, :math:`v`             | Dimensional and dimensionless    |
|                                  | velocity.                        |
+----------------------------------+----------------------------------+
| :math:`\mbox{\sf V}[X]`          | Variance of random variable      |
|                                  | :math:`X`.                       |
+----------------------------------+----------------------------------+
| :math:`V_{cf}`, :math:`V_{cr}`   | Crest front and crest rear       |
|                                  | velocity.                        |
+----------------------------------+----------------------------------+
| :math:`V_{rms}`                  | Root mean square value for       |
|                                  | velocity defined as              |
|                                  | :math:`2 H_{m0}/T_{m02}`.        |
+----------------------------------+----------------------------------+
| :math:`W_{age}`                  | Wave age.                        |
+----------------------------------+----------------------------------+
| :math:`W(x,t)`                   | Random Gassian field.            |
+----------------------------------+----------------------------------+
| :math:`X(t)`                     | Time series.                     |
+----------------------------------+----------------------------------+
| :math:`X_{i}`, :math:`Y_i`,      | Random variables.                |
| :math:`Z_{i}`                    |                                  |
+----------------------------------+----------------------------------+
| :math:`x_{c}`, :math:`y_c`,      | Truncation parameters.           |
| :math:`z_{c}`                    |                                  |
+----------------------------------+----------------------------------+

.. _greek-letters-1:

Greek letters
-------------

+----------------------------------+----------------------------------+
| :math:`\alpha`                   | Rayleigh scale parameter or      |
|                                  | JONSWAP normalization constant.  |
+----------------------------------+----------------------------------+
| :math:`\alpha`                   | Irregularity factor; spectral    |
|                                  | width measure.                   |
+----------------------------------+----------------------------------+
| :math:`\alpha(h)`,               | Weibull or Gamma parameters for  |
| :math:`\beta(h)`                 | scale and shape.                 |
+----------------------------------+----------------------------------+
| :math:`\alpha_{i}`               | Product correlation coefficient. |
+----------------------------------+----------------------------------+
| :math:`\Delta`                   | Forward difference operator.     |
+----------------------------------+----------------------------------+
| :math:`\delta_{i|1}`             | Residual process.                |
+----------------------------------+----------------------------------+
| :math:`\epsilon_{2}`             | Narrowness parameter defined as  |
|                                  | :math:                           |
|                                  | `\sqrt{m_{0}m_{2}/m_{1}^{2}-1}`. |
+----------------------------------+----------------------------------+
| :math:`\epsilon_{4}`             | Broadness factor defined as      |
|                                  | :math:`\s                        |
|                                  | qrt{1-m_{2}^{2}/(m_{0} m_{4})}`. |
+----------------------------------+----------------------------------+
| :math:`\epsilon`                 | Requested error tolerance for    |
|                                  | integration.                     |
+----------------------------------+----------------------------------+
| :math:`\epsilon_{c}`             | Requested error tolerance for    |
|                                  | Cholesky factorization.          |
+----------------------------------+----------------------------------+
| :math:`\eta(\cdot)`              | Surface elevation.               |
+----------------------------------+----------------------------------+
| :math:`\Gamma`                   | Gamma function.                  |
+----------------------------------+----------------------------------+
| :math:`\gamma`                   | JONSWAP peakedness factor or     |
|                                  | Weibull location parameter.      |
+----------------------------------+----------------------------------+
| :math:`\lambda_{i}`              | Eigenvalues or shape parameter   |
|                                  | of Ochi-Hubble spectrum.         |
+----------------------------------+----------------------------------+
| :math:`\mu_{X}(v)`               | Crossing intensity of level      |
|                                  | :math:`v` for time series        |
|                                  | :math:`X(t)`.                    |
+----------------------------------+----------------------------------+
| :math:`\mu_{X}^+(v)`             | Upcrossing intensity of level    |
|                                  | :math:`v` for time series        |
|                                  | :math:`X(t)`.                    |
+----------------------------------+----------------------------------+
| :math:`\Phi(\cdot),`             | CDF and PDF of a standard normal |
| :math:`\phi(\cdot)`              | variable.                        |
+----------------------------------+----------------------------------+
| :math:`\Theta_{n}`               | Phase function.                  |
+----------------------------------+----------------------------------+
| :math:`\rho_{3}`,                | Normalized cumulants, i.e.,      |
| :math:`\rho_{4}`                 | skewness and excess,             |
|                                  | respectively.                    |
+----------------------------------+----------------------------------+
| :math:`\rho_{ij}`                | Correlation between random       |
|                                  | variables :math:`X_{i}` and      |
|                                  | :math:`X_{j}`.                   |
+----------------------------------+----------------------------------+
| :math                            | Covariance matrix.               |
| :`\ensuremath{\mathbf{\Sigma} }` |                                  |
+----------------------------------+----------------------------------+
| :math:`\sigma_{X}^{2}`           | Variance of random variable      |
|                                  | :math:`X`.                       |
+----------------------------------+----------------------------------+
| :math:`\tau`                     | Shift variable of time.          |
+----------------------------------+----------------------------------+
| :math:`\tau_{i}`                 | Parameters defining the          |
|                                  | eigenvalues of                   |
|                                  | :math:`\en                       |
|                                  | suremath{\boldsymbol{\Sigma} }`. |
+----------------------------------+----------------------------------+
| :math:`\omega`                   | Wave angular frequency           |
|                                  | :math:`[rad/s]`.                 |
+----------------------------------+----------------------------------+
| :math:`\omega_{p}`               | Wave angular peak frequency      |
|                                  | :math:`[rad/s]`.                 |
+----------------------------------+----------------------------------+

.. _abbreviations-1:

Abbreviations
-------------

===== ===========================================
AMISE Asymptotic mean integrated square error.
CDF   Cumulative distribution function.
FFT   Fast Fourier Transform.
GEV   Generalized extreme value.
GPD   Generalized Pareto distribution.
HF    High frequency.
ISSC  International ship structures congress.
ITTC  International towing tank conference.
IQR   Interquartile range.
KDE   Kernel density estimate.
LS    Linear simulation.
MC    Markov chain.
MCTP  Markov chain of turning points.
ML    Maximum likelihood.
NLS   Non-linear simulation.
MISE  Mean integrated square error.
MWL   Mean water line.
PDF   Probability density function.
PSD   Power spectral density.
QTF   Quadratic transfer function.
SCIS  Sequential conditioned importance sampling.
TLP   Tension-leg platform.
TP    Turning points.
WAFO  Wave analysis for fatigue and oceanography.
===== ===========================================

.. _cha:KDE:

Kernel density estimation
=========================

Histograms are among the most popular ways to visually present data.
They are particular examples of density estimates and their appearance
depends on both the choice of origin and the width of the intervals
(bins) used. In order for the histogram to give useful information about
the true underlying distribution, a sufficient amount of data is needed.
This is even more important for histograms in two dimensions or higher.
Also the discontinuity of the histograms may cause problems, e.g., if
derivatives of the estimate are required.

An effective alternative to the histogram is the kernel density estimate
(KDE), which may be considered as a “smoothed histogram”, only depending
on the bin-width and not depending on the origin, see (Silverman 1986).

.. _the-univariate-kernel-density-estimator-1:

The univariate kernel density estimator
---------------------------------------

The univariate KDE is defined by

.. math::

   \label{eq:kdeformula}
     \hat{f}_{X}(x;h_{s}) = \frac{1}{n\,h_{s}}
                            \sum_{j=1}^{n} K_{d}\left( \frac{x-X_{j}}{h_{s}}\right),

where :math:`n` is the number of datapoints,
:math:`X_{1},X_{2},\ldots,X_{n}`, is the data set, and :math:`h_{s}` is
the smoothing parameter or window width. The kernel function
:math:`K_{d}` is usually a unimodal, symmetric probability density
function. This ensures that the KDE itself is also a density. However,
kernels that are not densities are also sometimes used (see (Wand and
Jones 1995)), but these are not implemented in the Wafo toolbox.

To illustrate the method, consider the kernel estimator as a sum of
“bumps” placed at the observations. The shape of the bumps are given by
the kernel function while the width is given by the smoothing parameter,
:math:`h_{s}`. Fig. `8.3 <#fig:kdedemo1>`__ shows a KDE constructed
using 7 observations from a standard Gaussian distribution with a
Gaussian kernel function. One should note that the 7 points used here,
is purely for clarity in illustrating how the kernel method works.
Practical density estimation usually involves much higher number of
observations.

Fig. `8.3 <#fig:kdedemo1>`__ also demonstrates the effect of varying the
smoothing parameter, :math:`h_{s}`. A too small value for :math:`h_{s}`
may introduce spurious bumps in the resulting KDE (top), while a too
large value may obscure the details of the underlying distribution
(bottom). Thus the choice of value for the smoothing parameter,
:math:`h_{s}`, is very important. How to select one will be elaborated
further in the next section.

.. figure:: kdedemo1f1
   :alt: Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).
   :name: fig:kdedemo1
   :height: 2.5in

   Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).

.. figure:: kdedemo1f2
   :alt: Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).
   :name: fig:kdedemo1
   :height: 2.5in

   Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).

.. figure:: kdedemo1f3
   :alt: Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).
   :name: fig:kdedemo1
   :height: 2.5in

   Smoothing parameter, :math:`h_{s}`, impact on KDE: True density
   (dotted) compared to KDE based on 7 observations (solid) and their
   individual kernels (dashed).

The particular choice of kernel function, on the other hand, is not very
important since suboptimal kernels are not suboptimal by very much, (see
pp. 31 in (Wand and Jones 1995)). However, the kernel that minimizes the
mean integrated square error is the Epanechnikov kernel, and is thus
chosen as the default kernel in the software, see
Eq. (`[eq:multivariateEpan] <#eq:multivariateEpan>`__). For a discussion
of other kernel functions and their properties, see (Wand and Jones
1995).

.. _smoothing-parameter-selection-1:

Smoothing parameter selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The choice of smoothing parameter, :math:`h_{s}`, is very important, as
exemplified in Fig.\ `8.3 <#fig:kdedemo1>`__. In many situations it is
satisfactory to select the smoothing parameter subjectively by eye,
i.e., look at several density estimates over a range of bandwidths and
selecting the density that is the most “pleasing” in some sense.
However, there are also many circumstances where it is beneficial to use
an automatic bandwidth selection from the data. One reason is that it is
very time consuming to select the bandwidth by eye. Another reason, is
that, in many cases, the user has no prior knowledge of the structure of
the data, and does not have any feeling for which bandwidth gives a good
estimate. One simple, quick and commonly used automatic bandwidth
selector, is the bandwidth that minimizes the mean integrated square
error (MISE) asymptotically. As shown in (Wand and Jones 1995 Section
2.5 and 3.2.1), the one dimensional AMISE [4]_-optimal normal scale rule
assuming that the underlying density is Gaussian, is given by

.. math::

   \label{eq:Hamise}
     h_{AMISE} =
     %  \left[\frac{8\,\sqrt{\pi}\,R(K_{d})}{3\,\mu_{2}^{2}(K_{d})
     %\,n}\right]^{1/5}\,\widehat{\sigma} =
   \left[\frac{4}{3\,n}\right]^{1/5}\,\widehat{\sigma} ,

where :math:`\widehat{\sigma}` is some estimate of the standard
deviation of the underlying distribution. Common choices of
:math:`\widehat{\sigma}` are the sample standard deviation,
:math:`\widehat{\sigma}_{s}`, and the standardized interquartile range
(denoted IQR):

.. math::

   \label{eq:Oiqr}
     \widehat{\sigma}_{IQR} = (\text{sample IQR})/
     (\Phi^{-1}(3/4)-\Phi^{-1}(1/4)) \approx  (\text{sample IQR})/1.349 ,

where :math:`\Phi^{-1}` is the standard normal quantile function. The
use of :math:`\widehat{\sigma}_{IQR}` guards against outliers if the
distribution has heavy tails. A reasonable approach is to use the
smaller of :math:`\widehat{\sigma}_{s}` and
:math:`\widehat{\sigma}_{IQR}` in order to lessen the chance of
oversmoothing, (see Silverman 1986, 47).

Various other automatic methods for selecting :math:`h_{s}` are
available and are discussed in (Silverman 1986) and in more detail in
(Wand and Jones 1995).

.. _transformation-kernel-denstity-estimator-1:

Transformation kernel denstity estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Densities close to normality appear to be the easiest for the kernel
estimator to estimate. The estimation difficulty increases with
skewness, kurtosis and multimodality (Chap. 2.9 in (Wand and Jones
1995)).

Thus, in the cases where the random sample
:math:`X_{1},X_{2},\ldots,X_{n}`, has a density, :math:`f`, which is
difficult to estimate, a transformation, :math:`t`, might give a good
KDE, i.e., applying a transformation to the data to obtain a new sample
:math:`Y_{1},Y_{2},\ldots,Y_{n}`, with a density :math:`g` that more
easily can be estimated using the basic KDE. One would then
backtransform the estimate of :math:`g` to obtain the estimate for
:math:`f`.

Suppose that :math:`Y_{i} = t(X_{i})`, where :math:`t` is an increasing
differentiable function defined on the support of :math:`f`. Then a
standard result from statistical distribution theory is that

.. math::

   \label{eq:transform}
     f(x) = g(t(x))\,t'(x),

where :math:`t'(x)` is the derivative. Backtransformation of the KDE of
:math:`g` based on :math:`Y_{1},Y_{2},\ldots,Y_{n}`, leads to the
explicit formula

.. math::

   \label{eq:transformkdeformula}
     \hat{f}_{X}(x;h_{s},t) = \frac{1}{n\,h_{s}} \sum_{j=1}^{n}
   K_{d}\left( \frac{t(x)-t(X_{j})}{h_{s}}\right)\,t'(x)

A simple illustrative example comes from the problem of estimating the
Rayleigh density. This density is very difficult to estimate by direct
kernel methods. However, if we apply the transformation
:math:`Y_{i} = \sqrt{X_{i}}` to the data, then the normal plot of the
transformed data, :math:`Y_{i}`, becomes approximately linear.
Fig. `8.5 <#fig:transformkde>`__ shows that the transformation KDE is a
better estimate around 0 than the ordinary KDE.

.. figure:: rayleightransformedkde
   :alt: True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.
   :name: fig:transformkde
   :width: 2.5in

   True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.

.. figure:: rayleighkde
   :alt: True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.
   :name: fig:transformkde
   :width: 2.5in

   True Rayleigh density (dotted) compared to transformation KDE
   (solid,left) and ordinary KDE (solid, right) based on 1000
   observations.

.. _sec:multivariateKDE:

The multivariate kernel density estimator
-----------------------------------------

The multivariate kernel density estimator is defined in its most general
form by

.. math::

   \label{eq:multivariateKDE}
     \widehat{f}_{\ensuremath{\mathbf{X} }}(\ensuremath{\mathbf{x} };\ensuremath{\mathbf{H} }) = \frac{|\ensuremath{\mathbf{H} }|^{-1/2}}{n}
    \sum_{j=1}^{n} K_{d}\left(\ensuremath{\mathbf{H} }^{-1/2}(\ensuremath{\mathbf{x} }-\ensuremath{\mathbf{X} }_{j})\right),

where :math:`\ensuremath{\mathbf{H} }` is a symmetric positive definite
:math:`d \times d` matrix called the *bandwidth matrix*. A
simplification of Eq. (`[eq:multivariateKDE] <#eq:multivariateKDE>`__)
can be obtained by imposing the restriction
:math:`\ensuremath{\mathbf{H} } = \text{diag}(h_{1}^{2}, h_{2}^{2}, \ldots ,
h_{d}^{2})`. Then Eq. (`[eq:multivariateKDE] <#eq:multivariateKDE>`__)
reduces to

.. math::

   \label{eq:multivariateKDE2}
     \widehat{f}_{\ensuremath{\mathbf{X} }}(\ensuremath{\mathbf{x} };\ensuremath{\mathbf{h} }) =
     \frac{1}{n\,\prod_{i=1}^{n}\,h_{i}}
     \sum_{j=1}^{n} K_{d}\left(\frac{x-X_{j\,1}}{h_{1}},
       \frac{x-X_{j\,2}}{h_{2}},\ldots,
       \frac{x-X_{j\,d}}{h_{d}} \right),

and is, in combination with a transformation, a reasonable solution to
visualize multivariate densities.

The multivariate Epanechnikov kernel also forms the basis for the
optimal spherically symmetric multivariate kernel and is given by

.. math::

   \label{eq:multivariateEpan}
     K_{d}(\ensuremath{\mathbf{x} }) = \frac{d+2}{2\,v_{d}}
     \left(1-\ensuremath{\mathbf{x} }^{T}\ensuremath{\mathbf{x} } \right)\ensuremath{\mathbf{1} }_{\ensuremath{\mathbf{x} }^{T}\ensuremath{\mathbf{x} }\le 1},

where :math:`v_{d}=2\,\pi^{d/2}/( \Gamma(d/2) \, d )` is the volume of
the unit :math:`d`-dimensional sphere.

In this tutorial we use the KDE to find a good estimator of the central
part of the joint densities of wave parameters extracted from time
series. Clearly, such data are dependent, so it is assumed that the time
series are ergodic and short range dependent to justify the use of
KDE:s, (see Chap. 6 in (Wand and Jones 1995)). Usually, KDE gives poor
estimates of the tail of the distribution, unless large amounts of data
is available. However, a KDE gives qualitatively good estimates in the
regions of sufficient data, i.e., in the main parts of the distribution.
This is good for visualization, e.g. detecting modes, symmetries of
distributions.

The kernel density estimation software is based on ``KDETOOL``, which is
a Matlab toolbox produced by Christian Beardah, (Beardah and Baxter
1996), which has been totally rewritten and extended to include the
transformation kernel estimator and generalized to cover any dimension
for the data. The computational speed has also been improved.

.. _cha:waveSpectra:

Standardized wave spectra
=========================

Knowledge of which kind of spectral density is suitable to describe sea
state data are well established from experimental studies. Qualitative
considerations of wave measurements indicate that the spectra may be
divided into 3 parts, (see Fig. `9.1 <#fig:qspecvar20>`__):

#. Sea states dominated by wind sea but significantly influenced by
   swell components.

#. More or less pure wind seas or, possibly, swell component located
   well inside the wind frequency band.

#. Sea states more or less dominated by swell but significantly
   influenced by wind sea.

.. figure:: qspecvar1
   :alt: Qualitative indication of spectral variability.
   :name: fig:qspecvar20
   :width: 4in

   Qualitative indication of spectral variability.

One often uses some parametric form of the spectral density. The three
most important parametric spectral densities implemented in Wafo will be
described in the following sections.

.. _sec:jonswap:

Jonswap spectrum
----------------

The Jonswap (JOint North Sea WAve Project) spectrum of (Hasselmann et
al. 1973) is a result of a multinational project to characterize
standardized wave spectra for the Southeast part of the North Sea. The
spectrum is valid for not fully developed sea states. However, it is
also used to represent fully developed sea states. It is particularly
well suited to characterize wind sea when
:math:`3.6 \sqrt{H_{m0}} < T_{p} < 5 \sqrt{H_{m0}}`. The Jonswap
spectrum is given in the form:

.. math::

   \begin{aligned}
      S^{+}(\ensuremath{\omega }) &= \frac{\alpha \, g^{2}}{\ensuremath{\omega }^{M}}
      \exp \Big( -\frac{M}{N} \, \big(   \frac{\ensuremath{\omega }_{p}}{\ensuremath{\omega }} \big)^{N} \Big) \,
      \gamma^{\exp \Big( \frac{-(\ensuremath{\omega }/ \ensuremath{\omega }_{p}-1)^{2}}{2 \, \sigma^{2}} \Big)},  \label{eq:jonswap} \\
   \intertext{where}
   \sigma &=  \begin{cases}
        0.07 & \text{if} \; \ensuremath{\omega }< \ensuremath{\omega }_{p}, \\
        0.09 & \text{if} \; \ensuremath{\omega }\ge \ensuremath{\omega }_{p},
       \end{cases} \nonumber \\
     M &= 5, \quad  N = 4, \nonumber \\
     \alpha &\approx 5.061 \frac{H_{m0}^{2}}{T_{p}^{4}} \,\Big\{ 1-0.287\, \ln(\gamma)  \Big\}. \nonumber
    \end{aligned}

A standard value for the peakedness parameter, :math:`\gamma`, is
:math:`3.3`. However, a more correct approach is to relate
:math:`\gamma` to :math:`H_{m0}` and :math:`T_{p}`, and use

.. math::

   \begin{gathered}
     \gamma = \exp \Big\{3.484 \,\big(1-0.1975\,(0.036-0.0056\,T_{p}/\sqrt{H_{m0}})
   \,T_{p}^{4}/H_{m0}^2\big) \Big\} .
   %\intertext{where}
   % D = 0.036-0.0056\,T_{p}/\sqrt{H_{m0}}\end{gathered}

Here :math:`\gamma` is limited by :math:`1 \le \gamma \le 7`. This
parameterization is based on qualitative considerations of deep water
wave data from the North Sea; see (Torsethaugen and others 1984) and
(Haver and Nyhus 1986).

The relation between the peak period and mean zero-upcrossing period may
be approximated by

.. math:: T_{m02} \approx T_{p}/\left(1.30301-0.01698\,\gamma+0.12102/\gamma \right)

The Jonswap spectrum is identical with the two-parameter
Pierson-Moskowitz, Bretschneider, ITTC (International Towing Tank
Conference) or ISSC (International Ship and Offshore Structures
Congress) wave spectrum, given :math:`H_{m0}` and :math:`T_{p}`, when
:math:`\gamma=1`. (For more properties of this spectrum, see the Wafo
function ``jonswap.m``.

.. _sec:torsethaugen:

Torsethaugen spectrum
---------------------

Torsethaugen, (Torsethaugen 1993, 1994, 1996), proposed to describe
bimodal spectra by

.. math::

   \begin{gathered}
    S^{+}(\ensuremath{\omega }) = \sum_{i=1}^{2} S_{J}^{+}(\ensuremath{\omega };H_{m0,i},\ensuremath{\omega }_{p,i},\gamma_{i},N_{i},M_{i},\alpha_{i})\end{gathered}

where :math:`S_{J}^{+}` is the Jonswap spectrum defined by
Eq. (`[eq:jonswap] <#eq:jonswap>`__). The parameters :math:`H_{m0,\,i}`,
:math:`\ensuremath{\omega }_{p,\,i}`, :math:`N_{i}`, :math:`M_{i}`, and
:math:`\alpha_{i}` for :math:`i=1,2`, are the significant wave height,
angular peak frequency, spectral shape and normalization parameters for
the primary and secondary peak, respectively.

These parameters are fitted to 20 000 spectra divided into 146 different
classes of :math:`H_{m0}` and :math:`T_{p}` obtained at the Statfjord
field in the North Sea in the period from 1980 to 1989. The measured
:math:`H_{m0}` and :math:`T_{p}` values for the data range from
:math:`0.5` to :math:`11` meters and from :math:`3.5` to :math:`19`
seconds, respectively.

Given :math:`H_{m0}` and :math:`T_{p}` these parameters are found by the
following steps. The borderline between wind dominated and swell
dominated sea states is defined by the fully developed sea, for which

.. math:: T_{p} = T_{f} = 6.6 \, H_{m0}^{1/3},

while for :math:`T_{p} < T_{f}`, the local wind sea dominates the
spectral peak, and if :math:`T_{p} > T_{f}`, the swell peak is
dominating.

For each of the three types a non-dimensional period scale is introduced
by

.. math::

   \begin{aligned}
   \epsilon_{l\,u} &= \frac{T_{f}-T_{p}}{T_{f}-T_{lu}},
   \intertext{where}
   T_{lu} &=
   \begin{cases}
        2 \, \sqrt{H_{m0}} & \text{if} \; T_{p} \le T_{f} \quad
        \text{(Lower limit)},\\
        25 & \text{if} \; T_{p} > T_{f} \quad
        \text{(Upper limit)},
       \end{cases}
   \intertext{defines the lower or upper value for $T_{p}$.
   The significant wave height for each peak is given as}
   H_{m0,1} &= R_{pp} \, H_{m0} \quad  H_{m0,2} = \sqrt{1-R_{pp}^{2}} \, H_{m0},
   \intertext{where}
   R_{pp} &= \big(1-A_{10}\big) \,\exp\Big(-\big(\frac{\epsilon_{l\,u}}{A_{1}}
   \big)^{2} \Big) +A_{10}, \\
   %\label{eq:A-values}
   A_{1} &= \begin{cases}
      0.5 & \text{if} \; T_{p} \le T_{f}, \\
      0.3 & \text{if} \; T_{p} > T_{f},
       \end{cases} \quad
   A_{10} = \begin{cases}
      0.7 & \text{if} \; T_{p} \le T_{f}, \\
      0.6 & \text{if} \; T_{p} > T_{f}.
       \end{cases}\end{aligned}

The primary and secondary peak periods are defined as

.. math::

   \begin{aligned}
   T_{p,\,1} &= T_{p}, \\
   T_{p,\,2} &=
   \begin{cases}
       T_{f} + 2  & \text{if} \; T_{p} \le T_{f}, \\[0.3em]
       \left( \frac{M_{2}\,(N_{2}/M_{2})^{(N_{2}-1)/M_{2}}/\Gamma((N_{2}-1)/M_{2} )}
       {1.28\,(0.4)^{N_{2}} \{1-\exp (-H_{m0,\,2}/3)
         \} } \right)^{1/(N_{2}-1)} & \text{if} \; T_{p} > T_{f},
   \end{cases}
   \intertext{where the spectral shape parameters are given as}
   N_{1} &= N_{2} = 0.5\, \sqrt{H_{m0}}+3.2, \\
   M_{i} &=  \begin{cases}
   4\, \Big( 1-0.7\exp
   \big(\frac{-H_{m0}}{3}\big)\Big) & \text{if} \; T_{p} > T_{f} \;
   \text{and} \; i=2, \\
   4 & \text{otherwise}.
     \end{cases}\end{aligned}

The peakedness parameters are defined as

.. math::

   \begin{aligned}
   \gamma_{1} &= 35 \,\Big(1+3.5\, \exp \big( -H_{m0} \big)\Big) \gamma_{T}, \qquad \gamma_{2} = 1,
   \intertext{where}
   \gamma_{T} &= \begin{cases}
   \Big( \frac{2 \, \pi \, H_{m0,\,1}}{g \, T_{p}^{2}}\Big)^{0.857} & \text{if} \; T_{p} \le T_{f}, \\
   \big(1+6\,\epsilon_{lu}\big) \Big( \frac{2 \,  \pi \, H_{m0}}{g \, T_{f}^{2}}\Big)^{0.857} & \text{if} \; T_{p} > T_{f}.
       \end{cases}\end{aligned}

Finally the normalization parameters :math:`\alpha_{i}` (:math:`i=1,2`)
are found by numerical integration so that

.. math::

   \begin{aligned}
   \int_{0}^{\infty}
   S_{J}^{+}(\ensuremath{\omega };H_{m0,i},\ensuremath{\omega }_{p,i},\gamma_{i},N_{i},M_{i},\alpha_{i})\,d\ensuremath{\omega }
   = H_{m0,\,i}^{2}/16.\end{aligned}

Preliminary comparisons with spectra from other areas indicate that the
empirical parameters in the Torsethaugen spectrum can be dependent on
geographical location. This spectrum is implemented as a matlab function
``torsethaugen.m`` in the Wafo toolbox.

.. _sec:ochi-hubble:

Ochi-Hubble spectrum
--------------------

Ochi and Hubble (Ochi and Hubble 1976), suggested to describe bimodal
spectra by a superposition of two modified Bretschneider
(Pierson-Moskovitz) spectra:

.. math::

   \begin{aligned}
      \label{eq:ohspec1} %
      S^{+}(\ensuremath{\omega }) &=\frac{1}{4} \sum_{i=1}^{2}
      \frac{\big( \big( \lambda_{i}+1/4 \big) \, \ensuremath{\omega }_{p,\,i}^{4}
   \big)^{\lambda_{i}}}{\Gamma(\lambda_{i})}
      \frac{H_{m0,\,i}^{2}}{\ensuremath{\omega }^{4 \, \lambda_{i}}+1}
      \exp \Big( \frac{-\big( \lambda_{i}+1/4 \big) \, \ensuremath{\omega }_{p,\,i}^{4}}{\ensuremath{\omega }^{4}}\Big),
    \end{aligned}

where :math:`H_{m0,\,i}`, :math:`\ensuremath{\omega }_{p,\,i}`, and
:math:`\lambda_{i}` for :math:`i=1,2`, are significant wave height,
angular peak frequency, and spectral shape parameter for the low and
high frequency components, respectively.

The values of these parameters are determined from an analysis of data
obtained in the North Atlantic. The source of the data is the same as
that for the development of the Pierson-Moskowitz spectrum, but analysis
is carried out on over :math:`800` spectra including those in partially
developed seas and those having a bimodal shape. In contrast to the
Jonswap and Torsethaugen spectra, which are parameterized as function of
:math:`H_{m0}` and :math:`T_{p}`, Ochi and Hubble, (Ochi and Hubble
1976) gave, from a statistical analysis of the data, a family of wave
spectra consisting of 11 members generated for a desired sea severity
(:math:`H_{m0}`) with the coefficient of :math:`0.95`.

The values of the six parameters as functions of :math:`H_{m0}` are
given as:

.. math::

   \begin{aligned}
    H_{m0,1} &= R_{p,1} \, H_{m0}, \\
    H_{m0,2} &= \sqrt{1-R_{p,1}^{2}} \, H_{m0}, \\
    \ensuremath{\omega }_{p,i} &= a_{i}\,\exp \big( -b_{i}\, H_{m0} \big), \\
    \lambda_{i} &= c_{i}\,\exp \big( -d_{i}\, H_{m0} \big),\end{aligned}

where :math:`d_{1}=0` and the remaining empirical constants
:math:`a_{i}`, :math:`b_{i}` (:math:`i=1,2`), and :math:`d_{2}`, are
given in Table `9.1 <#tab:oh-parameters>`__. (See also the function
``ochihubble.m`` in the Wafo toolbox.)

.. container::
   :name: tab:oh-parameters

   .. table:: Empirical parameter values for the Ochi-Hubble spectral
   model.

      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | M     | :math | :ma   | :ma   | :ma   | :ma   | :ma   | :ma   | :ma   |
      | ember | :`R_{ | th:`a | th:`a | th:`b | th:`b | th:`c | th:`c | th:`d |
      | no.   | p,1}` | _{1}` | _{2}` | _{1}` | _{2}` | _{1}` | _{2}` | _{2}` |
      +=======+=======+=======+=======+=======+=======+=======+=======+=======+
      | 1     | 0.84  | 0.70  | 1.15  | 0.046 | 0.039 | 3.00  | 1.54  | 0.062 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 2     | 0.84  | 0.93  | 1.50  | 0.056 | 0.046 | 3.00  | 2.77  | 0.112 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 3     | 0.84  | 0.41  | 0.88  | 0.016 | 0.026 | 2.55  | 1.82  | 0.089 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 4     | 0.84  | 0.74  | 1.30  | 0.052 | 0.039 | 2.65  | 3.90  | 0.085 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 5     | 0.84  | 0.62  | 1.03  | 0.039 | 0.030 | 2.60  | 0.53  | 0.069 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 6     | 0.95  | 0.70  | 1.50  | 0.046 | 0.046 | 1.35  | 2.48  | 0.102 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 7     | 0.65  | 0.61  | 0.94  | 0.039 | 0.036 | 4.95  | 2.48  | 0.102 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 8     | 0.90  | 0.81  | 1.60  | 0.052 | 0.033 | 1.80  | 2.95  | 0.105 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 9     | 0.77  | 0.54  | 0.61  | 0.039 | 0.000 | 4.50  | 1.95  | 0.082 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 10    | 0.73  | 0.70  | 0.99  | 0.046 | 0.039 | 6.40  | 1.78  | 0.069 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+
      | 11    | 0.92  | 0.70  | 1.37  | 0.046 | 0.039 | 0.70  | 1.78  | 0.069 |
      +-------+-------+-------+-------+-------+-------+-------+-------+-------+

Member no. 1 given in Table `9.1 <#tab:oh-parameters>`__ defines the
most probable spectrum, while member no. 2 to 11 define the :math:`0.95`
percent confidence spectra.

A significant advantage of using a family of spectra for design of
marine systems is that one of the family members yields the largest
response such as motions or wave induced forces for a specified sea
severity, while anothers yield the smallest response with confidence
coefficient of :math:`0.95`.

Rodrigues and Soares (Rodriguez and Guedes Soares 2000), used the
Ochi-Hubble spectrum with 9 different parameterizations representing 3
types of sea state categories: swell dominated (a), wind sea dominated
(b) and mixed wind sea and swell system with comparable energy (c). Each
category is represented by 3 different inter-modal distances between the
swell and the wind sea spectral components. These three subgroups are
denoted by I, II and III, respectively. The exact values for the six
parameters are given in Table `9.2 <#tab:soares-param>`__. (See the
function ``ohspec3.m`` in the Wafo toolbox.)

.. container::
   :name: tab:soares-param

   .. table:: Target spectra parameters for mixed sea states.

      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       |       |       |       |       |       |       |     |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | type  |       |       |       |       |       |       |     |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | group | :ma   | :ma   | :     | :     | :mat  | :mat  |     |
      |       | th:`H | th:`H | math: | math: | h:`\l | h:`\l |     |
      |       | _{m0, | _{m0, | `\ens | `\ens | ambda | ambda |     |
      |       | \,1}` | \,2}` | urema | urema | _{1}` | _{2}` |     |
      |       |       |       | th{\o | th{\o |       |       |     |
      |       |       |       | mega  | mega  |       |       |     |
      |       |       |       | }_{p, | }_{p, |       |       |     |
      |       |       |       | \,1}` | \,2}` |       |       |     |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | I     | 5.5   | 3.5   | 0.440 | 0.691 | 3.0   | 6.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | a     | II    | 6.5   | 2.0   | 0.440 | 0.942 | 3.5   | 4.0 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | III   | 5.5   | 3.5   | 0.283 | 0.974 | 3.0   | 6.0 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | I     | 2.0   | 6.5   | 0.440 | 0.691 | 3.0   | 6.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | b     | II    | 2.0   | 6.5   | 0.440 | 0.942 | 4.0   | 3.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | III   | 2.0   | 6.5   | 0.283 | 0.974 | 2.0   | 7.0 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | I     | 4.1   | 5.0   | 0.440 | 0.691 | 2.1   | 2.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      | c     | II    | 4.1   | 5.0   | 0.440 | 0.942 | 2.1   | 2.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+
      |       | III   | 4.1   | 5.0   | 0.283 | 0.974 | 2.1   | 2.5 |
      +-------+-------+-------+-------+-------+-------+-------+-----+

.. _cha:wave-models:

Wave models
===========

Generally the wind generated sea waves is a non-linear random process.
Non-linearities are important in the wave-zone, i.e., from the crest to
1-2 wave amplitudes below the trough. Below this zone linear theory is
acceptable. However, there are unsolved physical problems associated
with the modelling of breaking waves. In practice, linear theory is used
to simulate irregular sea and to obtain statistical estimates. Often a
transformation of the linear model is used to emulate the non-linear
behavior of the sea surface, but as faster computers are becoming
available also higher order approximations will become common in ocean
engineering practice. In the following sections we will outline these
wave models. Only long-crested sea is used here either recorded at a
fixed spatial location or at a fixed point in time.

.. _sec:linear-gaussian-wave:

The linear Gaussian wave model
------------------------------

Gaussian random surface can be obtained as a first order approximation
of the solutions to differential equations based on linear hydrodynamic
theory of gravity waves. The first order component is given by the
following Fourier series

.. math::

   \label{eq:linearcomponent}
    \eta_{l}(x,t) = \sum_{n=-N}^{N} \frac{A_{n}}{2} e^{i\psi_{n}}

where the phase functions are

.. math::

   \label{eq:phasefunction}
     \psi_{n} = \omega_{n}\,t-k_{n}\,x  %- \epsilon_{n}

If :math:`\eta_{l}` is assumed to be stationary and Gaussian then the
complex amplitudes :math:`A_{j}` are also Gaussian distributed. The mean
square amplitudes are related to the one-sided wave spectrum
:math:`S_{\eta\eta}^{+}(\omega)` by

.. math::

   \label{eq:304}
     E[|A_{n }|^{2}] = 2\,S_{\eta\eta}^{+}(|\omega_{n}|) \Delta \omega

The individual frequencies, :math:`\ensuremath{\omega }_{n}` and
wavenumbers, :math:`k_{n}` are related through the linear dispersion
relation

.. math::

   \label{eq:dispersionrelation}
     \ensuremath{\omega }^{2} = g \,k\, \tanh(k\,d)

where :math:`g` and :math:`d` are the acceleration of gravity and water
depth, respectively. For deep water
Eq. (`[eq:dispersionrelation] <#eq:dispersionrelation>`__) simplifies to

.. math::

   \label{eq:29}
     \ensuremath{\omega }^{2} = g\,k

It implies the following relation between the wave frequency spectrum
and the wave number spectrum

.. math::

   \label{eq:33}
     S_{\eta\eta}^{+}(\ensuremath{\omega }) = \frac{2\ensuremath{\omega }}{g} S_{\eta\eta}^{+}(k)

Without loss of generality it is assumed that :math:`\eta_{l}` has zero
expectation. It is also assumed that :math:`\eta` is ergodic, i.e., any
ensemble average may be replaced by the corresponding time-space
average. This means that one single realization of :math:`\eta` is
representative of the random field. Here it is also assumed
:math:`\ensuremath{\omega }_{-j} = -\ensuremath{\omega }_{j}`,
:math:`k_{-j} = -k_{j}` and :math:`A_{-j} = \bar{A}_{j}` where
:math:`\bar{A}_{j}` is the complex conjugate of :math:`A_{j}`. The
matlab program ``spec2sdat.m`` in WAFO use the Fast Fourier Transform
(FFT) to evaluate Eq. (`[eq:linearcomponent] <#eq:linearcomponent>`__).

.. _sec:second-order-non:

The Second order non-linear wave model
--------------------------------------

Real wave data seldom follow the linear Gaussian model perfectly. The
model can be corrected by including quadratic terms. Following (Langley
1987) the quadratic correction :math:`\eta_{q}` is given by

.. math::

   \label{eq:nonlinearcomponent}
     \eta_{q}(x,t) = \sum_{n=-N}^{N} \sum_{m=-N}^{N} \frac{A_{n}A_{m}}{4} E(\ensuremath{\omega }_{n},\ensuremath{\omega }_{m})\,e^{i\,(\psi_{n}+\psi_{m})}

where the quadratic transferfunction (QTF),
:math:`E(\ensuremath{\omega }_{n},\ensuremath{\omega }_{m})` is given by

.. math::

   \label{eq:QTF}
   E(\ensuremath{\omega }_{i},\ensuremath{\omega }_{j}) = \frac{\frac{gk_{i}k_{j}}{\ensuremath{\omega }_{i}\ensuremath{\omega }_{j}} -
     \frac{1}{2g}(\ensuremath{\omega }_{i}^{2}+\ensuremath{\omega }_{j}^{2}+\ensuremath{\omega }_{i}\ensuremath{\omega }_{j})+\frac{g}{2}\frac{\ensuremath{\omega }_{i}k_{j}^{2}+\ensuremath{\omega }_{j}k_{i}^{2}}{\ensuremath{\omega }_{i}\,\ensuremath{\omega }_{j}(\ensuremath{\omega }_{i}+\ensuremath{\omega }_{j})}}{1-g\frac{k_{i}+k_{j}}{(\ensuremath{\omega }_{i}+\ensuremath{\omega }_{j})^{2}}\tanh\bigl((k_{i}+k_{j})d\bigr)}
   -\frac{gk_{i}k_{j}}{2\ensuremath{\omega }_{i}\ensuremath{\omega }_{j}}+\frac{1}{2g}(\ensuremath{\omega }_{i}^{2}+\ensuremath{\omega }_{j}^{2}+\ensuremath{\omega }_{i}\ensuremath{\omega }_{j})

For deep water waves the QTF simplifies to

.. math::

   \label{eq:EsumAndEdiff}
     E(\ensuremath{\omega }_{i},\ensuremath{\omega }_{j}) = \frac{1}{2\,g}(\ensuremath{\omega }_{i}^{2}+\ensuremath{\omega }_{j}^{2}),
   \quad
     E(\ensuremath{\omega }_{i},-\ensuremath{\omega }_{j}) = -\frac{1}{2\,g}|\ensuremath{\omega }_{i}^{2}-\ensuremath{\omega }_{j}^{2}|

where :math:`\ensuremath{\omega }_{i}` and
:math:`\ensuremath{\omega }_{j}` are positive and satisfies the same
relation as in the linear model.

However, if the spectrum does not decay rapidly enough towards zero, the
contribution from the 2nd order wave components at the upper tail can be
very large and unphysical. The predicted non-linearities are sensitive
to how the input spectrum is treated (cut-off) as shown by (Stansberg
1994).

One method to ensure convergence of the perturbation series is to
truncate the upper tail of the spectrum at
:math:`\ensuremath{\omega }_{max}` in the calculation of the 1st and 2nd
order wave components. The (Nestegård and Stokka 1995) program *WAVSIM*
set :math:`\ensuremath{\omega }_{max}=\sqrt{2.0\,g/(0.95\, H_{m0})}`.
(Brodtkorb, Myrhaug, and Rue 2000) showed that this will have the side
effect of giving the medium to low wave-heights a too low steepness
(which may not be too serious in real application). However, using the
truncation method the spectrum for the simulated series will deviate
from the target spectrum in 2 ways: (1) no energy or wave components
exist above the upper frequency limit
:math:`\ensuremath{\omega }_{max}`, (2) the energy in the spectrum below
:math:`\ensuremath{\omega }_{max}` will be higher than the target
spectrum. In order to retain energy above
:math:`\ensuremath{\omega }_{max}` in the spectrum, one may only
truncate the upper tail of the spectrum for the calculation of the 2nd
order components. However, in a real application one usually wants the
simulated data to have a prescribed target spectrum. Thus a more correct
approach is to eliminate the second order effects from the spectrum
before using it in the non-linear simulation. One way to do this is to
extract the linear components from the spectrum by a fix-point iteration
on the spectral density using the non-linear simulation program so that
the simulated data will have approximately the prescribed target
spectrum. This method is implemented as matlab function
``spec2linspec.m`` available in the WAFO toolbox. To accomplish
convergence, the same seed is used in each call of the non-linear
simulation program.

.. figure:: spec6comparisonNew
   :alt: Target spectrum, :math:`S_{T}(\ensuremath{\omega })`, (solid)
   and its linear component, :math:`S_{L}(\ensuremath{\omega })`
   (dash-dot) compared with :math:`S_{T}^{NLS}` (dash) and
   :math:`S_{L}^{NLS}` (dot), i.e., spectra of non-linearly simulated
   data using input spectrum :math:`S_{T}(\ensuremath{\omega })` (method
   1) and :math:`S_{L}(\ensuremath{\omega })` (method 2), respectively.
   :name: fig:spec6comparison
   :width: 3in

   Target spectrum, :math:`S_{T}(\ensuremath{\omega })`, (solid) and its
   linear component, :math:`S_{L}(\ensuremath{\omega })` (dash-dot)
   compared with :math:`S_{T}^{NLS}` (dash) and :math:`S_{L}^{NLS}`
   (dot), i.e., spectra of non-linearly simulated data using input
   spectrum :math:`S_{T}(\ensuremath{\omega })` (method 1) and
   :math:`S_{L}(\ensuremath{\omega })` (method 2), respectively.

Fig.\ `10.1 <#fig:spec6comparison>`__ demonstrates the differences in
the spectra obtained from data simulated using these methods. The solid
line is the target spectrum, :math:`S_{T}(\ensuremath{\omega })`, and
the dash-dotted line is its linear component,
:math:`S_{L}(\ensuremath{\omega })`, obtained using method 2. The
spectra :math:`S_{T}^{NLS}` (dashed) and :math:`S_{L}^{NLS}` (dotted)
are estimated from non-linearly simulated data using the
:math:`S_{T}(\ensuremath{\omega })` and
:math:`S_{L}(\ensuremath{\omega })` spectra, respectively. As expected
:math:`S_{T}^{NLS}` is higher than :math:`S_{T}`, while
:math:`S_{L}^{NLS}` is indistinguishable from :math:`S_{T}`. It is also
worth noting that the difference between the spectra is small, but have
some impact on the higher order moments. For instance, the
:math:`\epsilon_{2}` and :math:`\epsilon_{4}` parameters calculated from
:math:`S_{T}^{NLS}` increase with :math:`6.1\%` and :math:`2.5\%`,
respectively. The corresponding values calculated from
:math:`S_{L}^{NLS}` increase with only :math:`0.5\%` and :math:`0.2\%`,
respectively.

The small difference between :math:`S_{T}(\ensuremath{\omega })` and
:math:`S_{L}(\ensuremath{\omega })` also lends some support to the view
noted earlier, that the difference frequency effect can not fully
explain the high values of the spectrum in the lower frequency part as
found by (Wist 2003).

The effects these methods have are discussed further in (Brodtkorb 2004)
and especially on wave steepness parameters. The second order non-linear
model explained here is implemented in WAFO as ``spec2nlsdat.m``. This
is a very efficient implementation that calculate
Eqs. (`[eq:nonlinearcomponent] <#eq:nonlinearcomponent>`__) to
(`[eq:EsumAndEdiff] <#eq:EsumAndEdiff>`__) in the bi-frequency domain
using a one-dimensional *FFT*. This is similar to the *WAVSIM* program
of (Nestegård and Stokka 1995), but is made even more efficient by
summing over non-zero spectral values only and by eliminating the need
for saving the computed results to the hard drive. *WAVSIM* use
:math:`40\, s` to evaluate a transform with :math:`32000` time
steps/frequencies compared with :math:`2\, s` for ``spec2nlsdat.m`` on a
Pentium M :math:`1700` *MHz* with :math:`1` *GB* of RAM. Thus the use of
second order random waves should now become common in ocean engineering
practice.

``spec2nlsdat.m`` also allows finite water depth and any spectrum as
input, in contrast to *WAVSIM*, which only uses infinite water depth and
the JONSWAP spectrum.

.. _sec:transf-line-gauss:

Transformed linear Gaussian model
---------------------------------

An alternative and faster method than including the quadratic terms to
the linear model, is to use a transformation. The non-Gaussian process,
:math:`\eta(x,t)`, is then a function of a single Gaussian process,
:math:`\eta_{l}(x,t)`

.. math::

   \label{eq:tran1}
     \eta(x,t)=G(\eta_{l}(x,t))

where :math:`G(\cdot)` is a continuously differentiable function with
positive derivative.

There are several ways to proceed when selecting the transformation. The
simplest alternative is to estimate :math:`G(\cdot)` by some parametric
or non-parametric means (see e.g. (Winterstein 1988; Ochi and Ahn 1994;
Rychlik, Johannesson, and Leadbetter 1997)).

The parametric formulas proposed by (Ochi and Ahn 1994) as well as
(Winterstein 1988) use the moments of :math:`\eta_{l}(x,t)` to compute
:math:`G(\cdot)`. Information about the moments can be obtained directly
from data or by using theoretical models. (Marthinsen and Winterstein
1992) derived an expression for the skewness and kurtosis of narrow
banded Stokes waves to the leading order and used these to define the
transformation. (Winterstein and Jha 1995) fitted a parametric model to
skewness and kurtosis of a second order model with a JONSWAP spectrum.

(Machado 2003) studied the performance of 6 transformation methods
including those mentioned above and concluded that the Hermite method in
general produces very good results.

.. _hermite-model-1:

Hermite model
~~~~~~~~~~~~~

The Hermite transformation model proposed by (Winterstein 1985)
approximates the true process by the following transformation of a
standard normal process :math:`Y(t)`:

.. math::

   \begin{gathered}
     G(y) = \mu + K \,\sigma \,[ y + c_{3}(y^2-1) + c_{4} \,(y^3-3\,y)] \\
   %\intertext{where}
        K  = 1/\sqrt{1+2\,c_{3}^2+6\,c_{4}^2}\end{gathered}

where :math:`\mu` and :math:`\sigma` are the mean and standard
deviation, respectively, of the true process. The unitless coefficients
:math:`c_{3}` and :math:`c_{4}` are chosen so that the transformed model
match the skewness, :math:`\rho_{3}`, and excess, :math:`\rho_{4}`, of
the true process. (Winterstein, Ude, and Kleiven 1994) improved the
parameterizations by minimizing lack-of-fit errors on :math:`\rho_{3}`
and :math:`\rho_{4}`, giving

.. math::

   \begin{aligned}
        c_{3}  &= \frac{\rho_{3}}{6} \,
   \frac{1-0.015\,|\rho_{3}|+ 0.3\, \rho_{3}^2}{1+0.2\,\rho_{4}} \\
        c_{4}  &= 0.1\,\left( \left( 1+1.25\,\rho_{4} \right)^{1/3}-1 \right)\,c_{41}                 \\
        c_{41} &= \left(1-\frac{1.43\,\rho_{3}^2}{\rho_{4}} \right)^{1-0.1\,(\rho_{4}+3)^{0.8}}\end{aligned}

These results apply for :math:`0\le 3/2\,\rho_{3}^{2}<\rho_{4}<12`,
which include most cases of practical interest. One may then estimate
:math:`c_{3}` and :math:`c_{4}` using the sample skewness,
:math:`\hat{\rho}_{3}`, but restrict :math:`\rho_{4}` so that
:math:`\rho_{4} =
\min(\max(3\,\hat{\rho}_{3}^{2}/2,\hat{\rho}_{4}),\min(4\,(4\hat{\rho}_{3}/3)^{2},12))`.
:math:`\hat{\rho}_{4}` is the sample excess and
:math:`(4\hat{\rho}_{3}/3)^{2}` is the leading excess contribution for
narrow banded Stokes waves as found by (Marthinsen and Winterstein
1992).

.. container:: references hanging-indent
   :name: refs

   .. container::
      :name: ref-BeardahBaxter1996

      Beardah, C. C., and M. J. Baxter. 1996. “MATLAB Routines for
      Kernel Density Estimation and the Graphical Representation of
      Archaeological Data.” *Analecta Praehistorica Leidensia*, 179–84.

   .. container::
      :name: ref-BrodtkorbEtal2000JointICCE

      Brodtkorb, P.A., D. Myrhaug, and H. Rue. 2000. “Joint
      Distributions of Wave Height and Wave Steepness Parameters.” In
      *Proc. 27’th Int. Conf. On Coastal Eng., Icce, Sydney, Australia*,
      1:545–58.

   .. container::
      :name: ref-Brodtkorb2004Probability

      Brodtkorb, Per A. 2004. “The Probability of Occurrence of
      Dangerous Wave Situations at Sea.” PhD thesis, Norwegian Univ. of
      Sci. and Technology, NTNU, Trondheim, Norway.

   .. container::
      :name: ref-HasselmannEtal1973Measurements

      Hasselmann, K., T. P. Barnett, E. Buows, H. Carlson, D. E.
      Carthwright, K. Enke, J. A. Ewing, et al. 1973. “Measurements of
      Wind-Wave Growth and Swell Decay During the Joint North Sea Wave
      Project.” *Deutschen Hydrografischen Zeitschrift* 12: 9–95.

   .. container::
      :name: ref-HaverAndNyhus1986Wave

      Haver, Sverre, and K. A. Nyhus. 1986. “A Wave Climate Description
      for Long Term Response Calculations.” In *Proc. 5’th Int. Symp. On
      Offshore Mech. And Arctic Eng., Omae, Tokyo, Japan*, 4:27–34.

   .. container::
      :name: ref-Langley1987Statistical

      Langley, R. S. 1987. “A Statistical Analysis of Non-Linear Random
      Waves.” *Ocean Eng.* 14 (5): 389–407.

   .. container::
      :name: ref-Machado2003Probability

      Machado, Ulla. 2003. “Probability Density Functions for Nonlinear
      Random Waves and Responses.” *Ocean Eng.* 3 (8): 1027–50.

   .. container::
      :name: ref-MarthinsenAndWinterstein1992Skewness

      Marthinsen, T., and S. R Winterstein. 1992. “On the Skewness of
      Random Surface Waves.” In *Proc. 2’nd Int. Offshore and Polar Eng.
      Conf., Isope, San Francisco, Usa*, III:472–78.

   .. container::
      :name: ref-NestegardAndStokka1995Third

      Nestegård, Arne, and Trond Stokka. 1995. “A Third-Order Random
      Wave Model.” In *Proc. 5’th Int. Offshore and Polar Eng. Conf.,
      Isope, the Hague, the Netherlands*, III:136–42.

   .. container::
      :name: ref-OchiAndHubble1976Six

      Ochi, M., and E. Hubble. 1976. “On Six Parameter Wave Spectra.” In
      *Proc. 15’th Int. Conf. On Coastal Eng., Icce*, 1:301–28.

   .. container::
      :name: ref-OchiAndAhn1994NonGaussian

      Ochi, M. K., and K. Ahn. 1994. “Non-Gaussian Probability
      Distribution of Coastal Waves.” In *Proc. 24’th Int. Conf. On
      Coastal Eng., Icce, Kobe, Japan*, 1:482–96.

   .. container::
      :name: ref-RodriguezAndSoares2000Wave

      Rodriguez, G. R., and C. Guedes Soares. 2000. “Wave Period
      Distribution in Mixed Sea States.” In *Proc. ETCE/Omae2000 Joint
      Conf. Energy for the New Millenium, New Orleans, La*.

   .. container::
      :name: ref-RychlikEtal1997Modelling

      Rychlik, I., P. Johannesson, and M. R. Leadbetter. 1997.
      “Modelling and Statistical Analysis of Ocean-Wave Data Using
      Transformed Gaussian Process.” *Marine Structures, Design,
      Construction and Safety* 10: 13–47.

   .. container::
      :name: ref-Silverman1986Density

      Silverman, B. W. 1986. *Density Estimation for Statistics and Data
      Analysis*. Monographs on Stat. and Appl. Prob. 26. Chapman; Hall.

   .. container::
      :name: ref-Stansberg1994SecondOrder

      Stansberg, C. T. 1994. “Second-Order Random Wave Kinematics.
      Description and Testing of a Numerical Estimation Method.” No.
      513041.01.01. MARINTEK, SINTEF Group, Trondheim, Norway.

   .. container::
      :name: ref-Torsethaugen1993Two

      Torsethaugen, Knut. 1993. “A Two Peak Wave Spectral Model.” In
      *Proc. 12’th Int. Conf. On Offshore Mech. And Arctic Eng., Omae,
      Glasgow, Scotland*.

   .. container::
      :name: ref-Torsethaugen1994Model

      ———. 1994. “Model for Doubly Peaked Wave Spectrum. Lifetime and
      Fatigue Strength Estimation Implications.” Int. Workshop on
      Floating structures in Coastal zone, Hiroshima.

   .. container::
      :name: ref-Torsethaugen1996Model

      ———. 1996. “Model for Doubly Peaked Wave Spectrum.” No. STF22
      A96204. SINTEF Civ. and Env. Eng., Trondheim, Norway.

   .. container::
      :name: ref-TorsethaugenEtal1984Characteristica

      Torsethaugen, K., and others. 1984. “Characteristica for Extreme
      Sea States on the Norwegian Continental Shelf.” No. STF60 A84123.
      Norwegian Hydrodyn. Lab., Trondheim.

   .. container::
      :name: ref-WAFO-group2000Wafo

      WAFO-group. 2000. “WAFO - a Matlab Toolbox for Analysis of Random
      Waves and Loads - a Tutorial.”
      http://www.maths.lth.se/matstat/wafo/.

   .. container::
      :name: ref-LindgrenPrevosto2017

      *Wafo Lagrange – a Wafo Module for Analysis of Random Lagrange
      Waves*. 2017. In Wafo module ``lagrange``.

   .. container::
      :name: ref-WandAndJones1995Kernel

      Wand, M. P., and M. C. Jones. 1995. *Kernel Smoothing*. Monographs
      on Stat. and Appl. Prob. 60. Chapman; Hall.

   .. container::
      :name: ref-Winterstein1985Nonnormal

      Winterstein, S. R. 1985. “Nonnormal Responses and Fatigue Damage.”
      *J. Eng. Mech., ASCE* 111 (10): 1291–5.

   .. container::
      :name: ref-Winterstein1988Nonlinear

      ———. 1988. “Nonlinear Vibration Models for Extremes and Fatigue.”
      *J. Eng. Mech., ASCE* 114 (10): 1772–90.

   .. container::
      :name: ref-WintersteinAndJha1995Random

      Winterstein, S. R., and A. K. Jha. 1995. “Random Models of Second
      Order Waves and Local Wave Statistics.” In *Proc. 10’th Int. Eng.
      Mech. Speciality Conf., Asce*, 1171–4.

   .. container::
      :name: ref-WintersteinEtal1994Springing

      Winterstein, S. R., T. C. Ude, and G. Kleiven. 1994. “Springing
      and Slow-Drift Responses: Predicted Extremes and Fatigue Vs.
      Simulation.” In *Proc. 7’th Int. Behaviour of Offshore Structures,
      Boss*, 3:1–15.

   .. container::
      :name: ref-Wist2003Statistical

      Wist, Hanne Therese. 2003. “Statistical Properties of Successive
      Ocean Wave Parameters.” PhD thesis, Norwegian Univ. of Sci. and
      Technology, NTNU, Trondheim, Norway.

.. [1]
   Norwegian Defense Research Establishment, Horten, Norway.

.. [2]
   AMISE = asymptotic mean integrated square error

.. [3]
   Norwegian Defense Research Establishment, Horten, Norway.

.. [4]
   AMISE = asymptotic mean integrated square error
