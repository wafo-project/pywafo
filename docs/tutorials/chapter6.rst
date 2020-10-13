.. _cha:6:

Extreme value analysis
======================

Of particular interest in wave analysis is how to find extreme quantiles
and extreme significant values for a wave series. Often this implies
going outside the range of observed data, i.e. to predict, from a
limited number of observations, how large the extreme values might be.
Such analysis is commonly known as *Weibull analysis* or *Gumbel
analysis*, from the names of two familiar extreme value distributions.
Both these distributions are part of a general family of extreme value
distributions, known as the *Generalized Extreme Value Distribution*,
(GEV). The *Generalized Pareto Distribution* (GPD) is another
distribution family, particularly adapted for *Peaks Over Threshold*
(POT), analysis. Wafo contains routines for fitting of such
distributions, both for the Weibull and Gumbel distributions, and for
the two more general classes of distributions. For a general
introduction to statistical extreme value analysis, the reader is
referred to (Coles 2001).

This chapter illustrates how Wafo can be used for elementary extreme
value analysis in the direct GEV method and in the POT method. The
example commands in ``Chapter6.m`` take a few seconds to run. We start
with a simple application of the classical Weibull and Gumbel analysis
before we turn to the general techniques.

.. _weibull-and-gumbel-papers-1:

Weibull and Gumbel papers
-------------------------

The Weibull and Gumbel distributions, the latter sometimes also called
“the” *extreme value distribution*, are two extreme value distributions
with distribution functions, respectively,

.. math::

   \begin{aligned}
   \mbox{Weibull:} \qquad F_W(x; a, c) & = & 1 - e^{-{(x/a)^c}}, \quad x > 0,
   \label{eq:Wei}
   \\[0.5em]
   \mbox{Gumbel:} \qquad F_G(x; a, b) & = & \exp\left( - e^{-(x-b)/a}\right),
   \quad -\infty < x < \infty. \label{eq:Gum}\end{aligned}

The Weibull distribution is often used as distribution for random
quantities which are the *minimum* of a large number of independent (or
weakly dependent) identically distributed random variables. In practice
it is used as a model for random strength of material, in which case it
was originally motivated by the principle of *weakest link*. Similarly,
the Gumbel distribution is used as a model for values which are *maxima*
of a large number of independent variables.

Since one gets the minimum of variables :math:`x_1, x_2, \ldots, x_n` by
changing the sign of the maximum of the sequence
:math:`-x_1, -x_2, \ldots , -x_n`, one realises that distributions
suitable for the analysis of maxima can also be used for analysis of
minima. Both the Weibull and the Gumbel distribution are members of the
class of Generalized Extreme Value distributions (GEV), which we shall
describe in Section `3.2 <#sec:GPD_GEV>`__.

.. _subsec:estimationandplotting:

Estimation and plotting
~~~~~~~~~~~~~~~~~~~~~~~

We begin here with an example of Weibull and Gumbel analysis, where we
plot data and empirical distribution and also estimate the parameters
:math:`a, b, c` in Eqs. (`[eq:Wei] <#eq:Wei>`__) and
(`[eq:Gum] <#eq:Gum>`__). The file ``atlantic.dat`` is included in Wafo,
and it contains significant wave-height data recorded approximately 14
times a month in the Atlantic Ocean in December to February during seven
years and at two locations. The data are stored in the vector ``Hs``. We
try to fit a Weibull distribution to this data set, by the Wafo-routine
``plotweib``, which performs both the estimation and the plotting.

::

         Hs = load('atlantic.dat');
         wei = plotweib(Hs)

This will result in a two element vector ``wei = [ahat chat]`` with
estimated values of the parameters :math:`(a, c)` in
(`[eq:Wei] <#eq:Wei>`__). The empirical distribution function of the
input data is plotted automatically in a Weibull diagram with scales
chosen to make the Weibull distribution function equal to a straight
line. The horizontal scale is logarithmic in the observations :math:`x`,
and the vertical scale is linear in the *reduced variable*
:math:`\log (-\log (1 - F(x)))`; see Figure `[fig7-1] <#fig7-1>`__\ (a).
Obviously, a Weibull distribution is not very well suited to describe
the significant wave-height data.

To illustrate the use of the Gumbel distribution we plot and estimate
the parameters :math:`(a, b)` in the Gumbel distribution
(`[eq:Gum] <#eq:Gum>`__) for the data in ``Hs``. The command

::

         gum = plotgumb(Hs)

results in a vector ``gum`` with estimated values ``[ahat bhat]`` and
the plot in Figure `[fig7-1] <#fig7-1>`__\ (b). Here the horizontal axis
is linear in the observations :math:`x` and the vertical axis carries
the reduced variable :math:`- \log (- \log(F(x)))`. The data shows a
much better fit to the Gumbel than to a Weibull distribution.

A distribution that is often hard to distinguish from the Gumbel
distribution is the Lognormal distribution, and making a Normal
probability plot of the logarithm of ``Hs`` in
Figure `[fig7-1] <#fig7-1>`__\ (c) also shows a good fit.

::

         plotnorm(log(Hs),1,0);

The parameter estimation in ``plotgumb`` and ``plotweib`` is done by
fitting a straight line to the empirical distribution functions in the
diagrams and using the relations

.. math:: \log\{-\log[1-F_{W}(x;a,c)]\}=c\log(x)-c\log(a),

and

.. math:: -\log\{-\log[F_{G}(x;a,b)]\}=x/a-b/a,

to relate parameters to intercepts and slopes of the estimated lines. In
the following section we shall describe some more statistical techniques
for parameter estimation in the Generalized Extreme Value distribution.

.. _subsec:returnvaluesWeibGumb:

Return value and return period 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The results of an extreme value analysis is often expressed in terms of
*return values* or *return levels*, which are simply related to the
quantiles of the distribution. A return value is always coupled to a
*return period*, expressed in terms of the length of an observation
period, or the number of (independent) observations of a random
variable.

Suppose we are interested in the return levels for the largest
significant wave height that is observed during one year at a measuring
site. Denote by :math:`M_{H_s}^k` the maximum during year number
:math:`k` and let its distribution function be :math:`F(x)`. Then the
:math:`N`-year return level, :math:`s_{N}`, is defined by

.. math::

   F(s_{N}) = 1 - 1/N.
   \label{eq:quantile}

For example, :math:`P(H_s > s_{100}) = 1 - F(s_{100}) = 1/100`, which
means that,

-  the probability that the level :math:`s_{100}` is exceeded during one
   particular year is :math:`0.01`,

-  on the average, the yearly maximum significant wave height exceeds
   :math:`s_{100}` one year in 100 years, (note that there may be
   several exceedances during that particular year),

-  the probability that :math:`s_{100}` is exceeded *at least* one time
   during a time span of 100 years is :math:`1-(1-0.01)^{100} \approx
   1-1/e = 0.6321`, provided years are independent.

To make it simple, we consider the Gumbel distribution, and get, from
(`[eq:quantile] <#eq:quantile>`__), the :math:`T`-year return value for
the yearly maximum in the Gumbel distribution (`[eq:Gum] <#eq:Gum>`__):

.. math::

   s_T = b - a \log (- \log (1 - 1/T)) \approx b + a \log T,
   \label{eq:GumbelreturnA}

where the last approximation is valid for large :math:`T`-values.

As an example we show a return value plot for the Atlantic data, as if
they represented a sequence of yearly maxima.
Figure `[fig7-1] <#fig7-1>`__\ (d) gives the return values as a function
of the return period for the Atlantic data. The Wafo-commands are:

::

         T = 1:100000;
         sT = gum(2) - gum(1)*log(-log(1-1./T));
         semilogx(T,sT), hold on
         N = 1:length(Hs); Nmax = max(N);
         plot(Nmax./N,sort(Hs,'descend'),'.')
         title('Return values in the Gumbel model')
         xlabel('Return priod')
         ylabel('Return value'), hold off

In the next section we shall see a more realistic example of return
value analysis. The Atlantic data did not represent yearly maxima and
the example was included only as an alternative way to present the
result of a Gumbel analysis.

.. _sec:GPD_GEV:

The GPD and GEV families
------------------------

The Generalized Pareto Distribution (GPD) has the distribution function

.. math::

   \mbox{GPD:} \qquad
     F(x; k, {\sigma})  =
     \left\{
     \begin{array}{ll}
     1 - \left( 1 - kx/{\sigma}\right)^{1/k},
     & \quad \mbox{if $k \neq 0$},\\[0.5em]
     1 - \exp \{-x/{\sigma}\}, & \quad \mbox{if $k = 0$},
     \end{array}
     \right.
     \label{eq:GPD}

for :math:`0 < x < \infty`, if :math:`k \leq 0`, and for
:math:`0 < x < {\sigma}/k`, if :math:`k
> 0`. The Generalized Extreme Value distribution (GEV) has distribution
function

.. math::

   \mbox{GEV:} \qquad F(x; k, {\mu}, {\sigma})  =
     \left\{
     \begin{array}{ll}
     \exp \left\{ - (1 - k(x-{\mu})/{\sigma})^{1/k}\right\},
     & \quad \mbox{if $k \neq 0$},\\[0.5em]
     \exp \left\{ - \exp \{ - (x-{\mu})/{\sigma}\} \right\},
     & \quad \mbox{if $k = 0$},
     \end{array}
     \right. \label{eq:GEV}

for :math:`k(x - {\mu}) < {\sigma}, \, {\sigma} > 0, \, k, \, {\mu}`
arbitrary. The case :math:`k=0` is interpreted as the limit when
:math:`k \to 0` for both distributions.

Note that the Gumbel distribution is a GEV distribution with :math:`k=0`
and that the Weibull distribution is equal to a reversed GEV
distribution with :math:`k=1/c`, :math:`{\sigma} = a/c`, and
:math:`{\mu} = -a`, i.e. if :math:`~W` has a Weibull distribution with
parameters :math:`(a,
c)` then :math:`- W` has a GEV distribution with :math:`k=1/c`,
:math:`{\sigma}= a/c`, and :math:`{\mu} = -a`.

The estimation of parameters in the GPD and GEV distributions is not a
simple matter, and no general method exists, which has uniformly good
properties for all parameter combinations. Wafo contains algorithms for
plotting of distributions and estimation of parameters with four
different methods, suitable in different regions.

.. _generalized-extreme-value-distribution-1:

Generalized Extreme Value distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the Generalized Extreme Value (GEV) distribution the estimation
methods used in the Wafo toolbox are the Maximum Likelihood (ML) method
and the method with Probability Weighted Moments (PWM), described in
(Prescott and Walden 1980) and (Hosking, Wallis, and Wood 1985). The
programs have been adapted to Matlab from a package of S-Plus routines
described in (Borg 1992).

We start with the significant wave-height data for the ``Atlantic``
data, stored in ``Hs``. The command

::

         gev = fitgev(Hs,'plotflag',2)

will give estimates ``gev.params = [khat sigmahat muhat]`` of the
parameters :math:`(k, {\sigma}, {\mu})` in the GEV
distribution (`[eq:GEV] <#eq:GEV>`__). The output matrix field
``gev.covariance`` will contain the estimated covariance matrix of the
estimates. The program also gives a plot of the empirical distribution
together with the best fitted distribution and two diagnostic plots that
give indications of the goodness of fit; see Figure `3.1 <#fig7-2>`__.

.. figure:: fig7-2ny
   :alt:  Empirical distribution (solid), cdf and pdf, of significant
   wave-height in ``atlantic`` data, with estimated (dashed) Generalized
   Extreme Value distribution, and two diagnostic plots og goodness of
   fit.
   :name: fig7-2
   :width: 110mm

   Empirical distribution (solid), cdf and pdf, of significant
   wave-height in ``atlantic`` data, with estimated (dashed) Generalized
   Extreme Value distribution, and two diagnostic plots og goodness of
   fit.

The routine ``plotkde``, which is a simplified version of the kernel
density estimation routines in ``kdetools``, is used to compare the GEV
density given estimated parameters with a non-parametric estimate (note
that ``plotkde`` can be slow for large data sets like ``Hs``). The
commands

::

         clf
         x = linspace(0,14,200);
         plotkde(Hs,[x;pdfgev(x,gev)]')

will give the upper right diagram in Figure `3.1 <#fig7-2>`__.

The default estimation algorithm for the GEV distribution is the method
with Probability Weighted Moments (PWM). An optional second argument,
``fitgev(Hs, method)``, allows a choice between the PWM-method (when
``method = ’pwm’``) and the alternative ML-method (when
``method = ’ml’``). The variances of the ML estimates are usually
smaller than those of the PWM estimates. However, it is recommended that
one first uses the PWM method, since it works for a wider range of
parameter values.

.. figure:: gevyura87
   :alt:  GEV analysis of 285 maxima over 5 minute intervals of sea
   level data Yura87.
   :name: fig7-2b
   :width: 110mm

   GEV analysis of 285 maxima over 5 minute intervals of sea level data
   Yura87.

**. *(Wave data from the Yura station)*\ 1em [yura87] The Wafo toolbox
contains a data set ``yura87`` of more than 23 hours of water level
registrations at the Poseidon platform in the Japan Sea; see
``help yura87``. Sampling rate is 1 Hz and to smooth data we interpolate
to 4 Hz, and then group the data into a matrix with 5 minutes of data in
each column, leaving out the last, unfinished period.**

::

        xn = load('yura87.dat');
        XI = 0:0.25:length(xn);
        N  = length(XI); N = N-mod(N,4*60*5);
        YI = interp1(xn(:,1),xn(:,2),XI(1:N),'spline');
        YI = reshape(YI,4*60*5,N/(4*60*5)); % Each column holds
                               %  5 minutes of interpolated data.

It turns out that the mean value and standard deviation change slowly
during the measuring period, and we therefore standardize each column to
zero mean and unit variance, before we take the maximum over each
5 minute interval and perform the GEV analysis; compare the results with
those in the simpler analysis in
Section `[sec:extreme_example] <#sec:extreme_example>`__.

::

        Y5 = (YI-ones(1200,1)*mean(YI))./(ones(1200,1)*std(YI));
        Y5M = max(Y5);
        Y5gev = fitgev(Y5M,'plotflag',2)

The estimated parameters in ``Y5gev.params`` are :math:`k = -0.314` with
a 95% confidence interval of :math:`(-0.12, 0.06)`, indicating that a
Gumbel distribution might be an acceptable choice. Location and scale
are estimated to :math:`\mu = 2.91` and :math:`\sigma = 0.34`.
Figure `3.2 <#fig7-2b>`__ shows a good fit to the GEV model for the
series of 5 minute maxima in the (standardized) Yura series, except for
the few largest values, which are underestimated by the model. This is
possibly due to a few short periods with very large variability in the
data. 1em :math:`\Box`

.. _generalized-pareto-distribution-1:

Generalized Pareto distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the Generalized Pareto distribution (GPD) the Wafo uses the method
with Probability Weighted Moments (PWM), described in (Hosking and
Wallis 1987), and the standard Method of Moments (MOM), as well as a
general method suggested by Pickands, in (Pickands 1975). S-Plus
routines for these methods are described in (Borg 1992).

The GPD is often used for exceedances over high levels, and it is well
suited as a model for significant wave heights. To fit a GPD to the
exceedances in the ``atlantic`` :math:`H_s` series over of thresholds 3
and 7, one uses the commands

::

         gpd3 = fitgenpar(Hs(Hs>3)-3,'plotflag',1);
         figure
         gpd7 = fitgenpar(Hs(Hs>7),'fixpar',...
                      [nan,nan,7],'plotflag',1);

This will give estimates ``gpd.params = [khat sigmahat]`` of the
parameters :math:`(k,
{\sigma})` in the Generalized Pareto
distribution (`[eq:GPD] <#eq:GPD>`__) based on exceedance data
``Hs(Hs>u)-u``. The optional output matrix ``gpd.covariance`` will
contain the estimated covariance matrix of the estimates. The program
also gives a plot of the empirical distribution together with the best
fitted distribution; see Figure `[fig7-3] <#fig7-3>`__. The fit is
better for exceedances over level 7 than over 3, but there are less data
available, and the confidence bounds are wider.

The choice of estimation method is rather dependent on the actual
parameter values. The default estimation algorithm in Wafo for
estimation in the Generalized Pareto distribution is the Maximum Product
of Spacings (MPS) estimator since it works for all values of the shape
parameter and have the same asymptotic properties as the Maximum
Likelihood (ML) method (when it is valid). The Pickands’ (PKD) and Least
Squares (LS) estimator also work for any value of the shape parameter
:math:`k` in Eq. (`[eq:GPD] <#eq:GPD>`__). The ML method is only useful
when :math:`k \leq 1`, the PWM when :math:`k>-0.5`, the MOM when
:math:`k>-0.25`. The variances of the ML estimates are usually smaller
than those of the other estimators. However, for small sample sizes it
is recommended to use the PWM, MOM or MPS if they are valid.

It is possible to simulate independent GEV and GPD observations in Wafo.
The command series

::

         Rgev = rndgev(0.3,1,2,1,100);
         gp = fitgev(Rgev,'method','pwm');
         gm = fitgev(Rgev,'method','ml','start',gp.params,...
                     'plotflag',0);
         x=sort(Rgev);
         plotedf(Rgev,gp,{'-','r-'}); hold on
         plot(x,cdfgev(x,gm),'--'); hold off

simulates 100 values from the GEV distribution with parameters
:math:`(0.3,1,2)`, then estimates the parameters using two different
methods and plots the estimated distribution functions together with the
empirical distribution. Similarly for the GPD distribution.

::

         Rgpd = rndgenpar(0.4,1,0,1,100);
         plotedf(Rgpd); hold on
         gp = fitgenpar(Rgpd,'method','pkd','plotflag',0);
         x=sort(Rgpd);
         plot(x,cdfgenpar(x,gp))
         gw = fitgenpar(Rgpd,'method','pwm','plotflag',0);
         plot(x,cdfgenpar(x,gw),'g:')
         gml = fitgenpar(Rgpd,'method','ml','plotflag',0);
         plot(x,cdfgenpar(x,gml),'--')
         gmps = fitgenpar(Rgpd,'method','mps','plotflag',0);
         plot(x,cdfgenpar(x,gmps),'r-.'); hold off

with four different methods of parameter estimation. The results are
shown in Figure `[fig7-4] <#fig7-4>`__\ (a) and (b).

.. _subsec_returnvalueanalysis:

Return value analysis
~~~~~~~~~~~~~~~~~~~~~

As in the Gumbel model, one can calculate the return levels in the GEV
by inverting (`[eq:quantile] <#eq:quantile>`__) with the GEV
distribution function (`[eq:GEV] <#eq:GEV>`__). The return level
corresponding to return period :math:`N` satisfies :math:`1-F(s_N)=1/N`,
so when :math:`F` is a GEV distribution function with shape parameter
:math:`k \neq 0`,

.. math::

   s_N = \mu + \frac{\sigma}{k} \left( 1 - (- \log (1 - 1/N) )^{k} \right)
   \approx \mu + \frac{\sigma}{k} \left( 1 - N^{-k} \right),
   \label{eq:return_gev}

where the last expression holds for :math:`N` large, so one can use
:math:`- \log (1 - 1/N) \approx 1/N`. As always in practice, the
parameters in the return level have to be replaced by their estimated
values, which introduces uncertainties in the computed level. 1ex1em
Applied to the ``yura87`` data and the estimated GEV-model, we perform
the return level extrapolation by the commands,

::

         T = 1:100000;
         k = Y5gev.params(1); mu=Y5gev.params(3);
         sigma = Y5gev.params(2);
         sT = mu + sigma/k*(1-(log(1-1./T))^k);
         semilogx(T,sT), hold
         N = 1:length(Y5M); Nmax=max(N);
         plot(Nmax./N,sort(Y5M,'descend'),'.')
         title('Return values in the GEV model')
         xlabel('Return priod')
         ylabel('Return value')
         grid on; hold off

The result is shown in Figure `3.3 <#fig_yurareturn>`__, consistent with
the quantile plot in Figure `3.2 <#fig7-2b>`__. 1em :math:`\Box`

.. figure:: fig5_6
   :alt:  Return level extrapolation in ``yura87`` data depends on the
   good fit in the main part of the distribution. A few deviating large
   observations are disturbing.
   :name: fig_yurareturn
   :width: 70mm

   Return level extrapolation in ``yura87`` data depends on the good fit
   in the main part of the distribution. A few deviating large
   observations are disturbing.

.. _pot-analysis-1:

POT-analysis
------------

Peaks Over Threshold analysis (POT) is a systematic way to analyse the
distribution of the exceedances over high levels in order to estimate
extreme quantiles outside the range of observed values. The method is
based on the observation that the extreme tail of a distribution often
has a rather simple and standardized form, regardless of the shape of
the more central parts of the distribution. One then fits such a simple
distribution only to those observations that exceed some suitable level,
with the hope that this fitted distribution gives an accurate fit to the
real distribution also in the more extreme parts. The level should be
chosen high enough for the tail to have approximately the standardized
form, but not so high that there remains too few observations above it.
After fitting a tail distribution one estimates the distribution of the
(random) number of exceedances over the level, and then combines the
tail distribution of the individual exceedances with the distribution
for the number of exceedances to find the total tail distribution.

.. _expected-exceedance-1:

Expected exceedance
~~~~~~~~~~~~~~~~~~~

The simplest distribution to fit to the exceedances over a level
:math:`u` is the Generalized Pareto distribution, GPD, with distribution
function (`[eq:GPD] <#eq:GPD>`__). Note that if a random variable
:math:`X` follows a Generalized Pareto distribution
:math:`F(x; k, {\sigma})`, then the exceedances over a level :math:`u`
is also GPD with distribution function :math:`F(x; k, {\sigma}-ku)`,
with the same :math:`k`-parameter but with different (if
:math:`k \neq 0`) scale parameter :math:`{\sigma} - ku`,

.. math::

   \mbox{\sf P}(X > u + y \mid X > u) = \frac{
   \left(1 - k \frac{u+y}{{\sigma}}\right)^{1/k}}
   {\left(1 - k \frac{u}{{\sigma}}\right)^{1/k}}
   = \left(1 - k \frac{y}{{\sigma}-ku}\right)^{1/k}.

Another important property of the Generalized Pareto Distribution is
that if :math:`k > -1`, then the mean exceedance over a level :math:`u`
is a linear function of :math:`u`:

.. math:: \mbox{\sf E}(X -u  \mid X > u) = \frac{{\sigma} - ku}{1+k}.

Plotting the mean exceedance as a function of :math:`u` can help on
decide on a proper threshold value. The resulting plot is called *Mean
residual life plot*, also referred to as mean excess plots in
statistical literature. The following command illustrate this for the
significant wave height ``atlantic`` data:

::

         plotreslife(Hs,'umin',2,'umax',10,'Nu',200);

The result is plotted in Figure `3.4 <#fig7-5>`__, and it seems to
exhibit an almost linear relationship for :math:`u\geq 7`.

.. figure:: fig7-5
   :alt: Expected exceedance over level :math:`u` of ``atlantic`` data
   as function of :math:`u`.
   :name: fig7-5
   :width: 70mm

   Expected exceedance over level :math:`u` of ``atlantic`` data as
   function of :math:`u`.

.. _poisson-gpd-gev-1:

Poisson + GPD = GEV
~~~~~~~~~~~~~~~~~~~

If one is successful in fitting a Generalized Pareto distribution to the
tail of data, one would like to use the GPD to predict how extreme
values might occur over a certain period of time. One could, for
example, want to predict the most extreme wave height that will appear
during a year. If the distribution of the individual significant wave
height exceedances is GPD one can easily find e.g., the distribution of
the largest value of a fixed number of exceedances. However, the number
of exceedances is not fixed but random, and then one has to combine the
distribution of the random size of individual exceedances with the
random number of exceedances :math:`N`, before one can say anything
about the total maximum. If the level :math:`u` is high we can, due to
the Poisson approximation of the Binomial distribution and neglecting
the dependence of nearby values, assume :math:`N` to have an approximate
Poisson distribution.

Now there is a nice relationship between the Generalized Pareto
distribution and the Generalized Extreme Value distribution in this
respect: *the maximum of a Poisson distributed number of independent GPD
variables has a GEV distribution*. This follows by simple summation of
probabilities: if :math:`N` is a Poisson distributed random variable
with mean :math:`{\mu}`, and :math:`M_{N} = \max (X_1, X_2,
\ldots, X_{N})` is the maximum of :math:`N` independent GPD variables
then,

.. math::

   \begin{aligned}
     \mbox{\sf P}(M_{N} \leq x) & = & \sum_{n=0}^{\infty} \mbox{\sf P}(N = n) \cdot
     \mbox{\sf P}(X_1 \leq x, X_2 \leq x, \ldots, X_n \leq x) \\
     & = & \sum_{n=0}^{\infty} e^{-{\mu}} \frac{{\mu}^n}{n!} \cdot
     \left( 1 - ( 1- k \frac{x}{{\sigma}})^{1/k}\right)^n \\
     & = & \exp \left\{- (1 - k(x-a)/b)^{1/k} \right\},\end{aligned}

which is the Generalized Extreme Value distribution with
:math:`b={\sigma}/{\mu}^k` and :math:`a={\sigma}(1 - {\mu}^{-k})/k`.

This means that we can estimate the distribution of the maximum
significant wave height during a winter (December – February) months
from our data set :math:`{\tt Hs}` by fitting a GPD to the exceedances
over some level :math:`u`, estimating :math:`{\mu}` by the number of
exceedances :math:`N` divided by the number of months
(:math:`7\times 3\times
2=42`) and use the above relation to fit a GEV distribution:

::

         gpd7 = fitgenpar(Hs(Hs>7)-7,'method','pwm','plotflag',0);
         khat = gpd7.params(1);
         sigmahat = gpd7.params(2);
         muhat = length(Hs(Hs>7))/(7*3*2);
         bhat = sigmahat/muhat^khat;
         ahat = 7-(bhat-sigmahat)/khat;
         x = linspace(5,15,200);
         plot(x,cdfgev(x,khat,bhat,ahat))

We have here used the threshold :math:`u=7` since the exceedances over
this level seem to fit well to a GPD distribution in
Figures `[fig7-3] <#fig7-3>`__\ (b) and `3.4 <#fig7-5>`__. A larger
value will improve the Poisson approximation to the number of
exceedances but give us less data to estimate the parameters.

Since we have approximately 14 data points for 41 complete months, we
can compute the monthly maxima ``mm`` and fit a GEV distribution
directly:

::

         mm = zeros(1,41);
         for i=1:41                    % Last month is not complete
           mm(i) = max(Hs(((i-1)*14+1):i*14));
         end
         gev = fitgev(mm);
         plotedf(mm), hold on
         plot(x,cdfgev(x,gev),'--'), hold off

The results of the two methods agree very well in this case as can be
seen in Figure `3.5 <#fig7-6>`__, where the estimated distributions are
plotted together with the empirical distribution of ``mm``.

.. figure:: fig7-6
   :alt: Estimated distribution functions of monthly maxima with the POT
   method (solid), fitting a GEV (dashed) and the empirical
   distribution.
   :name: fig7-6
   :width: 70mm

   Estimated distribution functions of monthly maxima with the POT
   method (solid), fitting a GEV (dashed) and the empirical
   distribution.

.. _declustering-1:

Declustering
~~~~~~~~~~~~

The POT method relies on two properties of peaks over the selected
threshold: they should occur randomly in time according to an
approximate Poisson process, and the exceedances should have an
approximate GPD distribution and be approximately independent. In
practice, one does not always find a Poisson distribution for the number
of exceedances. Since extreme values sometimes have a tendency to
cluster, some declustering algorithm could be applied to identify the
largest value in each of the clusters, and then use a Poisson
distribution for the number of clusters. The selected peaks should be
sufficiently far apart for the exceedances to be independent. The Wafo
toolbox contains the routine ``decluster`` to perform the declustering.

To select the clusters and check the Poisson character one can use the
*dispersion index*, which is the ratio between the variance and the
expectation of the number of peaks. For a Poisson distribution this
ratio is equal to one. An acceptable peak separation should give a
dispersion index near one.

**. *(Declustering sea data)*\ 1em [decluster] We will extract peaks
over threshold in the ``sea.dat``, which is a recording of almost 40
minutes of sea level data, sampled at a rate of ``4[Hz]``.**

We first define some parameters, ``Nmin,Tmin,Tb``, to control the
declustering, and to identify the peaks that exceed 90% of the median
peak size and are separated by at least ``Tmin``.

::

         Nmin = 7;                           % minimum number of extremes
         Tmin = 5;                           % minimum dist between extremes
         Tb = 15;                            % block period
         xx = load('sea.dat');
         timeSpan = (xx(end,1)-xx(1,1))/60;  % in minutes
         dt = xx(2,1)-xx(1,1);               % in seconds
         tc = dat2tc(xx);
         umin = median(tc(tc(:,2)>0,2));
         Ie0 = findpot(tc, 0.9*umin, Tmin);
         Ev = sort(tc(Ie0,2));
         Ne = numel(Ev)
         if Ne>Nmin && Ev(Ne-Nmin)>umin, umax = Ev(Ne-Nmin);
         else umax = umin;
         end

Next, we calculate the expected residual life and the dispersion index
for thresholds between ``umin`` and ``umax`` and select an interval
which is compatible with the Poisson distribution for the number of
peaks.

::

         Nu = floor((umax-umin)/0.025)+1;
         u = linspace(umin,umax,Nu);
         mrl = reslife(Ev, 'u',u);
         umin0 = umin;
         for io = numel(mrl.data):-1:1,
           CI = mrl.dataCI(io:end,:);
           if ~(max(CI(:,1))<=mrl.data(io) && mrl.data(io)<=min(CI(:,2))),
               umin0 = mrl.args(io); break;
           end
         end
         [di, threshold, ok_u] = ...
               disprsnidx(tc(Ie0,:), 'Tb', Tb, 'alpha',0.05, 'u',u);

The plots from the following commands are shown in
Figure `[fig:thresholds] <#fig:thresholds>`__. It seems as if
``threshold = 1.23[m]`` is a suitable threshold.

::

         figure(1); plot(di)
         vline(threshold)      % Threshold from dispersion index
         vline(umin0,'g')      % Threshold from mean residual life plot
         figure(2); plot(mrl)
         vline(threshold)      % Threshold from dispersion index
         vline(umin0,'g')      % Threshold from mean residual life plot

A GPD fit for peaks above ``1.23[m]`` with diagnostic plot is obtained
by the commands

::

         Ie = findpot(tc, threshold, Tmin);
         lambda = numel(Ie)/timeSpan; % # Y>threshold per minute
         varLambda = lambda*(1-(dt/60)*lambda)/timeSpan;
         stdLambd = sqrt(varLambda)
         Ev = tc(Ie,2);
         phat = fitgenpar(Ev, 'fixpar',[nan,nan,threshold], 'method','mps');
         figure(3); phat.plotfitsumry() % check fit to data

The diagnostic plots are found in Figure `3.6 <#fig:decluster3>`__. The
last step is to calculate the numerical value and some confidence
intervals for a return level, and we do so for a three hour period,
``180 min``.

::

         Tr = 3*60             % Return period in minutes
         [xr,xrlo,xrup] = invgenpar(1./(lambda*Tr),phat,...
           'lowertail',false,'alpha', 0.05) % return level + 95%CI
         [xr,xrlo5,xrup5] = invgenpar(1./(lambda*Tr),phat,...
           'lowertail',false,'alpha', 0.5)  % return level + 50%CI

The three hour return level is thus estimated to ``xr = 2.02[m]`` with a
95% confidence interval ``(1.30, 10.08)``. The 50% confidence bounds are
``(1.58, 3.05)``; as expected, a high confidence leads to a very high
upper limit. 1em :math:`\Box`

.. figure:: fig_decluster3b
   :alt: Diagnostic GPD plot for sea data return levels.
   :name: fig:decluster3
   :width: 110mm

   Diagnostic GPD plot for sea data return levels.

.. _sec:extremevaluestatistics:

Summary of statistical procedures in Wafo
-----------------------------------------

The extreme value analysis presented in this chapter is part of a
comprehensive library of statistical routines for random number
generation, probability distributions, and parameter and density
estimation and likelihood analysis, etc.

::

   help statistics

     Module STATISTICS in WAFO Toolbox.
     Version 2.5.2  07-Feb-2011

     What's new
       Readme           - New features, bug fixes, and changes
                           in STATISTICS.

     Parameter estimation
       fitbeta          - Parameter estimates for Beta data
       fitchi2          - Parameter estimates for
                                    Chi squared data
       fitexp           - Parameter estimates for
                                    Exponential data
       fitgam           - Parameter estimates for Gamma data
       fitgengam        - Parameter estimates for
                                    Generalized Gamma data
       fitgenpar        - Parameter estimates for
                                    Generalized Pareto data
       fitgenparml      - Internal routine for fitgenpar
                          (ML estimates for GPD data)
       fitgenparrange   - Parameter estimates for GPD model
                                    over a range of thresholds
       fitgev           - Parameter estimates for GEV data
       fitgumb          - Parameter estimates for Gumbel data
       fitinvnorm       - Parameter estimates for
                                    Inverse Gaussian data
       fitlognorm       - Parameter estimates for Lognormal data
       fitmarg2d        - Parameter estimates for MARG2D data
       fitmargcnd2d     - Parameter estimates for DIST2D data
       fitnorm          - Parameter estimates for Normal data
       fitray           - Parameter estimates for Rayleigh data
       fitraymod        - Parameter estimates for
                                    Truncated Rayleigh data
       fitt             - Parameter estimates for
                                    Student's T data
       fitweib          - Parameter estimates for Weibull data
       fitweib2d        - Parameter estimates for 2D Weibull data
       fitweibmod       - Parameter estimates for
                                    truncated Weibull data

       likgenpar        - Log likelihood function for GPD
       likweib2d        - 2D Weibull log-likelihood function

       loglike          - Negative Log-likelihood function.
       logps            - Moran's negative log Product
                                    Spacings statistic
       mlest            - Maximum Likelihood or Maximum Product
                          Spacing estimator

     Probability density functions (pdf)
       pdfbeta          - Beta PDF
       pdfbin           - Binomial PDF
       pdfcauchy        - Cauchy's PDF
       pdfchi2          - Chi squared PDF
       pdfdiscrete      - Discrete PDF
       pdfempirical     - Empirical PDF
       pdfexp           - Exponential PDF
       pdff             - Snedecor's F PDF
       pdffrech         - Frechet PDF
       pdfgam           - Gamma PDF
       pdfgengam        - Generalized Gamma PDF
       pdfgengammod     - Modified Generalized Gamma PDF (stable)
       pdfgenpar        - Generalized Pareto PDF
       pdfgev           - Generalized Extreme Value PDF
       pdfgumb          - Gumbel PDF.
       pdfhyge          - Hypergeometric probability mass function
       pdfinvnorm       - Inverse Gaussian PDF
       pdflognorm       - Lognormal PDF
       pdfmarg2d        - Joint 2D PDF due to Plackett given as
                          f{x1}*f{x2}*G(x1,x2;Psi).
       pdfmargcnd2d     - Joint 2D PDF computed as
                          f(x1|X2=x2)*f(x2)
       pdfnorm          - Normal PDF
       pdfnorm2d        - Bivariate Gaussian distribution
       pdfnormnd        - Multivariate Normal PDF
       pdfray           - Rayleigh PDF
       pdfraymod        - Truncated Rayleigh PDF
       pdft             - Student's T PDF
       pdfpois          - Poisson probability mass function
       pdfweib          - Weibull PDF
       pdfweib2d        - 2D Weibull PDF
       pdfweibmod       - Truncated Weibull PDF

     Cumulative distribution functions (cdf)
       cdfcauchy        - Cauchy CDF
       cdfdiscrete      - Discrete CDF
       cdfempirical     - Empirical CDF
       cdfmarg2d        - Joint 2D CDF due to Plackett
       cdfmargcnd2d     - Joint 2D CDF computed as
                          int F(X1<v|X2=x2).*f(x2)dx2
       cdfmargcnd2dfun  - is an internal function to cdfmargcnd2d
                          and prbmargcnd2d.
       cdfnormnd        - Multivariate normal CDF
       cdfweib2d        - Joint 2D Weibull CDF
       cdfbeta          - Beta CDF
       cdfbin           - Binomial CDF
       cdfchi2          - Chi squared CDF
       cdfexp           - Exponential CDF
       cdff             - Snedecor's F CDF
       cdffrech         - Frechet CDF
       cdfgam           - Gamma CDF
       cdfgengam        - Generalized Gamma CDF
       cdfgengammod     - Modified Generalized Gamma CDF
       cdfgenpar        - Generalized Pareto CDF
       cdfgev           - Generalized Extreme Value CDF
       cdfgumb          - Gumbel CDF
       cdfhyge          - The hypergeometric CDF
       cdfinvnorm       - Inverse Gaussian CDF
       cdflognorm       - Lognormal CDF
       cdfmargcnd2d     - Joint 2D CDF computed as
                          int F(X1<v|X2=x2).*f(x2)dx2
       cdfnorm          - Normal CDF
       cdfray           - Rayleigh CDF
       cdfraymod        - Modified Rayleigh CDF
       cdft             - Student's T CDF
       cdfpois          - Poisson CDF
       cdfweib          - Weibull CDF
       cdfweibmod       - Truncated Weibull CDF

       edf              - Empirical Distribution Function
       edfcnd           - Empirical Distribution Function
                          conditioned on X>=c

       prbmargcnd2d     - returns the probability for rectangular
                          regions
       prbweib2d        - returns the probability for rectangular
                          regions
       margcnd2dsmfun   - Smooths the MARGCND2D distribution
                          parameters
       margcnd2dsmfun2  - Smooths the MARGCND2D distribution
                          parameters

     Inverse cumulative distribution functions
       invbeta          - Inverse of the Beta CDF
       invbin           - Inverse of the Binomial CDF
       invcauchy        - Inverse of the Cauchy CDF
       invchi2          - Inverse of the Chi squared CDF
       invcmarg2d       - Inverse of the conditional CDF of
                          X2 given X1
       invcweib2d       - Inverse of the conditional 2D weibull
                          CDF of X2 given X1
       invdiscrete      - Disrete quantile
       invempirical     - Empirical quantile
       invexp           - Inverse of the Exponential CDF
       invf             - Inverse of Snedecor's F CDF
       invfrech         - Inverse of the Frechet CDF
       invgam           - Inverse of the Gamma CDF
       invgengam        - Inverse of the Generalized Gamma CDF
       invgengammod     - Inverse of the Generalized Gamma CDF
       invgenpar        - Inverse of the Generalized Pareto CDF
       invgev           - Inverse of the Generalized
                          Extreme Value CDF
       invgumb          - Inverse of the Gumbel CDF
       invhyge          - Inverse of the Hypergeometric CDF
       invinvnorm       - Inverse of the Inverse Ga(ussian CDF
       invlognorm       - Inverse of the Lognormal CDF
       invnorm          - Inverse of the Normal CDF
       invray           - Inverse of the Rayleigh CDF
       invt             - Inverse of the Student's T CDF
       invweib          - Inverse of the Weibull CDF
       invpois          - Inverse of the Poisson CDF
       invraymod        - Inverse of the modified Rayleigh CDF
       invweibmod       - Inverse of the modified Weibull CDF

     Random number generators
       rndalpha         - Random matrices from a symmetric
                          alpha-stable distribution
       rndbeta          - Random matrices from a Beta distribution
       rndbin           - Random numbers from the binomial
                          distribution
       rndboot          - Simulate a bootstrap resample from a
                          sample
       rndcauchy        - Random matrices a the Cauchy
                          distribution
       rndchi2          - Random matrices from a Chi squared
                          distribution
       rnddiscrete      - Random sample
       rndempirical     - Bootstrap sample
       rndexp           - Random matrices from an Exponential
                          distribution
       rndf             - Random matrices from Snedecor's F
                          distribution
       rndfrech         - Random matrices from a Frechet
                          distribution
       rndgam           - Random matrices from a Gamma distribution
       rndgengam        - Random matrices from a Generalized Gamma
                          distribution.
       rndgengammod     - Random matrices from a Generalized
                          Modified Gamma distribution.
       rndgenpar        - Random matrices from a Generalized Pareto
                          Distribution
       rndgev           - Random matrices from a Generalized
                          Extreme Value distribution
       rndgumb          - Random matrices from a Gumbel
                          distribution
       rndhyge          - Random numbers from the Hypergeometric
                          distribution
       rndinvnorm       - Random matrices from a Inverse Gaussian
                          distribution
       rndlognorm       - Random matrices from a Lognormal
                          distribution.
       rndmarg2d        - Random points from a MARG2D
                          distribution
       rndmargcnd2d     - Random points from a MARGCND2D
                          distribution
       rndnorm          - Random matrices from a Normal
                          distribution
       rndnormnd        - Random vectors from a multivariate
                          Normal distribution
       rndpois          - Random matrices from a Poisson
                          distribution
       rndray           - Random matrices from a Rayleigh
                          distribution
       rndraymod        - Random matrices from modified Rayleigh
                          distribution
       rndt             - Random matrices from a Student's T
                          distribution
       rndweib          - Random matrices a the Weibull
                          distribution
       rndweibmod       - Random matrices from the modified Weibull
                          distribution
       rndweib2d        - Random numbers from the 2D Weibull
                          distribution

     Moments
       mombeta          - Mean and variance for the Beta
                          distribution
       mombin           - Mean and variance for the Binomial
                          distribution
       momchi2          - Mean and variance for the Chi squared
                          distribution
       momexp           - Mean and variance for the Exponential
                          distribution
       momf             - Mean and variance for Snedecor's F
                          distribution
       momfrech         - Mean and variance for the Frechet
                          distribution
       momgam           - Mean and variance for the Gamma
                          distribution
       momgengam        - Mean and variance for the Generalized
                          Gamma distribution
       momgenpar        - Mean and variance for the Generalized
                          Pareto distribution
       momgev           - Mean and variance for the GEV
                          distribution
       momgumb          - Mean and variance for the Gumbel
                          distribution
       momhyge          - Mean and variance for the Hypergeometric
                          distribution
       mominvnorm       - Mean and variance for the Inverse
                          Gaussian distribution
       momlognorm       - Mean and variance for the Lognormal
                          distribution
       mommarg2d        - Mean and variance for the MARG2D
                          distribution
       mommargcnd2d     - Mean and variance for the MARGCND2D
                          distribution
       momnorm          - Mean and variance for the Normal
                          distribution
       mompois          - Mean and variance for the Poisson
                          distribution
       momray           - Mean and variance for the Rayleigh
                          distribution
       momt             - Mean and variance for the Student's T
                          distribution
       momweib          - Mean and variance for the Weibull
                          distribution
       momweib2d        - Mean and variance for the 2D Weibull
                          distribution

     Profile log likelihood functions
       lnkexp           - Link for x,F and parameters of
                          Exponential distribution
       lnkgenpar        - Link for x,F and parameters of
                          Generalized Pareto distribution
       lnkgev           - Link for x,F and parameters of
                          Generalized Extreme value distribution
       lnkgumb          - Link for x,F and parameters of Gumbel
                          distribution
       lnkgumbtrnc      - Link for x,F and parameters of truncated
                          Gumbel distribution
       lnkray           - Link for x,F and parameters of Rayleigh
                          distribution
       lnkweib          - Link for x,F and parameters of Weibull
                          distribution
       loglike          - Negative Log-likelihood function
       logps            - Moran's negative log Product Spacings
                          statistic
       ciproflog        - Confidence Interval using Profile Log-
                          likelihood or Product Spacing- function
       proflog          - Profile Log- likelihood or
                          Product Spacing-function
       findciproflog    - Find Confidence Interval from proflog
                          function

     Extremes
       decluster        - Decluster peaks over threshold values
       extremalidx      - Extremal Index measuring the dependence
                          of data
       findpot          - Find indices to Peaks over threshold
                          values
       fitgev           - Parameter estimates for GEV data
       fitgenpar        - Parameter estimates for Generalized
                          Pareto data
       prb2retper       - Return period from Probability of
                          exceedance
       retper2prb       - Probability of exceedance from return
                          period

     Threshold selection
       fitgenparrange   - Parameter estimates for GPD model vs
                          thresholds
       disprsnidx       - Dispersion Index vs threshold
       reslife          - Mean Residual Life, i.e., mean excesses
                          vs thresholds
       plotdisprsnidx   - Plot Dispersion Index vs thresholds
       plotreslife      - Plot Mean Residual Life
                          (mean excess vs thresholds)

     Regression models
       logit            - Logit function.
       logitinv         - Inverse logit function.
       regglm           - Generalized Linear Model regression
       reglm            - Fit multiple Linear Regression Model.
       reglogit         - Fit ordinal logistic regression model.
       regnonlm         - Non-Linear Model Regression
       regsteplm        - Stepwise predictor subset selection for
                          Linear Model regression

     Factor analysis
       princomp         -  Compute principal components of X

     Descriptive Statistics
       ranktrf          - Rank transformation of data material.
       spearman         - Spearman's rank correlation coefficient
       mean             - Computes sample mean (Matlab)
       median           - Computes sample median value (Matlab)
       std              - Computes standard deviation (Matlab)
       var              - Computes sample variance (Matlab)
       var2corr         - Variance matrix to correlation matrix
                          conversion
       cov              - Computes sample covariance matrix
                          (Matlab)
       corrcoef         - Computes sample correlation coefficients
                          (Matlab toolbox)
       skew             - Computes sample skewness
       kurt             - Computes sample kurtosis
       lmoment          - L-moment based on order statistics
       percentile       - Empirical quantile (percentile)
       iqrange          - Computes the Inter Quartile Range
       range            - Computes the range between the maximum
                          and minimum values

     Statistical plotting
       clickslct        - Select points in a plot by clicking
                          with the mouse
       histgrm          - Plot histogram
       plotbox          - Plot box-and-whisker diagram
       plotdensity      - Plot density.
       plotexp          - Plot data on Exponential distribution
                          paper
       plotedf          - Plot Empirical Distribution Function
       plotedfcnd       - Plot Empirical Distribution Function
                          CoNDitioned that X>=c
       plotfitsumry     - Plot diagnostic of fit to data
       plotgumb         - Plot data on Gumbel distribution paper
       plotkde          - Plot kernel density estimate of PDF
       plotmarg2dcdf    - Plot conditional CDF of X1 given X2=x2
       plotmarg2dmom    - Plot conditional mean and standard
                          deviation
       plotmargcnd2dcdf - Plot conditional empirical CDF of X1
                          given X2=x2
       plotmargcnd2dfit - Plot parameters of the conditional
                          distribution
       plotmargcnd2dmom - Plot conditional mean and
                          standard deviation
       plotnorm         - Plot data on a Normal distribution paper
       plotqq           - Plot empirical quantile of X vs empirical
                          quantile of Y
       plotray          - Plot data on a Rayleigh distribution
                          paper
       plotresprb       - Plot Residual Probability
       plotresq         - Plot Residual Quantile
       plotscatr        - Pairwise scatter plots
       plotweib         - Plot data on a Weibull distribution paper
       plotweib2dcdf    - Plot conditional empirical CDF of X1
                          given X2=x2
       plotweib2dmom    - Plot conditional mean and standard
                          deviation

     Hypothesis Tests
       anovan           - multi-way analysis of variance (ANOVA)
       testgumb         - Tests if shape parameter in a GEV is
                          equal to zero
       testmean1boot    - Bootstrap t-test for the mean equal to 0
       testmean1n       - Test for mean equals 0 using
                          one-sample T-test
       testmean2n       - Two-sample t-test for mean(x) equals
                          mean(y)
       testmean1r       - Wilcoxon signed rank test for
                          H0: mean(x) equals 0
       testmean2r       - Wilcoxon rank-sum test for
                          H0: mean(x) equals mean(y)

     Confidence interval estimation
       ciboot           - Bootstrap confidence interval.
       ciquant          - Nonparametric confidence interval for quantile
       momci1b          - Moment confidence intervals using
                          Bootstrap

     Bootstrap & jacknife estimates
       covboot          - Bootstrap estimate of the variance of
                          a parameter estimate.
       covjack          - Jackknife estimate of the variance of
                          a parameter estimate.
       stdboot          - Bootstrap estimate of the
                          standard deviation of a parameter
       stdjack          - Jackknife estimate of the
                          standard deviation of a parameter

     Design of Experiments
       yates            - Calculates main and interaction effects
                          using Yates' algorithm.
       ryates           - Reverse Yates' algorithm to give
                          estimated responses
       fitmodel         - Fits response by polynomial
       alias            - Alias structure of a fractional design
       cdr              - Complete Defining Relation
       cl2cnr           - Column Label to Column Number
       cnr2cl           - Column Number to Column Label
       ffd              - Two-level Fractional Factorial Design
       getmodel         - Return the model parameters
       sudg             - Some Useful Design Generators
       plotresponse     - Cubic plot of responses
       nplot            - Normal probability plot of effects

     Misc
       comnsize         - Calculates common size of all non-scalar
                          arguments
       dgammainc        - Incomplete gamma function with derivatives
       gammaincln       - Logarithm of incomplete gamma function.
       parsestatsinput  - Parses inputs to pdfxx, prbxx, invxx and
                          rndxx functions
       createfdata      - Distribution parameter struct constructor
       getdistname      - Return the distribution name

       stdize           - Standardize columns to have mean 0 and
                          standard deviation 1
       center           - Recenter columns to have mean 0

      Demo
       demofitgenpar    - Script to check the variance of estimated
                          parameters

.. container:: references hanging-indent
   :name: refs

   .. container::
      :name: ref-Borg1992XS

      Borg, S. 1992. “XS - a Statistical Program Package in Splus for
      Extreme-Value Analysis.” Dept. of Mathematical Statistics, Lund
      University.

   .. container::
      :name: ref-Coles2001

      Coles, S. 2001. *An Introduction to Statistical Modeling of
      Extreme Values*. London: Springer-Verlag.

   .. container::
      :name: ref-HoskingAndWallis1987Parameter

      Hosking, J. R. M., and J. R. Wallis. 1987. “Parameter and Quantile
      Estimation for the Generalized Pareto Distribution.”
      *Technometrics*.

   .. container::
      :name: ref-HoskingEtal1985Estimation

      Hosking, J. R. M., J. R. Wallis, and E. F Wood. 1985. “Estimation
      of the Generalized Extreme-Value Distribution by the Method of
      Probability-Weighted Moments.” *Technometrics*.

   .. container::
      :name: ref-Pickands1975Statistical

      Pickands, J. 1975. “Statistical Inference Using Extreme Order
      Statistics.” *Annals of Statistics* 3: 119–31.

   .. container::
      :name: ref-PrescottAndWalden1980Maximum

      Prescott, P., and A. T. Walden. 1980. “Maximum Likelihood
      Estimation of the Parameters of the Generalized Extreme-Value
      Distribution.” *Biometrika* 67: 723–24.
