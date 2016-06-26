## CHAPTER5 contains the commands used in Chapter 5 of the tutorial
#
# CALL:  Chapter5
#
# Some of the commands are edited for fast computation.
# Each set of commands is followed by a 'pause' command.
#

# Tested on Matlab 5.3
# History
# Added Return values by GL August 2008
# Revised pab sept2005
# Added sections -> easier to evaluate using cellmode evaluation.
# Created by GL July 13, 2000
# from commands used in Chapter 5
#

## Chapter 5 Extreme value analysis

## Section 5.1 Weibull and Gumbel papers
from __future__ import division
import numpy as np
import scipy.interpolate as si
from wafo.plotbackend import plotbackend as plt
import wafo.data as wd
import wafo.objects as wo
import wafo.stats as ws
import wafo.kdetools as wk
pstate = 'off'

# Significant wave-height data on Weibull paper,

fig = plt.figure()
ax = fig.add_subplot(111)
Hs = wd.atlantic()
wei = ws.weibull_min.fit(Hs)
tmp = ws.probplot(Hs, wei, ws.weibull_min, plot=ax)
plt.show()
#wafostamp([],'(ER)')
#disp('Block = 1'),pause(pstate)

##
# Significant wave-height data on Gumbel paper,
plt.clf()
ax = fig.add_subplot(111)
gum = ws.gumbel_r.fit(Hs)
tmp1 = ws.probplot(Hs, gum, ws.gumbel_r, plot=ax)
#wafostamp([],'(ER)')
plt.show()
#disp('Block = 2'),pause(pstate)

##
# Significant wave-height data on Normal probability paper,
plt.clf()
ax = fig.add_subplot(111)
phat = ws.norm.fit2(np.log(Hs))
phat.plotresq()
#tmp2 = ws.probplot(np.log(Hs), phat, ws.norm, plot=ax)

#wafostamp([],'(ER)')
plt.show()
#disp('Block = 3'),pause(pstate)

##
# Return values in the Gumbel distribution
plt.clf()
T = np.r_[1:100000]
sT = gum[0] - gum[1] * np.log(-np.log1p(-1./T))
plt.semilogx(T, sT)
plt.hold(True)
# ws.edf(Hs).plot()
Nmax = len(Hs)
N = np.r_[1:Nmax+1]

plt.plot(Nmax/N, sorted(Hs, reverse=True), '.')
plt.title('Return values in the Gumbel model')
plt.xlabel('Return period')
plt.ylabel('Return value')
#wafostamp([],'(ER)')
plt.show()
#disp('Block = 4'),pause(pstate)

## Section 5.2 Generalized Pareto and Extreme Value distributions
## Section 5.2.1 Generalized Extreme Value distribution

# Empirical distribution of significant wave-height with estimated
# Generalized Extreme Value distribution,
gev = ws.genextreme.fit2(Hs)
gev.plotfitsummary()
# wafostamp([],'(ER)')
# disp('Block = 5a'),pause(pstate)

plt.clf()
x = np.linspace(0,14,200)
kde = wk.TKDE(Hs, L2=0.5)(x, output='plot')
kde.plot()
plt.hold(True)
plt.plot(x, gev.pdf(x),'--')
# disp('Block = 5b'),pause(pstate)

# Analysis of yura87 wave data.
# Wave data interpolated (spline) and organized in 5-minute intervals
# Normalized to mean 0 and std = 1 to get stationary conditions.
# maximum level over each 5-minute interval analysed by GEV
xn = wd.yura87()
XI = np.r_[1:len(xn):0.25] - .99
N = len(XI)
N = N - np.mod(N, 4*60*5)

YI = si.interp1d(xn[:, 0],xn[:, 1], kind='linear')(XI)
YI = YI.reshape(4*60*5, N/(4*60*5))  # Each column holds 5 minutes of
                                     # interpolated data.
Y5 = (YI - YI.mean(axis=0)) / YI.std(axis=0)
Y5M = Y5.maximum(axis=0)
Y5gev = ws.genextreme.fit2(Y5M,method='mps')
Y5gev.plotfitsummary()
#wafostamp([],'(ER)')
#disp('Block = 6'),pause(pstate)

## Section 5.2.2 Generalized Pareto distribution

# Exceedances of significant wave-height data over level 3,
gpd3 = ws.genpareto.fit2(Hs[Hs>3]-3, floc=0)
gpd3.plotfitsummary()
#wafostamp([],'(ER)')

##
plt.figure()
# Exceedances of significant wave-height data over level 7,
gpd7 = ws.genpareto.fit2(Hs(Hs>7), floc=7)
gpd7.plotfitsummary()
# wafostamp([],'(ER)')
# disp('Block = 6'),pause(pstate)

##
#Simulates 100 values from the GEV distribution with parameters (0.3, 1, 2),
# then estimates the parameters using two different methods and plots the
# estimated distribution functions together with the empirical distribution.
Rgev = ws.genextreme.rvs(0.3,1,2,size=100)
gp = ws.genextreme.fit2(Rgev, method='mps');
gm = ws.genextreme.fit2(Rgev, *gp.par.tolist(), method='ml')
gm.plotfitsummary()

gp.plotecdf()
plt.hold(True)
plt.plot(x, gm.cdf(x), '--')
plt.hold(False)
#wafostamp([],'(ER)')
#disp('Block =7'),pause(pstate)

##
# ;
Rgpd = ws.genpareto.rvs(0.4,0, 1,size=100)
gp = ws.genpareto.fit2(Rgpd, method='mps')
gml = ws.genpareto.fit2(Rgpd, method='ml')

gp.plotecdf()
x = sorted(Rgpd)
plt.hold(True)
plt.plot(x, gml.cdf(x))
# gm = fitgenpar(Rgpd,'method','mom','plotflag',0);
# plot(x,cdfgenpar(x,gm),'g--')
#gw = fitgenpar(Rgpd,'method','pwm','plotflag',0);
#plot(x,cdfgenpar(x,gw),'g:')
#gml = fitgenpar(Rgpd,'method','ml','plotflag',0);
#plot(x,cdfgenpar(x,gml),'--')
#gmps = fitgenpar(Rgpd,'method','mps','plotflag',0);
#plot(x,cdfgenpar(x,gmps),'r-.')
plt.hold(False)
#wafostamp([],'(ER)')
#disp('Block = 8'),pause(pstate)

##
# Return values for the GEV distribution
T = np.logspace(1, 5, 10);
#[sT, sTlo, sTup] = invgev(1./T,Y5gev,'lowertail',false,'proflog',true);

#T = 2:100000;
#k=Y5gev.params(1); mu=Y5gev.params(3); sigma=Y5gev.params(2);
#sT1 = invgev(1./T,Y5gev,'lowertail',false);
#sT=mu + sigma/k*(1-(-log(1-1./T)).^k);
plt.clf()
#plt.semilogx(T,sT,T,sTlo,'r',T,sTup,'r')
#plt.hold(True)
#N = np.r_[1:len(Y5M)]
#Nmax = max(N);
#plot(Nmax./N, sorted(Y5M,reverse=True), '.')
#plt.title('Return values in the GEV model')
#plt.xlabel('Return priod')
#plt.ylabel('Return value')
#plt.grid(True)
#disp('Block = 9'),pause(pstate)

## Section 5.3 POT-analysis

# Estimated expected exceedance over level u as function of u.
plt.clf()

mrl = ws.reslife(Hs,'umin',2,'umax',10,'Nu',200);
mrl.plot()
#wafostamp([],'(ER)')
#disp('Block = 10'),pause(pstate)

##
# Estimated distribution functions of monthly maxima
#with the POT method (solid),
# fitting a GEV (dashed) and the empirical distribution.

# POT- method
gpd7 = ws.genpareto.fit2(Hs(Hs>7)-7, method='mps', floc=0)
khat, loc, sigmahat = gpd7.par

muhat = len(Hs[Hs>7])/(7*3*2)
bhat = sigmahat/muhat**khat
ahat = 7-(bhat-sigmahat)/khat
x = np.linspace(5,15,200);
plt.plot(x,ws.genextreme.cdf(x, khat,bhat,ahat))
# disp('Block = 11'),pause(pstate)

##
# Since we have data to compute the monthly maxima mm over
#42 months we can also try to fit a
# GEV distribution directly:
mm = np.zeros((1,41))
for i in range(41):
    mm[i] = max(Hs[((i-1)*14+1):i*14])


gev = ws.genextreme.fit2(mm)


plt.hold(True)
gev.plotecdf()

plt.hold(False)
#wafostamp([],'(ER)')
#disp('Block = 12, Last block'),pause(pstate)

