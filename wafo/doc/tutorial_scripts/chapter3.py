import numpy as np
from scipy import *
from pylab import *

#! CHAPTER3  Demonstrates distributions of wave characteristics
#!=============================================================
#!
#! Chapter3 contains the commands used in Chapter3 in the tutorial.
#! 
#! Some of the commands are edited for fast computation. 
#!
#! Section 3.2 Estimation of wave characteristics from data
#!----------------------------------------------------------
#! Example 1
#!~~~~~~~~~~ 
 
speed = 'fast'
#speed = 'slow'

import wafo.data as wd
import wafo.misc as wm
import wafo.objects as wo
xx = wd.sea() 
xx[:,1] = wm.detrend(xx[:,1])
ts = wo.mat2timeseries(xx)
Tcrcr, ix = ts.wave_periods(vh=0, pdef='c2c', wdef='tw', rate=8)
Tc, ixc = ts.wave_periods(vh=0, pdef='u2d', wdef='tw', rate=8)
 
#! Histogram of crestperiod compared to the kernel density estimate
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import wafo.kdetools as wk
clf()
print(Tc.mean())
print(Tc.max())

t = linspace(0.01,8,200);
ftc = wk.TKDE(Tc, L2=0, inc=128)

plot(t,ftc.eval_grid(t), t, ftc.eval_grid_fast(t),'-.') 
wm.plot_histgrm(Tc,normed=True)
title('Kernel Density Estimates')
axis([0, 8, 0, 0.5])
show() 
 
#! Extreme waves - model check: the highest and steepest wave
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clf()
S, H = ts.wave_height_steepness(method=0)
indS = S.argmax()
indH = H.argmax()
ts.plot_sp_wave([indH, indS],'k.')
show()

#! Does the highest wave contradict a transformed Gaussian model?
#!----------------------------------------------------------------

# TODO: Fix this

#clf
#inds1 = (5965:5974)'; #! points to remove
#Nsim = 10;
#[y1, grec1, g2, test, tobs, mu1o, mu1oStd] = ...
#   reconstruct(xx,inds1,Nsim);
#spwaveplot(y1,indA-10)
#hold on
#plot(xx(inds1,1),xx(inds1,2),'+')
#lamb = 2.;
#muLstd = tranproc(mu1o-lamb*mu1oStd,fliplr(grec1));
#muUstd = tranproc(mu1o+lamb*mu1oStd,fliplr(grec1));
#plot (y1(inds1,1), [muLstd muUstd],'b-')
#axis([1482 1498 -1 3]), 
#wafostamp([],'(ER)')
#disp('Block = 6'), 
#pause(pstate)
#
##!#! Expected value (solid) compared to data removed
#clf
#plot(xx(inds1,1),xx(inds1,2),'+'), hold on
#mu = tranproc(mu1o,fliplr(grec1));
#plot(y1(inds1,1), mu), hold off
#disp('Block = 7'), pause(pstate)

#! Crest height PDF
#!------------------
#! Transform data so that kde works better
clf()
wave_data = ts.wave_parameters()
Ac = wave_data['Ac']
L2 = 0.6;
import pylab
ws.probplot(Ac**L2, dist='norm', plot=pylab)
show()

#!#!
#!
fac = kde(Ac,{'L2',L2},linspace(0.01,3,200));
pdfplot(fac)
wafostamp([],'(ER)')
simpson(fac.x{1},fac.f)
disp('Block = 8'), pause(pstate)

#!#! Empirical crest height CDF
clf
Fac = flipud(cumtrapz(fac.x{1},flipud(fac.f)));
Fac = [fac.x{1} 1-Fac];
Femp = plotedf(Ac,Fac);
axis([0 2 0 1])
wafostamp([],'(ER)')
disp('Block = 9'), pause(pstate)

#!#! Empirical crest height CDF compared to a Transformed Rayleigh approximation
facr = trraylpdf(fac.x{1},'Ac',grec1);
Facr = cumtrapz(facr.x{1},facr.f);
hold on
plot(facr.x{1},Facr,'.')
axis([1.25 2.25 0.95 1])
wafostamp([],'(ER)')
disp('Block = 10'), pause(pstate)

#!#! Joint pdf of crest period and crest amplitude
clf
kopt2 = kdeoptset('L2',0.5,'inc',256);
Tc = Tcf+Tcb;
fTcAc = kdebin([Tc Ac],kopt2);
fTcAc.labx={'Tc [s]'  'Ac [m]'}
pdfplot(fTcAc)
hold on
plot(Tc,Ac,'k.')
hold off
wafostamp([],'(ER)')
disp('Block = 11'), pause(pstate)

#!#! Example 4:  Simple wave characteristics obtained from Jonswap spectrum 
clf
S = jonswap([],[5 10]);
[m,  mt]= spec2mom(S,4,[],0);
disp('Block = 12'), pause(pstate)

clf
spec2bw(S)
[ch Sa2] = spec2char(S,[1  3])
disp('Block = 13'), pause(pstate)

#!#! Section 3.3.2 Explicit form approximations of wave characteristic densities
#!#! Longuett-Higgins model for Tc and Ac
clf
t = linspace(0,15,100);
h = linspace(0,6,100);
flh = lh83pdf(t,h,[m(1),m(2),m(3)]);
disp('Block = 14'), pause(pstate)

#!#! Transformed Longuett-Higgins model for Tc and Ac
clf
[sk, ku ]=spec2skew(S);
sa = sqrt(m(1));
gh = hermitetr([],[sa sk ku 0]);
flhg = lh83pdf(t,h,[m(1),m(2),m(3)],gh);
disp('Block = 15'), pause(pstate)

#!#! Cavanie model for Tc and Ac
clf
t = linspace(0,10,100);
h = linspace(0,7,100);
fcav = cav76pdf(t,h,[m(1) m(2) m(3) m(5)],[]);
disp('Block = 16'), pause(pstate)

#!#! Example 5 Transformed Rayleigh approximation of crest- vs trough- amplitude
clf
xx = load('sea.dat');
x = xx;
x(:,2) = detrend(x(:,2));
SS = dat2spec2(x);
[sk, ku, me, si ] = spec2skew(SS);
gh = hermitetr([],[si sk ku me]);
Hs = 4*si;
r = (0:0.05:1.1*Hs)';
fac_h = trraylpdf(r,'Ac',gh);
fat_h = trraylpdf(r,'At',gh);
h = (0:0.05:1.7*Hs)';
facat_h = trraylpdf(h,'AcAt',gh);
pdfplot(fac_h)
hold on
pdfplot(fat_h,'--')
hold off
wafostamp([],'(ER)')
disp('Block = 17'), pause(pstate)

#!#!
clf
TC = dat2tc(xx, me);
tc = tp2mm(TC);
Ac = tc(:,2);
At = -tc(:,1);
AcAt = Ac+At;
disp('Block = 18'), pause(pstate)

#!#!
clf
Fac_h = [fac_h.x{1} cumtrapz(fac_h.x{1},fac_h.f)];
subplot(3,1,1)
Fac = plotedf(Ac,Fac_h);
hold on
plot(r,1-exp(-8*r.^2/Hs^2),'.')
axis([1. 2. 0.9 1])
title('Ac CDF')

Fat_h = [fat_h.x{1} cumtrapz(fat_h.x{1},fat_h.f)];
subplot(3,1,2)
Fat = plotedf(At,Fat_h);
hold on
plot(r,1-exp(-8*r.^2/Hs^2),'.')
axis([1. 2. 0.9 1])
title('At CDF')

Facat_h = [facat_h.x{1} cumtrapz(facat_h.x{1},facat_h.f)];
subplot(3,1,3)
Facat = plotedf(AcAt,Facat_h);
hold on
plot(r,1-exp(-2*r.^2/Hs^2),'.')
axis([1.5 3.5 0.9 1])
title('At+Ac CDF')

wafostamp([],'(ER)')
disp('Block = 19'), pause(pstate)

#!#! Section 3.4 Exact wave distributions in transformed Gaussian Sea
#!#! Section 3.4.1 Density of crest period, crest length or encountered crest period
clf
S1 = torsethaugen([],[6 8],1);
D1 = spreading(101,'cos',pi/2,[15],[],0);
D12 = spreading(101,'cos',0,[15],S1.w,1);
SD1 = mkdspec(S1,D1);
SD12 = mkdspec(S1,D12);
disp('Block = 20'), pause(pstate)

#!#! Crest period
clf
tic 
f_tc = spec2tpdf(S1,[],'Tc',[0 11 56],[],4);
toc
pdfplot(f_tc)
wafostamp([],'(ER)')
simpson(f_tc.x{1},f_tc.f)
disp('Block = 21'), pause(pstate)

#!#! Crest length

if strncmpi(speed,'slow',1)
  opt1 = rindoptset('speed',5,'method',3);
  opt2 = rindoptset('speed',5,'nit',2,'method',0);
else
  #! fast
  opt1 = rindoptset('speed',7,'method',3);
  opt2 = rindoptset('speed',7,'nit',2,'method',0);
end


clf
if strncmpi(speed,'slow',1)
  NITa = 5;
else
  disp('NIT=5 may take time, running with NIT=3 in the following')
  NITa = 3;
end
#!f_Lc = spec2tpdf2(S1,[],'Lc',[0 200 81],opt1); #! Faster and more accurate
f_Lc = spec2tpdf(S1,[],'Lc',[0 200 81],[],NITa);
pdfplot(f_Lc,'-.')
wafostamp([],'(ER)')
disp('Block = 22'), pause(pstate)


f_Lc_1 = spec2tpdf(S1,[],'Lc',[0 200 81],1.5,NITa);
#!f_Lc_1 = spec2tpdf2(S1,[],'Lc',[0 200 81],1.5,opt1);

hold on
pdfplot(f_Lc_1)
wafostamp([],'(ER)')

disp('Block = 23'), pause(pstate)
#!#!
clf
simpson(f_Lc.x{1},f_Lc.f)
simpson(f_Lc_1.x{1},f_Lc_1.f)
      
disp('Block = 24'), pause(pstate)
#!#!
clf
tic

f_Lc_d1 = spec2tpdf(rotspec(SD1,pi/2),[],'Lc',[0 300 121],[],NITa);
f_Lc_d12 = spec2tpdf(SD12,[],'Lc',[0 200 81],[],NITa);
#! f_Lc_d1 = spec2tpdf2(rotspec(SD1,pi/2),[],'Lc',[0 300 121],opt1);
#! f_Lc_d12 = spec2tpdf2(SD12,[],'Lc',[0 200 81],opt1);
toc
pdfplot(f_Lc_d1,'-.'), hold on
pdfplot(f_Lc_d12),     hold off
wafostamp([],'(ER)')

disp('Block = 25'), pause(pstate)

#!#!


clf
opt1 = rindoptset('speed',5,'method',3);
SD1r = rotspec(SD1,pi/2);
if strncmpi(speed,'slow',1)
  f_Lc_d1_5 = spec2tpdf(SD1r,[], 'Lc',[0 300 121],[],5);
  pdfplot(f_Lc_d1_5),     hold on
else
  #! fast
  disp('Run the following example only if you want a check on computing time')
  disp('Edit the command file and remove #!')
end
f_Lc_d1_3 = spec2tpdf(SD1r,[],'Lc',[0 300 121],[],3);
f_Lc_d1_2 = spec2tpdf(SD1r,[],'Lc',[0 300 121],[],2);
f_Lc_d1_0 = spec2tpdf(SD1r,[],'Lc',[0 300 121],[],0);
#!f_Lc_d1_n4 = spec2tpdf2(SD1r,[],'Lc',[0 400 161],opt1);

pdfplot(f_Lc_d1_3), hold on
pdfplot(f_Lc_d1_2)
pdfplot(f_Lc_d1_0)
#!pdfplot(f_Lc_d1_n4)

#!simpson(f_Lc_d1_n4.x{1},f_Lc_d1_n4.f)

disp('Block = 26'), pause(pstate)

#!#! Section 3.4.2 Density of wave period, wave length or encountered wave period
#!#! Example 7: Crest period and high crest waves
clf
tic
xx = load('sea.dat');
x = xx;
x(:,2) = detrend(x(:,2));
SS = dat2spec(x);
si = sqrt(spec2mom(SS,1));
SS.tr = dat2tr(x);
Hs = 4*si
method = 0;
rate = 2;
[S, H, Ac, At, Tcf, Tcb, z_ind, yn] = dat2steep(x,rate,method);
Tc = Tcf+Tcb;
t = linspace(0.01,8,200);
ftc1 = kde(Tc,{'L2',0},t);
pdfplot(ftc1)
hold on
#!      f_t = spec2tpdf(SS,[],'Tc',[0 8 81],0,4);
f_t = spec2tpdf(SS,[],'Tc',[0 8 81],0,2);
simpson(f_t.x{1},f_t.f)
pdfplot(f_t,'-.')
hold off
wafostamp([],'(ER)')
toc
disp('Block = 27'), pause(pstate)

#!#!
clf
tic

if strncmpi(speed,'slow',1)
  NIT = 4;
else
  NIT = 2;
end
#!      f_t2 = spec2tpdf(SS,[],'Tc',[0 8 81],[Hs/2],4);
tic
f_t2 = spec2tpdf(SS,[],'Tc',[0 8 81],Hs/2,NIT);
toc

Pemp = sum(Ac>Hs/2)/sum(Ac>0)
simpson(f_t2.x{1},f_t2.f)
index = find(Ac>Hs/2);
ftc1 = kde(Tc(index),{'L2',0},t);
ftc1.f = Pemp*ftc1.f;
pdfplot(ftc1)
hold on
pdfplot(f_t2,'-.')
hold off
wafostamp([],'(ER)')
toc
disp('Block = 28'), pause(pstate)

#!#! Example 8: Wave period for high crest waves 
#!      clf
      tic 
      f_tcc2 = spec2tccpdf(SS,[],'t>',[0 12 61],[Hs/2],[0],-1);
toc
      simpson(f_tcc2.x{1},f_tcc2.f)
      f_tcc3 = spec2tccpdf(SS,[],'t>',[0 12 61],[Hs/2],[0],3,5);
#!      f_tcc3 = spec2tccpdf(SS,[],'t>',[0 12 61],[Hs/2],[0],1,5);
      simpson(f_tcc3.x{1},f_tcc3.f)
      pdfplot(f_tcc2,'-.')
      hold on
      pdfplot(f_tcc3)
      hold off
       toc
disp('Block = 29'), pause(pstate)

#!#!
clf
[TC tc_ind v_ind] = dat2tc(yn,[],'dw');
N = length(tc_ind);
t_ind = tc_ind(1:2:N);
c_ind = tc_ind(2:2:N);
Pemp = sum(yn(t_ind,2)<-Hs/2 & yn(c_ind,2)>Hs/2)/length(t_ind)
ind = find(yn(t_ind,2)<-Hs/2 & yn(c_ind,2)>Hs/2);
spwaveplot(yn,ind(2:4))
wafostamp([],'(ER)')
disp('Block = 30'), pause(pstate)

#!#!
clf
Tcc = yn(v_ind(1+2*ind),1)-yn(v_ind(1+2*(ind-1)),1);
t = linspace(0.01,14,200);
ftcc1 = kde(Tcc,{'kernel' 'epan','L2',0},t);
ftcc1.f = Pemp*ftcc1.f;
pdfplot(ftcc1,'-.')
wafostamp([],'(ER)')
disp('Block = 31'), pause(pstate)

tic 
f_tcc22_1 = spec2tccpdf(SS,[],'t>',[0 12 61],[Hs/2],[Hs/2],-1);
toc
simpson(f_tcc22_1.x{1},f_tcc22_1.f)
hold on
pdfplot(f_tcc22_1)
hold off
wafostamp([],'(ER)')
disp('Block = 32'), pause(pstate)

disp('The rest of this chapter deals with joint densities.')
disp('Some calculations may take some time.') 
disp('You could experiment with other NIT.')
#!return

#!#! Section 3.4.3 Joint density of crest period and crest height
#!#! Example 9. Some preliminary analysis of the data
clf
tic
yy = load('gfaksr89.dat');
SS = dat2spec(yy);
si = sqrt(spec2mom(SS,1));
SS.tr = dat2tr(yy);
Hs = 4*si
v = gaus2dat([0 0],SS.tr);
v = v(2)
toc
disp('Block = 33'), pause(pstate)

#!#!
clf
tic
[TC, tc_ind, v_ind] = dat2tc(yy,v,'dw');
N       = length(tc_ind);
t_ind   = tc_ind(1:2:N);
c_ind   = tc_ind(2:2:N);
v_ind_d = v_ind(1:2:N+1);
v_ind_u = v_ind(2:2:N+1);
T_d     = ecross(yy(:,1),yy(:,2),v_ind_d,v);
T_u     = ecross(yy(:,1),yy(:,2),v_ind_u,v);

Tc = T_d(2:end)-T_u(1:end);
Tt = T_u(1:end)-T_d(1:end-1);
Tcf = yy(c_ind,1)-T_u;
Ac = yy(c_ind,2)-v;
At = v-yy(t_ind,2);
toc
disp('Block = 34'), pause(pstate)

#!#!
clf
tic
t = linspace(0.01,15,200);
kopt3 = kdeoptset('hs',0.25,'L2',0); 
ftc1 = kde(Tc,kopt3,t);
ftt1 = kde(Tt,kopt3,t);
pdfplot(ftt1,'k')
hold on
pdfplot(ftc1,'k-.')
f_tc4 = spec2tpdf(SS,[],'Tc',[0 12 81],0,4,5);
f_tc2 = spec2tpdf(SS,[],'Tc',[0 12 81],0,2,5);
f_tc = spec2tpdf(SS,[],'Tc',[0 12 81],0,-1);
pdfplot(f_tc,'b')
hold off
legend('kde(Tt)','kde(Tc)','f_{tc}')
wafostamp([],'(ER)')
toc
disp('Block = 35'), pause(pstate)

#!#! Example 10: Joint characteristics of a half wave:
#!#! position and height of a crest for a wave with given period
clf
tic
ind = find(4.4<Tc & Tc<4.6);
f_AcTcf = kde([Tcf(ind) Ac(ind)],{'L2',[1 .5]});
pdfplot(f_AcTcf)
hold on
plot(Tcf(ind), Ac(ind),'.');
wafostamp([],'(ER)')
toc
disp('Block = 36'), pause(pstate)

#!#!
clf
tic
opt1 = rindoptset('speed',5,'method',3);
opt2 = rindoptset('speed',5,'nit',2,'method',0);

f_tcfac1 = spec2thpdf(SS,[],'TcfAc',[4.5 4.5 46],[0:0.25:8],opt1);
f_tcfac2 = spec2thpdf(SS,[],'TcfAc',[4.5 4.5 46],[0:0.25:8],opt2);

pdfplot(f_tcfac1,'-.')
hold on
pdfplot(f_tcfac2)
plot(Tcf(ind), Ac(ind),'.');

simpson(f_tcfac1.x{1},simpson(f_tcfac1.x{2},f_tcfac1.f,1))
simpson(f_tcfac2.x{1},simpson(f_tcfac2.x{2},f_tcfac2.f,1))
f_tcf4=spec2tpdf(SS,[],'Tc',[4.5 4.5 46],[0:0.25:8],6);
f_tcf4.f(46)
toc
wafostamp([],'(ER)')
disp('Block = 37'), pause(pstate)

#!#!
clf
f_tcac_s = spec2thpdf(SS,[],'TcAc',[0 12 81],[Hs/2:0.1:2*Hs],opt1);
disp('Block = 38'), pause(pstate)

clf
tic
mom = spec2mom(SS,4,[],0);
t = f_tcac_s.x{1};
h = f_tcac_s.x{2};
flh_g = lh83pdf(t',h',[mom(1),mom(2),mom(3)],SS.tr);
clf
ind=find(Ac>Hs/2);
plot(Tc(ind), Ac(ind),'.');
hold on
pdfplot(flh_g,'k-.')
pdfplot(f_tcac_s)
toc
wafostamp([],'(ER)')
disp('Block = 39'), pause(pstate)

#!#!
clf
#!      f_tcac = spec2thpdf(SS,[],'TcAc',[0 12 81],[0:0.2:8],opt1);
#!      pdfplot(f_tcac)
disp('Block = 40'), pause(pstate)

#!#! Section 3.4.4 Joint density of crest and trough height
#!#! Section 3.4.5 Min-to-max distributions � Markov method
#!#! Example 11. (min-max problems with Gullfaks data)
#!#! Joint density of maximum and the following minimum
clf
tic
tp = dat2tp(yy);
Mm = fliplr(tp2mm(tp));
fmm = kde(Mm);
f_mM = spec2mmtpdf(SS,[],'mm',[],[-7 7 51],opt2);

pdfplot(f_mM,'-.')
hold on
pdfplot(fmm,'k-')
hold off
wafostamp([],'(ER)')
toc
disp('Block = 41'), pause(pstate)

#!#! The joint density of �still water separated�  maxima and minima.
clf
tic
ind = find(Mm(:,1)>v & Mm(:,2)<v);
Mmv = abs(Mm(ind,:)-v);
fmmv = kde(Mmv);
f_vmm = spec2mmtpdf(SS,[],'vmm',[],[-7 7 51],opt2);
clf
pdfplot(fmmv,'k-')
hold on
pdfplot(f_vmm,'-.')
hold off
wafostamp([],'(ER)')
toc
disp('Block = 42'), pause(pstate)


#!#!
clf
tic
facat = kde([Ac At]);
f_acat = spec2mmtpdf(SS,[],'AcAt',[],[-7 7 51],opt2);
clf
pdfplot(f_acat,'-.')
hold on
pdfplot(facat,'k-')
hold off
wafostamp([],'(ER)')
toc
disp('Block = 43'), pause(pstate)

