"""Example: scikits.statsmodels.discretemod
"""

import numpy as np
import scikits.statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt

#y = np.array([[300, 250, 200, 100, 90, 50, 20, 10, 5, 0],
#              [20, 70, 120, 220, 230, 270, 300, 310, 315, 320]]).T
#x = np.array([0, 1, 2, 3, 4, 5, 6, 7,8,9]).reshape((-1,1))

y = np.array([[2, 2, 3, 1, 4, 5, 6, 7, 5, 9],
              [8, 8, 7, 9, 6, 5, 4, 3, 5, 1]]).T * 10
x = np.array([0, 1, 2, 3, 4, 5, 6, 7,8,9]).reshape((-1,1))


# Logit Model
#logit_mod = sm.Logit(y, x)
#logit_res = logit_mod.fit()
#
#print(logit_res.params)
#print(logit_res.margeff())
#print(logit_res.predict(np.array([0,1,2,3,4]).reshape(-1,1)))
#print(y)
#
#logit_res.summary()
X = sm.add_constant(np.hstack((x,x**2)),prepend=False)
glm_binom = sm.GLM(y,  X, family=sm.families.Binomial()).fit()

print """The fitted values are
""", glm_binom.params
print """The corresponding t-values are
""", glm_binom.tvalues

print glm_binom.predict_bounds()
#check summary
print glm_binom.summary()

y = y[:,0]*1.0/y.sum(axis=1)
nobs = glm_binom.nobs
yhat = glm_binom.mu
yhat2 = glm_binom.predict(X)
ylo, yup = glm_binom.predict_bounds(X)

plt.figure()
plt.plot(x, y,'.', x,yhat,'r-', x, ylo,'r--', x, yup,'r--' )
plt.figure()
glm_binom.plot_fit_summary()
# Plot of yhat vs y
plt.figure()
glm_binom.plot_fit()

#plt.scatter(yhat, y)
#line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=False)).fit().params
#fit = lambda x: line_fit[1]+line_fit[0]*x # better way in scipy?
#plt.plot(np.linspace(0,1,nobs), fit(np.linspace(0,1,nobs)))
#plt.title('Model Fit Plot')
#plt.ylabel('Observed values')
#plt.xlabel('Fitted values')


# Plot of yhat vs. Pearson residuals
plt.figure()
glm_binom.plot_resid_dependence(kind='pearson')
#plt.scatter(yhat, glm_binom.resid_pearson)
#plt.plot([0.0, 1.0],[0.0, 0.0], 'k-')
#plt.title('Residual Dependence Plot')
#plt.ylabel('Pearson Residuals')
#plt.xlabel('Fitted values')

# Histogram of standardized deviance residuals
plt.figure()
glm_binom.plot_resid_histogram(kind='deviance')
#res = glm_binom.resid_deviance.copy()
#stdres = (res - res.mean())/res.std()
#plt.hist(stdres, bins=25)
#plt.title('Histogram of standardized deviance residuals')

# QQ Plot of Deviance Residuals
plt.figure()
glm_binom.plot_resid_qq(kind='deviance')
#res.sort()
#p = np.linspace(0 + 1./(nobs-1), 1-1./(nobs-1), nobs)
#quants = np.zeros_like(res)
#for i in range(nobs):
#    quants[i] = stats.scoreatpercentile(res, p[i]*100)
#mu = res.mean()
#sigma = res.std()
#y = stats.norm.ppf(p, loc=mu, scale=sigma)
#plt.scatter(y, quants)
#plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
#plt.title('Normal - Quantile Plot')
#plt.ylabel('Deviance Residuals Quantiles')
#plt.xlabel('Quantiles of N(0,1)')
# in branch *-skipper
#from scikits.statsmodels.sandbox import graphics
#img = graphics.qqplot(res)

plt.show()
#plt.close('all')
