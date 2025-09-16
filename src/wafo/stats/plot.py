'''
Created on 16. sep. 2025

@author: pab
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.interpolate import interp1d


def plotnorm(data, estimatelines=True, percentiles=True, df=np.inf, ax=None):
    """
    Plot data on a Normal or Student-t distribution probability paper.

    Parameters
    ----------
    data : array_like
        A vector with data, or a matrix with data from several groups,
        with one group per column.
    estimatelines : Bool, optional
        If True, estimated lines will be drawn. Defaults to True.
    percentiles : Bool, optional
        If True, percentiles will be drawn. Defaults to True.
    df : int or inf, optional
        Degrees of freedom in the t-distribution.
        If `df` is `np.inf`, a normal distribution paper will be used.
        If `0 < df < np.inf`, an approximation of the t-distribution will be used.
    ax : matplotlib.axex.Axes object
        Axes to plot

    Returns
    -------
    ax: matplotlib.axex.Axes object
        A handle to the plotted axes.

    See Also
    --------
    plotqq

    Notes
    -----
    Plotnorm uses an approximation for the t-distribution probability
    plot when `df` is finite.

    The original MATLAB code is distributed under the GNU Lesser General Public
    License.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.random.normal(0, 1, (100, 2))
    >>> plotnorm(R)
    >>> plotnorm(R, df=5)
    """

    if df <= 0 or (df != np.round(df) and not np.isinf(df)):
        raise ValueError('df must be a positive integer or infinity.')

    if ax is None:
        _fig, ax = plt.subplots()

    tdistr = not np.isinf(df)
    x = np.atleast_1d(data)

    # Handle single row vector input
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n, m = x.shape
    x_sorted = np.sort(x, axis=0)
    X = (np.arange(1, n + 1) - 0.5) / n
    Y = np.sqrt(2) * erfinv(2 * X - 1)

    if tdistr:
        g1 = 1 / 4 * (Y**3 + Y)
        g2 = 1 / 96 * (5 * Y**5 + 16 * Y**3 + 3 * Y)
        g3 = 1 / 384 * (3 * Y**7 + 19 * Y**5 + 17 * Y**3 - 15 * Y)
        g4 = 1 / 92160 * (79 * Y**9 + 776 * Y**7 + 1482 * Y**5 - 1920 * Y**3 - 945 * Y)
        Z = Y + g1 / df + g2 / (df**2) + g3 / (df**3) + g4 / (df**4)
        Y = Z

    linregx = Y
    SXX = np.sum((linregx - np.mean(linregx))**2)


    for i in range(m):
        linregy = x_sorted[:, i]
        SXY = np.dot((linregx - np.mean(linregx)), (linregy - np.mean(linregy)))
        b = SXY / SXX
        a = np.mean(linregy) - b * np.mean(linregx)

        ax.plot(x_sorted[:, i], Y, 'b.', markersize=12)

        if estimatelines:
            plot_x = np.array([x_sorted[0, i], x_sorted[n - 1, i]])
            plot_y = (plot_x - a) / b
            ax.plot(plot_x, plot_y, 'r--')

    span = np.max(x_sorted) - np.min(x_sorted)
    xx1 = np.min(x_sorted) - 0.1 * span
    xx2 = np.max(x_sorted) + 0.1 * span

    ax.set_xlim([xx1, xx2])
    ax.set_ylim([-4, 4])

    if not tdistr:
        ax.set_title('Normal Probability Plot')
        ax.set_ylabel('Quantiles of standard normal')
    else:
        ax.set_title('Student probability t-plot')
        ax.set_ylabel('Quantiles of Student t')

    if percentiles:
        levels = np.array([.5, .7, .9, .95, .98, .99, .995, .999, .9999])
        levels = np.concatenate((1 - np.flip(levels[1:]), levels))
        lev = np.sqrt(2) * erfinv(2 * levels - 1)

        ax.hlines(y=lev, xmin=xx1, xmax=xx2, color='k', linestyle='-')
        for i, lvl in enumerate(levels):
            ax.text(1.01*xx2, lev[i],
                    '{:.2f}%'.format(lvl*100),
                    verticalalignment='center',
                    fontsize=10,
                    fontstyle='italic')
    return ax


def plotqq(x, y, ps=None, method='linear', ax=None):
    """
    Plot empirical quantile of X versus empirical quantile of Y.

    If two distributions are the same (or possibly linearly transformed),
    the points should form an approximately straight line.

    Parameters
    ----------
    x : array_like
        First dataset.
    y : array_like
        Second dataset.
    ps : str, optional
        Plot symbol and style string (e.g., 'r--'). The default is '.'.
    method : str, optional
        Estimation method for quantiles. Defaults to 'linear'.
        Valid options are: 'inverted_cdf','averaged_inverted_cdf','closest_observation'
        'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear' (default), 'median_unbiased'
        'normal_unbiased'
        The first three methods are discontinuous. NumPy further defines the following
        discontinuous variations of the default 'linear' option:
        'lower', 'higher', 'midpoint' and 'nearest'.
    ax : matplotlib.axex.Axes object
        Axes to plot

    Returns
    -------
    ax:
        A handle to the plotted axes.

    See Also
    --------
    plotnorm
    scipy.stats.probplot : Similar function for plotting against a theoretical distribution.
    scipy.stats.iqr : Computes the interquartile range.

    Notes
    -----
    This function handles vectors of different lengths by interpolating
    the quantiles of the longer vector to match the length of the shorter one.
    This corresponds to the logic in the original MATLAB script.

    The original MATLAB code is distributed under the GNU Lesser General Public
    License.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import gumbel_r
    >>>
    >>> R1 = gumbel_r.rvs(loc=0, scale=1, size=50)
    >>> R2 = gumbel_r.rvs(loc=2, scale=1, size=100)
    >>>
    >>> # Plot two Gumbel-distributed datasets against each other
    >>> h = plotqq(R1, R2)
    >>> plt.show()
    """
    if ax is None:
        _fig, ax = plt.subplots()

    if ps is None:
        ps = '.'
    ps2 = 'r-.'

    x = np.sort(x)
    y = np.sort(y)

    nx = len(x)
    ny = len(y)

    n = min(nx, ny)
    ne = max(int(np.floor(n / 10)), 0)
    ix = np.arange(ne, n - ne)

    if nx < ny:
        yi = np.percentile(y, (np.arange(1, nx + 1) - 0.5)*100 / nx, method=method)
        ax.plot(x, yi, ps)

        # Linear fit for a straight line
        coeffs = np.polyfit(x[ix], yi[ix], 1)
        fit_line = coeffs[0] * x + coeffs[1]
        ax.plot(x, fit_line, ps2)

    else:
        xi = np.percentile(x, (np.arange(1, ny + 1) - 0.5)*100 / ny, method=method)
        ax.plot(xi, y, ps)

        # Linear fit for a straight line
        coeffs = np.polyfit(xi[ix], y[ix], 1)
        fit_line = coeffs[0] * xi + coeffs[1]
        ax.plot(xi, fit_line, ps2)

    ax.set_xlabel('Quantiles of X')
    ax.set_ylabel('Quantiles of Y')
    ax.set_title('Q-Q Plot')
    ax.grid(True)
    ax.set_aspect('equal')

    return ax


def test_percentile():
    from scipy.stats import gumbel_r

    # Single vector example
    x_vec = gumbel_r.rvs(loc=2, scale=1, size=100)
    q_vec = np.percentile(x_vec, [25, 50, 75])
    print(f"Quantiles for a single vector: {q_vec}")

    # Matrix example (2 columns)
    x_mat = np.random.rand(50, 2)
    q_mat = np.percentile(x_mat, [10, 90])
    print(f"\nQuantiles for each column of a matrix:\n{q_mat}")


def test_plot_qq():
    # Example from the original MATLAB code
    from scipy.stats import gumbel_r

    R1 = gumbel_r.rvs(loc=0, scale=1, size=50)
    R2 = gumbel_r.rvs(loc=2, scale=2, size=100)

    plotqq(R1, R2)
    plt.show()


def test_plotnorm():
    np.random.seed(0)
    R = np.random.normal(loc=0, scale=1, size=(100, 2))

    plotnorm(R)
    #plotnorm(R, df=5)
    plt.show()


if __name__ == '__main__':
    test_plot_qq()
    # test_percentile()