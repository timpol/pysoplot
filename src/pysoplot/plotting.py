"""
Plotting routines and functions.

"""


import warnings
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
from scipy import stats

from . import cfg, ludwig
from . import misc


#=================================
# 2-d data plotting
#=================================

def rline(ax, theta):
    """
    Plot regression line.

    Parameters
    -----------
    ax : matplotlib.pyplot.Axes
        Axes to plot to.
    theta: array-like
        Array with y-intercept as first element and slope as second.

    """
    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax],
            [np.dot(theta, [1, xmin]), np.dot(theta, [1, xmax])],
            label='regression line', **cfg.rline_kw)


def renv(ax, fit):
    """
    Plot uncertainty envelope about regression line indicating 95% confidence
    limits on fit. For classical regression fits, uses the approach of
    Ludwig (1980), for non-classical fits uses Monte Carlo simulation.

    Parameters
    -----------
    ax : matplotlib.pyplot.Axes
        Axes to plot to.
    fit: dict
        Linear regression fit results.
        
    References
    ----------
    Ludwig, K.R., 1980. Calculation of uncertainties of U-Pb
    isotope data. Earth and Planetary Science Letters 212–202.

    """
    # TODO: this approach fails when x at ymin and x at ymax are very close (?)

    if fit['type'] == 'classical':
        xx = np.linspace(*ax.get_xlim(), num=100, endpoint=True)

        a, b = fit['theta']
        a_95pm, b_95pm = fit['theta_95ci']
        xbar = fit['xbar']

        # Eq. 29 of Ludwig (1980); EPSL.
        ym = a + b * xx
        yhigh = ym + np.sqrt(a_95pm ** 2 + b_95pm ** 2 * xx * (xx - 2. * xbar))
        ylow = ym - np.sqrt(a_95pm ** 2 + b_95pm ** 2 * xx * (xx - 2. * xbar))

        ax.plot(xx, ylow, label='regression envelope line', **cfg.renv_line_kw)
        ax.plot(xx, yhigh, label='regression envelope line', **cfg.renv_line_kw)
        ax.fill_between(xx, yhigh, ylow, label='regression envelope', **cfg.renv_kw)

    else:
        # use Monte Carlo approach:
        renv_mc(ax, fit)


def renv_mc(ax, fit):
    """
    Use Monte Carlo simulation to plot regression error envelope.
    """
    n = 100                  # number of points to construct curve
    m = 10_000                 # number of simulated theta

    theta = fit['theta']
    covtheta = fit['covtheta']

    # simulate slope and intercept values
    ths = cfg.rng.multivariate_normal(theta, covtheta, m)

    # Find intercept points b/w regression line and axis frame, then
    # sort these by increasing y.
    w_intercepts = box_line_intersection(*ax.get_xlim(), *ax.get_ylim(), *theta)
    w_intercepts = w_intercepts[w_intercepts[:, 1].argsort()]

    if w_intercepts.size == 0:
        warnings.warn('regression line envelope could not be plotted as it appears to lie outside axis limits')
        return

    # If line intercepts window at ymin and ymax, then simulate x for given
    # vector of evenly spaced y spanning axis window:
    if (np.isclose(w_intercepts[0, 1], ax.get_ylim()[0]) and
            np.isclose(w_intercepts[1, 1], ax.get_ylim()[1])):

        ymin = w_intercepts[0, 1]
        ymax = w_intercepts[1, 1]
        yspan = ymax - ymin
        # increase span a bit to avoid edge effects
        yq = np.linspace(ymin - 1.25 * yspan, ymax + 1.25 * yspan, num=n,
                         endpoint=True)

        # calculate simulated x for each y and theta combination
        # x = (y - a) / b over each y and a, b combination
        xq = np.tile(yq, (m, 1)).T - ths[:, 0]
        xq /= np.tile(ths[:, 1], (n, 1))

        # get upper and lower limits
        xhigh = np.quantile(xq, 0.975, axis=1)
        xlow = np.quantile(xq, 0.025, axis=1)

        ax.plot(xlow, yq, label='regression envelope line', **cfg.renv_line_kw)
        ax.plot(xhigh, yq, label='regression envelope line', **cfg.renv_line_kw)
        ax.fill_betweenx(yq, xlow, xhigh, label='regression envelope', **cfg.renv_kw)

    # Simulate y for vector of evenly spaced x:
    else:
        xmin = w_intercepts[0, 0]
        xmax = w_intercepts[1, 0]
        xspan = xmax - xmin
        # increase span a bit to avoid edge effects
        xq = np.linspace(xmin - 1.25 * xspan, xmax + 1.25 * xspan, num=n,
                         endpoint=True)

        # calculate simulated y values
        # y = theta . X
        yq = ths @ np.array((np.ones(n), xq))

        # get upper and lower limits
        yhigh = np.quantile(yq, 0.975, axis=0)
        ylow = np.quantile(yq, 0.025, axis=0)

        ax.plot(xq, ylow, label='regression envelope line', **cfg.renv_line_kw)
        ax.plot(xq, yhigh, label='regression envelope line', **cfg.renv_line_kw)
        ax.fill_between(xq, yhigh, ylow, label='regression envelope', **cfg.renv_kw)


def plot_rfit(ax, fit):
    """
    Plot regression line and uncertainty envelope about regression line
    indicating 95% confidence limits on fit.

    Parameters
    -----------
    ax : matplotlib.pyplot.Axes
        Axes to plot fit to.
    fit: dict
        Linear regression fit results.

    """
    rline(ax, fit['theta'])
    renv(ax, fit)


def plot_cor207_projection(ax, x, sx, y, sy, r_xy, Pb76, t=None, A=None,
                           init=None):
    """
    Plot projection line from the common 207Pb/206Pb value through the center
    of the data ellipse. If t is supplied, then the line will be projected to the
    concordia intercept. Otherwise, it will continue to the edge of the plot.

    Parameters
    ----------
    t : array-like, optional
        Intercept age for data point. Must be same length as x and y.
    """
    n = x.shape[0]
    if t is not None:
        assert t.shape[0] == n
        assert A is not None and init is not None

    for i in range(n):
        a = Pb76
        b = (y[i] - a) / x[i]

        if t is not None:
            DC8 = np.array((cfg.lam238, cfg.lam234, cfg.lam230, cfg.lam226))
            DC5 = np.array((cfg.lam235, cfg.lam231))
            BC8 = ludwig.bateman(DC8)
            BC5 = ludwig.bateman(DC5, series='235U')
            xc = 1. / ludwig.f(t[i], A[:-1], DC8, BC8, init=init)
            yc = ludwig.g(t[i], A[-1], DC5, BC5) * xc / cfg.U
        else:
            yc = ax.get_ylim()[0]
            xc = (yc - a) / b

        ax.plot([0, xc], [a, yc],
                label='projection line', **cfg.pb76_line_kw)


def plot_dp(x, sx, y, sy, r_xy, labels=None, p=0.95, reset_axis_limits=True):
    """
    Plot 2-dimensional data points with assigned uncertainties as confidence
    ellipses.

    Parameters
    -----------
    x : np.ndarray
        x values (as 1-dimensional array)
    sx : np.ndarray
        uncertainty on x at :math:`1\sigma` abs.
    y : np.ndarray
        y values
    sy : np.ndaray
        uncertainty on y at :math:`1\sigma` abs.
    r_xy : np.ndarray
        x-y correlation coefficient

    Notes
    ------
    Assumes a large sample size was used to estimate data point uncertainties
    and covariances.

    """
    if labels is not None:
        assert len(labels) == x.shape[0]

    fig, ax = plt.subplots(**cfg.fig_kw, subplot_kw=cfg.subplot_kw)
    n = x.shape[0]

    for i in range(n):
        confidence_ellipse(ax, x[i], sx[i], y[i], sy[i], r_xy[i], p=p,
                           ellipse_kw=cfg.dp_ellipse_kw,
                           mpl_label='data ellipse')
        if labels is not None:
            ax.annotate(labels[i], (x[i], y[i]), **cfg.dp_label_kw,
                        label='data ellipse label')

    # set axis limits
    if reset_axis_limits:
        fig.canvas.draw()
        inv = ax.transData.inverted().transform        # display to data coords
        xy = np.array([inv(e.get_extents().get_points()).flatten() for e in
                       ax.patches if e.get_label() == 'data ellipse'])
        xmin = np.min(xy[:, 0::2])
        xmax = np.max(xy[:, 0::2])
        ymin = np.min(xy[:, 1::2])
        ymax = np.max(xy[:, 1::2])
        xpad = (xmax - xmin) / 5
        ypad = (ymax - ymin) / 5
        ax.set_xlim((xmin - xpad, xmax + xpad))
        ax.set_ylim((ymin - ypad, ymax + ypad))

        # turn off to prevent further auto updates (is this needed?)
        ax.autoscale(enable=False, axis='both')

    return fig


def confidence_ellipse(ax, x, sx, y, sy, r_xy, p=0.95, mpl_label='data ellipse',
                       ellipse_kw=None, outline_alpha=False):
    """
    Plot 2D correlated data point as covariance ellipse. Assumes large
    sample size was used to estimate data point uncertainties and covariance.

    Based partly on code implemented in Knighton and Tobin (2011), see:
    https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

    """
    if ellipse_kw is None:
        ellipse_kw = {}

    df = 2
    s = stats.chi2.ppf(p, df)

    # Compile covariance matrix
    cov = r_xy * sx * sy
    cm = np.array([[sx ** 2, cov], [cov, sy ** 2]])

    if not misc.pos_def(cm):
        raise ValueError('data point covariance matrix is not positive definite')

    # Get sorted eigenvalues and eigenvectors of variance-covariance matrix.
    w, v = misc.eigsorted(cm)

    # Calculate major and minor axis lengths.
    lx, ly = 2 * np.sqrt(s * w)

    # Angle of principle eigenvector from positive x-axis.
    theta = np.degrees(np.arctan2(*v[:, 0][::-1]))

    # Ignore alpha when plotting ellipse outline
    if outline_alpha == False:
        ec = None
        fc = None
        for k, v in ellipse_kw.items():
            a = 1. if (not 'alpha' in ellipse_kw.keys()) else ellipse_kw['alpha']
            if k in ('ec', 'edgecolor'):
                v = to_rgba(v)[:3]
                ec = to_rgba(v, 1.0)
            elif k in ('fc', 'facecolor'):
                v = to_rgba(v)[:3]
                fc = to_rgba(v, a)
        if ec is not None:
            ellipse_kw['ec'] = ec
        if fc is not None:
            ellipse_kw['fc'] = fc

    ellipse = Ellipse((x, y), width=lx, height=ly, angle=theta,
                      label=mpl_label, **ellipse_kw)
    return ax.add_patch(ellipse)


#=================================
# Weighted average plots
#=================================

def wav_plot(x, xpm, xb, xbpm, rand_pm=None, sorted=False, ylim=(None, None),
             x_multiplier=1., dp_labels=None):
    """
    Plot weighted average as line and uncertainty band and data points as
    uncertainty bars.

    Parameters
    ----------
    x : np.ndarray
        Values to be averaged as 1-dimensional array.
    xpm : np.ndarray
        Symmetrical +/- errors on x.
    xb : float
        Average value (x-bar).
    xbpm : float
        Symmetrical +/- error on xb.
    rand_pm : np.ndarray, optional
        Random symmetrical +/- errors on x.
    x_multiplier :
        Use for unit conversion (e.g. set to 1000 for Ma to ka conversion)
    dp_labels : array-like, optional
        Data point labels.

    """
    assert all([x >= 0 for x in xpm])
    fig, ax = plt.subplots(**cfg.wav_fig_kw, subplot_kw=cfg.subplot_kw)
    if x_multiplier != 1.:
        x = x * x_multiplier
        xpm = xpm * x_multiplier
        xb = xb * x_multiplier
        xbpm = xbpm * x_multiplier
        if rand_pm is not None:
            rand_pm = rand_pm * x_multiplier
    oned_dp(ax, x, xpm, rand_pm=rand_pm, sorted=sorted, labels=dp_labels)
    wav_line(ax, xb, xbpm)
    apply_plot_settings(fig, plot_type='wav', ylim=ylim)
    return fig


def oned_dp(ax, x, xpm, rand_pm=None, sorted=False, labels=None):
    """
    Plot 1-d data points (e.g. ages used to compute weighted mean).
    """
    assert len(x) == len(xpm)
    n = x.shape[0]
    if rand_pm is not None:
        assert len(x) == len(rand_pm)
    if sorted:
        ind = np.argsort(x)
        x = x[ind]
        xpm = xpm[ind]
        if rand_pm is not None:
            rand_pm = rand_pm[ind]

    ind = np.arange(n)

    # set x-limits
    xpad = cfg.wav_marker_width / 2
    ax.set_xlim((-1 + xpad, n - xpad))

    if rand_pm is not None:
        # random errors
        heights = 2. * rand_pm
        bottoms = x - rand_pm
        ax.bar(ind, heights, bottom=bottoms, width=cfg.wav_marker_width,
               **cfg.wav_markers_rand_kw, label='wav marker rand')
        # systematic errors
        if not np.all(xpm >= rand_pm):
            warnings.warn('random only analytical errors were greater than '
                          'or equal to total errors for one or more data points')
        # upper segment:
        heights = xpm - rand_pm
        ax.bar(ind, heights, bottom=(x + rand_pm), width=cfg.wav_marker_width,
               **cfg.wav_markers_kw, label='wav marker full')
        # lower segment:
        ax.bar(ind, heights, bottom=(x - xpm), width=cfg.wav_marker_width,
               **cfg.wav_markers_kw, label='wav marker full')
    else:
        # full errors
        heights = 2. * xpm
        bottoms = x - xpm
        ax.bar(ind, heights, bottom=bottoms, width=cfg.wav_marker_width,
               **cfg.wav_markers_kw, label='wav marker full')

    # set y-axis limits
    top = np.max(x + xpm)
    bottom = np.min(x - xpm)
    yspread = top - bottom
    ypad = 0.33 * yspread
    ax.set_ylim((bottom - ypad, top + ypad))

    # add labels
    if labels is not None:
        ax.set_xticks(ind)
        # Set ticks labels for x-axis
        ax.set_xticklabels(labels, rotation=45)
    else:
        # remove x-ticks/labels
        ax.xaxis.set_major_locator(plt.NullLocator())


def wav_line(ax, xb, xbpm, env=True):
    """ Add weighted average line and error envelope to plot of 1-D data points.
    """
    # plot weighted mean
    ax.axhline(y=xb, **cfg.wav_line_kw, label='wave line')
    # plot error envelope about weighted mean
    if env:
        y_low = xb - xbpm
        y_hi = xb + xbpm
        ax.axhspan(y_low, y_hi, **cfg.wav_env_kw)

        # ax.axhspan(y_low, y_hi, **cfg.wav_env_kw, label='wav env')
        # xx = ax.get_xlim()
        # ax.fill_between(xx, y_hi, y_low, **cfg.wav_env_kw)

#=================================
# Plot settings and formatting
#=================================

def apply_plot_settings(fig, plot_type='isochron', diagram=None,
        xlim=(None, None), ylim=(None, None), axis_labels=(None, None),
        norm_isotope='204Pb'):
    """
    Set axis labels and limits. Apply label, tick, grid formatting settings as
    defined in pysoplot.cfg.

    Parameters
    -----------
    fig : matplotlib.pyplot.Figure
        Figure to apply settings to.
    plot_type : {'isochron', 'intercept', 'wav', 'hist'}, optional
        Plot type.

    """
    assert plot_type in ('isochron', 'intercept', 'wav', 'hist')

    # ax = fig.axes[0]
    for ax in fig.axes:
        # set axis limits
        set_axis_limits(ax, xmin=xlim[0], xmax=xlim[1], ymin=ylim[0], ymax=ylim[1])

        # set axis labels
        if diagram is not None:
            set_axis_labels(ax, diagram=diagram, norm_isotope=norm_isotope,
                        axis_labels=axis_labels)

        # set spine params
        ax.spines[:].set_zorder(100)

        # Set major tick properties.
        # TODO: tick zorder still not working properly due a mpl bug.
        if plot_type == 'wav':
            ax.tick_params(axis='y', which='major', **cfg.major_ticks_kw)
            # Set minor tick properties.
            if cfg.show_minor_ticks:
                ax.minorticks_on()
                ax.tick_params(axis='y', which='minor', **cfg.minor_ticks_kw)
        else:
            ax.tick_params(axis='both', which='major', **cfg.major_ticks_kw)
            # Set minor tick properties.
            if cfg.show_minor_ticks:
                ax.minorticks_on()
                ax.tick_params(axis='both', which='minor', **cfg.minor_ticks_kw)

        # Set tick label formats
        if len(ax.get_xticks()) > 0 and plot_type != 'wav':
            xtick_formatter = tick_label_format(ax, ax.get_xticks(),
                    cfg.sci_limits, cfg.comma_sep_thousands, axis='x')
            ax.get_xaxis().set_major_formatter(xtick_formatter)

        if len(ax.get_yticks()) > 0:
            ytick_formatter = tick_label_format(ax, ax.get_yticks(),
                    cfg.sci_limits, cfg.comma_sep_thousands, axis='y')
            ax.get_yaxis().set_major_formatter(ytick_formatter)

        #  Major grid lines
        if cfg.show_major_gridlines:
            if plot_type == 'wav':
                ax.grid(True, which='major', axis='y', **cfg.gridlines_kw)
            else:
                ax.grid(True, which='major', axis='both', **cfg.gridlines_kw)
            ax.set_axisbelow(True)

        # Minor grid lines
        if cfg.show_minor_gridlines:
            if plot_type == 'wav':
                ax.grid(True, which='minor', axis='y', **cfg.gridlines_kw)
            else:
                ax.grid(True, which='minor', axis='both', **cfg.gridlines_kw)
            ax.set_axisbelow(True)

        # Hide spines
        if cfg.hide_top_spine:
            ax.spines['right'].set_visible(False)
            # hide ticks as well
            ax.tick_params(top=False)

        if cfg.hide_right_spine:
            ax.spines['top'].set_visible(False)
            # hide ticks as well
            ax.tick_params(right=False)

        # Set axis and tick label font sizes
        # note: this axis labels now done by set_axis_labels() function
        ax.xaxis.set_tick_params(labelsize=cfg.tick_label_size)
        ax.yaxis.set_tick_params(labelsize=cfg.tick_label_size)

        # Set axis exponent multiplier fontsize
        t = ax.yaxis.get_offset_text()
        t.set_size(cfg.exp_font_size)
        t = ax.xaxis.get_offset_text()
        t.set_size(cfg.exp_font_size)

    # tight layout
    # note - calling this if tight_layout is already True, may set it to False?
    # fig.tight_layout()


def set_axis_limits(ax, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Set axis limits if user specified limits are supplied.
    """
    xmin0, xmax0 = ax.get_xlim()
    ymin0, ymax0 = ax.get_ylim()

    # Set axis limits to user defined values.
    if xmin is not None:
        if xmax is not None:
            if xmin >= xmax:
                raise ValueError('xmin cannot be greater than xmax')
        elif xmin > xmax0:
            raise ValueError('xmin cannot be greater than xmax')
        ax.set_xlim(left=xmin)

    if xmax is not None:
        if xmin is not None:
            if xmin >= xmax:
                raise ValueError('xmin cannot be greater than xmax')
        elif xmax < xmin0:
            raise ValueError('xmin cannot be greater than xmax')
        ax.set_xlim(right=xmax)

    if ymin is not None:
        if ymax is not None:
            if ymin >= ymax:
                raise ValueError('ymin cannot be greater than ymax')
        elif ymin > ymax0:
            raise ValueError('ymin cannot be greater than ymax')
        ax.set_ylim(bottom=ymin)

    if ymax is not None:
        if ymin is not None:
            if ymin >= ymax:
                raise ValueError('ymin cannot be greater than ymax')
        elif ymax < ymin0:
            raise ValueError('ymin cannot be greater than ymax')
        ax.set_ylim(top=ymax)


def set_axis_labels(ax, diagram='tw', norm_isotope='204Pb',
                    axis_labels=(None, None)):
    """
    Set axis x, y axis labels for default diagram types, or use inputted
    axis labels.
    """
    xlabel, ylabel = None, None

    if diagram == 'Pb206*':
        ylabel = "$^{206}$Pb/$^{238}$U age (Ma)"

    elif diagram == 'Pb207*':
        ylabel = "$^{207}$Pb/$^{235}$U age (Ma)"

    elif diagram == 'cor-207Pb':
        ylabel = "$^{207}$Pb-corrected age (Ma)"

    elif diagram == "tw":
        xlabel = "$^{238}$U/$^{206}$Pb"
        ylabel = "$^{207}$Pb/$^{206}$Pb"

    elif diagram == 'wc':
        xlabel = "$^{207}$Pb/$^{235}$U"
        ylabel = "$^{206}$Pb/$^{238}$U"

    elif diagram == 'iso-206Pb' and norm_isotope == '208Pb':
        xlabel = "$^{238}$U/$^{208}$Pb"
        ylabel = "$^{206}$Pb/$^{208}$Pb"

    elif diagram == 'iso-206Pb' and norm_isotope == '204Pb':
        xlabel = "$^{238}$U/$^{204}$Pb"
        ylabel = "$^{206}$Pb/$^{204}$Pb"

    elif diagram == 'iso-207Pb' and norm_isotope == '208Pb':
        xlabel = "$^{235}$U/$^{208}$Pb"
        ylabel = "$^{207}$Pb/$^{208}$Pb"

    elif diagram == 'iso-207Pb' and norm_isotope == '204Pb':
        xlabel = "$^{235}$U/$^{204}$Pb"
        ylabel = "$^{207}$Pb/$^{204}$Pb"

    if axis_labels[0] is not None:
        xlabel = axis_labels[0]
    if axis_labels[1] is not None:
        ylabel = axis_labels[1]

    if xlabel is not None:
        ax.set_xlabel(xlabel, **cfg.axis_labels_kw)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **cfg.axis_labels_kw)


def tick_label_format(ax, ticks, sci_limits, comma_sep_thousands, axis='x'):
    """
    Format plot tick labels.

    Parameters
    -------------
    ticks : array-like
        List of ticks.
    sci_limits : array-like
        Axis scientific notation exponent limits as [lower, upper]. If the
        maximum axis value is greater than 10 ^ upper then scientific notation
        is used for tick labels. An equivalent test is applied to the lower
        limit.
    comma_sep_thousands : bool
        True if use comma to separate each 10^3 increment, e.g. "10,450".

    Returns
    -------
    tick_formatter :  matplotlib.ticker.Formatter
    """
    # Use scientiffic notation if axis values outside power limits.
    if misc.get_exponent(min(ticks)) < sci_limits[0] or \
            misc.get_exponent(max(ticks)) > sci_limits[1]:
        # TODO: consider improving this following e.g.:
        # <https://stackoverflow.com/questions/31517156/adjust-exponent-text-
        # after-setting-scientific-limits-on-matplotlib-axis>
        tick_formatter = ticker.ScalarFormatter(useMathText=True)
        tick_formatter.set_powerlimits((0, 0))

        if axis == 'x':
            ax.xaxis.get_offset_text().set_fontsize(
                ax.get_xticklabels()[0].get_fontsize() + 1)
        else:
            ax.yaxis.get_offset_text().set_fontsize(
                ax.get_yticklabels()[0].get_fontsize() + 1)

        return tick_formatter

    elif comma_sep_thousands:
        # Find number of decimal digits to show.
        ndp = 0
        for z in ticks:
            # mpl should give nice values, but just in case of rounding issues.
            z = misc.round_down(float(z), 12)
            nj = misc.num_dec_places(z)
            if nj > ndp:
                ndp = nj
        # format with comma sep thousands
        tick_formatter = ticker.StrMethodFormatter('{{x:,.{}f}}'.format(ndp))
        return tick_formatter

    else:
        # Just use standard formatting without sci notation.
        tick_formatter = ticker.ScalarFormatter()
        tick_formatter.set_powerlimits((-np.inf, np.inf))
        return tick_formatter


#=======================================
# Plot intersection and bbox functions
#=======================================

def box_line_intersection(xmin, xmax, ymin, ymax, a, b):
    """
    Find intercpet points between line defined by slope = a, int = b, and
    axis window limits.
    """
    p = np.full((4, 2), np.nan)
    bbox_lines = [[(xmin, ymax), (xmax, ymax)],
                  [(xmin, ymin), (xmax, ymin)],
                  [(xmin, ymin), (xmin, ymax)],
                  [(xmax, ymin), (xmax, ymax)]]
    for i, r in enumerate(bbox_lines):
        p[i] = line_intersection(a, b, r)
    p = p[np.any(~np.isnan(p), axis=1)]
    # can return repeated intercept points if they are at the box corners !!
    if p.shape[0] > 2:
        p = p[:2]
    return p


def line_intersection(a1, b1, L2):
    """
    Return intersection point between line defined by slope, b1, and y-int,
    a1, and a line segment defined by end points as L2 = ((x1, y1), (x2, y2)).
    """
    assert np.array(L2).shape == (2, 2)
    if L2[1][0] - L2[0][0] == 0:  # if b2 = inf.
        p = np.empty(2)
        p[0] = L2[1][0]
        p[1] = a1 + b1 * p[0]

    else:
        b2 = (L2[1][1] - L2[0][1]) / (L2[1][0] - L2[0][0])
        a2 = L2[0][1] - b2 * L2[0][0]

        A = np.array([[-a1],
                      [-a2]])
        B = np.array([[b1, -1.],
                      [b2, -1.]])

        # calculate intersection point
        p = np.dot(np.linalg.inv(B), A)
        p = p.flatten()

    # check if intersection within end points, otherwise return nan
    xmin = min(L2[0][0], L2[1][0])
    xmax = max(L2[0][0], L2[1][0])
    ymin = min(L2[0][1], L2[1][1])
    ymax = max(L2[0][1], L2[1][1])

    p = p if in_box(p, xmin, xmax, ymin, ymax) else np.nan
    return p


def in_box(p, xmin, xmax, ymin, ymax):
    """
    Check if point p = (x, y) is within bounding rectangle
    defined by x, y limits, allowing some tolerance for floating points numbers.
    """
    assert not xmin > xmax and not ymin > ymax
    if ((p[0] > xmin or np.isclose(p[0], xmin)) and
        (p[0] < xmax or np.isclose(p[0], xmax)) and
        (p[1] > ymin or np.isclose(p[1], ymin)) and
        (p[1] < ymax or np.isclose(p[1], ymax))):
        return True

    return False
