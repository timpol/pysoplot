"""
Package-wide settings and constants.

"""

import numpy as np


#==============================================================================
# Physical Constants
#==============================================================================
lam238 = 0.000155125479614  #: 238U decay constant [Ma^-1]. Default [JAFFEY1971]_.
lam235 = 0.000984849860843  #: 235U decay constant [Ma^-1]. Default [JAFFEY1971]_.
lam234 = 2.822030700105632  #: 234U decay constant [Ma^-1]. Default [CHENG2013]_.
lam231 = 21.15511004303205  #: 231Pa decay constant [Ma^-1]. Default [ROBERT1969]_.
lam230 = 9.170554357535263  #: 230Th decay constant [Ma^-1]. Default [CHENG2013]_.
lam227 = 31506.69
lam226 = 433.2169878499658  #: 226Ra decay constant [Ma^-1]
lam210 = 31506.69
lam232 = 4.947517348750502e-05  #: 232Th decay constant [Ma^-1] (default [HOLDEN1990]_)

s238 = 8.332053601458737e-08  #: 238U decay constant 1 sigma uncertainty [Ma^-1]. Default [JAFFEY1971]_.
s235 = 6.716698160081028e-07  #: 235U decay constant 1 sigma uncertainty [Ma^-1]. Default [JAFFEY1971]_.
s234 = 0.001493624261109568   #: 234U decay constant 1 sigma uncertainty [Ma^-1]. Default [CHENG2013]_.
s231 = 0.07102280191465059    #: 231Pa decay constant 1 sigma uncertainty [Ma^-1]. Default [ROBERT1969]_.
s230 = 0.006673111897550267   #: 230Th decay constant 1 sigma uncertainty [Ma^-1]. Default [CHENG2013]_.
s227 = 0.0
s226 = 1.8953243218436007     #: 226Ra decay constant 1 sigma uncertainty [Ma^-1]
s210 = 0.0
s232 = 3.531422306388949e-07  #: 232Th decay constant 1 sigma uncertainty [Ma^-1]. Default [HOLDEN1990]_.

# Decay constant error correlations
# Note: there are probably no practical cases where these error correlations
# are significant?
cor_238_234 = 0.95194         # estimated from Cheng et al. (2013)
cor_238_230 = 0.71046         # estimated from Cheng et al. (2013)
cor_234_230 = 0.67631         # estimated from Cheng et al. (2013)

# Natural uranium 238U/235U ratio
U = 137.818                   #: Modern natural 238U/235U ratio. Default [HIESS2012]_.
sU = 0.0225                   #: Modern natural 238U/235U ratio 1 sigma uncertainty. Default [HIESS2012]_.

# Equilibrium activity ratio values
a234_238_eq = 1.0
a230_238_eq = 1.0
a226_238_eq = 1.0
a231_235_eq = 1.0

a234_238_eq_1s = 0.
a230_238_eq_1s = 0.
a226_238_eq_1s = 0.
a231_235_eq_1s = 0.


#==============================================================================
# Computation parameters
#==============================================================================

#: bool: If False, U-series equations based on Bateman (1910) are implemented
#:    rather than the standard equations that rely on the assumption of negligible
#:   decay of 238U. Note, that the default Cheng et al. (2013) decay constants
#:   are computed assuming secular equilibrium so these should be re-computed
#:   using their measured eq. 234U/238U and 230Th/238U values if implementing
#:   this option. I.e., lam234 = 2.8221577 [Ma^-1] and lam230 = 9.1705 [Ma^-1].
#:   This is an *experimental* feature.
# TODO: double check the above calcs
secular_eq = True

#: :class:`numpy.Generator` : Random number generator used across all modules.
#:  Allows for reproducible Monte Carlo simulations results.
rng = np.random.default_rng()


#==============================================================================
# Statistical parameters
#==============================================================================

#: float : Spine h value (see [Powell2020]_).
h = 1.4

#: MSWD one-sided confidence interval thresholds for classical
#: regression fits. First element is the model 1 –> 1x threshold, second element
#: is the model 1x –> 2/3 threshold.
mswd_ci_thresholds = [0.85, 0.95]

#: MSWD one-sided confidence interval thresholds for classical wtd. averages.
#: First element is analytical –> analytical + observed scatter threshold.
#: Second element is not yet used. Equivalent Isoplot default for first
#: element is 0.70.
mswd_wav_ci_thresholds = [0.85, 0.95]


#=================================
# General plot options
#=================================
file_ext = '.pdf'
tight_layout = True
sort_ages = False                       # sort ages on wtd. average plots

comma_sep_thousands = False
exp_font_size = 9                       # axis exponent multiplier label
hide_right_spine = False
hide_top_spine = False
wav_marker_width = 0.6
sci_limits = [-3, 4]                    # sci notation used for axis values outside these exponent limits
show_major_gridlines = False
show_minor_gridlines = False
show_minor_ticks = False
tick_label_size = 9
wav_major_gridlines = True
wav_minor_gridlines = False


#==============================================================================
# Concordia plot settings
#
# every_second_threshold : int
#    Label every second marker if more than this number of concordia markers
#    within axis limits.
# label_markers : bool
#    Label markers if True.
# prefix_in_label : bool
#    Include age prefix (Ma or ka) in marker labels.
# prefix : str
#    Marker age label prefix. One of 'Ma' or 'ka'.
# individualised_labels: bool
#    Use individualised labelling routine if True.
# offset_factor : float
#    Label offset factor (multiple of text box height which will be
#    proportional to font size).
# rotate_conc_labels : bool
#    Rotate age marker labels according to perpendicular_rotation value.
# perpendicular_rotation : bool
#    If True, rotate perpendicular to concordia slope rather than parallel.
# remov_overlaps : bool
#    If True, remove all labels for markers older than the first overlapping
#    one.
#==============================================================================
conc_age_bounds = [0.010, 4600]                  # eq. age bounds (Ma)
# Diseq. concordia age upper and lower bounds (Ma) if
#   0) inital A08 and ar48
#   1) present ar48 only
#   2) present A08
diseq_conc_age_bounds = [[1e-03, 100.,],
                         [1e-03, 2.5],
                         [1e-03, 1.5]]

plot_age_markers = True
label_markers = True
ellipse_label_va = 'bottom'
ellipse_label_ha = 'left'
prefix_in_label = True
every_second_threshold = 8
individualised_labels = True
offset_factor = 0.8
rotate_conc_labels = True
perpendicular_rotation = False
remove_overlaps = False


#=================================
# Plot format settings
#=================================

#: Axis labels key-word arguments.
axis_labels_kw = {
    'color': 'black',
    'fontsize': 10,
}

#: Concordia age ellipse marker key-word arguments.
conc_age_ellipse_kw = {
    'alpha': 1.0,
    'edgecolor': 'black',
    'facecolor': "white",
    'linewidth': 0.5,
    'zorder': 10
}
#: Concordia uncertainty envelope fill key-word arguments.
conc_env_kw = {
    'alpha': 1.0,
    'edgecolor': 'none',
    'facecolor': 'white',
    'linestyle': '-',
    'linewidth': 0.0,
    'zorder': 8
}

#: Concordia envelope line key-word arguments.
conc_env_line_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': '--',
    'linewidth': 1.0,
    'zorder': 8
}

#: Concordia intercept ellipse key-word arguments.
conc_intercept_ellipse_kw = {
    'alpha': 0.60,
    'edgecolor': 'black',
    'facecolor': 'lightgrey',
    'linewidth': 1.0,
    'zorder': 30
}

#: Concordia intercept markers key-word arguments.
conc_intercept_markers_kw = {
    'alpha': 0.5,
    'markeredgecolor': 'none',
    'markerfacecolor': 'black',
    'linewidth': 0,
    'marker': ',',
    'markersize': 4,
    'zorder': 30,
}

#: Concordia line key-word arguments.
conc_line_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': '-',
    'linewidth': 0.80,
    'zorder': 9,
}

#: Concordia age markers key-word arguments.
conc_markers_kw = {
    'alpha': 1.0,
    'linewidth': 0,
    'marker': 'o',
    'markeredgecolor': 'black',
    'markerfacecolor': 'white',
    'markersize': 4,
    'zorder': 10,
}

#: Concordia marker labels key-word arguments.
#: Caution: Be careful changing the annotation_clip and clip_on settings.
conc_text_kw = {
    'annotation_clip': False,
    'clip_on': True,
    'color': 'black',
    'fontsize': 8,
    'horizontalalignment': 'left',
    'textcoords': 'offset points',
    'verticalalignment': 'center',
    'xytext': (3, 3),
    'zorder': 11,
}

#: Data point confidence ellipse key-word arguments.
dp_ellipse_kw = {
    'alpha': 1.0,
    'edgecolor': 'black',
    'facecolor': 'white',
    'linewidth': 0.80,
    'zorder': 40
}

#: Data point confidence ellipse key-word arguments.
dp_label_kw = {
    'color': 'black',
    'fontsize': 8,
    'horizontalalignment': 'center',
    'textcoords': 'offset points',
    'verticalalignment': 'center',
    'xytext': (10, 0),
    'zorder': 40
}

#: Figure key-word arguments.
fig_kw = {
    'dpi': 300,
    'facecolor': 'white',
    'figsize': (4.72, 4.012),
    'tight_layout': True,
}

#: Figure gridline key-word arguments.
gridlines_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': ':',
    'linewidth': 0.5,
}

#: Histogram bar key-word arguments.
hist_bars_kw = {
    'alpha': 0.75,
    'edgecolor': 'red',
    'facecolor': 'green',
    'histtype': 'step',
    'linewidth': 0.75,
}

#: Histogram figure key-word arguments.
hist_fig_kw = {
    'dpi': 300,
    'facecolor': 'white',
    'figsize': (4.72, 4.012),
    'tight_layout': True,
}

#: Major axis tick key-word arguments.
major_ticks_kw = {
    'color': 'black',
    'direction': 'in',
    'length': 4,
    'width': 0.5,
    'zorder': 100,
}

#: Minor axis tick key-word arguments.
minor_ticks_kw = {
    'color': 'black',
    'direction': 'in',
    'length': 2,
    'width': 0.5,
    'zorder': 100,
}

#: Common 207Pb/206Pb projection line key-word arguments.
pb76_line_kw = {
    'alpha': 0.5,
    'color': 'red',
    'linestyle': '--',
    'linewidth': 1.,
    'zorder': 10,

}

#: Regression envelope key-word arguments.
renv_kw = {
    'alpha': 0.30,
    'edgecolor': 'none',
    'facecolor': 'none',
    'linewidth': 0.,
    'zorder': 20,
}

#: Regression envelope line key-word arguments.
renv_line_kw = {
    'alpha': 1.0,
    'color': 'red',
    'linewidth': 0.8,
    'linestyle': '--',
    'zorder': 20,
}

#: Regression line key-word arguments.
rline_kw = {
    'alpha': 1.0,
    'color': 'red',
    'linestyle': '-',
    'linewidth': 0.75,
    'zorder': 21,
}

#: Scatter plot marker key-word arguments.
scatter_markers_kw = {
    "alpha": 0.3,
    "markeredgecolor": "none",
    "linewidth": 0,
    "markerfacecolor": "black",
    "marker": ",",
    "markersize": 4,
    "zorder": 1,
}

#: Figure axis spine key-word arguments. Spines are the axis lines.
spine_kw = {
    'color': 'red',
    'linewidth': 0.8,
    'zorder': 100
}

#: subplot_kw passed to matplotlib.pyplot.subplots().
subplot_kw = {
    'facecolor': 'white'
}

#: Weighted average envelope key-word arguments.
wav_env_kw = {
    'alpha': 0.8,
    'facecolor': 'lightgrey',
    'edgecolor': 'none',
    'linestyle': '-',
    'linewidth': 0.0,
    'zorder': 19,
}

#: Weighted average figure key-word arguments.
wav_fig_kw = {
    'dpi': 300,
    'facecolor': 'white',
    'figsize': (4.72, 4.012),
    'tight_layout': True,
}

#: Weighted average line key-word arguments.
wav_line_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': '-',
    'linewidth': 1.,
    'zorder': 20,
}

#: Weighted average data point marker key-word arguments.
wav_markers_kw = {
    'alpha': 1.0,
    'color': 'white',
    'edgecolor': 'blue',
    'linewidth': 1.,
    'zorder': 41

}

#: Weighted average random only uncertainty data point marker key-word arguments.
wav_markers_rand_kw = {
    'alpha': 1.0,
    'color': 'blue',
    'edgecolor': 'blue',
    'linewidth': 1.,
    'zorder': 40
}
