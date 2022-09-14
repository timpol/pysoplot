"""
Settings, constants and options used package-wide.

In future these settings may be stored in a configuration file instead.

References
----------
.. [Cheng2013]
    Cheng, H., Lawrence Edwards, R., Shen, C.-C., Polyak, V.J., Asmerom, Y.,
    Woodhead, J., Hellstrom, J., Wang, Y., Kong, X., Spötl, C., Wang, X.,
    Calvin Alexander, E., 2013. Improvements in 230Th dating, 230Th and
    234U half-life values, and U–Th isotopic measurements by multi-collector
    inductively coupled plasma mass spectrometry. Earth and Planetary
    Science Letters 371–372, 82–91. https://doi.org/10.1016/j.epsl.2013.04.006
.. [Hiess2012]
    Hiess, J., Condon, D.J., McLean, N., Science, S.N., 2012. 238U/235U
    systematics in terrestrial uranium-bearing minerals. Science.
    https://doi.org/10.1126/science.1215507
.. [Jaffey1971]
    Jaffey, A.H., Flynn, K.F., Glendeni, L.E., Bentley, W.C., Essling, A.M.,
    1971. Precision measurement of half-lives and specific activities of
    U-235 and U-238. Physical Review C 4, 1889–1906.
.. [Jerome2020]
    Jerome, S., Bobin, C., Cassette, P., Dersch, R., Galea, R., Liu, H., Honig,
    A., Keightley, J., Kossert, K., Liang, J., Marouli, M., Michotte, C.,
    Pommé, S., Röttger, S., Williams, R., Zhang, M., 2020. Half-life
    determination and comparison of activity standards of 231Pa. Applied
    Radiation and Isotopes 155, 108837.
    https://doi.org/10.1016/j.apradiso.2019.108837
.. [Ludwig1977]
    Ludwig, K.R., 1977. Effect of initial radioactive-daughter disequilibrium on
    U-Pb isotope apparent ages of young minerals. Journal of Research of
    the US Geological Survey 5, 663–667.
.. [Robert1969]
    Robert, J., Miranda, C.F., Muxart, R., 1969. Mesure de la période du
    protactinium 231 par microcalorimétrie. Radiochimica Acta 11, 104–108.

"""

import numpy as np


#==============================================================================
# Physical Constants
# All decay constants are in Ma^-1.
# Computations are performed in Ma as this tends to promote numerical stability
# over using a.
#==============================================================================

# Decay constants in Ma^-1
lam238 = np.log(2) / (4.4683E09 * 1E-6)    # Jaffey et al., (1971)
lam235 = np.log(2) / (7.0381E08 * 1E-6)    # Jaffey et al., (1971)
lam234 = np.log(2) / (245620 * 1E-6)       # Cheng et al., (2013)
lam231 = np.log(2) / (3.2765e4 * 1E-6)     # Robert et al., (1969)
# lam231 = np.log(2) / (3.2570e4 * 1E-6)     # Jerome et al., (2020)
lam230 = np.log(2) / (75584 * 1E-6)        # Cheng et al., (2013)
lam227 = np.log(2) / (22. * 1E-6)
lam226 = np.log(2) / (1600. * 1E-6)
lam210 = np.log(2) / (22. * 1E-6)

# Decay constant errors
# 1 sigma absolute in Ma^-1
s238 = lam238 ** 2 / np.log(2) * (0.0024E09 * 1E-6)   # Jaffey et al., (1971)
s235 = lam235 ** 2 / np.log(2) * (0.0048E08 * 1E-6)   # Jaffey et al., (1971)
s234 = lam234 ** 2 / np.log(2) * (130. * 1E-6)        # Cheng et al., (2013)
s231 = lam231 ** 2 / np.log(2) * (110. * 1E-6)        # Robert et al., (1969)
# s231 = lam231 ** 2 / np.log(2) * (130. * 1E-6 )       # Jerome et al., (2020)
s230 = lam230 ** 2 / np.log(2) * (55. * 1E-6)         # Cheng et al., (2013)
s227 = 0.
s226 = lam226 ** 2 / np.log(2) * (7. * 1E-6)
s210 = 0.

# Decay constant error correlations
# Note: there are probably no practical cases where these error correlations
# are significant ???
cor_238_234 = 0.95194         # estimated from Cheng et al. (2013)
cor_238_230 = 0.71046         # estimated from Cheng et al. (2013)
cor_234_230 = 0.67631         # estimated from Cheng et al. (2013)

# Natural uranium 238U/235U ratio
U = 137.818                               # Hiess, 2012 (approx. 'bulk Earth' value)
sU = 0.0225                               # Hiess, 2012

# Equilbrium activity ratio values
A48_eq = 1.0
A08_eq = 1.0
A68_eq = 1.0
A15_eq = 1.0


#==============================================================================
# Computation parameters
#
# secular_eq : bool
#   !!! Experimental feature !!!
#   If False, U-series equations derived following the approach of Ludwig (1977)
#   are implemented rather than the standard equations based on the assumption of
#   no decay of 238U. Note, that Cheng et al. (2013) decay constants
#   are computed assuming secular equilibrium so these should be re-computed
#   using their measured eq. 234U/238U and 230Th/238U values if implementing
#   this option. I.e. lam234 = 2.8221577 [Ma^-1] and lam230 = 9.1705 [Ma^-1].
#   TODO: double check these values
#
# cfg.rng : Generator object
#   Used across all modules.
#   Allows for reproducible Monte Carlo simulations results.
#   See: https://numpy.org/doc/stable/reference/random/generator.html
#
#==============================================================================\
secular_eq = True
rng = np.random.default_rng()


#==============================================================================
# Statistical parameters
# h : float
#   spine cut-off value (see Powell et al., 2020)
# mswd_ci_thresholds: array-like
#     MSWD one-sided confidence interval thresholds for classical regression
#     fits. First element is the model 1 –> 1x threshold, second element is the
#     model 1x –> 2/3 threshold.
# mswd_wav_ci_thresholds : array-like
#     MSWD one-sided confidence interval thresholds for classical wtd.averages.
#     First element is analytical –> analytical + observed scatter threshold.
#     Second element is not yet used. Equivalent Isoplot default for first
#     element is 0.70?
#==============================================================================
h = 1.4
mswd_ci_thresholds = (0.85, 0.95)
mswd_wav_ci_thresholds = (0.85, 0.95)


#===========================================
# Isoplot constants - for testing purposes
#===========================================
IsoLam238 = 1.55125E-10 * (10 ** 6)     # in Ma^-1
IsoLam234 = 2.8338E-6 * (10 ** 6)
IsoLam230 = 9.19525E-6 * (10 ** 6)
IsoLam235 = 9.8485E-10 * (10 ** 6)
IsoLam231 = 2.13276E-5 * (10 ** 6)
IsoS238 = 0.107 * IsoLam238 / 200.
IsoS234 = 0.2 * IsoLam234 / 200.
IsoS230 = 0.3 * IsoLam230 / 200.
IsoS235 = 0.136 * IsoLam235 / 200.
IsoU = 137.88


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

axis_labels_kw = {
    'color': 'k',
    'fontsize': 10,
}

conc_age_ellipse_kw = {
    'alpha': 1.0,
    'edgecolor': 'black',
    'facecolor': "white",
    'linewidth': 0.5,
    'zorder': 20
}

conc_env_kw = {
    'alpha': 1.0,
    'edgecolor': 'none',
    'facecolor': '#FFFFC0',
    'linestyle': '--',
    'linewidth': 0.0,
    'zorder': 18
}

conc_env_line_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': '--',
    'linewidth': 0.80,
    'zorder': 18
}

conc_intercept_ellipse_kw = {
    'alpha': 0.60,
    'edgecolor': 'black',
    'facecolor': '#C5F7C5',
    'linewidth': 1.0,
    'zorder': 25
}

conc_intercept_markers_kw = {
    'alpha': 0.5,
    'markeredgecolor': 'none',
    'markerfacecolor': 'black',
    'linewidth': 0,
    'marker': ',',
    'markersize': 4,
    'zorder': 25,
}

conc_line_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': '-',
    'linewidth': 0.80,
    'zorder': 19,
}

conc_markers_kw = {
    'alpha': 1.0,
    'linewidth': 0,
    'marker': 'o',
    'markeredgecolor': 'black',
    'markerfacecolor': 'white',
    'markersize': 4,
    'zorder': 20,
}

# Be careful changing the annotation_clip and clip_on settings!!!
# Note: Some settings will be filtered out if the individualised_labels
# routine is called.
conc_text_kw = {
    'annotation_clip': False,
    'clip_on': True,
    'color': 'black',
    'fontsize': 8,
    'horizontalalignment': 'left',
    'textcoords': 'offset points',
    'verticalalignment': 'center',
    'xytext': (3, 3),
    'zorder': 21,
}

dp_ellipse_kw = {
    'alpha': 0.8,
    'edgecolor': 'black',
    'facecolor': '#1FB714',
    'linewidth': 0.50,
    'zorder': 30
}

dp_label_kw = {
    'color': 'black',
    'fontsize': 8,
    'horizontalalignment': 'center',
    'textcoords': 'offset points',
    'verticalalignment': 'center',
    'xytext': (10, 0),
    'zorder': 30
}

fig_kw = {
    'dpi': 300,
    'facecolor': 'whitesmoke',
    'figsize': (4.72, 4.012),
    'tight_layout': True,
}

gridlines_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': ':',
    'linewidth': 0.5,
}

hist_bars_kw = {
    'alpha': 0.75,
    'edgecolor': 'red',
    'facecolor': 'green',
    'histtype': 'step',
    'linewidth': 0.75,
}

hist_fig_kw = {
    'dpi': 300,
    'facecolor': 'whitesmoke',
    'figsize': (4.72, 4.012),
    'tight_layout': True,
}

major_ticks_kw = {
    'color': 'black',
    'direction': 'in',
    'length': 4,
    'width': 0.5,
}

minor_ticks_kw = {
    'color': 'black',
    'direction': 'in',
    'length': 2,
    'width': 0.5,
}

pb76_line_kw = {
    'alpha': 0.5,
    'color': 'blue',
    'linestyle': '--',
    'linewidth': 1.,
    'zorder': 10,

}

renv_kw = {
    'alpha': 0.30,
    'edgecolor': 'none',
    'facecolor': 'blue',
    'linewidth': 0.,
    'zorder': 9,
}

renv_line_kw = {
    'alpha': 1.0,
    'color': 'blue',
    'linewidth': 0.,
    'zorder': 9,
}

rline_kw = {
    'alpha': 1.0,
    'color': 'blue',
    'linestyle': '-',
    'linewidth': 0.80,
    'zorder': 10,
}

scatter_markers_kw = {
    "alpha": 0.3,
    "markeredgecolor": "none",
    "linewidth": 0,
    "markerfacecolor": "black",
    "marker": ",",
    "markersize": 4,
    "zorder": 1,
}

spine_kw = {
    'color': 'k',
    'linewidth': 0.8,
}

# subplot_kw passed to matplotlib.pyplot.subplots()
subplot_kw = {
    'facecolor': 'white'
}

wav_env_kw = {
    'alpha': 0.8,
    'facecolor': 'palegreen',
    'edgecolor': 'none',
    'linestyle': '--',
    'linewidth': 0.0,
    'zorder': 9,
}

wav_fig_kw = {
    'dpi': 300,
    'facecolor': 'whitesmoke',
    'figsize': (4.72, 4.012),
    'tight_layout': True,
}

wav_line_kw = {
    'alpha': 1.0,
    'color': 'black',
    'linestyle': '-',
    'linewidth': 1.,
    'zorder': 10,
}


wav_markers_kw = {
    'alpha': 1.0,
    'color': 'lightblue',
    'zorder': 31
}

wav_markers_rand_kw = {
    'alpha': 1.0,
    'color': 'blue',
    'zorder': 30
}
