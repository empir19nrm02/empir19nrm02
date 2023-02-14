import numpy as np
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sigfig import round

from .draw_values import sumMC, sumMCV
from .help import label_font_size

__all__ = ['plotSelectedSPD','plotSPDs','plotYxy', 'plotHist', 'plotCorrMatrixSmall',
           'plotHistScales','plotHistScalesWl','plotHistScalesValue','plotHistGauss','confidence_ellipse','aspectratio_to_one','plot_2D',
           'array2analyse','analyse_stat','get_data_step','seaborn_plot_basedata', 'seaborn_plot_result']

def plotSPDs(SPDs, title='SDs', fileName=None,fontsize=None, normalize = True, log=False):
    for i in range(1,SPDs.shape[0]):
        if normalize:
            pyplot.plot(SPDs[0, :], SPDs[i, :]/np.max(SPDs[i, :]))
        else:
            pyplot.plot(SPDs[0, :], SPDs[i, :])
    pyplot.xlabel('$\lambda$ / nm', fontsize=fontsize)
    pyplot.ylabel('spectral distribution / A.U.', fontsize=fontsize)
    pyplot.title(title)
    if log:
        pyplot.yscale('log')
    if fileName != None: pyplot.savefig(fileName)

def plotSelectedSPD(SPD, iNumber, title='Selected SPD', fileName=None,fontsize=None):
    pyplot.plot(SPD[0, :], SPD[iNumber, :])
    pyplot.xlabel('$\lambda$ / nm', fontsize=fontsize)
    pyplot.ylabel('spectral distribution / A.U.', fontsize=fontsize)
    pyplot.title(title)
    if fileName != None: pyplot.savefig(fileName)


def plotYxy(Yxy, title='xy-plot', fileName=None, fontsize=None):
    pyplot.plot(Yxy[:, 1], Yxy[:, 2], '*')
    pyplot.xlabel('x', fontsize=fontsize)
    pyplot.ylabel('y', fontsize=fontsize)
    pyplot.title(title)
    if fileName != None: pyplot.savefig(fileName)


def plotHist(data, xLabel='x', yLabel='y', title='title', fileName=None, bins=50, fontsize=None):
    pyplot.hist(data, bins=bins)
    pyplot.xlabel(xLabel, fontsize=fontsize)
    pyplot.ylabel(yLabel, fontsize=fontsize)
    pyplot.title(title)
    if fileName != None: pyplot.savefig(fileName)


def plotCorrMatrixSmall(corr_data, data_labels, y_data_labels=None, iRaws=0, iCols=0, title=None, with_values=True, fileName=None):

    fig, ax = pyplot.subplots(figsize=(5, 5))
    ic=corr_data.shape[1]
    data2show=corr_data[iRaws:,:ic-iCols]
    im = ax.imshow(data2show, cmap=pyplot.get_cmap('summer'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    x_data_count = len(data_labels)
    if y_data_labels:
        y_data_count = len(y_data_labels)
    else:
        y_data_count = x_data_count
    ax.set_xticks(np.arange(x_data_count-iCols))
    ax.set_yticks(np.arange(y_data_count-iRaws))
    ax.set_xticklabels(data_labels[:ic-iCols])
    if y_data_labels:
        ax.set_yticklabels(y_data_labels[iRaws:])
    else:
        ax.set_yticklabels(data_labels[iRaws:])

    pyplot.setp(ax.get_xticklabels(), rotation=90, ha="right",
                rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    if with_values:
        data_corr=corr_data[iRaws:,:ic-iCols]
        for i in range(data_corr.shape[0]):
            for j in range(data_corr.shape[1]):
                strOut = round(str(data_corr[i, j]), decimals=2)
                ax.text(j, i, strOut, ha="center", va="center", color="r")
    ax.set_title(title)
    fig.tight_layout()
    pyplot.grid(False)
    if fileName is not None:
        pyplot.savefig(fileName)


def gauss( x, mean, stddev, relative = False):
    y = np.zeros_like(x)
    if stddev > np.finfo(float).eps:
        y = 1 / (stddev * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * stddev ** 2))
        if relative and abs(np.max(y))>np.finfo(float).eps:
            y /= np.max(y)
    return y
def plotHistGauss(data, ax=None, bins=50, density=True,
                   title='Title', xLabel='xLabel', yLabel=None,
                   fontsize=None, relative = True, color = 'r'):
    if ax == None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax

    n_hist, bins_hist = np.histogram(data[1:].flatten(), bins=bins, density=density)
    # stat over all
    [value, interval] = sumMC(data, Coverage=0.95)
    ax1.plot(bins_hist, gauss(bins_hist, value[0], value[1],relative=relative), color, linewidth=2, label=title)
    ax1.set_xlabel(xLabel, fontsize=fontsize)
    ax1.set_title(title)
    ax1.legend()
    if yLabel is None:
        ax1.set_ylabel('Probability')
    else:
        ax1.set_ylabel(yLabel)
    return ax1

def plotHistScales(data, fig=None, ax=None, bins=50, density=True,
                   title='Title', xLabel='xLabel', yLabel=None,
                   filename=None, fontsize=None, add_distribution = False):
    if ax == None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax

    n_hist, bins_hist, patches_hist =  ax1.hist(data[1:].flatten(), bins=bins, density=density)
    ax1.set_xlabel(xLabel, fontsize=fontsize)

    ax1.set_title(title)
    if yLabel is None:
        ax1.set_ylabel('Probability')
    else:
        ax1.set_ylabel(yLabel)


    # stat over all
    [value, interval] = sumMC(data, Coverage=0.95)
    print('Value=', value, 'Inteval(95%)=', interval[1] - interval[0])

    if add_distribution:
        ax1.plot(bins_hist, gauss(bins_hist, value[0], value[1]), linewidth=2, color='r')

    ax1.axvline(interval[0])
    ax1.axvline(interval[1])
    ax1.axvline(value[0], color='tab:red')
    if filename is not None:
        fig.savefig(filename)
#    if ax is None:
#        fig.show()

def plotHistScalesWl(data, fig=None, ax=None, bins=50, density=True,
                     title='Histogram of wavelength scale',
                     xLabel='$\Delta \lambda$ / nm',
                     filename = None, fontsize=None):
    plotHistScales(data, fig=fig, ax=ax, bins=bins, density=density, title=title, xLabel=xLabel, filename=filename, fontsize=fontsize)


def plotHistScalesValue(data, fig=None, ax=None, bins=50, density=True,
                        title='Histogram of value scale',
                        xLabel='$\Delta y$ / A.U.',
                        filename=None, fontsize=None):
    plotHistScales(data, fig=fig, ax=ax, bins=bins, density=density, title=title, xLabel=xLabel, filename=filename, fontsize=fontsize)

def array2analyse(spectrumMC, wavelength_stat=True, scale_to_ref=True):
    _trials = len(spectrumMC) - 1
    _res = len(spectrumMC[0].spd.wl)
    loc_analyse = np.zeros((_trials, _res))
    for i in range(_trials):
        if scale_to_ref:
            if wavelength_stat:
                loc_analyse[i] = spectrumMC[i + 1].spd.wl - spectrumMC[0].spd.wl
            else:
                loc_analyse[i] = spectrumMC[i + 1].spd.value - spectrumMC[0].spd.value
        else:
            if wavelength_stat:
                loc_analyse[i] = spectrumMC[i + 1].spd.wl
            else:
                loc_analyse[i] = spectrumMC[i + 1].spd.value
    return loc_analyse


def analyse_stat(spectrum_mc, wavelength_stat=True, scale_to_ref=True, fontsize=None):
    wavelength_array = spectrum_mc[0].spd.wl
    loc_analyse = array2analyse(spectrum_mc, wavelength_stat, scale_to_ref)

    [loc_result_sum_mcv, loc_interval] = sumMCV(loc_analyse, Coverage=0.95)
    corr_image = np.corrcoef(loc_analyse.T)
    cov_image = np.cov(loc_analyse.T)

    fig = plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)

    color = 'tab:red'
    ax3.margins(0.05)
    ax3.set_xlabel('$\lambda$ / nm', fontsize=fontsize)
    ax3.set_ylabel('$\mu$', color=color, fontsize=fontsize)
    ax3.plot(wavelength_array, loc_result_sum_mcv[0], color=color)
    ax3.fill_between(wavelength_array, loc_interval[0], loc_interval[1], color=color, alpha=.3)
    ax3.tick_params(axis='y', labelcolor=color)

    ax32 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax32.set_ylabel('$\sigma$', color=color, fontsize=fontsize)  # we already handled the x-label with ax1
    ax32.plot(wavelength_array, loc_result_sum_mcv[1], color=color)
    ax32.tick_params(axis='y', labelcolor=color)

    if wavelength_stat:
        plt.title('Data Statistics Wavelength')
    else:
        plt.title('Data Statistics Value')

    #vmin=-1, vmax=1,
    im1 = ax2.imshow(corr_image,
                     extent=[wavelength_array[0], wavelength_array[-1], wavelength_array[-1], wavelength_array[0]],
                     cmap="jet", interpolation="nearest")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    ax2.set_title('Correlation')
    ax2.set_xlabel('$\lambda$ / nm', fontsize=fontsize)
    ax2.set_ylabel('$\lambda$ / nm', fontsize=fontsize)

    if wavelength_stat:
        plotHistScalesWl(loc_analyse, ax=ax1)
    else:
        plotHistScalesValue(loc_analyse, ax=ax1)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    return wavelength_array, loc_result_sum_mcv, cov_image, corr_image

def get_data_step(size_to_minimize, max_data_to_display=1000):
    if size_to_minimize < max_data_to_display:
        step = 1
        disp_count = size_to_minimize
    else:
        step = int(size_to_minimize / max_data_to_display)
        disp_count = int(size_to_minimize / step)
    return disp_count, step

def seaborn_plot_basedata(loc_array, wavelength_to_observe=550, filename=None):
    pos, = np.where(np.isclose(loc_array[0].spd.wl, wavelength_to_observe))

    disp_array_count, step = get_data_step(loc_array.shape[0])
    disp_array = np.zeros((2, disp_array_count - 1))
    for i in range(disp_array_count - 1):
        disp_array[0, i] = loc_array[i * step + 1].spd.wl[pos] - loc_array[0].spd.wl[pos]
        disp_array[1, i] = loc_array[i * step + 1].spd.value[0, pos] - loc_array[0].spd.value[0, pos]

    sns.set_theme(style="ticks")
    df = pd.DataFrame(data=disp_array.T, columns=['$\Delta\lambda / nm$', '$\Delta value / A.U.$'])
    grid = sns.pairplot(df, corner=True)
    plotTitle = 'Observation @ $\lambda$ = {} nm'
    grid.fig.suptitle(plotTitle.format(wavelength_to_observe))
    if filename is not None:
        grid.fig.savefig(filename)

def seaborn_plot_result(loc_result, filename=None):
    disp_array_count, step = get_data_step(loc_result.shape[1])
    disp_array = np.zeros((4, disp_array_count - 1))
    for i in range(disp_array_count - 1):
        disp_array[0, i] = loc_result[0, i * step + 1] / loc_result[0, 0]
        disp_array[1, i] = loc_result[1, i * step + 1] - loc_result[1, 0]
        disp_array[2, i] = loc_result[2, i * step + 1] - loc_result[2, 0]
        disp_array[3, i] = loc_result[3, i * step + 1] - loc_result[3, 0]

    sns.set_theme(style="ticks")
    df = pd.DataFrame(data=disp_array.T, columns=['$Y_{\mathrm{rel}} / A.U.$', '$\Delta x$', '$\Delta y$', '$\Delta CCT$'])
    grid = sns.pairplot(df, corner=True)
    plotTitle = 'Observation Yxy, CCT result'
    grid.fig.suptitle(plotTitle.format())
    if filename is not None:
        grid.fig.savefig(filename)

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# THX: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    pearson = pearson if pearson <= 1. else 1.0
    pearson = pearson if pearson >= -1. else -1.0
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, fill=False, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# THX: https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
def aspectratio_to_one( ax):
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    print( x_left, x_right, y_low, y_high)
    print( (abs((x_right-x_left)/(y_low-y_high))))
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

#def plot_2D( inVec, number = 100):
#    fig, ax1 = pyplot.subplots()
#    _, step = get_data_step(len(inVec.val[:,0]), number)
#    ax1.plot(inVec.val[::step,0], inVec.val[::step,1], 'rx' , label = 'Label')
#    confidence_ellipse(inVec.val[:,0], inVec.val[:,1], ax1, n_std=2.45, edgecolor='k')
#    ax1.grid(visible=True)
#    ax1.legend()
#    ax1.set_xlabel('x', fontsize=label_font_size)
#    ax1.set_ylabel('y', fontsize=label_font_size)
#    ax1.set_aspect('equal')
#    #aspectratio_to_one(ax1)

def plot_2D(inVec, number:int=100, name:str=None, marker_color:str='r', ax1:pyplot.axes=None, center_data:bool=False, k_display=1, offset=0):
    """
    show 2D measurement data in a diagram with  .

    Args:
        :inVec:
            | result of a MCSimulation with data to be displayed
        :number:
            | number of points to be selected for display
            | if number == 0, there is no display of the data points itself
        :name:
            | name of the data, if None the name from the MCS is used
        :marker_color:
            | color for the marker in the diagram
        :ax1:
            | diagram to be continued, if None a new one is generated
        :center_data:
            | False, display the data as they are, True: put the mean value in the centre

    Returns:
        :returns:
            | axis for further operations

    Note:
        Attention: no error management
    """
    if not ax1:
        _, ax1 = pyplot.subplots()
    if not name:
        name = inVec.name.name
    data_x = inVec.val[:, 0+offset].copy()
    data_y = inVec.val[:, 1+offset].copy()
    if center_data:
        data_x = data_x - np.mean(data_x)
        data_y = data_y - np.mean(data_y)
    # display the mean value
    ax1.plot([np.mean(data_x)], [np.mean(data_y)], marker_color + 'o', label=name, markersize=10)
    if number:
        _, step = get_data_step(len(data_x), number)
        ax1.plot(data_x[::step], data_y[::step], marker_color + 'x')
    confidence_ellipse(data_x, data_y, ax1, n_std=k_display*2.45, edgecolor=marker_color, linewidth=2)
    ax1.grid(visible=True)
    ax1.legend()
    if center_data:
        ax1.set_xlabel(r'$x-\bar{x}$', fontsize=label_font_size)
        ax1.set_ylabel(r'$y-\bar{y}$', fontsize=label_font_size)
    else:
        ax1.set_xlabel(r'$x$', fontsize=label_font_size)
        ax1.set_ylabel(r'$y$', fontsize=label_font_size)

    ax1.set_aspect('equal')
    return ax1