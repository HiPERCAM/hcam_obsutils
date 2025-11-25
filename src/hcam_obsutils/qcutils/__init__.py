from typing import Callable

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip, sigma_clipped_stats
from hipercam import mpl
from hipercam.ccd import CCD
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike
from skimage.util import view_as_blocks


def block_measure(
    data: ArrayLike, block_size: int | ArrayLike[int] = 30, func: Callable = np.mean
):
    """
    Apply a function across blocks of an image.

    If ``data`` is not perfectly divisible by ``block_size`` along a
    given axis then the data will be trimmed (from the end) along that
    axis.

    Parameters
    ----------
    data : array_like
        The data to be resampled.

    block_size : int or array_like (int)
        The integer block size along each axis.  If ``block_size`` is a
        scalar and ``data`` has more than one dimension, then
        ``block_size`` will be used for for every axis.

    func : callable, optional
        The method to use to downsample the data.  Must be a callable
        that takes in a `~numpy.ndarray` along with an ``axis`` keyword,
        which defines the axis along which the function is applied. e.g np.mean
    """
    data = np.asanyarray(data)
    block_size = np.atleast_1d(block_size)
    if data.ndim > 1 and len(block_size) == 1:
        block_size = np.repeat(block_size, data.ndim)

    if len(block_size) != data.ndim:
        raise ValueError(
            "`block_size` must be a scalar or have the same length as `data.shape`"
        )

    block_size = np.array([int(i) for i in block_size])
    size_resampled = np.array(data.shape) // block_size
    size_init = size_resampled * block_size

    # trim data if necessary
    for i in range(data.ndim):
        if data.shape[i] != size_init[i]:
            data = data.swapaxes(0, i)
            data = data[: size_init[i]]
            data = data.swapaxes(0, i)

    view = view_as_blocks(np.ascontiguousarray(data), tuple(block_size))
    return func(view, axis=(-1, -2))


def block_stats(data: ArrayLike, block_size: int = 30):
    """
    Measures the mean, median and std. dev. accounting for local variation and outliers

    We measure each quantity over windows of size (block_size, block_size). Blocks which
    do not lie fully within the data are discarded. We then take the mean of all blocks,
    using sigma_clipping to discard outliers.
    """
    if any([dim <= block_size for dim in data.shape]):
        raise ValueError(
            "data is of shape {} and smaller than block size of {}".format(
                data.shape, block_size
            )
        )

    aggregators = (np.mean, np.median, np.std)
    measures = []
    for aggregator in aggregators:
        measure = block_measure(data, block_size, aggregator)
        measures.append(sigma_clip(measure).mean())
    return measures


def calc_and_plot(
    ccd: CCD,
    nccd: str,
    nwin: str,
    xleft: int,
    xstep: int,
    xsize: int,
    ylow: int,
    ystep: int,
    ysize: int,
):
    """
    Calculate statistics for the four patches in the bias window and plot them.

    Parameters
    ----------
    ccd : CCD
        The CCD object containing the data.
    nccd : str
        The CCD number as a string.
    nwin: int
        The window number of CCD
    xleft : int
        The left x-coordinate of the first patch.
    xstep : int
        The step size in x between patches.
    xsize : int
        The size in x of each patch.
    ylow : int
        The lower y-coordinate of the first patch.
    ystep : int
        The step size in y between patches.
    ysize : int
        The size in y of each patch.
    """

    if nwin == "1":
        win = "L"
    else:
        win = "R"
    window = ccd[nwin]

    meanList = []
    medianList = []
    sdList = []

    patches = []
    for i in np.arange(2):
        for j in np.arange(2):
            # define sub-window in binned coords
            xl = (xleft + xstep * (i % 2)) / window.xbin
            xr = xl + xsize / window.xbin
            ylo = (ylow + ystep * (j % 2)) / window.ybin
            yhi = ylo + ysize / window.ybin

            # convert to physical pixels
            xl, xr = window.x(np.array((xl, xr)))
            ylo, yhi = window.y(np.array((ylo, yhi)))

            # extract sub-window for patch
            sub_win = window.window(xl, xr, ylo, yhi)

            # create rectangle patch for plotting later
            patch = Rectangle(
                (xl, ylo), xr - xl, yhi - ylo, fill=False, color="r", lw=2
            )
            patches.append(patch)

            # calculate stats for this little patch
            data = sub_win.data
            mn, mdn, std = sigma_clipped_stats(data)

            meanList.append(mn)
            sdList.append(std)
            medianList.append(mdn)

    mean = np.mean(meanList)
    sd = np.mean(sdList)
    median = np.mean(medianList)
    print("CCD%s %s mean   = %5.0f" % (nccd, win, mean))
    print("CCD%s %s median   = %5.0f" % (nccd, win, median))
    print("CCD%s %s sigma  = %4.1f" % (nccd, win, sd))

    # and plot them at high contrast to check for pickup noise
    fig, axes = plt.subplots()
    plo = 0.999
    phi = 1.001
    mpl.pWind(axes, window, plo * window.median(), phi * window.median())
    axes.grid(False)
    # plot CCD boundary
    axes.plot(
        [0.5, ccd.nxtot + 0.5, ccd.nxtot + 0.5, 0.5, 0.5],
        [0.5, 0.5, ccd.nytot + 0.5, ccd.nytot + 0.5, 0.5],
    )
    for p in patches:
        axes.add_patch(p)
    plt.show()
    return (mean, sd)


def bias_measurement_to_dataframe_row(
    date: str,
    binning: int,
    readout: str,
    ccd_lut: dict[str, str],
    win_lut: dict[str, str],
    means: dict[str, dict[str, float]],
    sigmas: dict[str, dict[str, float]],
):
    """
    Convert bias measurement results to a dataframe row.

    Parameters
    ----------
    date : str
        The current date as a string.
    binning : int
        The current binning factor.
    readout : str
        The current readout speed.
    ccd_lut : dict[str, str]
        The CCD lookup table mapping CCD numbers to names.
    win_lut : dict[str, str]
        The window lookup table mapping window numbers to names.
    means : dict[str, dict[str, float]]
        The means for each CCD and window.
        The outer dict key is the CCD number as a string,
        the inner dict key is the window number as a string.
    sigmas : dict[str, dict[str, float]]
        The sigmas for each CCD and window.
        Same structure as means.
    """
    row = dict(readout=readout, date=date, binning=binning)
    for iccd, ccdmeans in means.items():
        for iwin, winmean in ccdmeans.items():
            ccd = ccd_lut[iccd]
            win_name = win_lut[iwin]
            winsigma = sigmas[iccd][iwin]
            row["{}{}Mean".format(ccd, win_name)] = winmean
            row["{}{}Sigma".format(ccd, win_name)] = winsigma
    return row


def plot_qc_bias_archive(
    date: str,
    binning: int,
    readout: str,
    ccd_lut: dict[str, str],
    win_lut: dict[str, str],
    means: dict[str, dict[str, float]],
    sigmas: dict[str, dict[str, float]],
    bias_df: pd.DataFrame,
):
    """
    Plot the bias level and readout noise archive, comparing current values.

    Parameters
    ----------
    date : str
        The current date as a string.
    binning : int
        The current binning factor.
    readout : str
        The current readout speed.
    ccd_lut : dict[str, str]
        The CCD lookup table mapping CCD numbers to names.
    win_lut : dict[str, str]
        The window lookup table mapping window numbers to names.
    means : dict[str, dict[str, float]]
        The current mean bias level for each CCD and window.
        The outer dict key is the CCD number as a string,
        the inner dict key is the window number as a string.
    sigmas : dict[str, dict[str, float]]
        The current sigma (readout noise) for each CCD and window.
        Same structure as means.
    bias_df : pd.DataFrame
        The bias data archive as a pandas DataFrame.
    """

    _, (bias_axis, rno_axis) = plt.subplots(nrows=2, sharex=True)
    print("\n\n")
    for iccd, ccdmeans in means.items():
        for iwin, winmean in ccdmeans.items():
            print(
                "{} readout speed, {} binning, {} CCD, {} channel".format(
                    readout, binning, ccd_lut[iccd], win_lut[iwin].lower()
                )
            )

            bias = bias_df["{}{}Mean".format(ccd_lut[iccd], win_lut[iwin])]
            rno = bias_df["{}{}Sigma".format(ccd_lut[iccd], win_lut[iwin])]
            current_bias = means[iccd][iwin]
            current_rno = sigmas[iccd][iwin]

            # plot date
            marker = ccd_lut[iccd][0]
            marker += "o" if iwin == "1" else "s"
            bias_axis.plot(bias, marker)
            bias_axis.plot([bias.size + 1], [current_bias], marker)
            rno_axis.plot(rno, marker)
            rno_axis.plot([rno.size + 1], [current_rno], marker)

        print("")
        print("Number of values in archive = ", bias.size)
        print(
            "Archival last recorded (bias, rno) value = {}, {}".format(
                bias.iloc[-1], rno.iloc[-1]
            )
        )
        print(
            "Archival minimum (bias,rno) value = {}, {}".format(bias.min(), rno.min())
        )
        print(
            "Archival maximum (bias,rno) value = {}, {}".format(bias.max(), rno.max())
        )
        print("Archival mean (bias,rno) value = {}, {}".format(bias.mean(), rno.mean()))
        print(
            "Archival standard deviation of (bias,rno) = {}, {}".format(
                bias.std(), rno.std()
            )
        )
        print(
            "Archival median (bias,rno) value = {}, {}".format(
                bias.median(), rno.median()
            )
        )
        bold = "\033[1m"
        reset = "\033[0;0m"
        print(
            bold
            + "Current median (bias,rno) value = {}, {}".format(
                current_bias, current_rno
            )
            + reset
        )
        print("")

    bias_axis.set_ylabel("Bias (counts)")
    bias_axis.text(
        bias.size + 1,
        2000,
        "current values",
        rotation="vertical",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
    )
    rno_axis.set_xlabel("Quality control archive entry number")
    rno_axis.set_ylabel("Readout noise (counts)")
    rno_axis.text(
        rno.size + 1,
        3.3,
        "current values",
        rotation="vertical",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
    )

    plt.show()
    return
