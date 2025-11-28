import dataclasses
from typing import Callable

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.time import Time
from astropy.visualization import time_support
from hipercam import mpl
from hipercam.ccd import CCD
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike
from skimage.util import view_as_blocks


# abstract base class for readout modes
class ReadoutMode:
    def asdict(self) -> dict:
        """
        Convert the ReadoutMode to a dictionary.

        Used for constructing a dataframe row
        """
        return dataclasses.asdict(self)

    def query_string(self) -> str:
        """
        Create an SQL query string from the ReadoutMode.

        Used for querying the database for rows matching this readout mode.
        """
        clauses = []
        for field, value in self.asdict().items():
            if isinstance(value, str):
                clauses.append(f"{field}='{value}'")
            else:
                clauses.append(f"{field}={value}")
        return " AND ".join(clauses)


def block_measure(
    data: ArrayLike, block_size: int | ArrayLike = 30, func: Callable = np.mean
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
    window_name: str,
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
    window_name : str
        The name of the window being analysed.
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
    print(f"CCD{nccd} {window_name} mean   = {mean:5.0f}")
    print(f"CCD{nccd} {window_name} median = {median:5.0f}")
    print(f"CCD{nccd} {window_name} sigma  = {sd:4.1f}")
    print("")

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
    readout_mode: ReadoutMode,
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
    readout_mode : ReadoutMode
        The readout mode used for the measurement.
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
    row = dict(date=date)
    row.update(readout_mode.asdict())
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
    readout_mode: ReadoutMode,
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
    readout_mode : ReadoutMode
        The readout mode used for the current measurement.
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
            print("{} CCD, {} channel".format(ccd_lut[iccd], win_lut[iwin].lower()))

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
            f"Archival last recorded (bias, rno) value = {bias.iloc[-1]:.1f}, {rno.iloc[-1]:.2f}"
        )
        print(f"Archival minimum (bias,rno) value = {bias.min():.1f}, {rno.min():.2f}")
        print(f"Archival maximum (bias,rno) value = {bias.max():.1f}, {rno.max():.2f}")
        print(f"Archival mean (bias,rno) value = {bias.mean():.1f}, {rno.mean():.2f}")
        print(
            f"Archival standard deviation of (bias,rno) = {bias.std():.1f}, {rno.std():.2f}"
        )
        print(
            f"Archival median (bias,rno) value = {bias.median():.1f}, {rno.median():.2f}"
        )

        bold = "\033[1m"
        reset = "\033[0;0m"
        print(
            bold
            + f"Current median (bias,rno) value = {current_bias:.1f}, {current_rno:.2f}"
            + reset
        )
        print("")

    bias_axis.set_ylabel("Bias (counts)")
    bias_lims = bias.median() + np.array([-2, 2]) * bias.std()
    bias_axis.text(
        bias.size + 1,
        2000,
        "current values",
        rotation="vertical",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
    )
    bias_axis.set_ylim(*bias_lims)

    rno_axis.set_xlabel("Quality control archive entry number")
    rno_axis.set_ylabel("Readout noise (counts)")
    rno_lims = rno.median() + np.array([-2, 2]) * rno.std()
    rno_axis.text(
        rno.size + 1,
        3.3,
        "current values",
        rotation="vertical",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
    )
    rno_axis.set_ylim(*rno_lims)
    plt.show()
    return


def plot_zeropoint_data(
    df: pd.DataFrame, bands: list[str], results: list[dict]
) -> None:
    df = df.sort_values("date")
    _, ax = plt.subplots()

    # make a dictionary of band vs colour for all bands
    time_support(format="iso")
    band_color_dict = {band: color for band, color in zip(bands, plt.cm.tab10.colors)}
    for band in bands:
        # plot archival data
        band_data = df[df["band"] == band]
        x = Time(band_data["date"].to_list())

        plt.scatter(
            x, band_data["mean"], marker=".", color=band_color_dict[band], label=band
        )
        mn, md, sd = sigma_clipped_stats(band_data["mean"])
        print(f"Band {band}: Archival mean ZP {mn:.2f} (SD = {sd:.2f})")

    # plot current result
    current_df = pd.DataFrame(results)
    for band in bands:
        band_data = current_df[current_df["band"] == band]
        x = Time(band_data["date"].to_list())
        plt.scatter(
            x,
            band_data["mean"],
            marker="o",
            color=band_color_dict[band],
            edgecolor="black",
        )
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Zeropoint")
    plt.show()
