import numpy as np
from hipercam import MCCD
from numpy.typing import ArrayLike


def gain_simple(mean: float | ArrayLike, sigma: float | ArrayLike) -> float | ArrayLike:
    """
    Simple gain calculation from mean and standard deviation of a flat field.

    The flat field must be bias subtracted before calculating the gain.

    Parameters
    ----------
    mean : float | ArrayLike
        Mean value(s) of small CCD region(s) in ADU
    sigmas : float | ArrayLike
        Standard deviation value(s) of small CCD region(s) in ADU

    Returns
    -------
    gains : float | ArrayLike
        Gain value(s) in e-/ADU.
    """
    return (np.sqrt(mean) / sigma) ** 2.0


def gain(
    flat1: MCCD,
    flat2: MCCD,
    bias: MCCD,
    nccd: str,
    nwin: str,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
) -> float:
    """
    Calculate the gain from two flat fields and a bias frame.

    Parameters
    ----------
    flat1 : MCCD
        First flat field frame.
    flat2 : MCCD
        Second flat field frame.
    bias : MCCD
        Bias frame.
    nccd : str
        CCD number as string
    nwin : str
        Window number as string
    xmin : int
        Minimum x pixel coordinate of region to use.
    xmax : int
        Maximum x pixel coordinate of region to use.
    ymin : int
        Minimum y pixel coordinate of region to use.
    ymax : int
        Maximum y pixel coordinate of region to use.

    Returns
    -------
    gains : ArrayLike
        Gain map in e-/ADU.
    """

    # helper to extract chunk of data we beed
    def select_region(mccd, nccd, nwin, xmin, xmax, ymin, ymax):
        win = mccd[nccd][nwin]
        sub_win = win.window(
            win.llx + xmin, win.llx + xmax, win.lly + ymin, win.lly + ymax
        )

        return sub_win

    # stats from bias
    bias_win = select_region(bias, nccd, nwin, xmin, xmax, ymin, ymax)
    bias_val = bias_win.median()
    rno = bias_win.std()

    # windows of flats
    flat1_win = select_region(flat1, nccd, nwin, xmin, xmax, ymin, ymax)
    flat2_win = select_region(flat2, nccd, nwin, xmin, xmax, ymin, ymax)

    # first measure variance in the difference between uncorrected flats
    diff = flat1 - flat2
    diff_win = select_region(diff, nccd, nwin, xmin, xmax, ymin, ymax)
    variance = diff_win.std() ** 2

    # now measure mean signal level in each flat after bias subtraction
    flat1_debias = flat1_win - bias_win
    flat2_debias = flat2_win - bias_win
    av_win = 0.5 * (flat1_debias + flat2_debias)
    mean = av_win.mean()

    # now calculate gain in e-/ADU
    gain = 1.0 / ((variance - (2.0 * (rno**2.0))) / (2.0 * mean))
    return gain
