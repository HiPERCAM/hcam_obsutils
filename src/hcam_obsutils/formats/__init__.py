from pathlib import Path
from typing import Iterable

from hipercam.hcam import Rhead as Hhead
from hipercam.ucam import Rhead as Uhead

UCAM_RE = "run[0-9][0-9][0-9].xml"
HCAM_RE = "run[0-9][0-9][0-9][0-9].fits"


def is_bias(header: Hhead | Uhead) -> bool:
    """
    Determine if a header corresponds to a bias frame.

    Parameters
    ----------
    header : Hhead or Uhead
        Header object to check.

    Returns
    -------
    bool
        True if the header corresponds to a bias frame, False otherwise.
    """
    try:
        # assume ucam first
        target = header.header["TARGET"].lower()
    except KeyError:
        target = header.header["OBJECT"].lower()
    return "bias" in target


def headers(dirpath: str, hcam: bool = False) -> Iterable[Hhead | Uhead]:
    """
    Generator yielding header objects from all runs in dirpath.

    ULTRACAM/ULTRASPEC Power ON/OFF runs are skipped.

    Parameters
    ----------
    dirpath : str
        Path to directory to search for runs.
    hcam : bool
        If True, process HiPERCAM runs, otherwise ULTRASPEC/ULTRACAM runs.

    Yields
    ------
    header : Hhead or Uhead
        Header object for each run found.
    """
    dirpath = Path(dirpath)
    if dirpath.is_dir():
        header_files = dirpath.glob(HCAM_RE) if hcam else dirpath.glob(UCAM_RE)
        for fn in header_files:
            fn = fn.with_suffix("")
            header = Hhead(str(fn)) if hcam else Uhead(str(fn))
            if not hcam and header.isPonoff():
                continue
            yield header


def uhead_equal(h1: Uhead, h2: Uhead, fussy: bool = False) -> bool:
    """
    Determine if two Uhead objects correspond to the same format, for the purposes of calibration.

    Parameters
    ----------
    h1 : Uhead
        First header object.
    h2 : Uhead
        Second header object.
    fussy : bool
        If True, include avalanche gain in the comparison (only relevant for ULTRASPEC).

    Returns
    -------
    bool
        True if the two headers correspond to the same format, False otherwise.
    """
    # binning, gain, instrument, etc
    ok = (
        (h1.xbin == h2.xbin)
        and (h1.ybin == h2.ybin)
        and (h1.instrument == h2.instrument)
        and (len(h1.win) == len(h2.win))
        and (h1.gainSpeed == h2.gainSpeed)
        and (
            h1.header.get("HVGAIN", None) == h2.header.get("HVGAIN", None)
            if fussy
            else True
        )
    )
    # check windows are the same
    if ok:
        for window in h1.win:
            if not any(w == window for w in h2.win):
                ok = False
                break
    return ok


def hhead_equal(h1: Hhead, h2: Hhead, **kwargs) -> bool:
    """
    Determine if two Hhead objects correspond to the same format, for the purposes of calibration.

    Parameters
    ----------
    h1 : Hhead
        First header object.
    h2 : Hhead
        Second header object.

    Returns
    -------
    bool
        True if the two headers correspond to the same format, False otherwise.
    """
    ok = (
        (h1.xbin == h2.xbin)
        and (h1.ybin == h2.ybin)
        and len(h1.windows) == len(h2.windows)
        # mode
        and h1.header.get("ESO DET READ CURNAME", None)
        == h2.header.get("ESO DET READ CURNAME", None)
        # readout speed
        and h1.header.get("ESO DET SPEED", None) == h2.header.get("ESO DET SPEED", None)
    )
    if ok:
        # check windows are the same in all CCDs
        ok = sorted(h1.wforms) == sorted(h2.wforms)
    return ok
