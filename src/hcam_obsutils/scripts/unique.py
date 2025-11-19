import argparse

from hcam_obsutils.formats import headers, hhead_equal, is_bias, uhead_equal

HELP = """
`unique` identifies the unique observational formats present in a night of observing data.
It can be used to identify how many different calibration frames are needed.
"""


def main():
    parser = argparse.ArgumentParser(description=HELP)
    parser.add_argument(
        "-i",
        "--include-caution",
        default=False,
        action="store_true",
        help="include runs marked 'data caution' when listing formats",
    )
    parser.add_argument(
        "--hcam",
        action="store_true",
        default=False,
        help="process HiPERCAM runs rather than ULTRASPEC and/or ULTRACAMruns",
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="directories to search for runs, subdirectories called 'data' will also be searched",
    )
    args = parser.parse_args()

    # choose comparison function
    compare = hhead_equal if args.hcam else uhead_equal

    unique_formats = {}
    dirs = set(["data"] + args.dirs)
    for dirpath in sorted(dirs):
        # all headers in this directory
        for header in headers(dirpath, hcam=args.hcam):
            # skip bias frames in favour of more intersting frames
            if is_bias(header):
                continue
            # skip data caution runs unless requested
            if (
                not args.include_caution
                and header.header["DTYPE"].lower() == "data caution"
            ):
                continue

            # see if we've seen this before
            new_format = True
            for _, rold in unique_formats.items():
                if compare(header, rold):
                    new_format = False
                    break

            if new_format:
                key = header.fname if args.hcam else header.run
                unique_formats[key] = header

    # print out unique formats
    print("Unique observational formats found:")
    for run, header in unique_formats.items():
        print(f"{run}: {header.mode} {header.xbin}x{header.ybin} {header.wforms}")
