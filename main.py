"""Generate the same manual visual gallery used by the regression tests."""

import argparse

from tests.visual import build_gallery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="visual", help="Gallery output directory")
    args = parser.parse_args()
    cases, path = build_gallery(args.output)
    print(f"{len(cases)} visual examples: {path.resolve()}")
