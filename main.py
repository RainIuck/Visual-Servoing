"""Compatibility entrypoint for the main pick-and-place demo."""

import runpy


if __name__ == "__main__":
    runpy.run_module("src.main", run_name="__main__")
