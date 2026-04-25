"""Compatibility entrypoint for the URDF scaling utility."""

import runpy


if __name__ == "__main__":
    runpy.run_module("src.scale_urdf", run_name="__main__")
