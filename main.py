"""Compatibility entrypoint for VPG visual-servo inference."""

import runpy


if __name__ == "__main__":
    runpy.run_module("src.vpg_visual_servo_main", run_name="__main__")
