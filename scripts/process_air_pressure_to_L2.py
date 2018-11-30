#!/usr/bin/env python3
"""
Processes pressure L1 data (TOA5) to L2 (NetCDF).

PRESSURE_DATA_PATH env variable must point to the L1 data path.
"""
from asist_nsf_2018.process_level1 import process_dp_to_level2
process_dp_to_level2()
