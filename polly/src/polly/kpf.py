"""
polly

kpf

Parameters specific to the KPF spectrograph and data filesystem
"""
import os
from pathlib import Path

THORIUM_ORDER_INDICES = [*list(range(12)), 35]
LFC_ORDER_INDICES = [i for i in range(67) if i not in THORIUM_ORDER_INDICES]
ALL_ORDER_INDICES = [*THORIUM_ORDER_INDICES, *LFC_ORDER_INDICES]
TEST_ORDER_INDICES = [0, 17, 34, 35, 51, 66]

ORDERLETS = ["SCI1", "SCI2", "SCI3", "CAL", "SKY"]
TIMESOFDAY = ["morn", "eve", "night", "midnight"]

MASTERS_DIR = os.getenv('KPF_POLLY_MASTERS_DIR')
if MASTERS_DIR is None:
    # Default directory is for execution inside of Docker container
    MASTERS_DIR = Path("/data/masters")

L1_DIR = os.getenv('KPF_POLLY_L1_DIR')
if L1_DIR is None:
    # Default directory is for execution inside of Docker container
    L1_DIR = Path("/data/L1")
