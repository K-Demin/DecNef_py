import sys

sys.argv = [
    "biopac_streamer",
    "--host", "115.145.189.30",
    "--port", "15000",
    "--tr", "0.9",
    "--phys-fs", "1000",
    "--mode", "biopac",
    "--mpdev-dll", r"D:\SIN_LAB RT-BIOPAC\DecNef_py\BIOPAC Hardware API 2.2.5 Research\VC10\x64\mpdev.dll",
    "--csv-path", r"D:\SIN_LAB\20251229_00067.csv",
    "--downsample-hz", "100",
    "--card-source", "ecg"
]

from biopac_rt.biopac_streamer import main

main()