import sys
import os

external_script_path = "D:/OneDrive - FHDO - PROD/Master BMIT/masterarbeit"

if external_script_path not in sys.path:
    sys.path.append(external_script_path)

from trajectory_calc import TrajectoryCalculator


tc = TrajectoryCalculator('abbiegen.pkl', 
    directory='D:/OneDrive - FHDO - PROD/Master BMIT/masterarbeit/', 
    rec_id='1247800 6754', 
    vis_only=True)
tc.visualize(wait_between_frames=300)