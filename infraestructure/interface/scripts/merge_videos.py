import argparse
import glob
import math
# import camera_info_manager
import numpy as np 
from numpy import linalg as LA
import subprocess, sys

if __name__ == '__main__':

    # argparse = argparse.ArgumentParser()

    # argparse.add_argument('-g', '--gt_filename', help='GT', required=True)
    # argparse.add_argument('-e', '--estimation_filename', help='Estimation', required=True)

    color = "green"
    vid_names = []
    vid_names.append(f"test_{color}_rgb.avi")
    vid_names.append(f"test_{color}_black.avi")
    vid_names.append(f"test_{color}_color.avi")
    vid_names.append(f"test_{color}_denoise.avi")
    vid_names.append(f"test_{color}_dilated.avi")
    vid_names.append(f"test_{color}_masked.avi")
    vid_names.append(f"test_{color}_positions.avi")

    min_dt = 10000

    lengths = []
    for vid in vid_names:
        print(f"vid {vid}")
        # result = subprocess.run(["ls",f"*{vid}"],shell=True, capture_output=True)
        # print(f"vidname: {result.stdout}")
        cmd = f'ffprobe -i *{vid} -show_entries format=duration -v quiet -of csv="p=0"'
        result = subprocess.run(cmd , shell=True, check=True, text=True, capture_output=True)
        print(f"dt: {result.stdout}")
        lengths.append(float(result.stdout))

        min_dt = min(min_dt, float(result.stdout))

    print(f"min_dt: {min_dt}")
    for i, vid in enumerate(vid_names):
        dt = lengths[i] - min_dt
        print(f"dt {dt}")
        cmd = f'ffmpeg -y -i *{vid} -ss {dt} -vcodec copy -acodec copy /tmp/{vid}'
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)

    cmd = f'ffmpeg -y -i /tmp/{vid_names[0]} -i /tmp/{vid_names[1]} -i /tmp/{vid_names[2]} -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" -c:v ffv1 output.avi'
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, text=True, capture_output=False)
