import os
import sys

projdir = sys.argv[1]
projname = projdir.split('/')[-1]
fps = sys.argv[2]

os.system("ffmpeg -framerate {} -pattern_type glob -i './{}/*.png' ./{}/_{}.mp4".format(fps, projdir, projdir, projname))