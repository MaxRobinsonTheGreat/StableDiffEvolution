import os
import sys

projdir = sys.argv[1]
projname = projdir.split('/')[-1]

os.system("ffmpeg -framerate 20 -pattern_type glob -i './{}/*.png' ./{}/_{}.mp4".format(projdir, projdir, projname))