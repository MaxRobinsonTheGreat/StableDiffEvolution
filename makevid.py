import os
import sys

projname = sys.argv[1]

os.system("ffmpeg -framerate 45 -pattern_type glob -i './walks/{}/*.png' ./walks/{}/_{}.mp4".format(projname, projname, projname))