import os
import sys

projname = sys.argv[1]

os.system("ffmpeg -framerate 15 -pattern_type glob -i './dreams/{}/*.jpg' ./dreams/{}/_{}.mp4".format(projname, projname, projname))