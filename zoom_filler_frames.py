import os, glob, math
from tqdm import tqdm
from PIL import Image
from util import cropToCenter

project_name = "atoms"
image_size = (512,512)
zoom_speed = 128
num_filler_frames = 64
resize_factor = 1

proj_dir = "./zooms/"+project_name
frames_dir = proj_dir+"/frames"
filler_frames_dir = proj_dir+"/filler_frames"
os.makedirs(filler_frames_dir, exist_ok=True)
im_files = sorted(filter(os.path.isfile, glob.glob(frames_dir + '/*.png')))

# total_frames = num_filler_frames * len(im_files-1) + 1 # +1 for the final frame
cur_im_size = image_size[0]*resize_factor
zoom_speed *= resize_factor
next_im_size = round((cur_im_size*cur_im_size)/(cur_im_size-(zoom_speed*2)))
upscaled_zoom_speed = round((next_im_size - cur_im_size) // 2)
exp_growth_base = next_im_size / cur_im_size
# function that describes rate of growth is f(x) = exp_growth_base^x
# we need to invert it to find x for cur_im and next_im
start_x = math.log(cur_im_size, exp_growth_base)
end_x = math.log(next_im_size, exp_growth_base)
x_step = (end_x - start_x)/num_filler_frames

next_im = None
frame_count = 0
loop = tqdm(total=(len(im_files)-1)*num_filler_frames)
for i in range(len(im_files)-1):
    cur_im = Image.open(im_files[i]).resize((cur_im_size,cur_im_size))
    next_im = Image.open(im_files[i+1]).resize((cur_im_size,cur_im_size))

    for filler_count in range(num_filler_frames):
        frame = next_im.resize((next_im_size,next_im_size))
        frame.paste(cur_im, (upscaled_zoom_speed, upscaled_zoom_speed))

        down_size = exp_growth_base**(start_x+(x_step*filler_count))

        frame = cropToCenter(frame, (down_size, down_size))
        frame.save(proj_dir+"/cropped.png")

        frame = frame.resize(image_size)

        frame.save(os.path.join(filler_frames_dir, '%06d.png' % frame_count))
        frame_count += 1
        loop.update(1)

next_im.save(os.path.join(filler_frames_dir, '%06d.png' % frame_count))
