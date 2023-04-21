# importing libraries
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from PIL import Image as im

from colorizers import *

from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    # files.sort(key = lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        # print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release() 


parser = argparse.ArgumentParser()
parser.add_argument('-v','--vid_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()
#base path for saving imgs to tempo
base_path = os.path.join('C:/Users/hp/Desktop/Main Project/f/colorization/tempo', 'frame')

#frame count for saving in tempo
n = 0

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(opt.vid_path)

#digit count for saving in tempo
digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(totalframes)

# Check if camera opened successfully
if (cap.isOpened()== False):
	print("Error opening video file")

#empty dir temp
dir = './tempo/'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

# Read until video is completed
while(cap.isOpened() and n<totalframes):
	
# Capture frame-by-frame
	ret, frame = cap.read()
	resFrame = cv2.resize(frame, (960, 540))

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	# img = load_img(opt.img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(resFrame, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	# out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
	if n==0:
		print("Processing frames. " + str(n))
	else:
		print(n)

	if ret == True:
	# Display the resulting frame
		# cv2.imshow('Frame', out_img_eccv16)
		cv2.imshow('Frame', out_img_siggraph17)
		
		# plt.imsave('{}_{}.{}'.format(base_path, str(n).zfill(digit), 'png'), out_img_eccv16)
		plt.imsave('{}_{}.{}'.format(base_path, str(n).zfill(digit), 'png'), out_img_siggraph17)
		
		# cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), 'png'), imag)
		n += 1
		
	# Press Q on keyboard to exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

# Break the loop
	else:
		break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

pathIn= './tempo/'
pathOut = 'final_out.avi'
fps = 14.28
print("Forming video....")
convert_frames_to_video(pathIn, pathOut, fps)
