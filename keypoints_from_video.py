import tensorflow as tf
import cv2
import numpy as np
import posenet
from pose import Pose
import pickle
import os

#USAGE : python3 keypoints_from_video.py --activity "punch - side" --video "test.mp4" 
def main(video, activity, lookup):
	pose = Pose()
	coords_list = []
	lookup_dict = {}
	
	with tf.compat.v1.Session() as sess:
		model_cfg, model_outputs = posenet.load_model(101, sess)
		
		cap = cv2.VideoCapture(video)
		i = 1

		if cap.isOpened() is False:
			print("error in opening video")
		while cap.isOpened():
			ret_val, image = cap.read()
			if ret_val:
				image = cv2.resize(image,(372,495))			
				input_points,input_black_image = pose.getpoints_vis(image,sess,model_cfg,model_outputs)
				input_points = input_points[0:34]
				# print(input_points)
				input_new_coords = pose.roi(input_points)
				input_new_coords = input_new_coords[0:34]
				input_new_coords = np.asarray(input_new_coords).reshape(17,2)
				coords_list.append(input_new_coords)
				# cv2.imshow("black", input_black_image)
				cv2.waitKey(1)
				i = i + 1
			else:
				break
		cap.release()
		

		coords_list = np.array(coords_list)
		
		cv2.destroyAllWindows
		# print(b)
		print(f'shape of coords array: {coords_list.shape}')
		print("Lookup Table Created")
		lookup_dict[activity] = coords_list
		# print(c)
		f = open(lookup,'wb')
		pickle.dump(lookup_dict,f)
		# pickle.dump()


