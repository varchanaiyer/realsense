#First we have to import the realsense library
import pyrealsense2 as rs
#Import Numpy and opencv for array and image related manipulations 
import numpy as np
import cv2
import os

#We need to create a pipeline and then give the configuration for the camera to start recording
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30) #This is for depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #This is for color

#Start streaming
profile = pipeline.start(config)

#We now need to get the depth sensors scale so that we can eliminate the background
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

#let us clip everything that is more than 0.5 meters away
clipping_distance_in_meters = 0.5
clipping_distance = clipping_distance_in_meters / depth_scale

#We want to align the image taken from the depth sensor with the actual captured image
#To do this we need to create an align object
#The "align_to" is the stream type to which we plan to align depth frames
#In this case, that is our color image
align_to = rs.stream.color
align = rs.align(align_to)
i=0
#Looping to capture all the images
try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Background Removed', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('p'):
            os.makedirs(f'images/{i}')
            cv2.imwrite(f'images/{i}/bg_removed.png', bg_removed)
            cv2.imwrite(f'images/{i}/depth.png', depth_colormap)
            i+=1
            print('Image Taken')
finally:
    pipeline.stop()
