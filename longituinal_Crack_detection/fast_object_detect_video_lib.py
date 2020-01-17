'''
Code to run tensorflow object detection API infrencing
Modified from code in repository
Also see FASTER: https://github.com/tensorflow/models/issues/4355
SEE tensorflow documentation for object detect
'''

import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util

# Image Frame Size (480x480 default)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
# DEBUG FLAG
DEBUG = 0

def setupgraph(path_to_model):
    '''
    import unserialized graph file
    '''
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def run_inference(image, sess):
    '''
    run interence
    '''
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}

    # make dictonary of tensor names
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])  #num detections not used in mobile model
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


def read_video(desired_frame, cap):
    '''
    Read the video
    '''
    img_resized = 0
    if cap.isOpened():
        ret, frame = cap.read()
        img1024 = frame[896:1920 , 26:1050]
        img_resized  = cv2.resize(img1024, (IMAGE_WIDTH, IMAGE_HEIGHT))   # Resize image to see all 
        if False:
            #convert to gray and then present it as RGB (to test if gray hurts performance)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    else:
        print('video file not open')
    return img_resized


def read_single_image(i, test_dir_path):
    '''
    read the image data
    '''
    i+=1
    path = test_dir_path+"/c3/image ({}).png".format(i)
    print(path)
    img = cv2.imread(path) # reading the img
    #get image shape
    width, height, ch = img.shape
    #select suqare part of image if needed
    if width != height:
        img = img[0:959 , 0:959]
    #resize the image if needed
    if width>IMAGE_WIDTH or height>IMAGE_HEIGHT:
        img  = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))   # Resize image to see all 
    return img


def visulization(image_np, output_dict, category_index, out, i, test_dir_path, save):
    '''
    Code To Generate Images and Videos With Results Drawn On
    '''
    print(f'test dir path is: {test_dir_path}')
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=2)
    #For DEBUG SAVE EACH FRAME with top score in name
    if True:
        test_score = str(int(100*output_dict['detection_scores'][0]))
        # save image as jpg    
        save_image_paths = test_dir_path+'/mobilenet_test/testCrackOut{}'.format(i)+'_Score_'+test_score+'.jpg'
        print(f'frame is saved at save_image_paths: {save_image_paths}')
        cv2.imwrite(save_image_paths, image_np)
        return image_np
    
    if save == 1:
        #for presentation uses, save frames to video
        print('saving video')
        out.write(image_np)


def get_videos():
    '''
    read video and creat output video holder
    '''
    # get video
    cap = cv2.VideoCapture('./Trailing3.MKV')
    # setup output
    out = cv2.VideoWriter('crack_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (IMAGE_WIDTH, IMAGE_HEIGHT))  
    return cap, out
