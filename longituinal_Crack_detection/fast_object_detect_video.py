'''
Code to run tensorflow object detection API infrencing
Modified from code in repository
Also see FASTER: https://github.com/tensorflow/models/issues/4355
SEE tensorflow documentation for object detect
'''

import time
import tensorflow as tf
from object_detection.utils import label_map_util
import fast_object_detect_video_lib as FOL

# Path to model graph
PATH_TO_MODEL = './fine_tuned_model/frozen_inference_graph.pb'
# Path to list of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './dataset/pascal_label_map.pbtxt'
# Total number of image classes
NUM_CLASSES = 3
# Path to image directory and to images
TEST_DIR_PATH = './testIMG_bw'
# Lables
LABEL_MAP = label_map_util.load_labelmap(PATH_TO_LABELS)
CATEGORIES = label_map_util.convert_label_map_to_categories(LABEL_MAP,
                                                            max_num_classes=
                                                            NUM_CLASSES,
                                                            use_display_name
                                                            =True)
CATEGORY_INDEX = label_map_util.create_category_index(CATEGORIES)
# Image Frame Size (480x480 default)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
# Debug flag
TIMER = 1
ITIMER = 0


def main():
    '''
    Main Code To Run
    '''
    # setup
    detection_graph = FOL.setupgraph(PATH_TO_MODEL)  
    cap, out = FOL.get_videos()  
    # get graph and start session
    with detection_graph.as_default():
        with tf.Session() as sess:
            # use session to run loop over all images
            startoverall = time.time()
            frames = 115
            for i, image_frame in enumerate(range(frames)):
                # Load Input Data (video or image)... THIS IS A SLOW STEP
                image_np = FOL.read_video(image_frame, cap)
                #image_np = read_single_image(i)
                # inference and check time
                start = time.time()
                output_dict = FOL.run_inference(image_np, sess)
                ttime = time.time()-start
                if ITIMER:
                    print('Finiahed Image: '+str(i)+'... Total time (sec): '+str(round(ttime,3))+'... FPS: '+str(round(1/ttime,3)))
                if True:
                    FOL.visulization(image_np, output_dict, CATEGORY_INDEX, out, i, TEST_DIR_PATH, save=0)
            if TIMER:
                #measure time completed
                endoverall = time.time()-startoverall
                print('Main Loop Average FPS: '+str(round(frames/endoverall,3)))
    # clean up
    cap.release() 
    out.release()


if __name__ == "__main__":
    main()
