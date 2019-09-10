####

# edicted by Huangdebo
# test the model using pb file

# ***

import numpy as np
import tensorflow as tf
from tensorflow.gfile import GFile
import cv2 as cv

from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

# The file path to save the data
save_file = 'xxx.pb'

num_class = 3
classes = ['bicycle', 'car', 'person']
color_table = get_color_table(num_class)


sess = tf.Session()
with GFile(save_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # import the net
    
 
# initializer 
sess.run(tf.global_variables_initializer())
 
 
# input
image = sess.graph.get_tensor_by_name('inputs:0')
phase_train = sess.graph.get_tensor_by_name('phase_train:0')
boxes = sess.graph.get_tensor_by_name('boxes:0')
confs = sess.graph.get_tensor_by_name('confs:0')
probs = sess.graph.get_tensor_by_name('probs:0')

pred_scores = confs * probs
boxes, scores, labels = gpu_nms(boxes, pred_scores, 3, max_boxes=50, score_thresh=0.4, iou_thresh=0.5)

img_ori = cv.imread('./data/demo_data/1.jpg')
height_ori, width_ori = img_ori.shape[:2]
size=[416, 416]
im = cv.resize(img_ori, size)
 
boxes_, scores_, labels_  = sess.run([boxes, confs, probs],  feed_dict={phase_train:False, image: im})


# rescale the coordinates to the original image
boxes_[:, 0] *= (width_ori/float(size[0]))
boxes_[:, 2] *= (width_ori/float(size[0]))
boxes_[:, 1] *= (height_ori/float(size[1]))
boxes_[:, 3] *= (height_ori/float(size[1]))

print("box coords:")
print(boxes_)
print('*' * 30)
print("scores:")
print(scores_)
print('*' * 30)
print("labels:")
print(labels_)

for i in range(len(boxes_)):
    x0, y0, x1, y1 = boxes_[i]
    plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
cv.imshow('Detection result', img_ori)
cv.imwrite('detection_result.jpg', img_ori)
cv.waitKey(0)



