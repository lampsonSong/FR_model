import tensorflow as tf
import cv2
import numpy as np

from center_face import CenterFace

import sys
from sys import argv

if len(argv) < 3:
    print("Usage: python demo.py img1_path img2_path")
    sys.exit(0)

#### Load Model ####
model_name = "./FR_small_FBN.pb"

output_graph_def = tf.GraphDef()
with open(model_name,"rb") as f:
  output_graph_def.ParseFromString(f.read())
  _ = tf.import_graph_def(output_graph_def, name="")

input_node_str = "input0:0"
output_node_str = "1238_classifier.0/BiasAdd:0"
#### END Load Model ####


def get_face_and_landmarks(img):
    detector = CenterFace()
    faces, lms = detector(img, img.shape[0], img.shape[1], threshold=0.25)
    
    if len(faces) > 1:
        print("Image should contain only ONE face!")
        sys.exit(0)
    
    return faces[0], lms[0]


def alignFace(landmarks):
    srcPts = np.array([[landmarks[0], landmarks[1]], [landmarks[2], landmarks[3]], [(landmarks[6] + landmarks[8])/2., (landmarks[7] + landmarks[9])/2.] ])
    dstPts = np.array([[70.7450, 112.0], [108.2370, 112.0], [89.4324, 153.5140]])

    assert(dstPts.shape == srcPts.shape)
    A = srcPts
    B = dstPts

    A_rows = A.shape[0]
    # rigid transform
    meanA = A.mean(0) # row mean
    meanB = B.mean(0) # row mean

    AA = A - meanA
    BB = B - meanB

    H = np.dot(AA.transpose() , BB) / A_rows
    U, S, V_t = np.linalg.svd(H)

    if cv2.determinant(U) * cv2.determinant(V_t) < 0:
        S[S.shape[0]-1, :] *= -1
        U[:, U.shape[1]-1] *= -1
   
    R = np.dot(U , V_t)
    varP = 0.
    for i in range(0, srcPts.shape[1]):
        mean, var = cv2.meanStdDev(A[:, i])
        varP += np.sum(var * var)
    
    scale = 1. / varP * np.sum(S)
    T = meanB - np.dot(meanA , np.dot(scale , R))
    T = np.expand_dims(T, axis=0)

    # continue
    R = R * scale

    diff = np.dot(A, R) + T - B

    trans = np.hstack((R.transpose(), T.transpose()))

    return trans

def preprocess(img, trans):
    # warp
    warped_img = cv2.warpAffine(img, trans, (178, 218))

    crop_size = 110
    crop_center_y_offset = 15
    final_size = 112

    ct_x = warped_img.shape[1] / 2.0
    ct_y = warped_img.shape[0] / 2.0 + crop_center_y_offset

    l = int(ct_x - crop_size / 2.)
    t = int(ct_y - crop_size / 2.) 
    input_data = warped_img[t:t+crop_size, l:l+crop_size]
    input_data = cv2.resize(input_data, (final_size, final_size))

    input_data = input_data.astype(np.float32)
    input_data = input_data * (3.2 / 255.) - 1.6

    return input_data

def extract_features(input_data):
    with tf.device("/cpu:0"):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config).as_default() as sess:
            node_in = sess.graph.get_tensor_by_name(input_node_str)
            model_out = sess.graph.get_tensor_by_name(output_node_str)
            
            feed_dict = {node_in:input_data}
            pred = sess.run(model_out, feed_dict)
    return pred            

def cal_similarity(v1, v2):
    product_v = np.sum(v1 * v2)
    v1_len = np.sum(v1 * v1)
    v2_len = np.sum(v2 * v2)

    vcos = product_v / np.sqrt(v1_len * v2_len)
    score = 1.0145 / (1 + np.exp(6.5 * (0.35 - vcos)))

    return score

# read images
# img 1
img1 = cv2.imread(argv[1])
face1, lm1 = get_face_and_landmarks(img1)

x, y, x1, y1 = face1[:4].astype(np.int32)

cv2.rectangle(img1, (x,y), (x1,y1), (255,0,0), 2)
for idx in range(0, len(lm1), 2):
    cv2.circle(img1, (lm1[idx], lm1[idx+1]), 1, (255,0,0), 8)


input_data1 = preprocess(img1, alignFace(lm1))
input_data1 = np.expand_dims(input_data1, 0)

feature1 = extract_features(input_data1)

# img 2
img2 = cv2.imread(argv[2])
face2, lm2 = get_face_and_landmarks(img2)

x, y, x1, y1 = face2[:4].astype(np.int32)

cv2.rectangle(img2, (x,y), (x1,y1), (255,0,0), 2)
for idx in range(0, len(lm2), 2):
    cv2.circle(img2, (lm2[idx], lm2[idx+1]), 1, (255,0,0), 8)


input_data2 = preprocess(img2, alignFace(lm2))
input_data2 = np.expand_dims(input_data2, 0)

feature2 = extract_features(input_data2)

score = cal_similarity(feature1, feature2)
print("--similarity score : ", score)
#cv2.imshow('frame', img2)
#cv2.waitKey(0)
