import gzip

import h5py
import numpy as np
import json
import os


current_path = os.getcwd()
path_to_h5_file   = 'download/VQA_image_features.h5'
path_to_json_file = 'download/VQA_img_features2id.json'

with open(current_path +'/data/imgid2imginfo.json', 'r') as file:
    imgid2info = json.load(file)

# load image features from hdf5 file and convert it to numpy array
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

# load mapping file
with open(path_to_json_file, 'r') as f:
     visual_feat_mapping = json.load(f)['VQA_imgid2id']

with gzip.GzipFile('data/vqa_annotatons_test.gzip', 'r') as file:
    annotations = json.loads(file.read())

with gzip.GzipFile('data/vqa_questions_test.gzip', 'r') as file:
    questions = json.loads(file.read())


for i in range(len(questions['questions'])):
    print(questions['questions'][i]['question'])
    for answer in annotations['annotations'][i]['answers'] :
        print("answer", answer['answer'], "confidence", answer['answer_confidence'])
    img_id = questions['questions'][i]['image_id']
    h5_id = visual_feat_mapping[str(img_id)]
    img_feat = img_features[h5_id]
    print(img_feat)
    print(imgid2info[str(img_id)])
    print("**************************")
