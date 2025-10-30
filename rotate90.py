import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from glob import glob
import cv2
from natsort import natsorted


src_dir = 'example/clip-002354/'
dst_dir = 'example/clip-002354-rot90/'
img_files = natsorted(glob(osp.join(src_dir, '*.jpg')))
for img_file in tqdm(img_files):
    image = cv2.imread(img_file)
    image = np.rot90(image, k=3)
    dst_file = osp.join(dst_dir, osp.basename(img_file))
    os.makedirs(osp.dirname(dst_file), exist_ok=True)
    cv2.imwrite(dst_file, image)