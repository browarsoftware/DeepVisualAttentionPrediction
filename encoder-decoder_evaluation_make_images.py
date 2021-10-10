"""
Author: Tomasz Hachaj, 2021
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/
Data source:

https://drive.google.com/file/d/13VIyqFNzQ6zIGmWll9tEHjOdXp5R2GZt/view
https://drive.google.com/file/d/1U8bwYA8PgNuNYQnv5TNtR2az3AleyrEZ/view
https://drive.google.com/file/d/1h5udf2tB64q6-N3lEh0vDhvfIyDOD43N/view
"""

import cv2
import numpy as np
import os

stimulus = ['ALG_1_v1_Page_1.jpg', 'ALG_1_v2_Page_1.jpg', 'ALG_2_v1_Page_1.jpg', 'ALG_2_v2_Page_1.jpg', 'BIO_Page_1.jpg',
    'FIZ_WB1_Page_1.jpg', 'FIZ_WB2.jpg', 'FIZ_WB3_v1_Page_1.jpg', 'FIZ_WB3_v2_Page_1.jpg', 'FIZ_WB4_stereo_Page_1.jpg',
    'FIZ_WZORY_Page_1.jpg', 'rz 1_Page_1.jpg', 'rz 2_Page_1.jpg', 'rz 3_Page_1.jpg']

#path_help = "_1pyramids"
#path_help = "_2pyramids"
#path_help = "_3pyramids"
#path_help = "_4pyramids"
path_help = "_places_3pyramids"

def make_dir_with_check(my_path):
    try:
        os.mkdir(my_path)
    except OSError:
        print(my_path + ' exists')
    else:
        print(my_path + ' created')

make_dir_with_check('res_results' + path_help)
for stim_name in stimulus:
    make_dir_with_check('res_results' + path_help + '/' + stim_name)


path = 'res' +path_help + '\\'
path_stimulus = 'data\\'
sample_id = 0
stimulus_id = 0

my_size = (512, 512)
for stimulus_id in range(len(stimulus)):
    dir_data = os.listdir(path +  stimulus[stimulus_id])
    print(int(len(dir_data) / 2))
    print(stimulus[stimulus_id])
    for sample_id in (range(int(len(dir_data) / 2))):
        print(sample_id)
        my_path_original = path +  stimulus[stimulus_id] + '\\original' + str(sample_id) + '.png'
        my_path_recon = path +  stimulus[stimulus_id] + '\\recon' + str(sample_id) + '.png'
        my_path_stimulus = path_stimulus + stimulus[stimulus_id]
        my_stimulus = cv2.resize(cv2.imread(my_path_stimulus), my_size)

        try:
            if os.path.exists(my_path_original) and os.path.exists(my_path_recon):
                img_original = cv2.resize(cv2.imread(my_path_original), my_size)
                img_recon = cv2.resize(cv2.imread(my_path_recon), my_size)

                my_stimulus_1 = np.copy(my_stimulus)
                my_stimulus_1 = my_stimulus_1 / 3
                my_stimulus_2 = np.copy(my_stimulus_1)

                for x in range(my_stimulus_1.shape[0]):
                    for y in range(my_stimulus_1.shape[1]):
                        if (img_recon[x,y,0] > 0):
                            if my_stimulus_1[x,y,2] + img_recon[x,y,0] > 255:
                                my_stimulus_1[x, y, 2] = 255
                            else:
                                my_stimulus_1[x, y, 2] = my_stimulus_1[x, y, 2] + img_recon[x, y, 0]

                        if (img_original[x,y,0] > 0):
                            if my_stimulus_2[x,y,2] + img_original[x,y,0] > 255:
                                my_stimulus_2[x, y, 2] = 255
                            else:
                                my_stimulus_2[x, y, 2] = my_stimulus_2[x, y, 2] + img_original[x, y, 0]


                my_stimulus_1 = my_stimulus_1.astype(np.uint8)
                my_stimulus_2 = my_stimulus_2.astype(np.uint8)

                cv2.imwrite('res_results' + path_help + '/' + stimulus[stimulus_id] + '/original' + str(sample_id) + '.png', my_stimulus_2)
                cv2.imwrite('res_results' + path_help + '/' + stimulus[stimulus_id] + '/recon' + str(sample_id) + '.png', my_stimulus_1)
        except OSError:
            print('image exists')
        else:
            print('image exist')
