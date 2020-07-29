import os
import sys
import glob

root = '/FULL/PATH/TO/DATA/ROOT/'

dirs_collect = glob.glob(root+'*')

count = 0
WHOLE_SPLIT = 10
TRAIN_SPLIT = 8
TEST_SPLIT = WHOLE_SPLIT - TRAIN_SPLIT

train_file = open('train.txt', 'w')
test_file = open('test.txt', 'w')
for ges in dirs_collect:
    print(ges)
    label = ges.split('/')[-1]
    images_list = glob.glob(ges+'/*')
    # print(images_list)
    for img_index in images_list:
        count = count + 1
        if count <= TRAIN_SPLIT:
            # WRITE TRIAN
            train_file.write(img_index+','+label+'\n')
        elif count <= WHOLE_SPLIT:
            # WRITE TEST
            test_file.write(img_index+','+label+'\n')
        if count == WHOLE_SPLIT:
            count = 0
train_file.close()
test_file.close()
