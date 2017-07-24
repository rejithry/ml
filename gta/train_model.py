# train_model.py

import sys
sys.path.insert(0, '/Users/raghr010/anaconda/lib/python2.7/site-packages')


import numpy as np
from alexnet import alexnet
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 2
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        #train_data = np.load('training_data-{}-balanced.npy'.format(i))
        train_data = np.load('balanaced_training_data.npy'.format(i))

        train = train_data[:-14]
        test = train_data[-14:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
             show_metric=True, run_id=MODEL_NAME, batch_size=16)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:C:/path/to/log




