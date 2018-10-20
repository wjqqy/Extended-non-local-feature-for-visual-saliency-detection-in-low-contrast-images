import cv2
import numpy as np
import NLDF2
import os
import sys
import tensorflow as tf
import time
import vgg16


def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = '/home/zd/wangjie1/NLDF-master/dataset/test1'
    elif dataset == 'HKU-IS':
        path = 'dataset/HKU-IS/imgs'
    elif dataset == 'DUT-OMRON':
        path = 'dataset/DUT-OMRON/DUT-OMRON-image'
    elif dataset == 'PASCAL-S':
        path = 'dataset/PASCAL-S/pascal'
    elif dataset == 'SOD': 
        path = 'dataset/SOD/images'
    elif dataset == 'ECSSD':
        path = 'dataset/ECSSD/images'

    imgs = os.listdir(path)
   
    return path, imgs


if __name__ == "__main__":
    
    num=0
    
    model = NLDF2.Model()
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = NLDF2.img_size
    label_size =np.int( NLDF2.label_size)
    print(123)
    ckpt = tf.train.get_checkpoint_state('')
   
    print(ckpt.model_checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    datasets = ['MSRA-B']
  #['MSRA-B', 'HKU-IS', 'DUT-OMRON','PASCAL-S', 'ECSSD', 'SOD']
    if not os.path.exists('Result3'):
        os.mkdir('Result3')
    
    for dataset in datasets:
        path, imgs = load_img_list(dataset)
       
           
        save_dir = 'Result3/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = 'Result3/' + dataset + '/NLDF'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
   
        for f_img in imgs:

            img = cv2.imread(os.path.join(path, f_img))
            print(path)
            img_name, ext = os.path.splitext(f_img)
                   
            if img is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))
 
                start_time = time.time()
                result = (sess.run(model.Prob,
                                  feed_dict={
                                          model.input_holder: img                                           
                                            }))#  model.keep_prob: 1
              
                print(result.shape,label_size,label_size)
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result,(label_size, label_size, 2))
                result = result[:,:,0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                save_name = os.path.join(save_dir, img_name+'_NLDF.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()
