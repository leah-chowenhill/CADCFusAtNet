# Import relevant libraries

import scipy.io as sio
import numpy as np
import tqdm
import gc

# The code takes the entire hsi/lidar image as input for 'X' and grounttruth file as input for 'y'
# and the patchsize as for 'windowSize'.
# The output are the patches centered around the groundtruth pixel, the corresponding groundtruth label and the
# pixel location of the patch.

def make_patches(X, y, windowSize):

  print('patches function called')
  shapeX = np.shape(X)

  margin = int((windowSize-1)/2)
  newX = np.zeros([shapeX[0]+2*margin,shapeX[1]+2*margin,shapeX[2]])

  newX[margin:shapeX[0]+margin:,margin:shapeX[1]+margin,:] = X

  index = np.empty([0,3], dtype = 'int')

  #cou = 0
  #dou = 0
  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    gc.collect()
    for i in range(shapeX[0]):
      for j in range(shapeX[1]):
        if y[i,j].any() == k:
          index = np.append(index,np.expand_dims(np.array([k,i,j]),0),0)
          #print(np.size(np.unique(y)), shapeX[0], shapeX[1])
          #print(cou)
          #cou += 1
      #print(dou)
      #dou += 1
  print('Index Done!')
  np.save('/content/drive/MyDrive/Perception_in_Snow', index)
  patchesX = np.empty([index.shape[0],2*margin+1,2*margin+1,shapeX[2]], dtype = 'float32')
  patchesY = np.empty([index.shape[0]],dtype = 'uint8')

  #print('Generating patches...')
  for i in range(index.shape[0]):
    #print('Iter: ', i)
    p = index[i,1]
    q = index[i,2]
    patchesX[i,:,:,:] = newX[p:p+windowSize,q:q+windowSize,:]
    #print('X complete!')
    patchesY[i] = index[i,0]
  #print('Patches Done!')

  del shapeX, margin, newX, p, q
  return patchesX, patchesY, index

# Reading data
with open('/content/drive/MyDrive/Perception_in_Snow/train_rgb.npy', 'rb') as f:
    train_rgb = np.load(f)
    f.close()
with open('/content/drive/MyDrive/Perception_in_Snow/train_lidar.npy', 'rb') as f:
    train_lidar = np.load(f)
    f.close()
with open('/content/drive/MyDrive/Perception_in_Snow/test_rgb.npy', 'rb') as f:
    test_rgb = np.load(f)
    f.close()
with open('/content/drive/MyDrive/Perception_in_Snow/test_lidar.npy', 'rb') as f:
    test_lidar = np.load(f)
    f.close()

with open('/content/drive/MyDrive/Perception_in_Snow/test_groundtruth.npy', 'rb') as t:
    test_groundtruth = np.load(t)
    #print("imported test groundtruth:", test_groundtruth.shape)
    t.close()
with open('/content/drive/MyDrive/Perception_in_Snow/train_groundtruth.npy', 'rb') as t:
    train_groundtruth = np.load(t) 
    t.close()  

# Concatenating HSI and LiDAR bands from the data and removing spurious pixels
train_feats = np.concatenate([train_rgb, np.expand_dims(train_lidar, axis = 2)], axis = 2)
test_feats = np.concatenate([test_rgb, np.expand_dims(test_lidar, axis = 2)], axis = 2)

del train_rgb, test_rgb, train_lidar, test_lidar

#print("test_rgb:", test_rgb.shape)
#print("test_lidar:", test_lidar.shape)
#print("test feats:", test_feats.shape)

# Normalising the bands using min-max normalization 
train_cols = 976896
test_cols = 119808

train_feats_norm = np.zeros([207,train_cols,3], dtype = 'float32') #1024?
test_feats_norm = np.zeros([207,test_cols,3], dtype = 'float32') #1024?
for i in tqdm.tqdm(range(3)):
  train_feats_norm[:,:,i] = train_feats[:,:,i]-np.min(train_feats[:,:,i])
  train_feats_norm[:,:,i] = train_feats_norm[:,:,i]/np.max(train_feats_norm[:,:,i])
  test_feats_norm[:,:,i] = test_feats[:,:,i]-np.min(test_feats[:,:,i])
  test_feats_norm[:,:,i] = test_feats_norm[:,:,i]/np.max(test_feats_norm[:,:,i])
#print("train feats norm shape:", train_feats_norm.shape)

del train_feats, test_feats

## Reading train and test groundtruth images
train_zeros = np.zeros((207, train_cols-train_groundtruth.shape[1], 4))
train_groundtruth = np.concatenate((train_groundtruth,train_zeros), axis=1)

train = train_groundtruth[:,:,:3]
test = test_groundtruth[:,:,:3]

print('groundtruth sets imported')
del train_groundtruth, test_groundtruth, train_zeros
gc.collect()

# Create train patches
train_batch_size = 5088
train_num_iter = int(train_cols / train_batch_size)

train_patches = np.empty([129868, 11, 11, 3])
train_labels = np.empty([129868,])
index_train = np.empty([129868, 3])

print("train patch iteration:")
for i in range(train_num_iter):
  print(i)
  gc.collect()
  start = i*train_batch_size
  end = (i+1)*train_batch_size
  tp, tl, it = make_patches(train_feats_norm[:,start:end,:], train[:,start:end,:], 11)
  #print("tp shape:", tp.shape)
  #print("tl shape:", tl.shape)
  #print("it shape:", it.shape)
  train_patches = np.vstack((train_patches, np.resize(tp,(129868, 11, 11, 3))))
  train_labels = np.vstack((train_labels, np.resize(tl,(129868,))))
  index_train = np.vstack((index_train, np.resize(it,(129868,3))))

  del tp, tl, it

  with open('/content/drive/MyDrive/Perception_in_Snow/train_patches.npy', 'wb') as f:
    np.save(f, train_patches)
    f.close()

  with open('/content/drive/MyDrive/Perception_in_Snow/train_labels.npy', 'wb') as f:
    np.save(f, train_labels)
    f.close()

  with open('/content/drive/MyDrive/Perception_in_Snow/index_train.npy', 'wb') as f:
    np.save(f, index_train)
    f.close()

# Create test patches
gc.collect()

test_batch_size = 3744
test_num_iter = int(test_cols / test_batch_size)

test_patches = np.empty([146953, 11, 11, 3])
test_labels = np.empty([146953,])
index_test = np.empty([146953, 3])

print("test patch iteration:")
for i in range(test_num_iter):
  print(i)
  gc.collect()
  start = i*test_batch_size
  end = (i+1)*test_batch_size
  tsp, tsl, its = make_patches(test_feats_norm[:,start:end,:], test[:,start:end,:], 11)
  #print("tsp shape:", tsp.shape)
  #print("tsl shape:", tsl.shape)
  #print("its shape:", its.shape)
  test_patches = np.vstack((test_patches,np.resize(tsp,(146953, 11, 11, 3))))
  test_labels = np.vstack((test_labels,np.resize(tsl,(146953,))))
  index_test = np.vstack((index_test, np.resize(its,(46953, 3))))

  del tsp, tsl, its
  with open('/content/drive/MyDrive/Perception_in_Snow/test_patches.npy', 'wb') as f:
    np.save(f, test_patches)
    f.close()

  with open('/content/drive/MyDrive/Perception_in_Snow/test_labels.npy', 'wb') as f:
    np.save(f, test_labels)
    f.close()

  with open('/content/drive/MyDrive/Perception_in_Snow/index_test.npy', 'wb') as f:
    np.save(f, index_test)
    f.close()