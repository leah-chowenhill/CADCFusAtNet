import scipy.io as sio
import numpy as np
import tqdm
import gc

# Reading data

with open('/content/drive/MyDrive/Perception_in_Snow/test_patches.npy', 'wb') as f:
    np.save(f, test_patches)
    f.close()

with open('/content/drive/MyDrive/Perception_in_Snow/test_labels.npy', 'wb') as f:
    np.save(f, test_labels)
    f.close()

with open('/content/drive/MyDrive/Perception_in_Snow/index_test.npy', 'wb') as f:
    np.save(f, index_test)
    f.close()

# Data augmentation by rotating patches by 90, 180 and 270 degrees

tr90 = np.empty([2832,11,11,3], dtype = 'float32')
tr180 = np.empty([2832,11,11,3], dtype = 'float32')
tr270 = np.empty([2832,11,11,3], dtype = 'float32')

for i in tqdm.tqdm(range(2832)):
  tr90[i,:,:,:] = np.rot90(train_patches[i,:,:,:])
  tr180[i,:,:,:] = np.rot90(tr90[i,:,:,:])
  tr270[i,:,:,:] = np.rot90(tr180[i,:,:,:])

train_patches = np.concatenate([train_patches, tr90, tr180, tr270], axis = 0)
train_labels = np.concatenate([train_labels,train_labels,train_labels,train_labels], axis = 0)

# Save the train patches/ test patches along with the labels

np.save('/content/drive/MyDrive/Perception_in_Snow/CADCFusAtNet/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-/data/train_patches',train_patches)
np.save('/content/drive/MyDrive/Perception_in_Snow/CADCFusAtNet/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-/data/test_patches',test_patches)
np.save('/content/drive/MyDrive/Perception_in_Snow/CADCFusAtNet/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-/data/train_labels',train_labels)
np.save('/content/drive/MyDrive/Perception_in_Snow/CADCFusAtNet/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-/data/test_labels',test_labels)
print('Done!')

# Save the normalised HSI and LiDAR images

np.save('/content/drive/MyDrive/Perception_in_Snow/CADCFusAtNet/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-/data/train/rgb',train_feats_norm[:,:,0:3])
np.save('/content/drive/MyDrive/Perception_in_Snow/CADCFusAtNet/FusAtNet-Dual-Attention-based-SpectroSpatial-Multimodal-Fusion-Network-for-Hyperspectral-and-LiDAR-/data/train/lidar',train_feats_norm[:,:,3])
print('Done!')