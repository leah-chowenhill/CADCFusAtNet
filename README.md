# CADCFusAtNet
dual cross attention RGB image and LiDAR sensor fusion model for AV perception in snow 

# To Load Data
1) Generate a concatenated matrix of RGB images and LiDAR data, as well as the corresponding labels by running the data_load.py
2) Generate patches by running the data_prepare.py
3) Augment the patches by running the data_augmentation.py
4) Update file paths in model.py according to the saved locations of the patch sets and groundtruth labels

# To Run Model
1) Confirm that file path locations in model.py are correct
2) Run model.py or run Colab notebook
3) Code to visualize model predictions per sample image available in last cell of Colab notebook
