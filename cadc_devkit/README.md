# cadcd_devkit
A devkit for the Canadian Adverse Driving Conditions dataset.

## download_cadcd.py
This will download all raw or labeled data into the given folder.

This dataset is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

## Transform naming convention
The scripts in this repository as well as the calibration files for the dataset follow this naming convention for transforms:

```
T_FRAME1_FRAME2
```
This could be described as a transform from FRAME2 to FRAME1.
For example, if you wanted to project lidar points onto an image and had the transforms `T_IMG_CAM` and `T_CAM_LIDAR` then this would be written:

```
T_IMG_LIDAR = T_IMG_CAM * T_CAM_LIDAR
```

## run_demo_vehicle_path.py
This script loads all GPS  messages in a drive, converts them to an ENU frame with the origin at the first message and plots each message as an axis frame.

![Vehicle path demo](images/2019_02_27_0027_vehicle_path.png) <!-- .element height="50%" width="50%" -->

## run_demo_tracklets.py
This script loads a camera image and the corresponding 3D annotation file in a drive, loads the calibration data, then creates and projects each cuboid within the frame onto the camera image.

![Vehicle path demo](images/2019_02_27_0033_tracklets_frame_12.png)

## run_demo_lidar.py
This script loads a camera image and the corresponding lidar file in a drive, loads the calibration data, then projects each lidar point onto the camera image. Point color is scaled by depth.

![Vehicle path demo](images/2019_02_27_0027_lidar_frame_90.png)

## run_demo_lidar_bev.py
This script loads lidar data and the corresponding 3D annotation file in a drive, then creates a birds eye view of the lidar point cloud with the cuboid boxes overlaid. Script created by [asvath](https://github.com/asvath) and also located [here](https://github.com/asvath/cadcd/blob/master/birds_eye_view.py).

![Vehicle path demo](images/2019_02_27_0033_lidar_bev_frame_12_cropped.png)

## OpenPCDet data loader
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet) is an open source project with multiple architectures implemented for lidar based 3D object detection and support of several different datasets. A CADC dataset loader has been implemented on the forked [cadc support branch](https://github.com/mpitropov/OpenPCDet/tree/cadc_support) with the [Getting started](https://github.com/mpitropov/OpenPCDet/blob/cadc_support/docs/GETTING_STARTED.md#cadc-dataset) document updated for the cadc dataset. Any issues with the data loader should be opened within this repository. As of right now there is only one difficulty level for test results.

## Additional scripts
Please view [asvath's](https://github.com/asvath) [cadcd](https://github.com/asvath/cadcd) repository.

## Citation
    @article{pitropov2021canadian,
        title={Canadian adverse driving conditions dataset},
        author={Pitropov, Matthew and Garcia, Danson Evan and Rebello, Jason and Smart, Michael and Wang, Carlos and Czarnecki, Krzysztof and Waslander, Steven},
        journal={The International Journal of Robotics Research},
        volume={40},
        number={4-5},
        pages={681--690},
        year={2021},
        publisher={SAGE Publications Sage UK: London, England}
    }
