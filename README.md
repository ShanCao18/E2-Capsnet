# E2-Capsnet
E2-Capsule Neural Networks for Facial Expression Recognition Using AU-Aware Attention

by Shan Cao, Yuqian Yao and Gaoyun An

The paper is under review.

To run the code, the file "shape_predictor_68_face_landmarks.dat" is required. You can download it online.


## The structure of our E2-Capsnet
E2-Capsnet takes a facial image as input and extracts rich feature maps with enhancement module1. Then the feature maps are fed to the capsule layers to be encoded. The three fully connected layers decode the feature maps. Finally, we get the results of facial expression recognition by squashing function. Our E2-Capsnet is trained end-to-end.

![]
(https://github.com/ShanCao18/E2-Capsnet/structure.jpg)
