# Rowing Instructor 

![app example](resources/videos/app_test.gif)

---------------------

This application helps on your indoor rowing training, also called **erging**, because an ergometer is being used (device to measure work performance). There are yearly events where you can compete, for example, in the 2000 metres competition.
Rowing is an effective full body excercise with low impact workout (if used correctly, this app can help you), you will use your full body, lower and upper part. Plus, you will burn calories faster any time of the year, no matter how cold or warm is outside.
For a low cost rowing machine visit (OpenErgo)[https://openergo.webs.com/].
Rowing instructor uses pose estimation and a neural network to detect bad back postures. The former makes sure you have straight arms when performing the stroke (when your legs are rotating). Also, makes sure when pulling you have the legs straight. The latter extracts the back of the individual in all the stroke process to detect bad posture.

## Pose Estimation
Pose estimation will allow us to describe the pose of an individual, by a set of coordinates that are connected. It has a variety of applications such as activity recognition, motion capture and augmented reality, training robots and many more.
We have tested lite movenet model and media pipe. We have decided to use Pose ML solution since has better hands and foot predictions, plus small latency. Movenet only detects 17 points, however, both approximations allow for multi-person detection and use a variation of heatmaps. Movenet through bottom-up, first detecting the objects of the image, then the joints. For that, MobileNetV2 is being used. A variety of steps follow such as detecting the center of the person, then a set of regression outputs are being produced for a "initial" keypoint set. Then a keypoint heatmap is generated together with the previous regressed points to decide on final local offsets, these are the precise locations. For further research, you can use movenet.ipynb notebook as a startting point.

### How pose media pipe works
Pose is a solution of media pipe. Is a two-step detection, first, the person is detected (ROI: Region Of Interest), then we can predict the pose using the COCO keypoint detection task, of persons keypoints. The 33 points of pose are detected using a regression approach over heatmaps (convolution part heatmap regression). Basically a regression encoder-decoder architecture is being used, by convolution blocks followed by deconv blocks. The deconv layers allow higher resolutions. After deconv layers there are **two heads**, classification (17 channels) and regression (2000 channels). The former, exclude points that make no sense, from the ones that make sense, for example, of a shoulder. Once we know the shoulder is being detected we use our regression head, to focus on small range, and detect an specific point. In a nutshell, first a person detector is needed, to reduce the space, then classification over 17 regions is applied, to discriminate over places where no ROI is detected, finally over the positive regions, regression is being applied, to detect the specific location of the keypoint (joint).

### Media pipe to detect bad technique
Rowing is about keeping your arms straight, specifically when catching, driving and recovering, except on the finish. It is also important to stress that on the finish your legs should be straight We use a three joint angle, plus a couple of variables to make decisions:
* test_values
* POSITION
* min_legs_angle_threshold
* min_arms_angle_threshold
* min_detection_confidence
* min_tracking_confidence
* min_visibility_values
* warning_threshold

**POSITION** can take either left or right and depends on the position of the camera with respect to the rowing machine. **test_values** print the angle for legs and arms, so you can pick the proper value on **min_legs_angle_threshold** and **min_arms_angle_threshold**. Should be between 130 and 180. **min_visibility_values** is the confidence (between 0 and 1) for the joints, finally **min_detection_confidence** and **min_tracking_confidence** are the values for pose class.
A warning image would be displayed when rowing has not been done properly plus a beep sound when instantaneously bad rowing has been done for **warning_threshold**.  
You can simply run **python rowing instructor** on your terminal after modifying some of the variables and you are ready to go.

### Transfer learning to detect back problems
You can use the creating_observations notebook that will store image on the img folder. Then you can use either the cnn_training_mobile or cnn_training notebook that will store the model in the models folder. So far we have not tuned the model, and our images base can get bigger. Finally the inference notebook was created to test the model inference on the rowing_instructor script. We make predictions every 10 frames to keep video flowing fast, there are around 20 frames in a second, then we make 2 predictions a second.

## Models tested
So far we have tested **mobilentev2** and **resnet**, but have theoretically good values, however, we are using mobilenetv2 since it resulted faster.

## TODO
Improve the folders structure.

## Learn more
* https://cocodataset.org/#home 
* https://worldrowing.com/events/indoor-events/
* https://fitatmidlife.com/indoor-rowing-exercise/
* https://openergo.webs.com/
* https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
* https://openaccess.thecvf.com/content_CVPRW_2019/papers/Augmented%20Human%20Human-centric%20Understanding%20and%202D-3D%20Synthesis/Zhang_Exploiting_Offset-guided_Network_for_Pose_Estimation_and_Tracking_CVPRW_2019_paper.pdf
* https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
* Alexandra Turner, pinterest
* https://www.youtube.com/watch?v=oP6OR-G7AxM
* https://clideo.com/
