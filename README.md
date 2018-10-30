# CarND-Term1-P3-BehavioralCloning
Self-Driving Car Engineer Nanodegree Program: Term 1 Project 3

## Introduction

This project employs the end-to-end deep learning technique described and implemented by NVIDIA Inc. (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It is an interesting approach which optimizes various goals such as lane detection, path planning and vehicle control under one roof using real images from on board a vehicle. Hence, we do not require lane detection algorithms created in assignment 1 (although image pre-processing to mark out lanes could be useful), nor do we need conditional algorithms to calculate steering angles under various driving scenarios. We do end up requiring a deep neural network with multiple convolutional and fully connected layers with non-linearity at each stage.

The architecture described by NVIDIA was a good suggested starting point. Initially, to get my model.py code up and running, I chose to use the LeNet architecture because I was familiar with its layers and input/output shapes. I developed both architectures in parallel for a while. I also began with the sample data provided and later collected new data. The sample data had 8,036 sets of center/left/right camera images, making a total of 24,108 images.

![Left camera image](/readme_images/intro_left.jpg)    ![Center camera image](/readme_images/intro_center.jpg)    ![Right camera image](/readme_images/intro_right.jpg)

     Left camera image	     Center camera image	  Right camera image

## Keras models

The model.py file contains my successful attempt with the NVIDIA architecture although I iterated with two Keras models, LeNet and NVIDIA. While I describe this code, the only difference in the LeNet model code is the different architecture.

I first loaded the driving_log.csv file that referenced the sample images (lines 12 to 17). Each row contained addresses to the images. I then split them into training and validation sets in an 80:20 ratio (lines 19 to 21). I used the generator function provided in the lectures to read each address and load images and corresponding steering angles in batches of 32 (lines 23 to 72). This function was also used to load left and right images for each center image with a steering correction if required (lines 49 to 63). It was also used to generate additional flipped center images if required (lines 43 to 47). I did not find any change in processing time by increasing the batch size hence maintained it throughout. I then set up my model architecture (lines 74 to 95) and compiled it to minimize the mean squared error loss between actual and predicted steering angles using an Adam optimizer (lines 97 to 98). I created the history_object parameter that would contain the training and validation losses over all epochs for visualization (lines 100 to 104). Finally I saved the model.h5 file to run on the simulator and generate a video (lines 118 to 119) using the video.p file provided without additional changes. The model file was used to drive the car in autonomous mode in the simulator using the drive.py file provided without additional changes.

Preceding both architectures, I used a Cropping2D function to crop out the top 50 and bottom 20 pixel rows from every image. I also used a Lambda function to normalize all image pixels between -0.5 and 0.5 to speed up model convergence.

## Solution Design Approach

I employed an iterative approach to arrive at the Keras model described above.

### LeNet architecture experimentation

My first iteration used just the center camera images with cropping and training over 5 epochs. This model scraped along the right wall over the bridge but corrected itself off the bridge to complete a full lap. Further modifications involving additional training images just for the bridge section, Dense layers of dimensions greater and smaller than 100 units, and use of left and right images failed to complete the lap even though the car crossed the bridge without incidents. The LeNet models were taking around 6 min per epoch to train with 8,036 center images. In contrast, the NVIDIA models would need just over 2 min per epoch under identical inputs. I then switched to using the NVIDIA architecture for quicker feedback on my changes.

### NVIDIA architecture experimentation

My first iteration used just the center camera images with cropping and training over 5 epochs. This model failed at the first sharp left turn after the bridge where the lane marking on the outside of the turn is missing. Presumably the model failed to turn hard enough thinking it is fine to keep going where no lane lines were present. The model thus failed to distinguish the edge of the road purely based on differences in surface color.

My first change was to introduce more images in the training stage by flipping all center images and inverting their corresponding steering angles. The assumption was that less data did not train the model hard enough. This iteration failed even before the bridge, proving the assumption wrong.

The first major improvement came in the form of a dropout layer with probability of 0.5 after all convolution layers. The model managed to perform well until the only right turn on the circuit. Thus it seemed to be overfitting previously. Thereafter I trained with all center images as is as well as flipped hoping that the additional right turn samples would help. This iteration failed again before the bridge, indicating that flipping images confused the model during the long left turn before the bridge. Long turns in either direction may have become its weakness. Adding ReLU activations to all dense layers as well as adding an identical dropout layer after the final dense layer Dense(10) did not help the car progress, indicating that the model had enough non-linearity baked into it through the convolution layers with ReLU activations. MSE training and validation losses reduced further with more epochs but also did not help. As didn’t adding a max pooling layer after all convolution layers instead of the dropout layer explained above.

In another attempt to increase the training data set size, I included the left and right camera images with a 0.2 (5 deg) steering correction factor based on the most successful iteration which used all 8,036 center camera images and a Dropout(0.5) layer. The model failed at the end of the bridge but both MSE plots seemed to be decreasing fast which indicated a need to increase number of epochs. But in the interest of saving processing time, I chose the first 3000 center, left and right image sets with 7 epochs. This model turned better until the first right turn where it did not turn sharp enough and went off but was the most progress so far. Increasing epoch size lowered both MSEs but somehow didn’t improve model performance.

In an effort to improve feature detection, I added a Dense(1500) layer after all convolution and dropout layers and replaced the default Dense(100) layer with a Dense(500) layer. This resulted in the model failing before the bridge. To enable sharper turns, especially at the right turn, I increased the correction factor to 0.25 (6.25 deg). This model finished a lap while crossing track bounds before the bridge and after the right turn on the straight towards the end of the lap. Thus the turning issues were sorted but the now car seemed to struggle on straights. Hence, I tried to fine tune the dropout layer by using probabilities of 0.75 and 0.25. The latter performed slightly better in terms of car stability but failed in the same way as the Dropout(0.5) model. Increasing epochs to 15 with this model lowered MSEs to the lowest observed so far. But the model seemed to overfit because the car because quite wobbly on the straights and failed in the same way as before.

At this point, I had gained sufficient trust in the default NVIDIA architecture with a dropout layer to start suspecting the quality of my sample data. Hence, I recorded 10,185 sets of center, left and right images (total 30,555 images) that included
- One lap each in the clockwise and counterclockwise directions while driving along the center of the road
- Recovery driving wherein I made sharp turns from the edge of the road

Recovery driving example:
     
![Left camera image](/readme_images/recovery_left.jpg)    ![Center camera image](/readme_images/recovery_center.jpg)    ![Right camera image](/readme_images/recovery_right.jpg)

     Left camera image	     Center camera image	  Right camera image

ReLU activation had to be added to all but the last dense layer to help the model complete a lap. But the car left track bounds on straights, long and sharp turns. It seemed to do great at correcting itself but not at avoiding recovery situations in the first place. That indicated too much recovery data or too less center driving data. Hence, my plan was to collect/ remove data and try an identical training iteration.

Before that, I came across a forum post (https://discussions.udacity.com/t/behavioral-cloning-non-spoiler-hints/233194) suggesting color scheme manipulation. The NVIDIA architecture was training using images in the YUV scheme which made me think if it was optimized for that particular scheme. I reverted back to the original sample data and chose the first 3000 image sets to begin with. A BGR to YUV conversion sent the model back to failing to finish a single lap. But a BGR to RGB conversion enabled the model to complete a full lap with a minor touch of the right lane on exiting the bridge. Using the full sample set enabled the model to complete two full identical laps without incident. The reason for such an immediate improvement was that the demo in the autonomous mode with the .h5 file did not convert images to the BGR color scheme for which the model was trained. Instead of doing so, it was easier to train the model for an RGB color scheme.

### Final mode architecture

Layer |	Input shape |	Output shape
----- | ----------- | ------------
Cropping | 160x320x3 |90x320x3
Normalization | 90x320x3 | 90x320x3
Convolution 1 | 90x320x3 | 43x158x24
Convolution 2 | 43x158x24 | 20x77x36
Convolution 3 | 20x77x36 | 8x37x48
Convolution 4 | 8x37x48 | 6x35x64
Convolution 5 | 6x35x64 | 4x33x64
Dropout | 4x33x64 | 4x33x64
Flatten | 4x33x64 | 8448x1
Dense 1 | 8448x1 | 100x1
Dense 2 | 100x1 | 50x1
Dense 3 | 50x1 | 10x1
Dense 4 | 10x1 | 1x1

## Conclusion

- In hindsight, changing the color scheme from BGR to RGB should have been one of my earlier attempts. There were signs that the car model struggled in regions where road color changed.
- Dropout layer helped reduce overfitting indicated during some iterations where the validation MSE was lower than the training MSE.
- Using left and right images additionally with a decent correction factor of 0.25 (6.25 deg) helped the car maintain a center location as well as make sharp turns (up to 13 deg) with ease.
- Number of epochs reduced training and validation MSEs but not model performance. Multiple such iteration results showed less correlation between MSE and model performance. Hence, 5-7 epochs seemed enough for this training exercise.
