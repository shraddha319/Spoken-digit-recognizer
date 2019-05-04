Spoken Digit Recognition
S Shraddha [1PE16CS136], shraddha.jain3219@gmail.com, github.com/shraddha319
Nithin M [1PE16CS104], nithinnithi0903@gmail.com, github.com/nithinm1999
Nishita V [1PE16CS102], nishitavaddem@gmail.com, github.com/housestar19
Problem Statement:
In this project, voice input was converted to spectrographic images. These images were fed into a K Nearest Neighbours machine learning algorithm and Logistic Regression algorithm. The purpose of this experiment is to get a sense of how effective combining KNN and Logistic Regression with spectrographic images might be for spoken digit recognition.
Later, we implemented Convolutional Neural Network to get better results compared to the two models considered above using same spectrogram images.
Data Description:
•	Data source: github.com/Jakobovski/free-spoken-digit-dataset [FSDD].
•	4 speakers.
•	2,000 recordings (50 of each digit per speaker).
•	English pronunciations.
Data Pre-processing:
Dataset contains 2000 audio recordings of digits (0-9) in the form of .wav files, these signals are originally in the time domain. Processing signals in the frequency domain is more convenient than that of time domain signals. Tasks such as noise elimination become much easier when the individual frequency components are isolated. 
A spectrogram is a convenient visualization of the frequencies present in an audio clip. Generating one involves obtaining the frequency components of each window of the audio via a Discrete Fourier Transform (DFT) of its waveform. To plot the spectrogram we break the audio signal into millisecond chunks and compute Short Time Fourier Transform (STFT) for each chunk. This time period is represented as a vertical line in the spectrogram.            

Proposed Machine Learning Models:
•	First model: Multiclass Logistic Regression.                                                                                          Multiclass logistic regression is a classification method that generalizes logistic regression to multiclass problems, i.e. with more than two possible discrete outcomes. It is a model that is used to predict the probabilities of the different possible outcomes of a categorically distributed dependent variable (Pixel values), given a set of independent variables (class labels).

•	Second model: K-Nearest Neighbours classifier. 	                 In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbour (which is been implemented). The Minkowski distance is used to calculate the nearest neighbour.	                              

Experiment Results:
 Logistic Regression:
•	Size of training and testing data split:                                                    Training data: 1600 spectrograms of dimensions 64x64.                                Testing data: 400 spectrograms of dimensions 64x64. 
•	Accuracy: On Test data is 88.5 %.
K-Nearest Neighbors: 
•	Size of training and testing data split:                                                    Training data: 1600 spectrograms of dimensions 64x64.                                Testing data: 400 spectrograms of dimensions 64x64. 
•	Best Accuracy (k=1) : 82.25 %

Discussion:
Comparing the above two models we can observe that Logistic Regression model classified our data better when compared to KNN. This is because KNN is simplest classification method and also known as a Lazy Learner which does not require much computation power as it only uses the Minkowski distance as  a parameter to find the nearest neighbour which is not well suited for image classification. Logistic regression on the other hand uses the Sigmoid function to classify based on One versus Rest method (ovr) which produced a better result compared to KNN.

Conclusion:
From the above Discussion we see that KNN and Logistic Regression are not suitable for image classification in terms of accuracy and efficiency.
A better learning algorithm would be CNN as it efficiently classifies images.
Convolutional Neural Networks (Bonus):
The capabilities of CNN for performing Machine Learning on images are well known and explored. Everyday some new research comes up to show an improvement in the algorithm or some new use case for it. It was with this backdrop that we decided to test something other than vision on CNN. We decided to test how CNN works for speech data. 

•	Size of training and testing data.
Training Data: 1600 Images of Spectrograms. Each has a size of 64x64 pixels. Training Data has 1600 labels corresponding to each of the images.                Test Data: 400 Images of Spectrograms. Each has a size of 64x64 pixels.       Test Data has 400 labels corresponding to each of the images.

•	Training and Testing the Network
Training:
10% of the training data is used for validation, i.e The model iterates over 10 epochs and improves its parameters until it gets its highest val_acc i.e. the test accuracy. We get 97% in this case
We make the network go through 10 epochs. Ideally one should have a larger number of epochs and should stop the network when test accuracy stops increasing.

•	Conclusion: Accuracy achieved: 97 %
 
Contribution:
Contribution of each member in the team:
•	Literature Survey and Data collection : Nishitha V
•	Research , data pre-processing : S Shraddha
•	Modelling, Analysis and testing : S Shraddha and Nithin M

References:
•	Audio Classification Using CNN — An Experiment: Medium.
•	Wiki-Pedia.
•	Scikit-Learn official doc.
•	Udemy-DeepLearning Course.






