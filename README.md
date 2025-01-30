**Week-1**
To develop a CNN model to classify images of plastic waste into different categories

**Week 2**
1. Model Architecture Setup:
The CNN model is built using a function called build_model(). The model consists of:
Three Conv2D layers with increasing filters (32, 64, 128) for feature extraction.
MaxPooling2D layers following each convolution to reduce spatial dimensions and retain significant features.
BatchNormalization layers after each convolution to stabilize the learning process and speed up convergence.
A Flatten layer to convert the 2D feature maps into a 1D vector.
A Dense layer with 512 units and ReLU activation for further feature extraction.
A Dropout layer with a 50% dropout rate to prevent overfitting.
The final Dense layer with softmax activation for multi-class classification, where the number of output neurons corresponds to the number of classes in the dataset (y_train.shape[1]).
2. Model Compilation:
The model is compiled using the Adam optimizer with a learning rate of 0.0001.
The categorical cross-entropy loss function is used for multi-class classification.
The metric accuracy is selected to evaluate the performance during training.
3. Model Training:
The model is trained using the fit() function with:
Training data (x_train, y_train) and validation data (x_val, y_val).
The training is set to run for 20 epochs with a batch size of 32.
The history of training (loss and accuracy) is recorded in history to track progress.
4. Model Evaluation:
After training, the model is evaluated on the validation data (x_val, y_val) using the evaluate() function.
The final validation loss and validation accuracy are printed.
5. Visualizing Training and Validation Accuracy:
The accuracy of the model during training and validation is visualized:
Training accuracy is plotted using the values from history.history['accuracy'].
Validation accuracy is plotted using history.history['val_accuracy'].
This plot provides insights into the model’s learning process and its ability to generalize to unseen data.
6. Prediction on Test Images:
A function called predict_image() is defined to handle predictions on test images:
The image is read using OpenCV’s cv2.imread() and converted from BGR to RGB.
It is resized to the input size of the model (150x150 pixels) and normalized by dividing pixel values by 255.0.
The image is reshaped to match the model’s expected input format using np.expand_dims().
The model predicts the class using model.predict(), and the predicted label is decoded with encoder.inverse_transform() to map the output to the correct class.
7. Displaying Predictions:
For test images like sample_image1 and sample_image2, the following steps are executed:
The images are read, and their predicted class labels are displayed.
The images are shown using plt.imshow(), and their predicted class is shown in the title using plt.title().
The axis of the images is hidden with plt.axis('off') to focus on the image content.
8. Final Output:
The images along with their predicted classes are displayed to verify the model's performance.
