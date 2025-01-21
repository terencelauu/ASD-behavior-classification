# Classification of Autism Spectrum Disorder (ASD) Using Deep Learning 

Autism Spectrum Disorders (ASD) are neurological disorders characterized by deficits in cognitive skills, social and communicative behaviors. A common way of diagnosing ASD is by studying behavioral cues, such as hand/arm flapping, head banging, and spinning. This project aims to leverage computer vision and machine learning techniques to improve the efficiency and accuracy of diagnosing ASD through behavioral cues.

### Dataset
The SSBD dataset can be downloaded from [Roland Goecke's research page](https://rolandgoecke.net/research/datasets/ssbd/) . The dataset contains about 25 videos of hand flapping, head banging, and spinning behaviors.


### Data Preparation and Preprocessing
1. Data Download: The dataset is downloaded from Google Drive if not already present.
2. Data Extraction: The downloaded zip file is extracted.
3. Frame Extraction: Frames are extracted from videos at regular intervals, resized to 128x128 pixels, normalized to the range [0, 1], and stored.
4. Dataset Creation: A dataset of sequences (each containing 60 frames) is created along with their corresponding labels.
5. Train-Test Split: The dataset is split into training and testing sets.

### Data Preparation and Preprocessing
ConvLSTM2D Model
- Initialization: Load and unzip the dataset, set constants (IMAGE_WIDTH, IMAGE_HEIGHT, SEQUENCE_LENGTH).
- Dataset Preparation: Extract 60 frames from each video, resize them.
- Model Structure:
  - 4 convolutional layers, each followed by MaxPooling3D.
  - Flatten layers and use a dense layer with softmax activation for classification.
  - Use categorical_crossentropy loss function and Adam optimizer.
  - Train for 10 epochs with a 75:25 train-validation split.

LRCN Model
- Initialization: Load and unzip the dataset, set constants (IMAGE_WIDTH, IMAGE_HEIGHT, SEQUENCE_LENGTH).
- Dataset Preparation: Extract 60 frames from each video, resize them.
- Model Structure:
  - 4 convolutional layers followed by MaxPooling2D.
  - Flatten layers, followed by an LSTM layer with 32 units to capture temporal dependencies.
  - Dense layer with softmax activation for multi-class prediction (3 classes).
  - Use categorical_crossentropy loss function and Adam optimizer.
  - Train for 10 epochs with a 75:25 train-validation split.
  
### Results
The models were evaluated on the test set, and performance metrics were calculated to determine the accuracy of classifying the behaviors.
