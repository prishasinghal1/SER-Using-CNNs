# SER-Using-CNNs

Name : Prisha Singhal
ID : 2024A7PS0515P

This project implements a Speech Emotion Recognition (SER) system using log-Mel spectrograms and a 2D Convolutional Neural Network trained on the RAVDESS dataset.

Dataset:
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Emotions:
Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

Each audio sample is approximately 3 seconds long and recorded by professional actors.

Preprocessing:
• Silence trimming, set top_db=25 after trying multiple different values and seeing that the audio was either getting too trimmed (at around 20 db), or silence was remaining (at around 30 db).
• Conversion to log-Mel spectrograms with 128 Mel bands, a 2048-point FFT window, and a hop length of 512 samples and max frames 128 to normalise frame size and pad all audios to make sure number of frames was the same in each.
• Compared angry and sad mel-spectrograms using visual analysis.
• Added noise, pitch shift, and time stretch for augmenting the data.

MODEL:
• Performed a stratified split (80% Train, 10% Val, 10% Test)
• The model is a 2D CNN with three convolutional blocks (32, 64, and 128 filters) followed by batch normalization, max pooling, and dropout. Global average pooling and a dense layer precede a softmax output for 8 emotion classes. Training was performed using the Adam optimizer (learning rate 0.001) and sparse categorical cross-entropy loss.
• Used 30 epochs with an early stop if needed, and a batch size of 32.
• Experimented a lot with the dropout rates for each layer, number of epochs, and batch sizes. Also with the learning rate of the Adam optimizer. 
• I was trying another model with a learning rate of 0.0005 that seemed to be performing better but stopped mid-way because I exceeded the Colab limit. I think that would have been my best performing model going by however much it ran, but as I said, only a very few epochs ran so I can't show the results for that yet.

Evaluation and confusion matrix:
• Macro F1-score: 0.35305887392101576
• Confusion matrix: <img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/e39a3f91-94d6-44a8-b637-b78cdb59e63f" />
• Accuracy and loss: <img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/f6eb9c5b-c830-489a-834f-41766bdf765a" />
• Male Macro F1: 0.348087608432436 and Female Macro F1: 0.3444964043309632

Files:
• AI Club Task (1).ipynb – Main notebook containing EDA, preprocessing,
  model training, and evaluation.
• ser_cnn_model.h5 (1) – Saved weights of the trained CNN model.
• predict.py (1) – Script for running live emotion prediction on unseen .wav files.
• README.md – Project overview and usage instructions.


