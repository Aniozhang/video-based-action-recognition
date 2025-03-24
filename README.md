# video-based-action-recognition
Build a deep learning model that classifies human actions (e.g., walking, jumping, waving) from video sequences using a combination of Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal sequence modeling.


## Preprocess the dataset

    1. Load videos and extract frames. 

    2. Normalize frames and convert them into sequences of fixed length (e.g., 16 frames per clip).

## CNN Feature Extraction

    1. Use a pre-trained CNN to extract features for each frame.

## LSTM Model for Temporal Dependency

    1. Build an LSTM network to process the CNN features sequentially.

## Training & Evaluation

    1. Train the model on labeled action data.

    2. Evaluate on test videos and analyze accuracy.
