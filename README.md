# Audio-Anomaly
Sound-based fault classifier for drones using multi-task learning

## Project Overview

This project aims to classify drone faults, their movement direction, and the drone model using sound data. The classifications include:

- Fault Status: N (Normal), MF `1~4` (Motor Failure), PC `1~4` (Propeller Cut) 
  - `1~4` indicates each motor/propeller of the quadcopter.
- Movement Direction: F (Front), B (Back), R (Right), L (Left), C (Clockwise), CC (Counter-clockwise).
- Drone Model: A, B, C.

The project leverages libraries such as librosa, numpy, pytorch, and pandas, sci-kit learn. The dataset used can be found [here](https://zenodo.org/records/7779574#.ZCOvfXZBwQ8).

## Audio Data Extraction

Initially, amplitude data from the audio files were extracted using librosa and then converted into spectrograms. These spectrograms were later used to train the model.

## The Model

The constructed model is a 2D-CNN ResNet with 18 layers. It was converted into a multi-task learning model utilizing parameter sharing to branch into three linear classifiers for the classifications mentioned above. The model was trained on the spectrogram data and tested using a separate section of the created spectrogram data. 

After 50 epochs of training, the model achieved 96.7% accuracy on the testing data.

## Difficulties Encountered

1. **MFCCs Data**: Initially, MFCCs data were used for training as it occupied less memory and trained faster. However, the accuracy was very low, leading to a switch to using spectrogram data directly.
2. **GPU Memory Issues**: Predicting the entire testing data at once caused the GPU memory to overflow. The solution was to predict in batches, which resolved the issue.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-anomaly.git
   cd audio-anomaly
   ```

2. Install the required libraries:
   ```bash
   pip install librosa numpy torch pandas matplotlib scikit-learn tqdm
   ```

3. Download and place the dataset in the specified directories as required by the notebook.

## Usage

1. Open the `integrated.ipynb` Jupyter notebook.
2. Run the notebook cells sequentially to process the data, train the models, and evaluate their performance.
