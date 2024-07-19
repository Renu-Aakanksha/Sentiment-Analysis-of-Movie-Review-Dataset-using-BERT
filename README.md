<img width="626" alt="image" src="https://github.com/user-attachments/assets/b9243742-196a-4390-a6f8-4e4954254106">

## Overview

In the modern era, businesses are increasingly relying on web and cloud-based platforms. One critical aspect of enhancing customer experience is understanding and reacting to customer emotions during interactions. This project presents a framework designed to improve customer service through emotion detection and sentiment analysis.

The framework uses a Convolutional Neural Network (CNN) architecture for Speech Emotion Recognition (SER) to classify emotions in speech into categories such as angry, neutral, disappointed, or happy. It also includes sentiment analysis of textual input using Long Short-Term Memory (LSTM) networks. This dual-phase system not only routes customer queries to the appropriate technical experts based on initial chat queries but also automatically grades customer service representatives based on the emotional tone of their speech.

## Contributors

- [@Renu Aakanksha Veesam](https://github.com/renu-aakanksha)
- [Preeti Singh](https://github.com/PreetiSingh)
- [Sai Sarath Reddy Koppula](https://github.com/SaiSarathKoppula)

## Built With

- [Python](https://www.python.org) ![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
- [MySQL](https://www.mysql.com/) ![MySQL](https://img.shields.io/badge/MySQL-005C84?style=for-the-badge&logo=mysql&logoColor=white)
- [TensorFlow](https://www.tensorflow.org/) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## Dataset

IMDB Dataset available in the Keras Module.

## Documentation

For comprehensive information about the project, please refer to the [Documentation](https://github.com/Renu-Aakanksha/Sentiment-Analysis-of-Movie-review-Dataset-using-LSTM-CNN/blob/main/Team_18%20Research%20Paper.pdf).

## Lessons Learned

Implementing the BERT model for sentiment analysis demonstrated the power of pre-trained language models in understanding complex text data. We observed significant improvements in classification accuracy and learned the importance of fine-tuning hyperparameters for optimal performance in NLP tasks. But as per the results, the LSTM - CNN model has yeild better results in less epochs that Bert model. 
The code using the LSTM and CNN is published i my repositories. Please refer that. 


## Key Components

1. **Data Collection and Preprocessing:**
   - Download the IMDB dataset from Stanford and preprocess it for sentiment analysis. This includes loading, cleaning, and organizing the data into a format suitable for BERT.

2. **BERT Model Architecture:**
   - Use TensorFlow Hub's BERT preprocessing and encoding layers. The architecture includes a BERT encoder to generate embeddings from text and a dense layer for final sentiment classification.

3. **Training Configuration:**
   - Configure the model with the Adam optimizer, binary crossentropy loss function, and binary accuracy metrics. Train the model for 10 epochs with a carefully chosen learning rate.

4. **Model Evaluation:**
   - Evaluate the model on a separate test dataset to measure its performance. Metrics such as accuracy and loss are used to assess the model’s effectiveness in sentiment prediction.

5. **Visualization:**
   - Visualize the model architecture and plot training metrics. Graphs of training loss and accuracy help in monitoring the model's learning process and performance over time.
