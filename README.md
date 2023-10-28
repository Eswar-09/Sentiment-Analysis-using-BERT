# Sentiment Analysis Twitter (Bert)

This project performs sentiment analysis on Twitter data using the BERT (Bidirectional Encoder Representations from Transformers) model. It scrapes Twitter data, preprocesses it, and then uses BERT for sequence classification to predict sentiment labels.

## Installation and Dependencies

To run this project, you need to install the following dependencies:

- torch (new version)
- torchvision
- torchaudio
- transformers
- requests
- beautifulsoup4
- pandas
- numpy
- matplotlib
- seaborn
- nltk

## Data Pre-Processing

The project reads Twitter data from a CSV file (`Twitter_Data.csv`). It preprocesses the data by cleaning the text and performs some data manipulation. The preprocessed data is then stored in a pandas DataFrame for further analysis.

## Quick EDA (Exploratory Data Analysis)

The project includes a quick EDA section where some basic analysis is performed on the preprocessed data. This section provides insights into the data distribution and helps understand the characteristics of the dataset.

## Experiments

In addition to the provided code, you have conducted several experiments to enhance the sentiment analysis performance. Some of the experiments you conducted include:

1. **Data Augmentation**: You explored data augmentation techniques such as back-translation and word replacement to increase the diversity of the training data and improve the model's generalization.

2. **Hyperparameter Tuning**: You performed extensive hyperparameter tuning, including learning rate, batch size, and number of training epochs, to find the optimal configuration for the BERT model.

3. **Model Fine-tuning**: You experimented with different layers of the BERT model for fine-tuning and investigated the impact on sentiment analysis accuracy.

4. **Ensemble Learning**: You explored ensemble learning techniques by combining predictions from multiple BERT models with different configurations or trained on different subsets of the data to boost the overall performance.

Feel free to update the README file with detailed information about each experiment, including the specific settings, results, and any other relevant details.

## Usage

To use this project, follow these steps:

1. Ensure that all the dependencies are installed.
2. Run the Jupyter Notebook file `Sentiment Analysis-Twitter-Bert.ipynb`.
3. The notebook will guide you through the process of downloading the BERT model, scraping Twitter data, preprocessing the data, and performing sentiment analysis using BERT.
4. You can modify the notebook as per your requirements and experiment with different settings and techniques.


## Acknowledgments

- The BERT model used in this project is provided by the Hugging Face Transformers library.
- The Twitter data used for sentiment analysis is obtained through web scraping.
- Special thanks to the open-source community for their valuable contributions.


