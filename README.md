# SarcasmTweetDetect

In this program, we detect sarcasm in tweets using different classification methods. Namely, we used Logistic Regression, SVM, and RNN models. The dataset we used was from surajr's repository, which can be found here https://github.com/surajr/SarcasmDetection/blob/master/Data/dataset_csv.csv.

We clean the data in preprocessing by removing all punctuation/numbers, lemmatizing the data, removing stop words, and finally converting everything to lowercase. For the SVM and logistic models, we use TF-IDF vectorization. By comparing the TF-IDF values, we were able to determine which words are more important to this binary classification task. We found that the top 10 words were (from most important to least important): speak, love, get, like, know, people, go, much, good, and want.

For both SVM and logistic models, we first test the models using no regularization. Then, using cross-validation, we find the optimal hyperparameters that yield the highest test accuracy and compare results.

For our final model (RNN), we tokenized the cleaned data using Keras Tokenizer and padded the data before splitting it into train and test sets. The architecture of our RNN model consists of an initial embedding layer followed by two LSTM layers, two BatchNormalization layers, one Dropout layer, and two Dense layers. Our initial hyperparameter choice was arbitrary (manually chosen after a few runs). We then conduct hyperparameter tuning in later code segments to improve the model's performance.


Future Work:
This project was created at UCLA Summer Institute: Computer Science Intermediate Track. The objective of this project was to apply the methods we studied in class to a machine-learning task of our choice and compare the results. We found that due to the small size of the dataset we used, the RNN model trained on our data does not generalize well to sarcastic statements in general. Thus, one potential improvement may be finding a larger dataset to train the model on. Another potential improvement is tuning more hyperparameters and at greater precision. The precision of our hyperparameter tuning process was limited by the speed at which my computer was able to execute the code. Thus, we were only able to search through a very limited grid of potential hyperparameter values.
