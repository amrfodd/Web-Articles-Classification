# News Articles Categorization

## Project Motivation

Categorization of web articles for the student enrolled in various courses on an online learning platform.
This will helps students to access atricles related to their courses without wasting time. 

## File Descriptions
#### Data

The file contains a dataset of web articles that classified into three categories: 'Engineering', 'Product & Design', 'Startups & Business'.

-The folder Data contains two data files:

1)articles.json - The file contains a dataset of the articles
2)cleaned_articles.json.

#### Notebooks

-The folder NOTEBOOKS contains 4 notebooks.

One notebook file 1_Preprocessing.ipynb which contains the loading and preprocessing the data code.
One notebook file 2_Feature_extraction.ipynb which contains the feature extraction code.
One notebook file 3_Modeling.ipynb which contains the modeling code.
One notebook file 4_Prediction.ipynb which contains the inference code.

### Steps to build the model:

###### 1) Data Exploration and Text Processing
Common issues that we generally face during the data preparation phase:

1) too many spelling mistakes in the text.
2) too many numbers and punctuations.
3) too many emojis and emoticons and username and links too.
4) Some of the text parts are not in the English language. Data is having a mixture of more than one language.
4) Some of the words are combined with the hyphen or data having contractions word or Repetitions of words.


Here i will clean the text by doing the following steps:

1) Ensure category name consistency.
2) Lowecasing the data.
3) Removing Puncuatations.
4) Removing Numbers.
5) Removing extra space.
6) Removing Contractions.
7) Removing HTML tags.
8) Removing & Finding URL and Email id.
9) Removing Stop Words
10)Removing Extra-spaces

###### 2) Feature Extraction

We cannot work on texts directly when using machine learning algorithms.So, we need to convert the text to numbers.

I used TF-IDF feature extraction algorithm. 
Term Frequency (TF): Frequency of a term appearing in one document
Inverse Document Frequency (IDF): TFrequency of a term appearing a lot across documents.
TF-IDF are word frequency scores that try to highlight words that are more interesting.

The vectorized data is included in VECTORS file.


###### 3) Modeling
In this project, I used Support vector machine (SVM) Algorithm to classify the articles.

The best trained model is saved in MODELS file.


###### 4) Inference or Prediction    
The inference process in any ML project done by testing the model on the unseen data(Test data)
and to test the model we use some metric in the classifications task.

###### Metrics 
1) Accuracy

2) Confusion Matrix

3) Classification Report


## Summary
In this project, I preprocessed the data. I used TF-IDF feature extraction algorithm as a fearture extraction model. I used SVM to classify the articles. and used evaluation metric to justify accuracy.  Support vector machine scored a better accuracy then MNB.



