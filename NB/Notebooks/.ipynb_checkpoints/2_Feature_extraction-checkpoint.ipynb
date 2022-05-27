{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c2d11a44",
   "metadata": {},
   "source": [
    "# Articles Recommendation Categorization\n",
    "\n",
    "Recommending web articles for the learners for different study programs"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e79b41d",
   "metadata": {},
   "source": [
    "### 1) Import libraries\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fe9096b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "import re \n",
    "import nltk\n",
    "import string\n",
    "import pickle\n",
    "from collections import defaultdict\n",
    "from nltk.tokenize import RegexpTokenizer\n",
    "from collections import Counter \n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5339ed37",
   "metadata": {},
   "source": [
    "### 2) Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8a09f518",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>body</th>\n",
       "      <th>title</th>\n",
       "      <th>category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>protecting netflix viewing privacy scale open ...</td>\n",
       "      <td>Protecting Netflix Viewing Privacy at Scale</td>\n",
       "      <td>Engineering</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>introducing winston event driven diagnostic re...</td>\n",
       "      <td>Introducing Winston - Event driven Diagnostic ...</td>\n",
       "      <td>Engineering</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>performance usage instagram instagram treat pe...</td>\n",
       "      <td>Performance &amp; Usage at Instagram</td>\n",
       "      <td>Engineering</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>simple example calculating formatting bill vid...</td>\n",
       "      <td>Refactoring a javascript video store</td>\n",
       "      <td>Engineering</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>billing applications transactions need acid co...</td>\n",
       "      <td>Netflix Billing Migration to AWS - Part III</td>\n",
       "      <td>Engineering</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                body  \\\n",
       "0  protecting netflix viewing privacy scale open ...   \n",
       "1  introducing winston event driven diagnostic re...   \n",
       "2  performance usage instagram instagram treat pe...   \n",
       "3  simple example calculating formatting bill vid...   \n",
       "4  billing applications transactions need acid co...   \n",
       "\n",
       "                                               title     category  \n",
       "0        Protecting Netflix Viewing Privacy at Scale  Engineering  \n",
       "1  Introducing Winston - Event driven Diagnostic ...  Engineering  \n",
       "2                   Performance & Usage at Instagram  Engineering  \n",
       "3               Refactoring a javascript video store  Engineering  \n",
       "4        Netflix Billing Migration to AWS - Part III  Engineering  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Read the dataset from csv file\n",
    "df = pd.read_json(r'../Data/cleaned_articles.json')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be599b45",
   "metadata": {},
   "source": [
    "### 3) Feature Extraction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "16dd80b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Engineering', 'Engineering', 'Engineering', ..., 'Engineering',\n",
       "       'Product & Design', 'Startups & Business'], dtype=object)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Determine data and target\n",
    "X = df['body']\n",
    "y = df.iloc[:, -1].values\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "49c26668",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the dataset into the Training, validation and Test sets\n",
    "\n",
    "X_train, X_res, y_train, y_res = train_test_split(X, y, test_size = 0.2)\n",
    "\n",
    "\n",
    "X_val, X_test, y_val, y_test = train_test_split(X_res, y_res, test_size = 0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a64fae8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding the Dependent Variable\n",
    "\n",
    "encoder = LabelEncoder()\n",
    "encoder.fit(y_train)\n",
    "Ytr = encoder.transform(y_train)\n",
    "Yde = encoder.transform(y_val)\n",
    "Yte = encoder.transform(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9a0ed4d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of X_train: (1968,)\n",
      "Shape of X_test: (247,)\n",
      "Shape of X_val: (246,)\n",
      "Shape of train_vectors: (1968, 2000)\n",
      "Shape of test_vectors: (247, 2000)\n",
      "Shape of val_vectors: (246, 2000)\n"
     ]
    }
   ],
   "source": [
    "# I will use TF-IDF method to extract the text features.\n",
    "\n",
    "# Use TF-IDF\n",
    "\n",
    "tf_vec = TfidfVectorizer(tokenizer=None, stop_words=None, max_df=0.75, max_features=2000, lowercase=False,\n",
    "                         ngram_range=(1,2), use_idf=False, sublinear_tf=True, min_df=5, norm='l2',\n",
    "                         encoding='latin-1')\n",
    "\n",
    "\n",
    "train_features = tf_vec.fit(X_train)\n",
    "train_features = tf_vec.transform(X_train)\n",
    "val_features = tf_vec.transform(X_val)\n",
    "test_features = tf_vec.transform(X_test)\n",
    "\n",
    "\n",
    "print('Shape of X_train:',X_train.shape)\n",
    "print('Shape of X_test:',X_test.shape)\n",
    "print('Shape of X_val:',X_val.shape)\n",
    "print('Shape of train_vectors:',train_features.shape)\n",
    "print('Shape of test_vectors:',test_features.shape)\n",
    "print('Shape of val_vectors:',val_features.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a2c40c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Vectorization of data\n",
    "## Vectorize the data using Bag of words (BOW)\n",
    "\n",
    "tokenizer = nltk.tokenize.RegexpTokenizer(r\"\\w+\")\n",
    "stop_words = nltk.corpus.stopwords.words(\"english\")\n",
    "tf_vec = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)\n",
    "\n",
    "train_features = tf_vec.fit(X_train)\n",
    "train_features = tf_vec.transform(X_train)\n",
    "val_features = tf_vec.transform(X_valid)\n",
    "test_features = tf_vec.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0ce417fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "#  save tfidf vectorizer\n",
    "with open('../vectors/train_vector.pkl', 'wb') as handle:\n",
    "    pickle.dump(train_features, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "    \n",
    "with open('../vectors/val_vector.pkl', 'wb') as handle:\n",
    "    pickle.dump(val_features, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "    \n",
    "with open('../vectors/test_vector.pkl', 'wb') as handle:\n",
    "    pickle.dump(test_features, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7cdc7558",
   "metadata": {},
   "outputs": [],
   "source": [
    "#  save Target Varab\n",
    "\n",
    "with open('../vectors/train_label.pkl', 'wb') as handle:\n",
    "    pickle.dump(Ytr, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "    \n",
    "with open('../vectors/val_label.pkl', 'wb') as handle:\n",
    "    pickle.dump(Yde, handle, protocol=pickle.HIGHEST_PROTOCOL)    \n",
    "\n",
    "with open('../vectors/test_label.pkl', 'wb') as handle:\n",
    "    pickle.dump(Yte, handle, protocol=pickle.HIGHEST_PROTOCOL)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c918b2e1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
