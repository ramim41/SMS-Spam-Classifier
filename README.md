# SMS Spam Classifier

A machine learning project that classifies SMS/email messages as **Spam** or **Not Spam (Ham)** using text preprocessing, TF-IDF feature extraction, and a Multinomial Naive Bayes classifier. The trained model is integrated into a Streamlit web application for interactive prediction.

## Project Overview

The goal of this project is to build an NLP-based text classification system that can identify whether an SMS/email message is spam or ham.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Text preprocessing using NLTK
- Stopword removal
- Punctuation filtering
- Stemming with Porter Stemmer
- Word-frequency analysis
- Spam and ham word clouds
- TF-IDF feature extraction
- Comparison of multiple machine learning classifiers
- Selection of the final model based on test performance
- Model serialization using Pickle
- Streamlit web application for interactive prediction
- Manual testing with 30 unseen messages

## Dataset

The notebook loads the dataset from:

`email.csv`

The dataset contains:

- `Category` — original message class
- `Message` — SMS/email message text

During preprocessing:

- `Category` is renamed to `Target`
- Labels are encoded as:
  - `0` = Ham
  - `1` = Spam
- Duplicate rows are removed
- The last row is removed during data cleaning

## Text Preprocessing

The project uses the following text preprocessing pipeline:

1. Convert the message to lowercase
2. Tokenize the message using NLTK
3. Remove non-alphanumeric tokens
4. Remove English stopwords
5. Filter punctuation
6. Apply Porter Stemming
7. Join the processed tokens back into a single text string

The same preprocessing logic is applied to new messages in the Streamlit application before prediction.

## Exploratory Data Analysis

The notebook performs several analyses to understand the dataset and message patterns, including:

- Class distribution analysis
- Character count
- Word count
- Sentence count
- Ham vs. spam distribution
- Histograms
- Pairplot
- Correlation analysis
- Spam word cloud
- Ham word cloud
- Most common words in spam messages
- Most common words in ham messages

The analysis also shows that the dataset is imbalanced between ham and spam messages.

## Feature Extraction

The project experiments with two common text feature extraction techniques:

- `CountVectorizer`
- `TfidfVectorizer`

The final text features used for model training are generated using:

`TfidfVectorizer(max_features=3000)`

TF-IDF converts the processed text messages into numerical features that can be used by machine learning algorithms.

## Machine Learning Models

Multiple machine learning classifiers were trained and evaluated to find the best-performing model for this spam classification problem.

The models tested include:

- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes
- Decision Tree
- Logistic Regression
- Random Forest
- AdaBoost
- Bagging Classifier
- Extra Trees Classifier
- Gradient Boosting
- XGBoost

The models were evaluated using:

- Accuracy
- Precision
- Classification report
- Confusion matrix

## Final Model Selection

Several machine learning algorithms were trained and evaluated to identify the most suitable model for the SMS Spam Classification task. The models were compared using **Accuracy** and **Precision**.

| Model | Accuracy | Precision |
|---|---:|---:|
| SVC | 98.55% | 99.00% |
| KNN | 93.12% | 100.00% |
| **Multinomial Naive Bayes** | **98.55%** | **100.00%** |
| Decision Tree | 93.90% | 75.00% |
| Logistic Regression | 96.41% | 96.34% |
| Random Forest | 98.06% | 96.97% |
| AdaBoost | 93.60% | 75.82% |
| Bagging Classifier | 96.41% | 82.76% |
| Extra Trees Classifier | 98.45% | 96.19% |
| Gradient Boosting | 95.74% | 88.76% |
| XGBoost | 97.09% | 90.29% |

After comparing all the models, **Multinomial Naive Bayes** was selected as the final model.

### Why Multinomial Naive Bayes Was Selected

- It achieved **98.55% overall accuracy**, which was among the highest results obtained.
- It achieved **100% precision**, meaning every message predicted as Spam was actually Spam in the test set.
- It achieved **100% recall for the Ham class**, correctly identifying all 919 Ham messages in the test set.
- The model produced **zero false positives**, meaning no Ham message was incorrectly classified as Spam.
- **SVC also achieved 98.55% accuracy**, but its precision was slightly lower at 99%.
- KNN achieved 100% precision as well, but its overall accuracy was considerably lower at 93.12%.
- Extra Trees and Random Forest also performed strongly, but both had lower precision and/or accuracy than the selected Multinomial Naive Bayes model.

## Final Multinomial Naive Bayes Performance

The final model achieved the following results on the test set:

- **Accuracy:** 98.55%
- **Ham Precision:** 98%
- **Ham Recall:** 100%
- **Ham F1-Score:** 99%
- **Spam Precision:** 100%
- **Spam Recall:** 87%
- **Spam F1-Score:** 93%

### Confusion Matrix

```text
[[919   0]
 [ 15  98]]
```

The confusion matrix shows that:

- **919 Ham messages** were correctly classified as Ham.
- **0 Ham messages** were incorrectly classified as Spam.
- **98 Spam messages** were correctly classified as Spam.
- **15 Spam messages** were incorrectly classified as Ham.

Therefore, **Multinomial Naive Bayes provided the best overall combination of accuracy, precision, and class-wise performance for this dataset**, and it was selected as the final model for deployment in the Streamlit application.

The final model was saved as:

`model.pkl`

The TF-IDF vectorizer used during training was saved as:

`vectorizer.pkl`

Both files are loaded by the Streamlit application to make predictions on new messages.

## Streamlit Web Application

The Streamlit application provides an interactive interface where a user can enter an SMS/email message and click **Predict**.

The application follows this prediction flow:

1. Accept a new SMS/email message from the user
2. Apply the same text preprocessing used during training
3. Load the saved TF-IDF vectorizer
4. Convert the processed message into TF-IDF features
5. Load the trained Multinomial Naive Bayes model
6. Make a prediction
7. Display either:
   - **Spam**
   - **Not Spam**

The application uses:

- `vectorizer.pkl` — saved TF-IDF vectorizer
- `model.pkl` — saved Multinomial Naive Bayes model

## Manual Testing on Unseen Messages

After completing the model evaluation, the Streamlit application was tested with **30 new messages** that were manually entered and were separate from the original model test-set evaluation.

The purpose of this testing was to verify the complete prediction pipeline on new messages, including preprocessing, TF-IDF transformation, model prediction, and result display.

### Manual Testing Summary

| Test Category | Number of Messages | Correct Predictions |
|---|---:|---:|
| Clear Ham/Spam Messages | 20 | 20/20 |
| Tricky / Real-world Messages | 10 | 10/10 |
| **Total** | **30** | **30/30** |

The application correctly classified all 30 manually tested messages during local testing.

> **Important:** The 30/30 manual testing result is separate from the model's test-set accuracy. The model achieved **98.55% accuracy on the test set**, while the application correctly classified all 30 messages used during this additional manual test.

### Test Cases 1–20: Clear Ham and Spam Messages

| # | Message | Prediction |
|---|---|---|
| 1 | Hey, are you coming to class today? | Not Spam |
| 2 | Please call me when you reach home. | Not Spam |
| 3 | I will be late for the meeting today. | Not Spam |
| 4 | Can you send me the project file tonight? | Not Spam |
| 5 | Don't forget to bring your notebook tomorrow. | Not Spam |
| 6 | What time should we meet at the station? | Not Spam |
| 7 | I have finished the work. You can check it now. | Not Spam |
| 8 | Mom asked you to call her when you are free. | Not Spam |
| 9 | Thanks for helping me with the assignment. | Not Spam |
| 10 | I am on my way. I will be there in ten minutes. | Not Spam |
| 11 | Congratulations! You have won a cash prize of $5000. Claim it now! | Spam |
| 12 | You are selected for a special reward. Click the link to claim your prize. | Spam |
| 13 | WINNER! Your mobile number has been chosen for a free gift. Reply NOW. | Spam |
| 14 | Get an exclusive discount today! Limited offer, claim your reward now. | Spam |
| 15 | You have won a free vacation. Call now to receive your prize. | Spam |
| 16 | Urgent! Your reward is waiting. Click here before the offer expires. | Spam |
| 17 | Congratulations! You are eligible for a special cash bonus. Claim today. | Spam |
| 18 | Free gift available for selected customers. Reply YES to receive it. | Spam |
| 19 | You have been chosen to receive a $1000 bonus. Contact us now. | Spam |
| 20 | Special promotion! Claim your free prize before midnight. | Spam |

### Test Cases 21–30: Tricky / Real-world Messages

| # | Message | Prediction |
|---|---|---|
| 21 | You have been selected for a special offer. Call me when you get a chance. | Spam |
| 22 | Your account has been updated. Please contact customer support if this wasn't you. | Spam |
| 23 | Congratulations on your achievement! We would like to offer you a free gift. | Not Spam |
| 24 | Hey, I won something today! I will tell you about it when we meet. | Not Spam |
| 25 | Please check this link when you have time. I need your opinion. | Not Spam |
| 26 | Limited time offer! Get 50% off your next purchase. Click to claim. | Spam |
| 27 | I am outside your house. Please come downstairs when you are free. | Not Spam |
| 28 | You have won a prize. Send your details now to receive your reward. | Spam |
| 29 | Your appointment has been confirmed for tomorrow at 10 AM. | Not Spam |
| 30 | Urgent! Your account will be suspended unless you verify your information now. | Not Spam |

### Manual Testing Result

- **Total messages tested:** 30
- **Correct predictions:** 30
- **Incorrect predictions:** 0
- **Manual testing result:** 30/30

The manual tests were used as an additional application-level validation after the model had already been evaluated on the original test dataset.

## Example Predictions

### Example 1 — Spam

Input:

```text
Congratulations! You have won a prize. Claim your reward now.
```

Output:

```text
Spam
```

### Example 2 — Ham

Input:

```text
Hey, are you coming home today?
```

Output:

```text
Not Spam
```

## Project Structure

```text
SMS-Spam-Classifier/
│
├── app.py
├── email.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── SMS_Spam_Classifier.ipynb
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd SMS-Spam-Classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- WordCloud
- XGBoost
- Streamlit
- Pickle

## Key Learning Outcomes

Through this project, the following practical concepts were covered:

- NLP text preprocessing
- Tokenization
- Stopword removal
- Stemming
- TF-IDF feature extraction
- Text classification
- Exploratory Data Analysis
- Comparing multiple machine learning classifiers
- Model evaluation using classification metrics
- Confusion matrix analysis
- Selecting a final model based on test performance
- Model serialization using Pickle
- Building a Streamlit ML application
- Using a trained machine learning model to predict new text data
- Testing an ML application with unseen messages

## Future Improvements

Possible improvements include:

- Add probability/confidence score to the prediction
- Improve the Streamlit user interface
- Add batch message prediction
- Add model performance visualizations to the application
- Add better error handling for user input
- Deploy the Streamlit application online
- Experiment with more advanced NLP techniques
- Evaluate the final model on additional unseen datasets

## Conclusion

This project demonstrates a complete machine learning workflow for SMS/email spam classification, starting from data preprocessing and exploratory analysis through feature extraction, model comparison, final model selection, serialization, and deployment through Streamlit.

The final Multinomial Naive Bayes model achieved **98.55% accuracy** on the test set and was successfully integrated into a Streamlit application. Additional manual testing with 30 unseen messages resulted in **30/30 correct predictions** during local application testing.
