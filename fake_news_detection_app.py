import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add class labels
df_fake["class"] = 0
df_true["class"] = 1

# Remove last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)

# Add class labels for manual testing data
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# Combine manual testing data
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv")

# Combine fake and true data
df_marge = pd.concat([df_fake, df_true], axis=0)

# Drop unnecessary columns
df = df_marge.drop(["title", "subject", "date"], axis=1)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply text preprocessing
df["text"] = df["text"].apply(wordopt)

# Split data into features and labels
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
lr_accuracy = LR.score(xv_test, y_test)

# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
dt_accuracy = DT.score(xv_test, y_test)

# Gradient Boosting Classifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
gbc_accuracy = GBC.score(xv_test, y_test)

# Random Forest Classifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
rfc_accuracy = RFC.score(xv_test, y_test)

# Function for label output
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Streamlit App
st.title("Fake News Detection")

st.write("Logistic Regression Accuracy: ", lr_accuracy)
st.write("Decision Tree Accuracy: ", dt_accuracy)
st.write("Gradient Boosting Classifier Accuracy: ", gbc_accuracy)
st.write("Random Forest Classifier Accuracy: ", rfc_accuracy)

user_input = st.text_area("Enter news text for classification")

if st.button("Classify"):
    if user_input:
        # Preprocess the input
        processed_input = wordopt(user_input)
        vectorized_input = vectorization.transform([processed_input])
        
        # Predictions
        pred_LR = LR.predict(vectorized_input)
        pred_DT = DT.predict(vectorized_input)
        pred_GBC = GBC.predict(vectorized_input)
        pred_RFC = RFC.predict(vectorized_input)
        
        st.write(f"Logistic Regression Prediction: {output_label(pred_LR[0])}")
        st.write(f"Decision Tree Prediction: {output_label(pred_DT[0])}")
        st.write(f"Gradient Boosting Classifier Prediction: {output_label(pred_GBC[0])}")
        st.write(f"Random Forest Classifier Prediction: {output_label(pred_RFC[0])}")
    else:
        st.write("Please enter some text to classify.")
