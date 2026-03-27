from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # ✅ auto create folder
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

df_global = None
model = None
accuracy = None
feature_columns = None   # ✅ store features


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    global df_global

    file = request.files['file']

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        df_global = pd.read_csv(filepath)

        columns = df_global.columns.tolist()

        return render_template("select.html", columns=columns)

    return "No file uploaded"


@app.route('/train', methods=['POST'])
def train():
    global df_global, model, accuracy, feature_columns

    target = request.form['target']
    df = df_global.copy()

    # 🔹 Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 🔹 REMOVE ID-like columns (IMPORTANT FIX)
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() == len(df):  # unique values → ID column
                df.drop(columns=[col], inplace=True)

    # 🔹 Encode categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))

    # 🔹 Features and target
    X = df.drop(columns=[target])
    y = df[target]

    feature_columns = X.columns  # ✅ save for prediction

    # 🔹 Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 🔹 Visualization 1: Target Distribution
    plt.figure()
    sns.histplot(y, kde=True)
    plt.title("Target Distribution")
    plt.savefig("static/plot1.png")
    plt.close()

    # 🔹 Visualization 2: Correlation Matrix
    plt.figure(figsize=(8,6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig("static/plot2.png")
    plt.close()

    # 🔹 Visualization 3: Pairplot (IMPORTANT)
    sns.pairplot(df.select_dtypes(include=['number']))
    plt.savefig("static/plot3.png")
    plt.close()

    # 🔹 Visualization 4: Boxplot
    plt.figure(figsize=(10,6))
    df.select_dtypes(include=['number']).boxplot()
    plt.title("Boxplot of Features")
    plt.savefig("static/plot4.png")
    plt.close()
    
    return render_template("index.html",
                           accuracy=round(accuracy * 100, 2),
                           trained=True,
                           features=feature_columns)


@app.route('/predict', methods=['POST'])
def predict():
    global model, feature_columns

    values = []

    try:
        for col in feature_columns:
            val = float(request.form[col])
            values.append(val)

        prediction = model.predict([values])

        return render_template("index.html",
                               prediction=prediction[0],
                               accuracy=round(accuracy * 100, 2),
                               trained=True,
                               features=feature_columns)

    except:
        return "⚠️ Please enter valid numeric values"


if __name__ == "__main__":
    app.run(debug=True)