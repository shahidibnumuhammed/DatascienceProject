# DatascienceProject
# Data Science Model Training & Prediction Platform

A user-friendly Flask-based web application that allows users to upload datasets, train machine learning models dynamically, visualize data distributions, and make real-time predictions.

## 🚀 Features
- **CSV Data Upload**: Easily upload any structured tabular dataset (`.csv`).
- **Dynamic Model Training**: Select a target column, and the application will automatically preprocess the data and train a **Logistic Regression** model.
- **Automated Preprocessing**:
  - Handles missing values automatically.
  - Removes unique ID-like columns that do not contribute to the model.
  - Automatically encodes categorical text variables to numeric values using `LabelEncoder`.
- **Data Visualization**: Generates rich exploratory data analysis (EDA) plots automatically:
  - Target Distribution (Histogram & KDE)
  - Correlation Matrix (Heatmap)
  - Pairplot for feature relationships
  - Boxplots to detect outliers
- **Real-Time Predictions**: Input new data directly through the generated web interface to get instant predictions using the trained model.

## 📁 Project Structure
```text
DatascienceProject/
│
├── code/
│   ├── app.py                  # Main Flask application file
│   ├── requirements.txt        # Python dependencies
│   ├── static/                 # Directory for generated visual plots and CSS
│   ├── templates/              # HTML templates (index.html, select.html)
│   └── uploads/                # Directory where uploaded CSVs are saved
│
├── datasets/                   # Sample datasets for testing
├── ppt/                        # Project presentations
├── report/                     # Project documentation/reports
├── screenshot/                 # Screenshots of the application
├── video/                      # Demo videos
└── README.md                   # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites
Make sure you have Python installed on your system.

### 1. Navigate to the Code Directory
Open your terminal or command prompt and go to the code folder:
```bash
cd code
```

### 2. Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask development server:
```bash
python app.py
```

### 4. Access the App
Open your web browser and go to:
`http://127.0.0.1:5000/`

## 📊 How to Use
1. **Upload Dataset**: On the home page, upload a `.csv` dataset file.
2. **Select Target**: Choose the column you want the model to predict from the dropdown list.
3. **Train Model**: The app will automatically preprocess the data, train the model, display the accuracy score, and generate data visualizations.
4. **Predict**: Once the model is trained, a form will dynamically generate based on your feature columns. Enter values to get a live prediction.

## 🧰 Tech Stack
- **Backend Framework**: Python / Flask
- **Machine Learning**: Scikit-Learn
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
