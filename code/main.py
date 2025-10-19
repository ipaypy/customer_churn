import pandas as pd    # Data processing, CSV file I/O
import numpy as np     # Numerical computations
from sklearn.model_selection import train_test_split    # Split data
from sklearn.ensemble import RandomForestClassifier     # Main model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Data preprocessing
import matplotlib.pyplot as plt   # Basic plotting
import seaborn as sns             # Statistical visualizations

df = pd.read_csv('customer churn\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].astype(float)


def InfoData():
  print(df.shape)
  print('=' * 60)
  print(df.info())
  print('=' * 60)
  print(df.describe())
  print('=' * 60)
  print(df.columns)
InfoData()


# explore target variable

print("=== CHURN DISTRIBUTION ===")
churn_counts =df['Churn'].value_counts()
churn_percentage = df['Churn'].value_counts(normalize=True)*100


print("Counts:")
print(churn_counts)
print("\nPercentage:")
print(churn_percentage)

# visualize
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Churn')
plt.title('distribution churn')

def VisualizeFeatures(df):
    # Pisahkan kolom numerik dan kategorikal
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    float_features = df.select_dtypes(include=['object'])

    # Visualisasi histogram untuk fitur numerik
    # print("=== Histogram untuk Fitur Numerik ===")
    # for col in numeric_features:
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(df[col], kde=True, bins=30, color='blue')
    #     plt.title(f'Distribusi {col}')
    #     plt.xlabel(col)
    #     plt.ylabel('Frekuensi')
    #     plt.grid(True)
    #     plt.show()
# VisualizeFeatures(df)

# handle categorical columns
float_features = df.select_dtypes(include=['object']).columns

print(float_features.tolist())

df_processed = df.copy()
# drop customer id
df_processed = df.drop('customerID', axis=1).copy()

df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

# Pilih features categorical untuk encoding
features_to_encode = ['gender', 'InternetService', 'Contract', 'PaymentMethod']

# label encoding untuk categorical features

label_encoders = {}
for col in features_to_encode:
   le = LabelEncoder()
   df_processed[col] = le.fit_transform(df_processed[col])
   label_encoders[col] = le

print("=== AFTER ENCODING ===")
print(df_processed[features_to_encode].head())

# handle numerical columns
# Cek numerical columns
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("=== NUMERICAL COLUMNS ===")
print(df_processed[numerical_columns].describe())

# Convert TotalCharges ke numeric (handle errors)
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

# handle missing values
if df_processed['TotalCharges'].isnull().sum() > 0 :
   df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)

# handle missing value column churn
if df_processed['Churn'].isnull().sum() > 0:
    print("Nilai kosong ditemukan di kolom 'Churn'. Menghapus baris dengan nilai kosong...")
    df_processed = df_processed.dropna(subset=['Churn'])

print('======missing value churn======')
x = df_processed.drop('Churn', axis= 1)
y = df_processed['Churn']
print(df_processed['Churn'].isnull().sum())

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")

categorical_cols = x.select_dtypes(include=['object']).columns
numerical_cols = x.select_dtypes(include=[np.number]).columns

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {x_train.shape[0]} samples")
print(f"Testing set: {x_test.shape[0]} samples")

# 1. Cek semua kolom categorical yang masih ada
print("=== ALL COLUMNS ===")
print(x_train.columns.tolist())

print("\n=== DATA TYPES ===")
print(x_train.dtypes)

# 2. Identifikasi SEMUA kolom categorical
categorical_cols = x_train.select_dtypes(include=['object']).columns
print(f"\n=== COLUMNS NEED ENCODING: {len(categorical_cols)} ===")
print(categorical_cols.tolist())

# 3. ENCODE SEMUA KOLOM CATEGORICAL
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    x_train[col] = le.fit_transform(x_train[col].astype(str))
    x_test[col] = le.transform(x_test[col].astype(str))
    label_encoders[col] = le

print("✅ All categorical columns encoded!")

# 4. Cek hasil encoding
print("\n=== AFTER ENCODING ALL ===")
print(x_train.head())
print(f"Data types after encoding:\n{x_train.dtypes}")

# train model
model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced',max_depth=10)
model.fit(x_train,y_train)
print("model training complete ✅")
# predictions
y_pred = model.predict(x_test)

# evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f' 🎯Akurasi: {accuracy:.1f}%')

# Classification Report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

feature_importances = pd.DataFrame({
   'feature': x_train.columns,
    'importance': model.feature_importances_
   }).sort_values(by='importance', ascending=False)

print("\n=== FEATURE IMPORTANCES ===")
feature_importances = feature_importances.sort_values('importance', ascending=False)
print(feature_importances)


