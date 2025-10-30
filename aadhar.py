# ================================================
# 🧠 Aadhaar Update Center Recommender — Logic Only
# ================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ------------------------------------------------
# 🧩 Part 1: Load and Prepare Datasets
# ------------------------------------------------
aadhaar_file = "/content/drive/MyDrive/aadhaar_updates_with_target.csv"
pincode_file = "/content/drive/MyDrive/India_pincode.csv"

df = pd.read_csv(aadhaar_file)
pincode_df = pd.read_csv(pincode_file, low_memory=False)

# Clean pincode dataset
pincode_df.columns = [col.strip().capitalize() for col in pincode_df.columns]
pincode_df = pincode_df.drop_duplicates(subset=['Pincode'])
pincode_df = pincode_df.dropna(subset=['Pincode', 'Area', 'District', 'State'])

print("✅ Datasets Loaded and Cleaned")

# ------------------------------------------------
# ⚙️ Part 2: Encode Categorical Columns
# ------------------------------------------------
label_encoders = {}
for col in ['State', 'District', 'Update_Volume_Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("✅ Encoding Completed")

# ------------------------------------------------
# 🧠 Part 3: Train Random Forest Model
# ------------------------------------------------
X = df[['State', 'District', 'Pincode', 'Bio_age_5_17', 'Bio_age_17+', 'Total_Updates']]
y = df['Update_Volume_Category']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("✅ Model Trained")

# ------------------------------------------------
# 📍 Part 4: User Input (for backend usage)
# ------------------------------------------------
user_state = "Andhra Pradesh"
user_district = "Visakhapatnam"
user_pincode = 530001
age_group = "17+"   # or "5-17"

if age_group == "5-17":
    bio_5_17, bio_17_plus = 200, 50
else:
    bio_5_17, bio_17_plus = 50, 200
total_updates = bio_5_17 + bio_17_plus

# ------------------------------------------------
# 🔮 Part 5: Prediction Logic
# ------------------------------------------------
encoded_state = label_encoders['State'].transform([user_state])[0]
encoded_district = label_encoders['District'].transform([user_district])[0]

user_data = np.array([[encoded_state, encoded_district, user_pincode, bio_5_17, bio_17_plus, total_updates]])
prediction = model.predict(user_data)[0]
predicted_category = label_encoders['Update_Volume_Category'].inverse_transform([prediction])[0]

print(f"📍 Predicted Aadhaar Update Volume Category: {predicted_category}")

# ------------------------------------------------
# 🏙️ Part 6: Find Nearby High-Volume Centers
# ------------------------------------------------
nearby = df[df['Pincode'].between(user_pincode - 10, user_pincode + 10)]
high_centers = nearby[nearby['Update_Volume_Category'] == label_encoders['Update_Volume_Category'].transform(['High'])[0]]

if len(high_centers) > 0:
    high_centers = high_centers.copy()
    high_centers['Distance'] = abs(high_centers['Pincode'] - user_pincode)
    high_centers = high_centers.merge(pincode_df[['Pincode', 'Area']], on='Pincode', how='left')

    decoded_state = label_encoders['State'].inverse_transform(high_centers['State'])
    decoded_district = label_encoders['District'].inverse_transform(high_centers['District'])

    print("\n🏙️ Nearby High-Volume Centers:")
    for i, (state, district, pincode, area, dist) in enumerate(
        zip(decoded_state, decoded_district, high_centers['Pincode'], high_centers['Area'], high_centers['Distance'])
    ):
        print(f"{i+1}. {district}, {state} — {area if pd.notna(area) else 'Unknown Area'} — Pincode: {pincode} — Distance: {dist}")

    # Simple distance visualization
    plt.figure(figsize=(7, 4))
    plt.bar(high_centers['Area'].fillna('Unknown Area'), high_centers['Distance'])
    plt.title("📊 Distance to Nearby High-Volume Aadhaar Centers")
    plt.xlabel("Area")
    plt.ylabel("Distance (Pincode Difference)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

else:
    print("\n⚠️ No nearby high-priority centers found.")

# ------------------------------------------------
# 🚫 Part 7: Avoid Low-Volume Centers
# ------------------------------------------------
low_centers = nearby[nearby['Update_Volume_Category'] == label_encoders['Update_Volume_Category'].transform(['Low'])[0]]

if len(low_centers) > 0:
    low_centers = low_centers.merge(pincode_df[['Pincode', 'Area']], on='Pincode', how='left')
    decoded_state = label_encoders['State'].inverse_transform(low_centers['State'])
    decoded_district = label_encoders['District'].inverse_transform(low_centers['District'])

    print("\n🚫 Avoid These Locations (Low Efficiency):")
    for state, district, pincode, area in zip(decoded_state, decoded_district, low_centers['Pincode'], low_centers['Area']):
        print(f"❌ {district}, {state} — {area if pd.notna(area) else 'Unknown Area'} — Pincode: {pincode}")

