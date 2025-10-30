# ================================================
# 🧭 Aadhaar Update Location Recommender (CLI Version)
# ================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Step 1: Load and prepare data
# ---------------------------
file_path = "/content/drive/MyDrive/aadhaar_updates_with_target.csv"  # your dataset
df = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully\n")

# Encode categorical columns
label_encoders = {}
for col in ['State', 'District', 'Update_Volume_Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and labels
X = df[['State', 'District', 'Pincode', 'Bio_age_5_17', 'Bio_age_17+', 'Total_Updates']]
y = df['Update_Volume_Category']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("🧠 Model trained successfully!\n")

# ---------------------------
# Step 2: Take user input
# ---------------------------
user_state = input("Enter your State: ").strip()
user_district = input("Enter your District: ").strip()
user_pincode = int(input("Enter your Pincode: "))
age_group = input("Enter Age Group (5-17 or 17+): ").strip()

# ---------------------------
# Step 3: Determine update counts
# ---------------------------
if age_group == "5-17":
    bio_5_17, bio_17_plus = 200, 50
else:
    bio_5_17, bio_17_plus = 50, 200
total_updates = bio_5_17 + bio_17_plus

# ---------------------------
# Step 4: Encode and Predict
# ---------------------------
try:
    encoded_state = label_encoders['State'].transform([user_state])[0]
    encoded_district = label_encoders['District'].transform([user_district])[0]
except ValueError:
    print("\n❌ Invalid state or district name. Please check your spelling.")
    exit()

user_data = np.array([[encoded_state, encoded_district, user_pincode, bio_5_17, bio_17_plus, total_updates]])
prediction = model.predict(user_data)[0]
predicted_category = label_encoders['Update_Volume_Category'].inverse_transform([prediction])[0]

# ---------------------------
# Step 5: Show result
# ---------------------------
print("\n📍 Recommended Aadhaar Update Center Category:", predicted_category)

if predicted_category.lower() == "high":
    print("✅ This center has a HIGH update volume — likely efficient and faster.")
else:
    print("⚠️ This center has a LOW update volume — possible delays.\n")

# ---------------------------
# Step 6: Find nearby centers
# ---------------------------
print("------------------------------------------------------------")
print("🏙️ Nearby Centers (within ±10 pincode range):")
print("------------------------------------------------------------")

nearby = df[(df['Pincode'].between(user_pincode - 10, user_pincode + 10))]

# Decode for display
df_state_decoded = label_encoders['State'].inverse_transform(df['State'])
df_district_decoded = label_encoders['District'].inverse_transform(df['District'])
df['Decoded_State'] = df_state_decoded
df['Decoded_District'] = df_district_decoded

high_centers = nearby[nearby['Update_Volume_Category'] == label_encoders['Update_Volume_Category'].transform(['High'])[0]]

if len(high_centers) > 0:
    print("\n⭐ High Priority Centers Near You:")
    for i, row in high_centers.iterrows():
        state = label_encoders['State'].inverse_transform([row['State']])[0]
        district = label_encoders['District'].inverse_transform([row['District']])[0]
        print(f"  {district}, {state} — Pincode: {row['Pincode']}")
else:
    print("\n_No nearby high-priority centers found._")

# ---------------------------
# Step 7: Avoided Centers
# ---------------------------
low_centers = nearby[nearby['Update_Volume_Category'] == label_encoders['Update_Volume_Category'].transform(['Low'])[0]]
if len(low_centers) > 0:
    print("\n🚫 Avoided Centers (Crowded / Low Efficiency):")
    for i, row in low_centers.iterrows():
        state = label_encoders['State'].inverse_transform([row['State']])[0]
        district = label_encoders['District'].inverse_transform([row['District']])[0]
        print(f"  {district}, {state} — Pincode: {row['Pincode']} (High crowd or low throughput)")

print("\n✅ Recommendation complete.")
