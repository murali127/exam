# ============================================
# 📊 GOOGLE PLAY STORE DATA ANALYSIS
# Sentiment Distribution | App Distribution | Rating Distribution
# ============================================

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------
# 1️⃣ SENTIMENT DISTRIBUTION
# --------------------------------------------

# Load user reviews dataset
reviews = pd.read_csv("googleplaystore_user_reviews.csv")

# Check available columns
print("Reviews Columns:", reviews.columns.tolist())

# Count sentiment occurrences
sentiment_counts = reviews['Sentiment'].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(6,4))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Type")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()


# --------------------------------------------
# 2️⃣ APP DISTRIBUTION
# --------------------------------------------

# (a) Top 20 Apps by Number of Reviews
app_counts = reviews['App'].value_counts().head(20)

plt.figure(figsize=(8,6))
plt.barh(app_counts.index[::-1], app_counts.values[::-1])
plt.title("Top 20 Apps by Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("App Name")
plt.tight_layout()
plt.show()

# (b) Apps per Category from main dataset
apps = pd.read_csv("googleplaystore.csv")
category_counts = apps['Category'].value_counts().head(20)

plt.figure(figsize=(8,6))
plt.barh(category_counts.index[::-1], category_counts.values[::-1])
plt.title("Top 20 Categories by Number of Apps")
plt.xlabel("Number of Apps")
plt.ylabel("Category")
plt.tight_layout()
plt.show()


# --------------------------------------------
# 3️⃣ RATING DISTRIBUTION
# --------------------------------------------

# Load main app dataset
apps = pd.read_csv("googleplaystore.csv")

# Drop missing ratings
ratings = apps['Rating'].dropna()

# Plot histogram of app ratings
plt.figure(figsize=(6,4))
plt.hist(ratings, bins=20, edgecolor='black')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Apps")
plt.tight_layout()
plt.show()

# ============================================
# ✅ END OF SCRIPT
# ============================================
