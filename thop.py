# --------------- Rating prediction model ----------------
apps_model = apps[pd.notna(apps['Rating'])].copy()
features = ['Category', 'Installs_num', 'Size_mb', 'Price_num', 'Reviews_num']
features = [f for f in features if f in apps_model.columns]

X = apps_model[features]
y = apps_model['Rating']

num_feats = [f for f in features if X[f].dtype != 'object']
cat_feats = [f for f in features if X[f].dtype == 'object']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_feats),
    ('cat', cat_transformer, cat_feats)
])

rating_pipe = Pipeline([
    ('pre', preprocessor),
    ('model', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rating_pipe.fit(X_train, y_train)
y_pred = rating_pipe.predict(X_test)

# ✅ Compatible RMSE computation for all sklearn versions
try:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
except TypeError:
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)
print("\n🎯 Rating Model → RMSE: %.3f | R²: %.3f" % (rmse, r2))
joblib.dump(rating_pipe, "/content/rating_model_pipeline.joblib")

# --------------- Save summary ----------------
with open("/content/ml_project_summary.txt", "w") as f:
    f.write("GOOGLE PLAY ML PROJECT SUMMARY\n\n")
    f.write(f"License: {license_text.strip()[:200]}...\n\n")
    f.write(f"Sentiment samples: {reviews.shape[0]}\n")
    f.write(f"Apps used: {apps_model.shape[0]}\n")
    f.write(f"RMSE: {rmse:.3f} | R²: {r2:.3f}\n")

print("\n✅ All done! Files created in /content:")
print("- sentiment_pipeline.joblib (if trained)")
print("- rating_model_pipeline.joblib")
print("- ml_project_summary.txt")

import matplotlib.pyplot as plt
import seaborn as sns

# Rating distribution
plt.figure(figsize=(6,4))
sns.histplot(apps['Rating'], bins=20, kde=True, color='skyblue')
plt.title('App Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Top 10 categories by average rating
top_cats = apps.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=top_cats.values, y=top_cats.index, palette='viridis')
plt.title('Top 10 Categories by Average Rating')
plt.xlabel('Average Rating')
plt.show()

# Sentiment proportion (if available)
if 'Sentiment_label' in reviews.columns:
    plt.figure(figsize=(5,4))
    reviews['Sentiment_label'].value_counts().plot(kind='bar', color=['limegreen','tomato','gold'])
    plt.title('Sentiment Distribution')
    plt.ylabel('Number of Reviews')
    plt.show()

import joblib
import pandas as pd

model = joblib.load('/content/rating_model_pipeline.joblib')

# Example new app data (replace with your own)
new_app = pd.DataFrame([{
    'Category': 'GAME',
    'Installs_num': 5000000,
    'Size_mb': 45.0,
    'Price_num': 0.0,
    'Reviews_num': 12000
}])

pred = model.predict(new_app)
print("Predicted rating:", round(pred[0], 2))
