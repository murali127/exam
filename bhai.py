import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ipywidgets import Dropdown, IntSlider, FloatSlider, Button, VBox, HBox, HTML, Layout
from IPython.display import display, clear_output
from google.colab import drive

drive.mount('/content/drive')

file_path = "/content/Sleep_health_and_lifestyle_dataset.csv"
df = pd.read_csv(file_path)

bp_split = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(bp_split[0])
df['Diastolic_BP'] = pd.to_numeric(bp_split[1])
df.drop('Blood Pressure', axis=1, inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Good Sleep')

categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

feature_columns = [
    'Age', 'Gender', 'Occupation', 'BMI Category', 'Sleep Duration',
    'Physical Activity Level', 'Stress Level', 'Systolic_BP', 'Diastolic_BP',
    'Heart Rate', 'Daily Steps', 'Quality of Sleep'
]

X = df[feature_columns]
y = df['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

title = HTML("<h2 style='color:#2E86C1;text-align:center;'>💤 Sleep Disorder Prediction Tool</h2>")
subtitle = HTML("<p style='text-align:center;'>Provide your information and click <b>Predict</b> to view the result.</p>")

widgets_with_texts = [
    ("Age", IntSlider(min=18, max=80, step=1, value=30), "Enter your age (years)"),
    ("Gender", Dropdown(options=list(le_dict['Gender'].classes_)), "Select your gender"),
    ("Occupation", Dropdown(options=list(le_dict['Occupation'].classes_)), "Select your occupation"),
    ("BMI", Dropdown(options=list(le_dict['BMI Category'].classes_)), "Select BMI category"),
    ("Sleep (hrs)", FloatSlider(min=3, max=12, step=0.1, value=7), "Average sleep hours per day"),
    ("Activity", IntSlider(min=0, max=200, step=1, value=50), "Daily physical activity level"),
    ("Stress", IntSlider(min=0, max=10, step=1, value=5), "Stress level (0=none, 10=high)"),
    ("Systolic", IntSlider(min=90, max=180, step=1, value=120), "Top blood pressure value"),
    ("Diastolic", IntSlider(min=60, max=120, step=1, value=80), "Bottom blood pressure value"),
    ("Heart Rate", IntSlider(min=50, max=120, step=1, value=70), "Resting heart rate (bpm)"),
    ("Steps", IntSlider(min=0, max=20000, step=100, value=5000), "Average daily steps"),
    ("Quality", IntSlider(min=1, max=10, step=1, value=6), "Sleep quality (1=poor, 10=excellent)")
]

styled_boxes = [
    VBox([
        HTML(f"<b style='font-size:14px;color:#1A5276;'>{label}</b>"),
        w,
        HTML(f"<p style='font-size:12px;color:gray;margin-top:5px;'>{desc}</p>")
    ], layout=Layout(width='100%'))
    for label, w, desc in widgets_with_texts
]

half = len(styled_boxes)//2
left_col = VBox(styled_boxes[:half], layout=Layout(width='45%'))
right_col = VBox(styled_boxes[half:], layout=Layout(width='45%'))
form = HBox([left_col, right_col], layout=Layout(justify_content='center', gap='30px'))

predict_btn = Button(
    description="🔮 Predict Sleep Disorder",
    button_style='success',
    layout=Layout(width='260px', align_self='center', margin='30px auto')
)
button_box = VBox([predict_btn], layout=Layout(display='flex', align_items='center'))

output = HTML("<h4 style='color:#117A65;text-align:center;'>Prediction Result:</h4>")

def on_predict_clicked(b):
    clear_output(wait=True)
    display(title, subtitle, form, button_box, output)
    Gender_enc = le_dict['Gender'].transform([widgets_with_texts[1][1].value])[0]
    Occupation_enc = le_dict['Occupation'].transform([widgets_with_texts[2][1].value])[0]
    BMI_enc = le_dict['BMI Category'].transform([widgets_with_texts[3][1].value])[0]

    input_df = pd.DataFrame([[
        widgets_with_texts[0][1].value, Gender_enc, Occupation_enc, BMI_enc,
        widgets_with_texts[4][1].value, widgets_with_texts[5][1].value, widgets_with_texts[6][1].value,
        widgets_with_texts[7][1].value, widgets_with_texts[8][1].value, widgets_with_texts[9][1].value,
        widgets_with_texts[10][1].value, widgets_with_texts[11][1].value
    ]], columns=feature_columns)

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    result = le_dict['Sleep Disorder'].inverse_transform(pred)[0]
    display(HTML(f"<h3 style='color:#AF7AC5;text-align:center;'>🔹 Predicted Sleep Disorder: <b>{result}</b></h3>"))

predict_btn.on_click(on_predict_clicked)
display(title, subtitle, form, button_box, output) explain 
