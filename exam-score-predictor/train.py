import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("student_habits_performance.csv")
df.drop(['age', 'netflix_hours', 'part_time_job', 'extracurricular_participation','diet_quality'
,'sleep_hours','gender','internet_quality','attendance_percentage','parental_education_level','student_id'], axis=1, inplace=True)
print(df.head())
X = df[['social_media_hours', 'exercise_frequency', 'mental_health_rating','study_hours_per_day']]
y = df['exam_score'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(
   n_estimators=200,
    max_depth=None, 
    min_samples_split=5, 
    random_state=42
)
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score (Accuracy): {r2:.2f}")
print(df.corr()['exam_score'].sort_values())
new_data = [[3.9, 4,1,1]]  
predicted_score = model.predict(new_data)
print(f"Predicted Exam Score: {predicted_score[0]:.2f}")

import joblib
joblib.dump(model, 'exam_score_predictor.pkl')