import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 100

data = {
    'Hours_Studied': np.random.randint(1, 45, n_samples),
    'Attendance': np.random.randint(60, 101, n_samples),
    'Parental_Involvement': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Access_to_Resources': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n_samples),
    'Sleep_Hours': np.random.randint(4, 11, n_samples),
    'Previous_Scores': np.random.randint(50, 101, n_samples),
    'Motivation_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Internet_Access': np.random.choice(['Yes', 'No'], n_samples),
    'Tutoring_Sessions': np.random.randint(0, 9, n_samples),
    'Family_Income': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'School_Type': np.random.choice(['Public', 'Private'], n_samples),
    'Peer_Influence': np.random.choice(['Negative', 'Neutral', 'Positive'], n_samples),
    'Physical_Activity': np.random.randint(0, 7, n_samples),
    'Learning_Disabilities': np.random.choice(['Yes', 'No'], n_samples),
    'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], n_samples),
    'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
}

df = pd.DataFrame(data)

# Generate realistic Exam_Score based on some features
base_score = 50
df['Exam_Score'] = (
    base_score +
    (df['Hours_Studied'] * 0.5) +
    ((df['Attendance'] - 60) * 0.2) +
    ((df['Previous_Scores'] - 50) * 0.3) +
    np.random.normal(0, 5, n_samples)
).astype(int)

# Cap score between 0 and 100
df['Exam_Score'] = df['Exam_Score'].clip(0, 100)

df.to_csv('data/StudentPerformanceFactors.csv', index=False)
print("Generated data/StudentPerformanceFactors.csv successfully!")
