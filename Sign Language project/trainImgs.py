import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Print metrics
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Perform cross-validation
cv_scores = cross_val_score(model, data, labels, cv=5)
print('\nCross-validation scores:', cv_scores)
print('Average CV score: {:.2f}% (+/- {:.2f}%)'.format(
    cv_scores.mean() * 100, cv_scores.std() * 2 * 100))

# Save the model and scaler
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)