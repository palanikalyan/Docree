import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv("Doceree-HCP_Train.csv")
test_data = pd.read_csv("Doceree-HCP_Test.csv")

train_data = train_data.dropna()
train_data = train_data.drop(['ID'], axis=1)
train_data['USERAGENT_OS'] = train_data['USERAGENT'].str.split('(').str[1].str.split(';').str[0]

X = train_data.drop(['TAXONOMY'], axis=1)
y = train_data['TAXONOMY']

encoder = LabelEncoder()
X_encoded = X.apply(encoder.fit_transform)

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
classification_report = classification_report(y_val, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report)

X_test = test_data.drop(['ID'], axis=1)
X_test_encoded = X_test.apply(encoder.transform)

test_predictions = model.predict(X_test_encoded)

output_scoring = pd.DataFrame({
    'USERPLATFORMUID': test_data['USERPLATFORMUID'],
    'IS_HCP': test_predictions
})
output_scoring.to_csv('output_scoring.csv', index=False)

output_complete = pd.DataFrame({
    'USERPLATFORMUID': test_data['USERPLATFORMUID'],
    'IS_HCP': test_predictions,
    'TAXONOMY': test_data['TAXONOMY']
})
output_complete.to_csv('output_complete.csv', index=False)
