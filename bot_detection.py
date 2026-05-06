import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv(r'C:\Users\ghasq\AppData\Local\Temp\9ef98a10-f4bb-4757-84c6-52f416d70c1f_archive (3).zip.c1f\twitter_human_bots_dataset.csv')

# تحويل الـ target
df['label'] = df['account_type'].apply(lambda x: 1 if x == 'bot' else 0)

features = ['followers_count', 'friends_count', 'favourites_count',
            'statuses_count', 'average_tweets_per_day', 'account_age_days']

df = df[features + ['label']].dropna()

X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
print(classification_report(y_test, model.predict(X_test)))
import joblib
joblib.dump(model, r'C:\Users\ghasq\bot_model.pkl')
print("Model saved!")