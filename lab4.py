# LAB 4
import os, shutil, random, re
from datetime import datetime, timedelta
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

nltk.download('vader_lexicon')

# create demo folder
demo_folder = r"./stock_data_demo"
if os.path.exists(demo_folder):
    shutil.rmtree(demo_folder)
os.makedirs(demo_folder, exist_ok=True)

# create demo tickers and dates
tickers = ['AAPL','MSFT','GOOG']
start_date = datetime(2022,9,1)
dates = [start_date + timedelta(days=i) for i in range(30)]  # 30 days

# generate demo prices.csv
price_rows = []
for t in tickers:
    price = 100.0 + random.random()*20
    for d in dates:
        open_p = price + np.random.normal(0,1)
        close_p = open_p + np.random.normal(0,2)
        vol = int(1e6 + np.random.randint(-100000,100000))
        price_rows.append({'Date': d.strftime('%Y-%m-%d'), 'Stock Name': t, 'Open': round(open_p,2), 'Close': round(close_p,2), 'Volume': vol})
        price = close_p
pd.DataFrame(price_rows).to_csv(os.path.join(demo_folder,'prices.csv'), index=False)

# generate demo tweets.csv (several tweets per day per ticker)
positive = ["great earnings", "strong buy", "beat expectations", "rally expected", "bullish"]
negative = ["missed guidance", "lawsuit", "selloff incoming", "bearish", "downgrade"]
tweet_rows = []
for t in tickers:
    for d in dates:
        n = random.randint(1,6)
        for i in range(n):
            samp = random.choices([positive, negative, positive+negative], weights=[0.4,0.4,0.2])[0]
            text = ' '.join(random.choices(samp, k=random.randint(1,3)))
            tweet_rows.append({'Date': d.strftime('%Y-%m-%d'), 'Stock Name': t, 'Tweet': text})
pd.DataFrame(tweet_rows).to_csv(os.path.join(demo_folder,'tweets.csv'), index=False)

print("Demo files created in:", os.path.abspath(demo_folder))
print("Files:", os.listdir(demo_folder))

# Now run the (simplified) pipeline on the demo files
df_tweets = pd.read_csv(os.path.join(demo_folder,'tweets.csv'))
df_prices = pd.read_csv(os.path.join(demo_folder,'prices.csv'))

# prepare
df_tweets['Date'] = pd.to_datetime(df_tweets['Date'])
df_tweets['date'] = df_tweets['Date'].dt.date
def clean_text(txt):
    return re.sub(r'[^A-Za-z0-9\s]','', str(txt)).lower()
df_tweets['clean_tweet'] = df_tweets['Tweet'].apply(clean_text)
sia = SentimentIntensityAnalyzer()
df_tweets['sentiment'] = df_tweets['clean_tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])

df_daily = df_tweets.groupby(['date','Stock Name'])['sentiment'].mean().reset_index().rename(columns={'sentiment':'score'})

df_prices['Date'] = pd.to_datetime(df_prices['Date'])
df_prices['date'] = df_prices['Date'].dt.date
df = pd.merge(df_prices, df_daily, how='left', on=['date','Stock Name']).dropna(subset=['score']).sort_values(['Stock Name','date']).reset_index(drop=True)

df['next_close'] = df.groupby('Stock Name')['Close'].shift(-1)
df['return'] = (df['next_close'] - df['Close']) / df['Close']
df = df.dropna(subset=['return'])
df['class'] = (df['return'] >= 0).astype(int)
df['change'] = df['Close'] - df['Open']

X = df[['score','change']]
y_class = df['class']
y_reg = df['return']

X_train, X_test, y_train, y_test, y_train_reg, y_test_reg = train_test_split(X, y_class, y_reg, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred_test = knn.predict(X_test_scaled)
print("Confusion (test):\n", confusion_matrix(y_test, y_pred_test))
print("Precision, Recall, F1:", precision_score(y_test, y_pred_test), recall_score(y_test, y_pred_test), f1_score(y_test, y_pred_test))

# regression
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_scaled, y_train_reg)
y_pred_test_reg = knn_reg.predict(X_test_scaled)
mse_test = mean_squared_error(y_test_reg, y_pred_test_reg)
rmse_test = mse_test**0.5
r2_test = r2_score(y_test_reg, y_pred_test_reg)
print("Reg test MSE, RMSE, R2:", round(mse_test,6), round(rmse_test,6), round(r2_test,6))

# simple scatter plot of sentiment vs change
plt.figure(figsize=(6,4))
plt.scatter(df['score'], df['change'], c=df['class'], cmap='bwr', s=20)
plt.xlabel('Sentiment score'); plt.ylabel('Price change'); plt.title('Demo: sentiment vs change (colored by class)')
plt.colorbar(label='class(0=down,1=up)')
plt.show()
