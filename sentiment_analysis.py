import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load Dataset
df = pd.read_csv("amazon_reviews.csv")

# Display Dataset
print("========== AMAZON REVIEWS DATASET ==========")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Function to classify sentiment
def get_sentiment(review):
    polarity = TextBlob(str(review)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review"].apply(get_sentiment)

print("\n========== SENTIMENT RESULTS ==========")
print(df[["Review", "Sentiment"]])

# Sentiment Count
sentiment_count = df["Sentiment"].value_counts()

print("\nSentiment Summary:")
print(sentiment_count)

# Save Output
df.to_csv("sentiment_results.csv", index=False)

# Bar Chart
plt.figure(figsize=(6,5))
sentiment_count.plot(kind="bar")
plt.title("Sentiment Analysis of Amazon Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.grid(axis="y")
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
sentiment_count.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.ylabel("")
plt.title("Sentiment Distribution")
plt.show()

# Final Insights
print("\n========== INSIGHTS ==========")

most_common = sentiment_count.idxmax()

if most_common == "Positive":
    print("Most customers are satisfied with the product.")
elif most_common == "Negative":
    print("Most customers are dissatisfied with the product.")
else:
    print("Customer opinions are mostly neutral.")

print("\nTask 4 (Sentiment Analysis) Completed Successfully!")