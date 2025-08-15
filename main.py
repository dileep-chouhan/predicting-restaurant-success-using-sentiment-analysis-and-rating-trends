import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_restaurants = 50
num_months = 24
dates = pd.date_range(start='2021-01-01', periods=num_months, freq='M')
restaurants = [f'Restaurant {i+1}' for i in range(num_restaurants)]
data = {
    'Restaurant': np.repeat(restaurants, num_months),
    'Date': np.tile(dates, num_restaurants),
    'Rating': np.random.uniform(2.5, 4.5, size=num_restaurants * num_months),
    'Review_Sentiment': np.random.normal(0, 1, size=num_restaurants * num_months) # 0 = neutral, +/- indicates sentiment
}
df = pd.DataFrame(data)
# Add some realistic trends (simulated growth/decline)
for rest in restaurants[:10]: #Simulate growth for first 10 restaurants
    df.loc[df['Restaurant'] == rest, 'Rating'] += np.linspace(0, 1, num_months)
for rest in restaurants[10:20]: #Simulate decline for next 10 restaurants
    df.loc[df['Restaurant'] == rest, 'Rating'] -= np.linspace(0, 1, num_months)
    
df['Review_Sentiment'] = df['Review_Sentiment'].clip(-2,2) #Cap sentiment scores
# --- 2. Data Cleaning and Feature Engineering ---
# (In a real-world scenario, this would involve handling missing data, outliers, etc.)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# --- 3. Analysis ---
# Calculate monthly average rating and sentiment for each restaurant
monthly_data = df.groupby(['Restaurant', 'Year', 'Month'])[['Rating', 'Review_Sentiment']].mean().reset_index()
#Simple linear regression to predict future rating based on past trend.  This is a simplified example.
results = []
for rest in restaurants:
    restaurant_data = monthly_data[monthly_data['Restaurant'] == rest]
    slope, intercept, r_value, p_value, std_err = linregress(restaurant_data['Month'], restaurant_data['Rating'])
    results.append([rest, slope, r_value])
prediction_df = pd.DataFrame(results, columns=['Restaurant', 'Slope', 'R_value'])
# --- 4. Visualization ---
# Plot average rating trend for a few restaurants
plt.figure(figsize=(12, 6))
for rest in restaurants[:5]:
    restaurant_data = monthly_data[monthly_data['Restaurant'] == rest]
    plt.plot(restaurant_data['Date'], restaurant_data['Rating'], label=rest)
plt.xlabel('Month')
plt.ylabel('Average Rating')
plt.title('Average Rating Trend for Selected Restaurants')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('restaurant_rating_trends.png')
print("Plot saved to restaurant_rating_trends.png")
#Plot the regression slopes to show which restaurants are growing/declining
plt.figure(figsize=(10, 6))
sns.barplot(x='Restaurant', y='Slope', data=prediction_df)
plt.xticks(rotation=90)
plt.title('Regression Slopes of Rating Trends')
plt.tight_layout()
plt.savefig('regression_slopes.png')
print("Plot saved to regression_slopes.png")