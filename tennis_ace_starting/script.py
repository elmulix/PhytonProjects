#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
stats = pd.read_csv('tennis_stats.csv')
print(stats.head())
print(len(stats))
print(stats.columns)
print(stats.describe())
print(stats.isna().any())
# Print the correlations between columns to identify which ones have strong correlations
print(stats.corr())

# perform exploratory analysis here:
plt.scatter(stats.BreakPointsOpportunities, stats.Wins, alpha=.2)
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(stats.Winnings, stats.Wins, alpha=.2)
plt.xlabel('Winnings')
plt.ylabel('Wins')
plt.show()
plt.clf()

## perform single feature linear regressions here:
# Assign features and outcome from stats chosen columns
features = stats[['FirstServeReturnPointsWon']]
outcome = stats[['Winnings']]
title = 'Predicted Winnings vs. Actual Winnings - 1 Feature'
# Define function to model features
def model_these_features(features,outcome,title):
  # Assign train and  test data
  X_train, X_test, y_train, y_test = train_test_split(features, outcome, train_size=0.8)
  
  print(X_train.shape)
  # Create model and fit with train data
  model = LinearRegression()
  model.fit(X_train, y_train)
  # Score the model with the test data
  print('Features: ', list(features.columns))
  print('Test score: ', model.score(X_test, y_test))
  print('Train score:', model.score(X_train, y_train))

  # Calculate the outcomes predictions 
  y_prediction = model.predict(X_test)
  # Plot prediction vs test data
  plt.scatter(y_test, y_prediction, alpha=.4)
  plt.xlabel('Test')
  plt.ylabel('Prediction')
  plt.title(title)
  plt.show()
  plt.clf()
# Call function to model features
model_these_features(features,outcome,title)

# Create new features
features = stats[['BreakPointsOpportunities']]
title = 'Predicted Winnings vs. Actual Winnings - 1 Feature'
# Call function
model_these_features(features,outcome,title)

# Create new features
features = stats[['Aces']]
title = 'Predicted Winnings vs. Actual Winnings - 1 Feature'

# Calls function
model_these_features(features,outcome,title)

## perform two feature linear regressions here:
# Set new features
features = stats[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
title = 'Predicting Winnings with 2 Features Test Score:'
# Call function
model_these_features(features,outcome,title)

# Set new features
features = stats[['Aces', 'ServiceGamesWon']]
title = 'Predicting Winnings with 2 Features Test Score:'
# Call function
model_these_features(features,outcome,title)


## perform multiple feature linear regressions here:
# Create multiples features
features = stats[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = stats[['Winnings']]
title = 'Predicted Winnings vs. Actual Winnings - Multiple Features'

# Call function
model_these_features(features,outcome,title)



















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
