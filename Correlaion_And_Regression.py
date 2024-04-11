import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, LinearDiscriminantAnalysis


print("\n********* Assignment 4"
      "\nName: Sachin Chhetri\n")
# Load the dataset
df = pd.read_csv("weatherAUS.csv")
# start and end dates
print("************************************* Start and End Dates")
print("Start Date:", df['Date'].min())
print("End Date:", df['Date'].max())

# calculates average MinTemp based on month
print("\n************************************* Top five rainiest cities")
df['Month'] = pd.to_datetime(df['Date']).dt.month
avgMinTemp = df.groupby('Month')['MinTemp'].mean()
print("Average MinTemp based on month:")
print(avgMinTemp)

# Bar chart for average MinTemp based on month
plt.bar(avgMinTemp.index, avgMinTemp.values)
plt.xlabel("Month")
plt.ylabel("Average MinTemp")
plt.title("Average MinTemp Based on Month")
plt.show()

# Print the number of unique cities
print("\n************************************* The number of unique cities")
uniqueCities = df['Location'].nunique()
print("Number of unique cities:", uniqueCities)

# Print out the top five rainiest cities
print("\n************************************* Top five rainiest cities")
rainiestCities = df.groupby('Location')['Rainfall'].mean().nlargest(5)
print("Top five rainiest cities:")
print(rainiestCities)

# Univariate values : observations on only a single characteristic or attribute
print("\n************************************* Univariate values")
print("Mean of Pressure9am: ", df['Pressure9am'].mean())
print("Mode of Humidity9am: ", df['Humidity9am'].mode()[0])
print("Median of WindGustSpeed: ", df['WindGustSpeed'].median())
print("SD of WindSpeed9am: ", df['WindSpeed9am'].std())
print("Maximum Temperature: ", df['MaxTemp'].max())

# Pearson correlation & Spearman correlation
print("\n************************************* Pearson correlation & Spearman correlation")
pearsonCorr = df['MinTemp'].corr(df['Rainfall'], method='pearson')
spearmanCorr = df['MinTemp'].corr(df['Rainfall'], method='spearman')
print("Pearson correlation between MinTemp and Rainfall:", pearsonCorr)
print("Spearman correlation between MinTemp and Rainfall:", spearmanCorr)

# Scatterplot for MinTemp and Rainfall
plt.scatter(df['MinTemp'], df['Rainfall'])
plt.xlabel("MinTemp")
plt.ylabel("Rainfall")
plt.title("Scatterplot of MinTemp and Rainfall")
plt.show()

# Explanation
print("\n************************************* Explanation")
print("The Pearson correlation cofficient between MinTemp and Rainfall is about 0.104, which is close to 0. We can say\n"
      "there is a weak positive linear relationship in between MinTemp and Rainfall. The Spearman correlation cofficient\n"
      "is also close to 0. We can say there monotonic relationship between them. So, there is a weak and negligible\n"
      "relationship between the MinTemp and Rainfall.\n")

# Correlation matrix for five cities
print("\n************************************* Correlation matrix for five cities")
cities = ['Albury', 'Sydney', 'Melbourne', 'Brisbane', 'Perth']
selectedCities = df[df['Location'].isin(cities)]
numericColumns = selectedCities.select_dtypes(include=['number']).copy()
numericColumns['Location'] = selectedCities['Location']

correlationMatrix = numericColumns.groupby('Location').corr()
print("Correlation Matrix for Selected Cities:")
print(correlationMatrix)

print("\nExplanation:")# Explanation
print("A correlation coefficient close to 1 shows us that there is a strong positive correlation, while close to\n"
      "-1 indicates a strong negative correlation. So, we can say that the coefficient close to 0 suggests no linear correlation.\n"
      "By analyzing the correlation matrix, we can observe the relationships between rainfall and other variables in each cities.\n")

sns.scatterplot(data=selectedCities[selectedCities['Location'] == 'Brisbane'], x='MaxTemp', y='Rainfall')
plt.title('Scatterplot of MaxTemp vs Rainfall for Brisbane')
plt.xlabel('Max Temperature')
plt.ylabel('Rainfall')
plt.show()

# Part 1:
# Drop rows with missing values
df.dropna(inplace=True)
xaxis = df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
          'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
          'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']]
yaxis = df['RainTomorrow']
le = LabelEncoder()
xaxis.loc[:, 'RainToday'] = le.fit_transform(xaxis['RainToday'])
X_train, X_test, y_train, y_test = train_test_split(xaxis, yaxis, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=len(df.axes[0])) # iterate to the end row
model.fit(X_train, y_train)
yPrediction = model.predict(X_test)
accuracy = accuracy_score(y_test, yPrediction)
confusionMatrix = confusion_matrix(y_test, yPrediction)
print("\n************************************* Features, Accuracy, Confusion Matrix of Logistical Regression Model")
print("Features used: MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, WindSpeed9am, WindSpeed3pm,\n"
      "Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix: ")
print(confusionMatrix)

# Part 2:
logreg = LogisticRegression(max_iter=len(df.axes[0]))
rfe = RFE(estimator=logreg, n_features_to_select=3)
rfe.fit(X_train, y_train)
top_features = xaxis.columns[rfe.support_]
logreg.fit(X_train[top_features], y_train)
yPredictionRFE = logreg.predict(X_test[top_features])
accuracyRFE = accuracy_score(y_test, yPredictionRFE)
confusionMatrixRFE = confusion_matrix(y_test, yPredictionRFE)

print("\n************************************* Top 3 Features, Accuracy, Confusion Matrix of RFE")
print("Top 3 features from RFE algorithm:", top_features)
print("Features used in the model:", top_features)
print("Accuracy: {:.2f}%".format(accuracyRFE * 100))
print("Confusion Matrix:")
print(confusionMatrixRFE)

# Part 3:
lda = LinearDiscriminantAnalysis()
X_LDA = lda.fit_transform(X_train, y_train)
logreg_LDA = LogisticRegression(max_iter=len(df.axes[0]))
(logreg_LDA.fit(X_LDA, y_train))
xTestLDA = lda.transform(X_test)
yPreditionLDA = logreg_LDA.predict(xTestLDA)
accuracyLDA = accuracy_score(y_test, yPreditionLDA)
confusionMatrixLDA = confusion_matrix(y_test, yPreditionLDA)

print("\n************************************* Using LDA, Accuracy, Confusion Matrix of LDA")
print("Using LDA")
print("Accuracy: {:.2f}%".format(accuracyLDA * 100))
print("Confusion Matrix:")
print(confusionMatrixLDA)

