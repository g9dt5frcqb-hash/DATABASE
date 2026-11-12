import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------------
# 1. DATABASE AND STOCK SETTINGS
# -----------------------------------

DB_PATH = Path(r"\\192.168.1.95\CS-CLASSROOM\intraday_5m.sqlite")

DB_URI = "file:////192.168.1.95/CS-CLASSROOM/intraday_5m.sqlite?mode=ro&immutable=1"

STOCK = "AAPL"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# -----------------------------------
# 2. CHECK DATABASE CONNECTION
# -----------------------------------

if not DB_PATH.is_file():
    raise FileNotFoundError(
        "Cannot see the database. Connect to the classroom NAS in PowerShell first."
    )

# -----------------------------------
# 3. RETRIEVE STOCK DATA
# -----------------------------------

with sqlite3.connect(DB_URI, uri=True) as conn:
    conn.execute("PRAGMA query_only = ON")

    sql = """
        SELECT ts_utc, close
        FROM bars_5m
        WHERE symbol = ?
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc ASC
    """

    df = pd.read_sql_query(
        sql,
        conn,
        params=(STOCK, START_DATE, END_DATE)
    )

# Check that data was found
if df.empty:
    raise ValueError(
        "No data was found. Try a different stock or date range."
    )

print("\nFIRST FIVE 5-MINUTE RECORDS:")
print(df.head())

# -----------------------------------
# 4. CONVERT 5-MINUTE DATA TO DAILY DATA
# -----------------------------------

df["ts_utc"] = pd.to_datetime(df["ts_utc"])

# Create a column containing only the date
df["date"] = df["ts_utc"].dt.date

# Take the last available closing price from each trading day
daily = df.groupby("date").last().reset_index()

# Keep only the date and closing price
daily = daily[["date", "close"]]

# Make sure the data is in chronological order
daily = daily.sort_values("date").reset_index(drop=True)

# -----------------------------------
# 5. CREATE THE NEXT-DAY TARGET
# -----------------------------------

# The next day's close becomes what the model tries to predict
daily["next_close"] = daily["close"].shift(-1)

# Remove the last row because there is no next trading day to predict
daily = daily.dropna().reset_index(drop=True)

print("\nDAILY DATA:")
print(daily.head())

# -----------------------------------
# 6. SAVE THE PREPARED DATA
# -----------------------------------

daily.to_csv("training_data.csv", index=False)

print("\ntraining_data.csv has been saved.")

# -----------------------------------
# 7. CREATE INPUT AND TARGET VARIABLES
# -----------------------------------

# X is the input: today's closing price
X = daily[["close"]]

# y is the target: next trading day's closing price
y = daily["next_close"]

# -----------------------------------
# 8. CREATE A CHRONOLOGICAL 80/20 SPLIT
# -----------------------------------

split_index = int(len(daily) * 0.80)

# Oldest 80% = training data
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Newest 20% = testing data
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("\nTRAINING ROWS:", len(X_train))
print("TESTING ROWS:", len(X_test))

# -----------------------------------
# 9. CREATE AND TRAIN THE LINEAR REGRESSION MODEL
# -----------------------------------

model = LinearRegression()

# Train the model using the historical training data
model.fit(X_train, y_train)

# Make predictions using the test data
predictions = model.predict(X_test)

# -----------------------------------
# 10. EVALUATE THE MODEL
# -----------------------------------

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n----- LINEAR REGRESSION RESULTS -----")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# -----------------------------------
# 11. CREATE A BASELINE FOR COMPARISON
# -----------------------------------

# The baseline assumes tomorrow's close will be the same as today's close
baseline_predictions = X_test["close"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\n----- BASELINE RESULTS -----")
print("Baseline Mean Absolute Error:", baseline_mae)

# -----------------------------------
# 12. DISPLAY THE PREDICTIONS
# -----------------------------------

results = pd.DataFrame({
    "Actual Next Close": y_test.values,
    "Model Prediction": predictions,
    "Baseline Prediction": baseline_predictions.values
})

print("\n----- FIRST 10 PREDICTIONS -----")
print(results.head(10))

# -----------------------------------
# 13. FINAL CONCLUSION
# -----------------------------------

print("\n----- CONCLUSION -----")

if mae < baseline_mae:
    print("The linear regression model performed better than the baseline.")
else:
    print("The baseline performed as well as or better than the linear regression model.")

# -----------------------------------
# 14. GRAPH THE LINEAR REGRESSION MODEL
# -----------------------------------

# Put the test data and predictions into a table
graph_data = pd.DataFrame({
    "Today's Close": X_test["close"].values,
    "Actual Next Close": y_test.values,
    "Predicted Next Close": predictions
})

# Sort by today's closing price so the regression line is smooth
graph_data = graph_data.sort_values("Today's Close")

# Create the graph
plt.figure(figsize=(10, 6))

# Plot the actual stock data as dots
plt.scatter(
    graph_data["Today's Close"],
    graph_data["Actual Next Close"],
    label="Actual Stock Data"
)

# Plot the linear regression model as a line
plt.plot(
    graph_data["Today's Close"],
    graph_data["Predicted Next Close"],
    label="Linear Regression Line"
)

# Label the graph
plt.xlabel("Today's Closing Price ($)")
plt.ylabel("Next Trading Day's Closing Price ($)")
plt.title(f"Linear Regression Model for {STOCK}")
plt.legend()
plt.grid(True)

# Display the graph
plt.show()