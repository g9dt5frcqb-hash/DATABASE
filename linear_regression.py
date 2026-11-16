import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

DB_PATH = Path(r"\\192.168.1.95\CS-CLASSROOM\intraday_5m.sqlite")

DB_URI = "file:////192.168.1.95/CS-CLASSROOM/intraday_5m.sqlite?mode=ro&immutable=1"

STOCK = "AAPL"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

if not DB_PATH.is_file():
    raise FileNotFoundError(
        "Cannot see the database. Connect to the classroom NAS in PowerShell first."
    )

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
if df.empty:
    raise ValueError(
        "No data was found. Try a different stock or date range."
    )

print("\nFIRST FIVE 5-MINUTE RECORDS:")
print(df.head())

df["ts_utc"] = pd.to_datetime(df["ts_utc"])

df["date"] = df["ts_utc"].dt.date

daily = df.groupby("date").last().reset_index()

daily = daily[["date", "close"]]

daily = daily.sort_values("date").reset_index(drop=True)


daily["next_close"] = daily["close"].shift(-1)

daily = daily.dropna().reset_index(drop=True)

print("\nDAILY DATA:")
print(daily.head())


daily.to_csv("training_data.csv", index=False)

print("\ntraining_data.csv has been saved.")


X = daily[["close"]]

y = daily["next_close"]


split_index = int(len(daily) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("\nTRAINING ROWS:", len(X_train))
print("TESTING ROWS:", len(X_test))


model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n----- LINEAR REGRESSION RESULTS -----")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)


baseline_predictions = X_test["close"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\n----- BASELINE RESULTS -----")
print("Baseline Mean Absolute Error:", baseline_mae)


results = pd.DataFrame({
    "Actual Next Close": y_test.values,
    "Model Prediction": predictions,
    "Baseline Prediction": baseline_predictions.values
})

print("\n----- FIRST 10 PREDICTIONS -----")
print(results.head(10))


print("\n----- CONCLUSION -----")

if mae < baseline_mae:
    print("The linear regression model performed better than the baseline.")
else:
    print("The baseline performed as well as or better than the linear regression model.")


graph_data = pd.DataFrame({
    "Today's Close": X_test["close"].values,
    "Actual Next Close": y_test.values,
    "Predicted Next Close": predictions
})

graph_data = graph_data.sort_values("Today's Close")

plt.figure(figsize=(10, 6))

plt.scatter(
    graph_data["Today's Close"],
    graph_data["Actual Next Close"],
    label="Actual Stock Data"
)

plt.plot(
    graph_data["Today's Close"],
    graph_data["Predicted Next Close"],
    label="Linear Regression Line"
)

plt.xlabel("Today's Closing Price ($)")
plt.ylabel("Next Trading Day's Closing Price ($)")
plt.title(f"Linear Regression Model for {STOCK}")
plt.legend()
plt.grid(True)

plt.show()