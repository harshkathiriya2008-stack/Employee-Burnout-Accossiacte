import pandas as pd
import dtale as d
import sweetviz as s
import matplotlib.pyplot as pt
import seaborn as sns

# Read CSV file
df = pd.read_csv("Dataset\\train.csv")
df.info()

# Histogram: Burn_Rate
pt.figure(figsize=(8, 5))
sns.histplot(df["Burn_Rate"].dropna(), kde=True)
pt.title("Distribution of Burn_Rate")
pt.xlabel("Burn_Rate")
pt.ylabel("Count")
pt.tight_layout()
pt.show()

# D-Tale
dtale_app = d.show(df)
dtale_app.open_browser()

# Sweetviz report
report = s.analyze(df)
report.show_html("sweetviz_report.html")

# Scatter plot: Mental Fatigue Score vs Burn_Rate
pt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Mental Fatigue Score", y="Burn_Rate")
pt.title("Mental Fatigue Score vs Burn_Rate")
pt.tight_layout()
pt.show()

# Box plots for all numeric columns
numeric_df = df.select_dtypes(include=["number"])
if not numeric_df.empty:
    ncols = 3
    nrows = (len(numeric_df.columns) + ncols - 1) // ncols
    pt.figure(figsize=(15, 5 * nrows))

    for i, col in enumerate(numeric_df.columns, start=1):
        pt.subplot(nrows, ncols, i)
        sns.boxplot(y=numeric_df[col].dropna())
        pt.title(col)
        pt.ylabel(col)

    pt.tight_layout()
    pt.show()
