import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Footfall vs Sales", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def make_sample_data():
    rng = np.random.default_rng(42)
    days = pd.date_range("2026-01-01", periods=120, freq="D")
    rows = []

    for d in days:
        for h in range(9, 22):
            peak = (
                1
                + 0.9 * np.exp(-((h - 13) ** 2) / 6)
                + 1.1 * np.exp(-((h - 19) ** 2) / 5)
            )
            weekend = 1.4 if d.weekday() >= 5 else 1.0
            footfall = max(int(rng.normal(40 * peak * weekend, 8)), 0)
            sales = max(200 + 42 * footfall + rng.normal(0, 350), 0)

            rows.append(
                (
                    d + pd.Timedelta(hours=h),
                    footfall,
                    round(sales, 2),
                )
            )

    return pd.DataFrame(
        rows,
        columns=["datetime", "footfall", "sales"],
    )


@st.cache_data
def clean_data(df, remove_outliers):
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce"
    )

    df = (
        df.dropna(subset=["datetime"])
        .drop_duplicates("datetime")
        .sort_values("datetime")
    )

    df[["footfall", "sales"]] = df[
        ["footfall", "sales"]
    ].apply(pd.to_numeric, errors="coerce")

    df[["footfall", "sales"]] = (
        df[["footfall", "sales"]].interpolate()
    )

    df = df.dropna(subset=["footfall", "sales"])

    # Remove outliers using the IQR rule
    if remove_outliers:
        for col in ["footfall", "sales"]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1

            df = df[
                df[col].between(
                    q1 - 1.5 * iqr,
                    q3 + 1.5 * iqr,
                )
            ]

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()
    df["is_weekend"] = (
        df["datetime"].dt.weekday >= 5
    ).astype(int)
    df["month"] = df["datetime"].dt.month_name()

    return df


st.title("🛍️ Footfall Impact on Revenue")
st.caption("Time-based footfall & sales analysis using regression")

st.sidebar.header("Data")

file = st.sidebar.file_uploader("Upload CSV", type="csv")

if file:
    raw = pd.read_csv(file)
    cols = list(raw.columns)

    time_col = st.sidebar.selectbox("Date/time column", cols)
    foot_col = st.sidebar.selectbox(
        "Footfall column",
        cols,
        index=min(1, len(cols) - 1),
    )
    sales_col = st.sidebar.selectbox(
        "Sales column",
        cols,
        index=min(2, len(cols) - 1),
    )

    raw = raw[[time_col, foot_col, sales_col]]
    raw.columns = ["datetime", "footfall", "sales"]

else:
    st.sidebar.info(
        "Using sample data. Upload a CSV to use your own."
    )
    raw = make_sample_data()

drop_out = st.sidebar.checkbox(
    "Remove outliers (IQR)",
    value=False,
)

df = clean_data(raw, drop_out)

if len(df) < 30:
    st.error("Not enough valid rows (need at least 30).")
    st.stop()


# Date filter
picked = st.sidebar.date_input(
    "Date range",
    (df["date"].min(), df["date"].max()),
)

if len(picked) < 2:
    st.info("Pick an end date in the sidebar.")
    st.stop()

start, end = picked

df = df[
    (df["date"] >= start)
    & (df["date"] <= end)
]

if len(df) < 30:
    st.warning(
        "Too few rows in this date range (need at least 30)."
    )
    st.stop()


# Summary metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Footfall",
    f"{df['footfall'].sum():,.0f}",
)

c2.metric(
    "Total Sales",
    f"{df['sales'].sum():,.0f}",
)

total_footfall = df["footfall"].sum()

sales_per_visitor = (
    df["sales"].sum() / total_footfall
    if total_footfall > 0
    else 0
)

c3.metric("Sales / Visitor", f"{sales_per_visitor:,.2f}")

corr = (
    pearsonr(df["footfall"], df["sales"])[0]
    if df["footfall"].nunique() > 1
    and df["sales"].nunique() > 1
    else 0
)

c4.metric("Correlation", f"{corr:.2f}")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Trends",
        "🕒 Time Patterns",
        "🔬 Regression",
        "🎯 Predict",
    ]
)


# Tab 1: Trends
with tab1:
    daily = (
        df.groupby("date")[["footfall", "sales"]]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        daily["date"],
        daily["footfall"],
        color="tab:blue",
        label="Footfall",
    )
    ax.set_ylabel("Footfall", color="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(
        daily["date"],
        daily["sales"],
        color="tab:orange",
        label="Sales",
    )
    ax2.set_ylabel("Sales", color="tab:orange")
    ax2.grid(False)

    ax.set_title("Daily Footfall and Sales")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(11, 4))

    sns.histplot(
        df["sales"],
        kde=True,
        ax=ax,
        color="tab:green",
    )

    ax.set_title("Sales Distribution")
    st.pyplot(fig)


# Tab 2: Time Patterns
with tab2:
    left, right = st.columns(2)

    hourly = (
        df.groupby("hour")[["footfall", "sales"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=hourly,
        x="hour",
        y="footfall",
        ax=ax,
        color="tab:blue",
    )

    ax.set_title("Avg Footfall by Hour")
    left.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=hourly,
        x="hour",
        y="sales",
        ax=ax,
        color="tab:orange",
    )

    ax.set_title("Avg Sales by Hour")
    right.pyplot(fig)

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    heat = (
        df.pivot_table(
            index="weekday",
            columns="hour",
            values="sales",
            aggfunc="mean",
        )
        .reindex(order)
    )

    fig, ax = plt.subplots(figsize=(11, 4))

    sns.heatmap(
        heat,
        cmap="YlOrRd",
        ax=ax,
    )

    ax.set_title("Avg Sales: Weekday x Hour")
    st.pyplot(fig)


# Tab 3: Regression
with tab3:
    # Model A: footfall only
    # Model B: footfall + time features
    y = df["sales"]
    XA = df[["footfall"]]

    XB = pd.get_dummies(
        df[["footfall", "hour", "is_weekend"]],
        columns=["hour"],
        drop_first=True,
    )

    results = []
    models = {}

    for name, X in [
        ("Footfall only", XA),
        ("Footfall + hour + weekend", XB),
    ]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )

        m = LinearRegression().fit(X_tr, y_tr)
        pred = m.predict(X_te)

        results.append(
            {
                "Model": name,
                "R² (test)": round(
                    r2_score(y_te, pred), 3
                ),
                "RMSE": round(
                    np.sqrt(mean_squared_error(y_te, pred)),
                    2,
                ),
            }
        )

        models[name] = (m, X_te, y_te, pred)

    st.subheader("Model comparison")

    st.dataframe(
        pd.DataFrame(results),
        hide_index=True,
    )

    m_simple, X_te, y_te, pred = models["Footfall only"]

    slope = m_simple.coef_[0]
    intercept = m_simple.intercept_

    st.success(
        f"**Impact:** each extra visitor adds about "
        f"**{slope:,.2f}** in sales "
        f"(baseline sales with zero footfall ≈ "
        f"{intercept:,.0f})."
    )

    st.info(
        f"A 10% rise in average footfall ≈ "
        f"**{0.10 * df['footfall'].mean() * slope:,.0f}** "
        f"extra sales per hour."
    )

    left, right = st.columns(2)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.regplot(
        data=df,
        x="footfall",
        y="sales",
        ax=ax,
        scatter_kws={"alpha": 0.3, "s": 12},
        line_kws={"color": "red"},
    )

    ax.set_title("Footfall vs Sales (regression line)")
    left.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        pred,
        y_te - pred,
        alpha=0.4,
        s=12,
    )

    ax.axhline(0, color="red")
    ax.set_xlabel("Predicted sales")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")

    right.pyplot(fig)

    st.caption(
        "Tip: residuals scattered randomly around 0 "
        "mean the linear model may fit well."
    )


# Tab 4: Prediction
with tab4:
    st.subheader("What-if: predict sales from footfall")

    f_min = int(df["footfall"].min())
    f_max = int(df["footfall"].max())

    f = st.slider(
        "Expected footfall (per hour)",
        f_min,
        f_max,
        int(df["footfall"].mean()),
    )

    m_simple = models["Footfall only"][0]

    predicted_sales = m_simple.predict(
        pd.DataFrame({"footfall": [f]})
    )[0]

    st.metric(
        "Predicted sales",
        f"{predicted_sales:,.0f}",
    )


# Download cleaned data
st.download_button(
    "Download cleaned data",
    df.to_csv(index=False),
    "cleaned_data.csv",
)