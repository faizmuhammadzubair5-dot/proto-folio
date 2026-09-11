# %%
get_ipython().run_line_magic("pip", "install streamlit seaborn")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales & Inventory Dashboard", layout="wide")

st.title("Sales and Inventory Dashboard")
st.caption("Clean messy sales data, explore performance, and forecast the next 3 months.")

uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"])


def clean_data(file):
    df = pd.read_csv(file)

    # Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Rename common messy column names
    aliases = {
        "product_name": "product",
        "item": "product",
        "item_name": "product",
        "order_date": "date",
        "sale_date": "date",
        "units": "quantity",
        "qty": "quantity",
        "revenue": "sales",
        "amount": "sales",
        "stock": "inventory",
        "stock_level": "inventory",
    }

    df = df.rename(columns=aliases)

    required_columns = ["date", "product", "quantity", "sales", "inventory"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info(
            "Required columns: date, product, quantity, sales, inventory"
        )
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product"] = df["product"].astype(str).str.strip()

    for col in ["quantity", "sales", "inventory"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "product", "quantity", "sales"])
    df = df[df["quantity"] >= 0]
    df = df[df["sales"] >= 0]
    df["inventory"] = df["inventory"].fillna(0)

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return df


if uploaded_file:
    df = clean_data(uploaded_file)

    st.sidebar.header("Filters")

    products = st.sidebar.multiselect(
        "Select products",
        sorted(df["product"].unique()),
        default=sorted(df["product"].unique()),
    )

    filtered_df = df[df["product"].isin(products)]

    if filtered_df.empty:
        st.warning("No data available for the selected products.")
        st.stop()

    total_sales = filtered_df["sales"].sum()
    total_units = filtered_df["quantity"].sum()
    average_order_value = (
        filtered_df["sales"].sum() / len(filtered_df)
        if len(filtered_df) > 0
        else 0
    )

    monthly_demand = filtered_df.groupby("month")["quantity"].sum()
    average_monthly_demand = monthly_demand.mean()
    current_inventory = filtered_df["inventory"].sum()

    # Simple overstock rule
    overstock_limit = average_monthly_demand * 1.5
    overstock_units = max(current_inventory - overstock_limit, 0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", f"${total_sales:,.2f}")
    col2.metric("Units Sold", f"{total_units:,.0f}")
    col3.metric("Average Order Value", f"${average_order_value:,.2f}")
    col4.metric("Overstock Units", f"{overstock_units:,.0f}")

    st.subheader("Sales and Inventory Overview")

    left, right = st.columns(2)

    with left:
        monthly_sales = (
            filtered_df.groupby("month", as_index=False)["sales"].sum()
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.lineplot(
            data=monthly_sales,
            x="month",
            y="sales",
            marker="o",
            ax=ax,
        )
        ax.set_title("Monthly Sales")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with right:
        product_sales = (
            filtered_df.groupby("product", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(
            data=product_sales,
            x="sales",
            y="product",
            palette="viridis",
            ax=ax,
        )
        ax.set_title("Top Products by Sales")
        ax.set_xlabel("Sales")
        ax.set_ylabel("Product")
        st.pyplot(fig)

    st.subheader("Top 5 Selling Products")

    top_5 = (
        filtered_df.groupby("product", as_index=False)
        .agg(
            sales=("sales", "sum"),
            units_sold=("quantity", "sum"),
            inventory=("inventory", "sum"),
        )
        .sort_values("sales", ascending=False)
        .head(5)
    )

    st.dataframe(top_5, use_container_width=True)

    st.subheader("Inventory Risk Analysis")

    inventory_table = (
        filtered_df.groupby("product", as_index=False)
        .agg(
            units_sold=("quantity", "sum"),
            inventory=("inventory", "sum"),
        )
    )

    inventory_table["average_monthly_demand"] = (
        inventory_table["units_sold"]
        / max(filtered_df["month"].nunique(), 1)
    )

    inventory_table["months_of_stock"] = (
        inventory_table["inventory"]
        / inventory_table["average_monthly_demand"].replace(0, np.nan)
    )

    inventory_table["status"] = np.select(
        [
            inventory_table["months_of_stock"] > 3,
            inventory_table["months_of_stock"] < 1,
        ],
        ["Overstocked", "Low Stock"],
        default="Healthy",
    )

    st.dataframe(inventory_table, use_container_width=True)

    st.subheader("Three-Month Sales Forecast")

    monthly_sales = (
        filtered_df.groupby("month")["sales"]
        .sum()
        .sort_index()
    )

    if len(monthly_sales) >= 2:
        x = np.arange(len(monthly_sales))
        y = monthly_sales.values

        slope, intercept = np.polyfit(x, y, 1)

        future_x = np.arange(len(monthly_sales), len(monthly_sales) + 3)
        forecast_values = np.maximum(
            slope * future_x + intercept,
            0,
        )

        future_dates = pd.date_range(
            monthly_sales.index.max() + pd.offsets.MonthBegin(1),
            periods=3,
            freq="MS",
        )

        forecast = pd.DataFrame(
            {
                "month": future_dates,
                "forecast_sales": forecast_values,
            }
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            monthly_sales.index,
            monthly_sales.values,
            marker="o",
            label="Historical Sales",
        )
        ax.plot(
            forecast["month"],
            forecast["forecast_sales"],
            marker="o",
            linestyle="--",
            label="Forecast",
        )
        ax.set_title("Next 3 Months Sales Forecast")
        ax.set_ylabel("Sales")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.dataframe(forecast, use_container_width=True)
    else:
        st.warning("At least two months of data are required for forecasting.")

    st.subheader("Business Summary")

    st.markdown(
        """
        **Project impact statement:**

        - Identified the top 5 selling products.
        - Highlighted products with more than three months of inventory.
        - Created a three-month sales forecast to support purchasing decisions.
        - Overstock reduction of **18%** should only be reported when it is
          verified using before-and-after inventory data.
        """
    )

    st.download_button(
        "Download Cleaned Data",
        filtered_df.to_csv(index=False),
        "cleaned_sales_data.csv",
        "text/csv",
    )

else:
    st.info(
        "Upload a CSV file with these columns: "
        "`date`, `product`, `quantity`, `sales`, and `inventory`."
    )

# %%
from io import StringIO

rng = np.random.default_rng(42)

dates = pd.date_range("2024-01-01", periods=12, freq="MS")
products = ["Laptop", "Phone", "Tablet", "Headphones", "Monitor"]

sample_rows = []

for date in dates:
    for product in products:
        quantity = int(rng.integers(10, 80))
        price = float(rng.integers(25, 500))
        inventory = int(rng.integers(20, 250))

        sample_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "product": product,
                "quantity": quantity,
                "sales": round(quantity * price, 2),
                "inventory": inventory,
            }
        )

sample_data = pd.DataFrame(sample_rows)

# Convert the generated data into an in-memory CSV file
uploaded_file = StringIO(sample_data.to_csv(index=False))

# Clean the generated data using the existing function
df = clean_data(uploaded_file)

df.head()

# %%
# Notebook version of the Sales and Inventory Dashboard

total_sales = df["sales"].sum()
total_units = df["quantity"].sum()
average_order_value = df["sales"].mean()
current_inventory = df["inventory"].sum()

display(pd.DataFrame({
    "Metric": ["Total Sales", "Units Sold", "Average Order Value", "Inventory"],
    "Value": [
        f"${total_sales:,.2f}",
        f"{total_units:,.0f}",
        f"${average_order_value:,.2f}",
        f"{current_inventory:,.0f}",
    ],
}))

monthly_sales = df.groupby("month", as_index=False)["sales"].sum()
product_sales = (
    df.groupby("product", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.lineplot(
    data=monthly_sales,
    x="month",
    y="sales",
    marker="o",
    ax=axes[0],
)
axes[0].set_title("Monthly Sales")
axes[0].tick_params(axis="x", rotation=45)

sns.barplot(
    data=product_sales,
    x="sales",
    y="product",
    hue="product",
    palette="viridis",
    legend=False,
    ax=axes[1],
)
axes[1].set_title("Sales by Product")

plt.tight_layout()
plt.show()

top_5 = (
    df.groupby("product", as_index=False)
    .agg(
        sales=("sales", "sum"),
        units_sold=("quantity", "sum"),
        inventory=("inventory", "sum"),
    )
    .sort_values("sales", ascending=False)
    .head(5)
)

display(top_5)


