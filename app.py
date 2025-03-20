import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Revenue Forecasting & Analysis", layout="wide")

# Title and description
st.title("Revenue Forecasting & Analysis Dashboard")
st.markdown("Analyze and optimize revenue forecasts through KPI adjustments and sensitivity analysis")

# Initialize session state for data persistence
if 'macrotable_df' not in st.session_state:
    # Generate monthly data for the year
    months = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
    shops = [1]
    data = []

    # Base parameters from example
    base_traffic = np.array([1000, 1200, 1100, 1300, 1250, 1400, 1300, 1250, 1400, 1300, 1250, 1400])
    base_params = {
        'prob_prospect_generation': 0.30,
        'tourist_prob_prospect_generation': 0.0,
        'local_prob_direct_customer_conversion': 0.1,
        'tourist_prob_direct_customer_conversion': 0.05,
        'local_prob_existing_clients_conversion': 0.03,
        'tourist_prob_existing_clients_conversion': 0.01,
        'existing_local_customers': 5000,
        'existing_tourist_customers': 1000,
        'local_avg_ticket_new_from_prospects': 50,
        'local_avg_ticket_new_direct': 60,
        'local_avg_ticket_existing': 40,
        'tourist_avg_ticket_new_from_prospects': 0,
        'tourist_avg_ticket_new_direct': 80,
        'tourist_avg_ticket_existing': 45,
        'budget_sales': 500000
    }

    for shop in shops:
        shop_traffic = base_traffic
        params = base_params.copy()

        for month in months:
            month_idx = months.get_loc(month)
            row = {
                "shop_id": shop,
                "month": month,
                "total_traffic": int(shop_traffic[month_idx]),
                "prev_year_total_traffic": int(shop_traffic[month_idx] * 0.95),
                "ytd_total_traffic": int(shop_traffic[month_idx] * 0.8),
                "locals_new_effectiveness": params['local_prob_direct_customer_conversion'],
                "prospect_generation": params['prob_prospect_generation'],
                "tourist_new_effectiveness": params['tourist_prob_direct_customer_conversion'],
                "local_come_back": params['local_prob_existing_clients_conversion'],
                "tourist_come_back": params['tourist_prob_existing_clients_conversion'],
                "avg_amt_ticket": params['local_avg_ticket_new_direct'],
                "avg_num_ticket_per_customer": 1.0,
                "db_buyers_locals": params['existing_local_customers'],
                "db_buyers_tourist": params['existing_tourist_customers'],
                "prev_year_sales": int(params['budget_sales'] * 0.9),
                "ytd_sales": int(params['budget_sales'] * 0.8),
                "budget_sales": int(params['budget_sales'])
            }
            data.append(row)

    st.session_state.macrotable_df = pd.DataFrame(data)

def compute_customer_forecast(n_months, traffic, prob_prospect_generation,
                            prob_prospect_conversion,
                            prob_direct_customer_conversion, retention_prob,
                            prob_existing_clients_conversion,
                            existing_customers):
    prospects = np.zeros(n_months)
    new_customers_from_prospects = np.zeros(n_months)
    new_customers_direct = np.zeros(n_months)
    total_existing_customers = np.zeros(n_months)

    prospects = traffic * prob_prospect_generation
    rows, cols = np.meshgrid(np.arange(n_months), np.arange(n_months), indexing="ij")
    M_conversion_matrix_ops = np.where(cols <= rows, prob_prospect_conversion[rows-cols], 0)
    new_customers_from_prospects = M_conversion_matrix_ops @ prospects
    new_customers_direct = traffic * prob_direct_customer_conversion
    M_retention_matrix_ops = np.where(cols <= rows, retention_prob[rows-cols], 0)
    retained_customers_from_prospects = M_retention_matrix_ops @ new_customers_from_prospects
    retained_customers_direct = M_retention_matrix_ops @ new_customers_direct
    existing_customers_vector = np.full(n_months, existing_customers * prob_existing_clients_conversion)
    total_existing_customers = existing_customers_vector + retained_customers_from_prospects + retained_customers_direct

    forecast_df = pd.DataFrame({
        "prospects": prospects,
        "new_customers_from_prospects": new_customers_from_prospects,
        "new_customers_direct": new_customers_direct,
        "total_existing_customers": total_existing_customers
    }, index=[f"Month {i+1}" for i in range(n_months)])

    return forecast_df

def compute_revenue_forecast(local_customer_forecast_df, tourist_customer_forecast_df, avg_ticket_df):
    total_revenue_df = pd.DataFrame(index=local_customer_forecast_df.index)

    total_revenue_df['revenue_local_new_from_prospects'] = local_customer_forecast_df['new_customers_from_prospects']*avg_ticket_df['local_avg_ticket_new_from_prospects'].iloc[0]
    total_revenue_df['revenue_local_new_from_direct'] = local_customer_forecast_df['new_customers_direct']*avg_ticket_df['local_avg_ticket_new_direct'].iloc[0]
    total_revenue_df['revenue_local_existing'] = local_customer_forecast_df['total_existing_customers']*avg_ticket_df['local_avg_ticket_existing'].iloc[0]
    total_revenue_df['revenue_local_total'] = total_revenue_df['revenue_local_new_from_prospects']+total_revenue_df['revenue_local_new_from_direct']+total_revenue_df['revenue_local_existing']

    total_revenue_df['revenue_tourist_new_from_prospects'] = tourist_customer_forecast_df['new_customers_from_prospects']*avg_ticket_df['tourist_avg_ticket_new_from_prospects'].iloc[0]
    total_revenue_df['revenue_tourist_new_from_direct'] = tourist_customer_forecast_df['new_customers_direct']*avg_ticket_df['tourist_avg_ticket_new_direct'].iloc[0]
    total_revenue_df['revenue_tourist_existing'] = tourist_customer_forecast_df['total_existing_customers']*avg_ticket_df['tourist_avg_ticket_existing'].iloc[0]
    total_revenue_df['revenue_tourist_total'] = total_revenue_df['revenue_tourist_new_from_prospects']+total_revenue_df['revenue_tourist_new_from_direct']+total_revenue_df['revenue_tourist_existing']

    total_revenue_df['revenue_total'] = total_revenue_df['revenue_local_total']+total_revenue_df['revenue_tourist_total']

    return total_revenue_df

def compute_kpi(traffic, existing_local_customers, existing_tourist_customers, total_revenue_df, local_customer_forecast_df, tourist_customer_forecast_df):
    local_new_closing_ratio = local_customer_forecast_df['new_customers_direct'].sum() / traffic.sum()
    prospect_closing_ratio = local_customer_forecast_df['new_customers_from_prospects'].sum() / traffic.sum()
    local_come_back = local_customer_forecast_df['total_existing_customers'].sum() / (
        existing_local_customers + local_customer_forecast_df['new_customers_direct'].sum() + local_customer_forecast_df['new_customers_from_prospects'].sum()
    )
    tourist_new_closing_ratio = tourist_customer_forecast_df['new_customers_direct'].sum() / traffic.sum()
    tourist_come_back = tourist_customer_forecast_df['total_existing_customers'].sum() / (
        existing_tourist_customers + tourist_customer_forecast_df['new_customers_direct'].sum()
    )

    kpi_df = pd.DataFrame({
        "KPI": ["Local New Closing Ratio", "Prospect Closing Ratio", "Local Comeback Rate", "Tourist New Closing Ratio", "Tourist Comeback Rate"],
        "Value": [local_new_closing_ratio, prospect_closing_ratio, local_come_back, tourist_new_closing_ratio, tourist_come_back]
    })

    return kpi_df

def calculate_sales(df):
    """Calculate sales based on input parameters"""
    n_months = len(df)
    traffic = df["total_traffic"].values
    prob_prospect_generation = df["prospect_generation"].values[0]
    prob_prospect_conversion = np.array([0.3, 0.2, 0.1] + [0] * (n_months-3))
    prob_direct_customer_conversion = df["locals_new_effectiveness"].values[0]
    retention_prob = np.array([0.2, 0.1, 0.06, 0.05, 0.04, 0.03] + [0.03] * (n_months-6))
    prob_existing_clients_conversion = df["local_come_back"].values[0]
    existing_customers = df["db_buyers_locals"].values[0]

    forecast = compute_customer_forecast(
        n_months, traffic, prob_prospect_generation,
        prob_prospect_conversion, prob_direct_customer_conversion,
        retention_prob, prob_existing_clients_conversion,
        existing_customers
    )

    return forecast["new_customers_from_prospects"] * df["avg_amt_ticket"].values[0]

def perform_sensitivity_analysis(traffic, prob_prospect_generation, prob_prospect_conversion,
                               local_prob_direct_customer_conversion, local_retention_prob,
                               local_prob_existing_clients_conversion, tourist_prob_direct_customer_conversion,
                               tourist_prob_existing_clients_conversion, tourist_retention_prob,
                               existing_local_customers, existing_tourist_customers, avg_ticket_df):
    """Perform sensitivity analysis on key variables"""
    variables = {
        "prob_prospect_generation": prob_prospect_generation,
        "local_prob_prospect_conversion": prob_prospect_conversion,
        "local_prob_direct_customer_conversion": local_prob_direct_customer_conversion,
        "local_prob_existing_clients_conversion": local_prob_existing_clients_conversion,
        "local_retention_prob": local_retention_prob,
        "tourist_prob_direct_customer_conversion": tourist_prob_direct_customer_conversion,
        "tourist_prob_existing_clients_conversion": tourist_prob_existing_clients_conversion,
        "tourist_retention_prob": tourist_retention_prob
    }

    # Calculate baseline forecasts
    local_baseline_forecast = compute_customer_forecast(
        len(traffic), traffic, prob_prospect_generation, prob_prospect_conversion,
        local_prob_direct_customer_conversion, local_retention_prob,
        local_prob_existing_clients_conversion, existing_local_customers
    )

    tourist_baseline_forecast = compute_customer_forecast(
        len(traffic), traffic, 0, prob_prospect_conversion,
        tourist_prob_direct_customer_conversion, tourist_retention_prob,
        tourist_prob_existing_clients_conversion, existing_tourist_customers
    )

    baseline_revenue = compute_revenue_forecast(
        local_baseline_forecast, tourist_baseline_forecast, avg_ticket_df
    )['revenue_total'].sum()

    sensitivity_results = {}
    for var in variables:
        modified_variables = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in variables.items()}
        modified_variables[var] = variables[var] * 1.01  # 1% increase

        local_forecast = compute_customer_forecast(
            len(traffic), traffic,
            modified_variables["prob_prospect_generation"],
            modified_variables["local_prob_prospect_conversion"],
            modified_variables["local_prob_direct_customer_conversion"],
            modified_variables["local_retention_prob"],
            modified_variables["local_prob_existing_clients_conversion"],
            existing_local_customers
        )

        tourist_forecast = compute_customer_forecast(
            len(traffic), traffic, 0,
            modified_variables["local_prob_prospect_conversion"],
            modified_variables["tourist_prob_direct_customer_conversion"],
            modified_variables["tourist_retention_prob"],
            modified_variables["tourist_prob_existing_clients_conversion"],
            existing_tourist_customers
        )

        new_revenue = compute_revenue_forecast(
            local_forecast, tourist_forecast, avg_ticket_df
        )['revenue_total'].sum()

        sensitivity_results[var] = ((new_revenue - baseline_revenue) / baseline_revenue) * 100

    return pd.DataFrame.from_dict(sensitivity_results, orient='index', columns=['Impact %']).sort_values(by='Impact %', ascending=False)

# Sidebar for shop selection and analysis type
st.sidebar.title("Analysis Controls")
shop_id = st.sidebar.selectbox("Select Shop ID", st.session_state.macrotable_df["shop_id"].unique())
selected_month = st.sidebar.selectbox(
    "Select Month",
    options=st.session_state.macrotable_df["month"].unique(),
    format_func=lambda x: x.strftime('%B %Y')
)
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Revenue Forecast", "Sensitivity Analysis", "Gap Analysis", "KPI Optimization"]
)

# Get data for selected shop and month
shop_data = st.session_state.macrotable_df[
    (st.session_state.macrotable_df["shop_id"] == shop_id) &
    (st.session_state.macrotable_df["month"] == selected_month)
].copy()

# Main content area
if analysis_type == "Revenue Forecast":
    st.header("Revenue Forecast Analysis")

    # Get shop-specific data
    shop_monthly_data = st.session_state.macrotable_df[
        st.session_state.macrotable_df["shop_id"] == shop_id
    ].copy()

    # Get shop-specific parameters from the generated data
    n_months = 12
    traffic = shop_monthly_data["total_traffic"].values  # Already includes shop-specific variations
    prob_prospect_generation = shop_monthly_data["prospect_generation"].values[0]  # Shop-specific value
    prob_prospect_conversion = np.array([0.3, 0.2, 0.1] + [0] * 9)  # Common conversion pattern
    local_prob_direct_customer_conversion = shop_monthly_data["locals_new_effectiveness"].values[0]  # Shop-specific value
    tourist_prob_direct_customer_conversion = shop_monthly_data["tourist_new_effectiveness"].values[0]  # Shop-specific value
    local_retention_prob = np.array([0.2, 0.1, 0.06, 0.05, 0.04, 0.03] + [0.03] * 6)  # Common retention pattern
    tourist_retention_prob = np.array([0.12, 0.6, 0.5, 0.3, 0.02, 0.01] + [0.01] * 6)  # Common retention pattern
    local_prob_existing_clients_conversion = shop_monthly_data["local_come_back"].values[0]  # Shop-specific value
    tourist_prob_existing_clients_conversion = shop_monthly_data["tourist_come_back"].values[0]  # Shop-specific value
    existing_local_customers = shop_monthly_data["db_buyers_locals"].values[0]  # Shop-specific value
    existing_tourist_customers = shop_monthly_data["db_buyers_tourist"].values[0]  # Shop-specific value

    # Calculate forecasts with shop-specific parameters
    local_forecast = compute_customer_forecast(
        n_months, traffic, prob_prospect_generation, prob_prospect_conversion,
        local_prob_direct_customer_conversion, local_retention_prob,
        local_prob_existing_clients_conversion, existing_local_customers
    )

    tourist_forecast = compute_customer_forecast(
        n_months, traffic, 0, prob_prospect_conversion,
        tourist_prob_direct_customer_conversion, tourist_retention_prob,
        tourist_prob_existing_clients_conversion, existing_tourist_customers
    )

    # Create average ticket dataframe

    local_avg_ticket_new_from_prospects = 50
    local_avg_ticket_new_direct = 60
    local_avg_ticket_existing = 40

    tourist_avg_ticket_new_from_prospects = 0
    tourist_avg_ticket_new_direct = 80
    tourist_avg_ticket_existing = 45

    avg_ticket_df = pd.DataFrame({
        "local_avg_ticket_new_from_prospects": [local_avg_ticket_new_from_prospects],
        "local_avg_ticket_new_direct": [local_avg_ticket_new_direct],
        "local_avg_ticket_existing": [local_avg_ticket_existing],
        "tourist_avg_ticket_new_from_prospects": [tourist_avg_ticket_new_from_prospects],
        "tourist_avg_ticket_new_direct": [tourist_avg_ticket_new_direct],
        "tourist_avg_ticket_existing": [tourist_avg_ticket_existing]
    })

    # Calculate revenue forecast
    revenue_forecast = compute_revenue_forecast(local_forecast, tourist_forecast, avg_ticket_df)

    # Create accumulated monthly data for comparison
    date_range = pd.date_range(start=selected_month, periods=12, freq='M')
    monthly_data = pd.DataFrame(index=date_range)
    monthly_data['Total Revenue Forecast'] = revenue_forecast['revenue_total'].cumsum().values
    monthly_data['Budget'] = [shop_data['budget_sales'].iloc[0]] * 12
    monthly_data['Previous Year Sales'] = [(shop_data['prev_year_sales'].iloc[0])] * 12
    monthly_data.index = monthly_data.index.strftime('%Y %b')

    # Plot accumulated revenue comparison
    fig_trend = px.line(
        monthly_data,
        title="Accumulated Revenue Forecast vs Budget and Previous Year",
        labels={"value": "Accumulated Revenue (€)", "index": "Month"},
        y=["Total Revenue Forecast", "Budget", "Previous Year Sales"]
    )
    fig_trend.update_layout(
        yaxis=dict(
            tickformat="€,.0f",
            title="Revenue (€)"
        )
    )
    st.plotly_chart(fig_trend, use_container_width=True)

elif analysis_type == "Sensitivity Analysis":
    st.header("Sensitivity Analysis")

    # Initialize parameters for sensitivity analysis
    n_months = 12
    traffic = np.array([1000, 1200, 1100, 1300, 1250, 1400, 1300, 1250, 1400, 1300, 1250, 1400])
    prob_prospect_generation = 0.30
    prob_prospect_conversion = np.array([0.3, 0.2, 0.1] + [0] * 9)
    local_prob_direct_customer_conversion = 0.1
    tourist_prob_direct_customer_conversion = 0.05
    local_retention_prob = np.array([0.2, 0.1, 0.06, 0.05, 0.04, 0.03] + [0.03] * 6)
    tourist_retention_prob = np.array([0.12, 0.6, 0.5, 0.3, 0.02, 0.01] + [0.01] * 6)
    local_prob_existing_clients_conversion = 0.03
    tourist_prob_existing_clients_conversion = 0.01
    existing_local_customers = 5000
    existing_tourist_customers = 1000

    # Create average ticket dataframe
    avg_ticket_df = pd.DataFrame({
        'local_avg_ticket_new_from_prospects': [50],
        'local_avg_ticket_new_direct': [60],
        'local_avg_ticket_existing': [40],
        'tourist_avg_ticket_new_from_prospects': [0],
        'tourist_avg_ticket_new_direct': [80],
        'tourist_avg_ticket_existing': [45]
    })

    sensitivity_df = perform_sensitivity_analysis(
        traffic, prob_prospect_generation, prob_prospect_conversion,
        local_prob_direct_customer_conversion, local_retention_prob,
        local_prob_existing_clients_conversion, tourist_prob_direct_customer_conversion,
        tourist_prob_existing_clients_conversion, tourist_retention_prob,
        existing_local_customers, existing_tourist_customers, avg_ticket_df
    )

    # Plot sensitivity analysis
    fig = px.bar(
        sensitivity_df,
        title=f"% Impact on Total Revenue",
        labels={"index": "Variable", "Impact %": "% Impact on Total Revenue"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sensitivity_df.style.format({"Impact %": "{:.2f}%"}))

elif analysis_type == "Gap Analysis":
    st.header("Gap Analysis")
    # Calculate revenue forecast for all months
    n_months = 12
    monthly_data = st.session_state.macrotable_df[
        st.session_state.macrotable_df["shop_id"] == shop_id
    ].copy()

    local_forecast = compute_customer_forecast(
        n_months, monthly_data["total_traffic"].values,
        monthly_data["prospect_generation"].values[0],
        np.array([0.3, 0.2, 0.1] + [0] * (n_months-3)),
        monthly_data["locals_new_effectiveness"].values[0],
        np.array([0.2, 0.1, 0.06, 0.05, 0.04, 0.03] + [0.03] * (n_months-6)),
        monthly_data["local_come_back"].values[0],
        monthly_data["db_buyers_locals"].values[0]
    )

    tourist_forecast = compute_customer_forecast(
        n_months, monthly_data["total_traffic"].values,
        0,
        np.array([0.3, 0.2, 0.1] + [0] * (n_months-3)),
        monthly_data["tourist_new_effectiveness"].values[0],
        np.array([0.12, 0.6, 0.5, 0.3, 0.02, 0.01] + [0.01] * (n_months-6)),
        monthly_data["tourist_come_back"].values[0],
        monthly_data["db_buyers_tourist"].values[0]
    )

    avg_ticket_df = pd.DataFrame({
        "local_avg_ticket_new_from_prospects": [50],
        "local_avg_ticket_new_direct": [60],
        "local_avg_ticket_existing": [40],
        "tourist_avg_ticket_new_from_prospects": [0],
        "tourist_avg_ticket_new_direct": [80],
        "tourist_avg_ticket_existing": [45]
    })

    revenue_forecast = compute_revenue_forecast(local_forecast, tourist_forecast, avg_ticket_df)
    current_month_idx = monthly_data[monthly_data["month"] == selected_month].index[0] - 1
    current_forecast = revenue_forecast["revenue_total"].iloc[:current_month_idx + 1].sum()
    remaining_forecast = revenue_forecast["revenue_total"].iloc[current_month_idx + 1:].sum()
    budget_target = monthly_data["budget_sales"].iloc[0]
    gap = current_forecast + remaining_forecast - budget_target

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sales (YTD)", f"€{current_forecast:,.0f}")
    col2.metric("Remaining Months Projection", f"€{remaining_forecast:,.0f}")
    col3.metric("Budget Target", f"€{budget_target:,.0f}")
    col4.metric("Gap", f"€{gap:,.0f}")

    # Calculate remaining months forecast
    remaining_months = 13 - selected_month.month  # Number of remaining months
    monthly_data = monthly_data[monthly_data["month"] >= selected_month].copy()
    monthly_data["forecast"] = revenue_forecast["revenue_total"].values[:remaining_months]
    monthly_data["monthly_budget"] = monthly_data["budget_sales"]/12
    monthly_data["gap"] = monthly_data["monthly_budget"] - monthly_data["forecast"]

    # Plot monthly gaps
    monthly_data["month"] = monthly_data["month"].dt.strftime('%Y %b')
    fig_monthly = px.bar(
        monthly_data,
        x="month",
        y="gap",
        title="Monthly Budget Gaps for Remaining Months",
        labels={"gap": "Gap (€)", "month": "Month"}
    )
    fig_monthly.update_layout(
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=pd.date_range(start=selected_month, 
                                      end='2020-12-31', 
                                      freq='M').strftime('%Y %b')
        )
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

else:  # KPI Optimization
    st.header("KPI Optimization")

    # Get initial values from the data
    initial_prospect_gen = shop_data["prospect_generation"].values[0]
    initial_local_effectiveness = shop_data["locals_new_effectiveness"].values[0]
    initial_tourist_effectiveness = shop_data["tourist_new_effectiveness"].values[0]
    initial_local_comeback = shop_data["local_come_back"].values[0]
    initial_tourist_comeback = shop_data["tourist_come_back"].values[0]

    # Calculate the required increase to meet budget
    n_months = 12
    traffic = shop_data["total_traffic"].values[0]
    prob_prospect_conversion = np.array([0.3, 0.2, 0.1] + [0] * 9)
    local_retention_prob = np.array([0.2, 0.1, 0.06, 0.05, 0.04, 0.03] + [0.03] * 6)
    tourist_retention_prob = np.array([0.12, 0.6, 0.5, 0.3, 0.02, 0.01] + [0.01] * 6)
    budget_target = shop_data["budget_sales"].iloc[0]

    # Calculate baseline revenue with current KPIs
    baseline_local_forecast = compute_customer_forecast(
        n_months, np.full(n_months, traffic), initial_prospect_gen,
        prob_prospect_conversion, initial_local_effectiveness,
        local_retention_prob, initial_local_comeback,
        shop_data["db_buyers_locals"].values[0]
    )

    baseline_tourist_forecast = compute_customer_forecast(
        n_months, np.full(n_months, traffic), 0,
        prob_prospect_conversion, initial_tourist_effectiveness,
        tourist_retention_prob, initial_tourist_comeback,
        shop_data["db_buyers_tourist"].values[0]
    )

    # Create average ticket dataframe
    avg_ticket_df = pd.DataFrame({
        "local_avg_ticket_new_from_prospects": [50],
        "local_avg_ticket_new_direct": [60],
        "local_avg_ticket_existing": [40],
        "tourist_avg_ticket_new_from_prospects": [0],
        "tourist_avg_ticket_new_direct": [80],
        "tourist_avg_ticket_existing": [45]
    })

    baseline_revenue = compute_revenue_forecast(
        baseline_local_forecast, baseline_tourist_forecast, avg_ticket_df
    )['revenue_total'].sum()

    # Calculate required increase factor
    increase_factor = budget_target / baseline_revenue

    # Calculate optimized KPI values (increase the most impactful KPIs more)
    optimized_prospect_gen = min(initial_prospect_gen * increase_factor, 1.0)
    optimized_local_effectiveness = min(initial_local_effectiveness * increase_factor, 1.0)
    optimized_tourist_effectiveness = min(initial_tourist_effectiveness * increase_factor, 1.0)
    optimized_local_comeback = min(initial_local_comeback * increase_factor, 1.0)
    optimized_tourist_comeback = min(initial_tourist_comeback * increase_factor, 1.0)

    # Add KPI adjustment controls in sidebar
    st.sidebar.subheader("KPI Adjustments")

    # Create sliders for KPI adjustments with optimized values as defaults
    adjusted_prospect_gen = st.sidebar.slider(
        "Prospect Generation Rate",
        min_value=0.0,
        max_value=1.0,
        value=float(optimized_prospect_gen),
        step=0.01,
        format="%.2f"
    )

    adjusted_local_effectiveness = st.sidebar.slider(
        "Local New Closing Ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(optimized_local_effectiveness),
        step=0.01,
        format="%.2f"
    )

    adjusted_tourist_effectiveness = st.sidebar.slider(
        "Tourist New Closing Ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(optimized_tourist_effectiveness),
        step=0.01,
        format="%.2f"
    )

    adjusted_local_comeback = st.sidebar.slider(
        "Local Comeback Rate",
        min_value=0.0,
        max_value=1.0,
        value=float(optimized_local_comeback),
        step=0.01,
        format="%.2f"
    )

    adjusted_tourist_comeback = st.sidebar.slider(
        "Tourist Comeback Rate",
        min_value=0.0,
        max_value=1.0,
        value=float(optimized_tourist_comeback),
        step=0.01,
        format="%.2f"
    )

    # Forecast with adjusted KPIs
    adjusted_local_forecast = compute_customer_forecast(
        n_months,
        np.full(n_months, traffic),
        adjusted_prospect_gen,
        prob_prospect_conversion,
        adjusted_local_effectiveness,
        local_retention_prob,
        adjusted_local_comeback,
        shop_data["db_buyers_locals"].values[0]
    )

    adjusted_tourist_forecast = compute_customer_forecast(
        n_months,
        np.full(n_months, traffic),
        0,
        prob_prospect_conversion,
        adjusted_tourist_effectiveness,
        tourist_retention_prob,
        adjusted_tourist_comeback,
        shop_data["db_buyers_tourist"].values[0]
    )

    # Calculate adjusted revenue forecast
    adjusted_revenue = compute_revenue_forecast(
        adjusted_local_forecast,
        adjusted_tourist_forecast,
        avg_ticket_df
    )

    # Calculate KPIs and impact metrics
    shop_monthly_data = st.session_state.macrotable_df[
        st.session_state.macrotable_df["shop_id"] == shop_id
    ].copy()
    current_month_idx = shop_monthly_data[shop_monthly_data["month"] == selected_month].index[0] - 1
    ytd_revenue = adjusted_revenue["revenue_total"].iloc[:current_month_idx + 1].sum()
    remaining_forecast = adjusted_revenue["revenue_total"].iloc[current_month_idx + 1:].sum()
    gap = ytd_revenue + remaining_forecast - budget_target

    # Display impact analysis using metrics
    st.subheader("Impact Analysis")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sales (YTD)", f"€{ytd_revenue:,.0f}")
    col2.metric("Remaining Months Projection", f"€{remaining_forecast:,.0f}")
    col3.metric("Budget Target", f"€{budget_target:,.0f}")
    col4.metric("Gap", f"€{gap:,.0f}")

    # Plot revenue comparison with accumulated values
    revenue_comparison = pd.DataFrame({
        "Month": [f"Month {i+1}" for i in range(n_months)],
        "Adjusted Revenue": adjusted_revenue["revenue_total"].cumsum(),
        "Monthly Budget": [shop_data["budget_sales"].values[0]] * 12
    })

    fig = px.line(
        revenue_comparison,
        x="Month",
        y=["Adjusted Revenue", "Monthly Budget"],
        title="Accumulated Revenue Projection with Adjusted KPIs",
        labels={"value": "Accumulated Revenue (€)", "variable": "Type"}
    )
    fig.update_layout(
        yaxis=dict(
            tickformat="€,.0f",
            title="Revenue (€)"
        )
    )
    st.plotly_chart(fig, use_container_width=True)