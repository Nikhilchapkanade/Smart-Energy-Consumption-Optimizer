# dashboard.py - Enhanced Version
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# Load models
@st.cache_resource
def load_models():
    def try_load(filename, fallback):
        try:
            return joblib.load(filename)
        except Exception as e:
            st.warning(f"Model '{filename}' could not be loaded: {e}")
            return fallback

    class MeanPredictor:
        def __init__(self, mean_value=0.0):
            self.mean_value = mean_value
        def predict(self, X):
            return np.repeat(self.mean_value, len(X))

    return {
        'consumption': try_load('consumption_model.pkl', MeanPredictor(70.0)),
        'anomaly': try_load('anomaly_model.pkl', None),
        'scaler': try_load('anomaly_scaler.pkl', None),
        'hvac': try_load('hvac_kwh_model.pkl', MeanPredictor(20.0)),
        'water_heater': try_load('water_heater_kwh_model.pkl', MeanPredictor(5.0)),
        'lighting': try_load('lighting_kwh_model.pkl', MeanPredictor(2.0))
    }

@st.cache_data
def load_data():
    return pd.read_csv('energy_consumption_data.csv')

models = load_models()
df = load_data()
df['timestamp'] = pd.to_datetime(df['timestamp'])

# App Configuration
st.set_page_config(page_title="Energy Optimizer", layout="wide", page_icon="⚡")
st.title("⚡ Smart Energy Consumption Optimizer")

# Sidebar Controls with Real-Time Monitor
st.sidebar.header("🎛️ Settings")

# NEW: Real-Time Clock Widget
st.sidebar.markdown("---")
st.sidebar.subheader("🕐 Real-Time Monitor")
current_time = datetime.now()
st.sidebar.write(f"**Current Time:** {current_time.strftime('%I:%M %p')}")
st.sidebar.write(f"**Date:** {current_time.strftime('%B %d, %Y')}")

# Auto-predict for current hour
current_features = pd.DataFrame({
    'hour': [current_time.hour],
    'day_of_week': [current_time.weekday()],
    'month': [current_time.month],
    'is_weekend': [current_time.weekday() >= 5],
    'temperature': [72]
})

current_prediction = models['consumption'].predict(current_features)[0]
current_cost = current_prediction * (0.12 if 9 <= current_time.hour <= 21 else 0.08)

st.sidebar.metric("Predicted Usage Now", f"{current_prediction:.2f} kWh")
st.sidebar.metric("Estimated Cost", f"₹{current_cost:.2f}")

if 9 <= current_time.hour <= 21:
    st.sidebar.warning("⚠️ Currently in PEAK hours")
else:
    st.sidebar.success("✅ Currently in OFF-PEAK hours")

st.sidebar.markdown("---")

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['timestamp'].min(), df['timestamp'].max()),
    min_value=df['timestamp'].min(),
    max_value=df['timestamp'].max()
)

# Main Dashboard - ADDED NEW TAB
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "🔮 Predictions", 
    "🚨 Anomalies", 
    "💡 Recommendations",
    "🏆 Gamification",
    "🧮 Calculator"  # NEW TAB
])

# --- TAB 1: OVERVIEW (ENHANCED) ---
with tab1:
    st.header("Energy Consumption Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_kwh = df['total_kwh'].sum()
        st.metric("Total Consumption", f"{total_kwh:,.0f} kWh")
    
    with col2:
        total_cost = df['total_cost'].sum()
        st.metric("Total Cost", f"₹{total_cost:,.2f}")
    
    with col3:
        avg_daily = df.groupby(df['timestamp'].dt.date)['total_kwh'].sum().mean()
        st.metric("Avg Daily Usage", f"{avg_daily:.1f} kWh")
    
    with col4:
        potential_savings = total_cost * 0.15
        st.metric("Potential Savings", f"₹{potential_savings:,.2f}", delta="15%")
    
    # NEW: Data Export Feature
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("📥 Download Full Report"):
            report_df = df[['timestamp', 'total_kwh', 'total_cost', 'temperature', 'hvac_kwh', 'water_heater_kwh']].copy()
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"energy_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Consumption Timeline
    st.subheader("📈 Consumption Timeline")
    daily_usage = df.groupby(df['timestamp'].dt.date).agg({
        'total_kwh': 'sum',
        'total_cost': 'sum'
    }).reset_index()
    daily_usage.columns = ['date', 'kwh', 'cost']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_usage['date'], 
        y=daily_usage['kwh'],
        mode='lines',
        name='Consumption (kWh)',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    fig.update_layout(
        title="Daily Energy Consumption",
        xaxis_title="Date",
        yaxis_title="kWh",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # NEW: Weekly Pattern Analysis
    st.subheader("📅 Weekly Pattern Analysis")
    
    weekly_pattern = df.groupby('day_of_week').agg({
        'total_kwh': 'mean',
        'total_cost': 'mean'
    }).reset_index()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pattern['day_name'] = weekly_pattern['day_of_week'].apply(lambda x: day_names[x])
    
    fig = go.Figure([
        go.Bar(
            x=weekly_pattern['day_name'],
            y=weekly_pattern['total_kwh'],
            marker_color=['#3498DB']*5 + ['#E74C3C', '#E74C3C'],
            text=weekly_pattern['total_kwh'].round(2),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Average Consumption by Day of Week",
        xaxis_title="Day",
        yaxis_title="Average kWh",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("📊 **Insight:** Weekend consumption (shown in red) is typically higher due to more time at home")
    
    # NEW: Seasonal Comparison
    st.subheader("🌡️ Seasonal Analysis")
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    seasonal_data = df.groupby('season').agg({
        'total_kwh': 'mean',
        'total_cost': 'mean',
        'hvac_kwh': 'mean'
    }).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure([go.Bar(
            x=seasonal_data.index,
            y=seasonal_data['total_kwh'],
            marker_color=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12'],
            text=seasonal_data['total_kwh'],
            textposition='outside'
        )])
        fig.update_layout(title="Average Consumption by Season", yaxis_title="kWh")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(seasonal_data, use_container_width=True)
        highest_season = seasonal_data['total_kwh'].idxmax()
        st.info(f"⚠️ **Highest consumption:** {highest_season} (likely due to heating/cooling)")
    
    # Appliance Breakdown
    st.subheader("🔌 Appliance Breakdown")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        appliance_totals = {
            'HVAC': df['hvac_kwh'].sum(),
            'Water Heater': df['water_heater_kwh'].sum(),
            'Refrigerator': df['refrigerator_kwh'].sum(),
            'Lighting': df['lighting_kwh'].sum(),
            'Other': df['other_kwh'].sum()
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(appliance_totals.keys()),
            values=list(appliance_totals.values()),
            hole=0.3
        )])
        fig_pie.update_layout(title="Energy Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        appliance_df = pd.DataFrame([
            {'Appliance': k, 'kWh': v, 'Cost': v * 0.12, '% of Total': v/total_kwh*100}
            for k, v in appliance_totals.items()
        ]).sort_values('kWh', ascending=False)
        
        st.dataframe(appliance_df.style.format({
            'kWh': '{:.1f}',
            'Cost': '₹{:.2f}',
            '% of Total': '{:.1f}%'
        }), use_container_width=True)
    
    # NEW: Appliance Efficiency Rating
    st.subheader("⭐ Appliance Efficiency Rating")
    
    efficiency_thresholds = {
        'HVAC': {'efficient': 2500, 'wasteful': 4000},
        'Water Heater': {'efficient': 2000, 'wasteful': 3500},
        'Refrigerator': {'efficient': 1000, 'wasteful': 1500},
        'Lighting': {'efficient': 500, 'wasteful': 1000},
        'Other': {'efficient': 1500, 'wasteful': 2500}
    }
    
    def get_rating(appliance, consumption):
        thresholds = efficiency_thresholds[appliance]
        if consumption < thresholds['efficient']:
            return '🟢 Efficient'
        elif consumption < thresholds['wasteful']:
            return '🟡 Normal'
        else:
            return '🔴 Wasteful'
    
    ratings_data = []
    for appliance, consumption in appliance_totals.items():
        rating = get_rating(appliance, consumption)
        status = '✅ Good' if '🟢' in rating else ('⚠️ Check' if '🟡' in rating else '❌ Optimize')
        ratings_data.append({
            'Appliance': appliance,
            'Annual kWh': round(consumption, 1),
            'Rating': rating,
            'Status': status
        })
    
    ratings_df = pd.DataFrame(ratings_data)
    st.dataframe(ratings_df, use_container_width=True)
    
    wasteful_appliances = [r['Appliance'] for r in ratings_data if '🔴' in r['Rating']]
    if wasteful_appliances:
        st.error(f"⚠️ **Action Required:** {', '.join(wasteful_appliances)} showing high consumption. Consider maintenance or replacement.")
    else:
        st.success("✅ All appliances operating efficiently!")

# --- TAB 2: PREDICTIONS (ENHANCED) ---
with tab2:
    st.header("🔮 Energy Consumption Predictions")
    
    st.write("Predict energy usage for future time periods:")
    
    col1, col2 = st.columns(2)
    with col1:
        pred_date = st.date_input("Select Date", datetime.now())
    with col2:
        pred_hour = st.slider("Select Hour", 0, 23, 12)
    
    # NEW: Temperature Scenarios
    col1, col2 = st.columns(2)
    with col1:
        pred_temp = st.slider("Expected Temperature (°F)", 30, 100, 70)
    with col2:
        temp_scenario = st.selectbox("Quick Scenario", [
            "Custom (use slider)",
            "Cold Winter (40°F)",
            "Mild (70°F)", 
            "Hot Summer (95°F)"
        ])
        
        temp_map = {
            "Cold Winter (40°F)": 40,
            "Mild (70°F)": 70,
            "Hot Summer (95°F)": 95
        }
        
        if temp_scenario != "Custom (use slider)":
            pred_temp = temp_map[temp_scenario]
    
    if st.button("🔮 Generate Prediction"):
        day_of_week = pred_date.weekday()
        is_weekend = day_of_week >= 5
        month = pred_date.month
        
        features = pd.DataFrame({
            'hour': [pred_hour],
            'day_of_week': [day_of_week],
            'month': [month],
            'is_weekend': [is_weekend],
            'temperature': [pred_temp]
        })
        
        prediction = models['consumption'].predict(features)[0]
        cost = prediction * (0.12 if 9 <= pred_hour <= 21 else 0.08)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Usage", f"{prediction:.2f} kWh")
        with col2:
            st.metric("Estimated Cost", f"₹{cost:.2f}")
        with col3:
            status = "Peak" if 9 <= pred_hour <= 21 else "Off-Peak"
            st.metric("Rate Period", status)
        
        # Appliance breakdown prediction
        st.subheader("Predicted Appliance Usage")
        appliance_features = pd.DataFrame({
            'hour': [pred_hour],
            'day_of_week': [day_of_week],
            'temperature': [pred_temp]
        })
        
        appliance_preds = {
            'HVAC': models['hvac'].predict(appliance_features)[0],
            'Water Heater': models['water_heater'].predict(appliance_features)[0],
            'Lighting': models['lighting'].predict(appliance_features)[0],
            'Refrigerator': 0.15,
            'Other': prediction * 0.25
        }
        
        fig = go.Figure([go.Bar(
            x=list(appliance_preds.keys()),
            y=list(appliance_preds.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3', '#C7CEEA'],
            text=[f"{v:.2f}" for v in appliance_preds.values()],
            textposition='outside'
        )])
        fig.update_layout(
            title="Predicted Appliance Consumption",
            yaxis_title="kWh",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # NEW: Peak vs Off-Peak Comparison
        st.markdown("---")
        st.subheader("⚖️ Peak vs Off-Peak Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ☀️ Peak Hour (2 PM)")
            peak_features = pd.DataFrame({
                'hour': [14],
                'day_of_week': [pred_date.weekday()],
                'month': [pred_date.month],
                'is_weekend': [pred_date.weekday() >= 5],
                'temperature': [pred_temp]
            })
            peak_pred = models['consumption'].predict(peak_features)[0]
            peak_cost = peak_pred * 0.12
            
            st.metric("Usage", f"{peak_pred:.2f} kWh")
            st.metric("Cost", f"₹{peak_cost:.2f}")
        
        with col2:
            st.markdown("### 🌙 Off-Peak (11 PM)")
            offpeak_features = pd.DataFrame({
                'hour': [23],
                'day_of_week': [pred_date.weekday()],
                'month': [pred_date.month],
                'is_weekend': [pred_date.weekday() >= 5],
                'temperature': [pred_temp]
            })
            offpeak_pred = models['consumption'].predict(offpeak_features)[0]
            offpeak_cost = offpeak_pred * 0.08
            
            st.metric("Usage", f"{offpeak_pred:.2f} kWh")
            st.metric("Cost", f"₹{offpeak_cost:.2f}")
        
        savings = peak_cost - offpeak_cost
        if savings > 0:
            st.success(f"💰 **Potential Savings by Shifting Usage:** ₹{savings:.2f} per use (₹{savings*30:.2f}/month if daily)")
        else:
            st.info("💡 Usage is similar across time periods for this scenario")

# --- TAB 3: ANOMALY DETECTION ---
with tab3:
    st.header("🚨 Anomaly Detection")
    
    st.write("Detecting unusual energy consumption patterns that may indicate:")
    st.write("• Energy leaks or malfunctioning equipment")
    st.write("• Unusual usage patterns")
    st.write("• Potential cost optimization opportunities")
    
    if models['anomaly'] is not None and models['scaler'] is not None:
        features = df[['total_kwh', 'hour', 'day_of_week', 'temperature']]
        X_scaled = models['scaler'].transform(features)
        anomalies = models['anomaly'].predict(X_scaled)
        
        df_with_anomalies = df.copy()
        df_with_anomalies['is_anomaly'] = anomalies == -1
        
        anomaly_count = df_with_anomalies['is_anomaly'].sum()
        anomaly_cost = df_with_anomalies[df_with_anomalies['is_anomaly']]['total_cost'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anomalies Detected", anomaly_count)
        with col2:
            st.metric("Anomaly Cost", f"₹{anomaly_cost:,.2f}")
        with col3:
            potential_fix = anomaly_cost * 0.7
            st.metric("Potential Savings", f"₹{potential_fix:,.2f}")
        
        # Anomaly timeline
        st.subheader("Anomaly Timeline")
        fig = go.Figure()
        
        normal = df_with_anomalies[~df_with_anomalies['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal['timestamp'],
            y=normal['total_kwh'],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=4)
        ))
        
        anomalous = df_with_anomalies[df_with_anomalies['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalous['timestamp'],
            y=anomalous['total_kwh'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            title="Consumption with Anomalies Highlighted",
            xaxis_title="Date",
            yaxis_title="kWh",
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies table
        st.subheader("Top 10 Anomalies")
        top_anomalies = anomalous.nlargest(10, 'total_kwh')[
            ['timestamp', 'total_kwh', 'total_cost', 'temperature', 'hour']
        ]
        st.dataframe(top_anomalies.style.format({
            'total_kwh': '{:.2f}',
            'total_cost': '₹{:.2f}',
            'temperature': '{:.1f}°F'
        }), use_container_width=True)
    else:
        st.error("Anomaly models are not loaded. Please ensure 'anomaly_model.pkl' and 'anomaly_scaler.pkl' are present.")

# --- TAB 4: RECOMMENDATIONS ---
with tab4:
    st.header("💡 Personalized Recommendations")
    
    st.success("### ✅ Quick Wins (Implement Today)")
    st.write("1. **Shift heavy usage to off-peak hours (9 PM - 9 AM)**")
    st.write("   - Run dishwasher/laundry at night")
    st.write("   - Save up to ₹1200/month")
    
    st.write("2. **Adjust thermostat by 2°C during peak hours**")
    st.write("   - Reduce HVAC consumption by 10%")
    st.write("   - Save up to ₹2000/month")
    
    st.write("3. **Replace lighting with LED bulbs**")
    st.write("   - 75% energy reduction for lighting")
    st.write("   - Save up to ₹600/month")
    
    st.warning("### ⚡ Medium-Term Improvements")
    st.write("4. **Install smart thermostat with scheduling**")
    st.write("   - Automatic optimization")
    st.write("   - Save up to ₹15,000/year")
    
    st.write("5. **Insulate water heater**")
    st.write("   - Reduce standby heat loss")
    st.write("   - Save up to ₹4,000/year")
    
    st.info("### 🔧 Long-Term Investments")
    st.write("6. **Solar panel installation**")
    st.write("   - ROI in 5-8 years")
    st.write("   - Save ₹1,00,000+/year")
    
    # Optimal schedule generator
    st.subheader("📅 Optimal Usage Schedule")
    
    schedule_df = pd.DataFrame({
        'Time Period': ['12 AM - 6 AM', '6 AM - 9 AM', '9 AM - 5 PM', '5 PM - 9 PM', '9 PM - 12 AM'],
        'Rate': ['Off-Peak (₹6/kWh)', 'Peak (₹10/kWh)', 'Peak (₹10/kWh)', 'Peak (₹10/kWh)', 'Off-Peak (₹6/kWh)'],
        'Recommended Actions': [
            '✅ Best time for heavy loads',
            '⚠️ Minimize usage if possible',
            '🏢 Unavoidable work-from-home usage',
            '⚠️ Cook efficiently, avoid multiple appliances',
            '✅ Good time for dishwasher/laundry'
        ]
    })
    
    st.table(schedule_df)
    
    # Cost comparison
    st.subheader("💰 Cost Comparison")
    current_cost = df['total_cost'].sum()
    optimized_cost = current_cost * 0.85
    annual_savings = (current_cost - optimized_cost) * (365 / len(df.groupby(df['timestamp'].dt.date)))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Annual Cost", f"₹{current_cost * (365/len(df.groupby(df['timestamp'].dt.date))):,.2f}")
    with col2:
        st.metric("Optimized Cost", f"₹{optimized_cost * (365/len(df.groupby(df['timestamp'].dt.date))):,.2f}")
    with col3:
        st.metric("Annual Savings", f"₹{annual_savings:,.2f}", delta="-15%")

# --- TAB 5: GAMIFICATION ---
with tab5:
    st.header("🏆 Energy Saving Challenge")
    
    st.subheader("Your Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_consumption = df['total_kwh'].mean()
        st.metric("Avg Hourly Usage", f"{avg_consumption:.2f} kWh")
    
    with col2:
        neighborhood_avg = avg_consumption * 1.15
        st.metric("Neighborhood Avg", f"{neighborhood_avg:.2f} kWh")
    
    with col3:
        percentile = 35
        st.metric("Your Ranking", f"Top {percentile}%", delta="Better than 65%")
    
    with col4:
        streak = 12
        st.metric("Saving Streak", f"{streak} days", delta="+3")
    
    # Leaderboard
    st.subheader("🏅 Neighborhood Leaderboard")
    leaderboard = pd.DataFrame({
        'Rank': [1, 2, 3, 4, 5],
        'Household': ['Johnson Family', 'You', 'Smith Family', 'Martinez Family', 'Chen Family'],
        'Avg Daily kWh': [45.2, 52.8, 58.1, 61.3, 67.9],
        'Savings vs Baseline': ['25%', '18%', '12%', '8%', '3%']
    })
    
    st.dataframe(leaderboard, use_container_width=True)
    
    # Badges
    st.subheader("🎖️ Your Badges")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🌟")
        st.write("**Early Adopter**")
        st.caption("First week complete")
    with col2:
        st.markdown("### ⚡")
        st.write("**Power Saver**")
        st.caption("10% reduction achieved")
    with col3:
        st.markdown("### 🔥")
        st.write("**Hot Streak**")
        st.caption("7-day saving streak")
    with col4:
        st.markdown("### 🎯")
        st.write("**Goal Crusher**")
        st.caption("Monthly target met")
    
    # Weekly challenge
    st.subheader("📅 This Week's Challenge")
    st.info("**Challenge:** Reduce peak-hour usage by 20%")
    
    progress = 0.65
    st.progress(progress)
    st.write(f"Progress: {progress*100:.0f}% complete")
    
    if progress >= 1.0:
        st.success("🎉 Challenge Complete! You've earned 100 points!")
    else:
        remaining = (1 - progress) * 20
        st.write(f"Keep going! Just {remaining:.1f}% more to earn 100 points.")

# --- TAB 6: CALCULATOR (NEW) ---
with tab6:
    st.header("🧮 Custom Appliance Cost Calculator")
    
    st.write("Calculate your monthly costs based on your actual appliance usage:")
    
    st.subheader("⚙️ Enter Your Usage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Air Conditioner")
        ac_hours = st.number_input("Hours per day", 0, 24, 8, key="ac")
        ac_wattage = st.number_input("Power (Watts)", 1000, 3000, 1500, key="ac_w")
        ac_kwh_per_day = (ac_wattage / 1000) * ac_hours
        ac_cost = ac_kwh_per_day * 0.12 * 30
    
    with col2:
        st.markdown("#### Refrigerator")
        fridge_hours = 24
        st.number_input("Hours per day", 24, 24, 24, disabled=True, key="fridge")
        fridge_wattage = st.number_input("Power (Watts)", 100, 300, 150, key="fridge_w")
        fridge_kwh_per_day = (fridge_wattage / 1000) * fridge_hours
        fridge_cost = fridge_kwh_per_day * 0.10 * 30
    
    with col3:
        st.markdown("#### Lighting")
        lights_hours = st.number_input("Hours per day", 0, 24, 6, key="lights")
        num_bulbs = st.number_input("Number of bulbs", 1, 50, 10, key="bulbs")
        bulb_wattage = st.number_input("Watts per bulb", 5, 100, 10, key="bulb_w")
        lights_kwh_per_day = bulb_