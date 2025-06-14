import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import folium
from streamlit_folium import st_folium


st.set_page_config(
    page_title="NYC Taxi Pro - Interactive",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
    <style>
        /* Main background - Clean White */
        .stApp { background-color: #FFFFFF; }
        /* Main containers - Subtle shadow for depth */
        .st-emotion-cache-1r4qj8v {
            background-color: #F8F9FA;
            border-radius: 10px; border: 1px solid #E0E0E0;
            padding: 2rem !important; margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        /* Main title and headers - Dark text for readability */
        h1, h2, h3 { color: #1E293B; text-shadow: none; }
        /* Button styling */
        .stButton > button {
            color: #FFFFFF; background-image: linear-gradient(to right, #FFC107, #FF8F00);
            border: none; border-radius: 8px; padding: 12px 24px;
            font-size: 1.1rem; font-weight: bold; cursor: pointer;
            transition: all 0.2s ease; width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-2px); box-shadow: 0 6px 15px rgba(255, 165, 0, 0.4);
        }
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] { gap: 16px; border-bottom: 2px solid #E0E0E0; }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent; border-radius: 0; padding: 10px;
            border-bottom: 2px solid transparent; color: #6c757d;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 2px solid #FF8F00; color: #1E293B; font-weight: 600;
        }
        /* Result display boxes */
        .result-box {
            padding: 2rem; border-radius: 15px; text-align: center;
            border: none; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .fare-box { background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%); color: #1E293B; }
        .hotspot-box { background: linear-gradient(135deg, #fceabb 0%, #f8b500 100%); color: #1E293B; }
        .coldspot-box { background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); color: #28313b; }
    </style>
    """, unsafe_allow_html=True)

load_css()


if 'pickup_coords' not in st.session_state:
    st.session_state.pickup_coords = {'lat': 40.768, 'lng': -73.982} 
if 'dropoff_coords' not in st.session_state:
    st.session_state.dropoff_coords = {'lat': 40.730, 'lng': -73.980} 
if 'driver_coords' not in st.session_state:
    st.session_state.driver_coords = {'lat': 40.75, 'lng': -73.99} 


@st.cache_resource
def load_models():
    models = {}
    try:
        with open('taxi_fare_predictor.pkl', 'rb') as f1:
            models['regression'] = pickle.load(f1)
        with open('driver_hotspot_classifier.pkl', 'rb') as f2:
            models['classification'] = pickle.load(f2)
        return models
    except FileNotFoundError as e:
        st.error(f"🚨 Model file not found! Please make sure both .pkl files are present. Missing: {e.filename}")
        return None

models = load_models()


with st.sidebar:
    st.image("Taxi.png", width=100)
    st.title("About")
    st.info("This app provides AI-powered tools for both passengers and drivers in the NYC taxi ecosystem.")
    st.warning("Click the map to set your location!")


st.title("👑 NYC Taxi Pro")
st.markdown("### Your Interactive AI Co-Pilot for Fare Estimates & Hotspot Analysis")

if models:
    passenger_tab, driver_tab = st.tabs(["Passenger Mode: Fare Estimator", "Driver Mode: Hotspot Finder"])

    # ============================ PASSENGER MODE ============================
    with passenger_tab:
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("📍 Interactive Trip Map")
            m_pass = folium.Map(location=[st.session_state.pickup_coords['lat'], st.session_state.pickup_coords['lng']], zoom_start=13, tiles="cartodbpositron")
            folium.Marker([st.session_state.pickup_coords['lat'], st.session_state.pickup_coords['lng']], popup="Pickup", icon=folium.Icon(color='green', icon='play')).add_to(m_pass)
            if st.session_state.dropoff_coords:
                folium.Marker([st.session_state.dropoff_coords['lat'], st.session_state.dropoff_coords['lng']], popup="Dropoff", icon=folium.Icon(color='red', icon='stop')).add_to(m_pass)
            map_data_pass = st_folium(m_pass, key="pass_map", width=700, height=500)

        with col2:
            st.subheader("⚙️ Trip Configuration")
            selection_mode = st.radio("Map Click Mode:", ["Set Pickup", "Set Dropoff"], horizontal=True, label_visibility="collapsed")
            st.write("---")
            p_lat = st.number_input("Pickup Latitude", value=st.session_state.pickup_coords['lat'], format="%.5f", key="p_lat")
            p_lon = st.number_input("Pickup Longitude", value=st.session_state.pickup_coords['lng'], format="%.5f", key="p_lon")
            d_lat = st.number_input("Dropoff Latitude", value=st.session_state.dropoff_coords['lat'], format="%.5f", key="d_lat")
            d_lon = st.number_input("Dropoff Longitude", value=st.session_state.dropoff_coords['lng'], format="%.5f", key="d_lon")
            st.write("---")
            pass_count = st.slider("Passengers", 1, 6, 1, key="pass_count")
            p_date = st.date_input("Date", datetime.date.today(), key="p_date")
            p_time = st.time_input("Time", datetime.time(18, 30), key="p_time")
            
        if map_data_pass and map_data_pass['last_clicked']:
            clicked_coords = map_data_pass['last_clicked']
            if selection_mode == "Set Pickup":
                st.session_state.pickup_coords = clicked_coords
            else:
                st.session_state.dropoff_coords = clicked_coords
            st.rerun()

        if st.button("✨ Forecast My Fare!", key="fare_btn"):
            with st.spinner('Forecasting your fare...'):
                pickup_datetime = datetime.datetime.combine(p_date, p_time)
                pickup_hour = pickup_datetime.hour; pickup_day_of_week = pickup_datetime.weekday()
                is_weekend = 1 if pickup_day_of_week >= 5 else 0; is_night = 1 if pickup_hour <= 6 or pickup_hour >= 22 else 0
                def haversine_distance(lon1, lat1, lon2, lat2):
                    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2]); dlon, dlat = lon2 - lon1, lat2 - lat1
                    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                    return 6371 * 2 * np.arcsin(np.sqrt(a)) * 0.621371
                trip_distance = haversine_distance(p_lon, p_lat, d_lon, d_lat)
                estimated_duration_min = (trip_distance / 15) * 60 if trip_distance > 0 else 0
                trip_distance_log = np.log1p(trip_distance); trip_duration_log = np.log1p(estimated_duration_min)
                final_features_list = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'RatecodeID', 'payment_type', 'pickup_hour', 'pickup_day_of_week', 'is_weekend', 'is_night', 'trip_distance_log', 'trip_duration_log']
                input_reg_data = pd.DataFrame({'passenger_count': [pass_count], 'pickup_longitude': [p_lon], 'pickup_latitude': [p_lat], 'dropoff_longitude': [d_lon], 'dropoff_latitude': [d_lat], 'RatecodeID': [1], 'payment_type': [1], 'pickup_hour': [pickup_hour], 'pickup_day_of_week': [pickup_day_of_week], 'is_weekend': [is_weekend], 'is_night': [is_night], 'trip_distance_log': [trip_distance_log], 'trip_duration_log': [trip_duration_log]})[final_features_list]
                prediction = models['regression'].predict(input_reg_data)[0]
                st.balloons(); st.markdown(f'<div class="result-box fare-box"><h3>Your Estimated Fare is:</h3><h1>${prediction:.2f}</h1></div>', unsafe_allow_html=True)

    # ============================ DRIVER MODE ============================
    with driver_tab:
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📍 Where Are You? (Click map to set location)")
            m_driver = folium.Map(location=[st.session_state.driver_coords['lat'], st.session_state.driver_coords['lng']], zoom_start=14, tiles="cartodbpositron")
            folium.Marker([st.session_state.driver_coords['lat'], st.session_state.driver_coords['lng']], popup="Your Location", icon=folium.Icon(color='blue', icon='car', prefix='fa')).add_to(m_driver)
            map_data_driver = st_folium(m_driver, key="driver_map", width=700, height=500)

        with col4:
            st.subheader("⚙️ Analysis Configuration")
            drv_lat = st.number_input("Your Latitude", value=st.session_state.driver_coords['lat'], format="%.5f", key="drv_lat")
            drv_lon = st.number_input("Your Longitude", value=st.session_state.driver_coords['lng'], format="%.5f", key="drv_lon")
            st.write("---")
            st.subheader("⏰ When Are You Driving?")
            drv_date = st.date_input("Date", datetime.date.today(), key="drv_date")
            drv_hour = st.slider("Hour of Day (24h format)", 0, 23, 19)

        if map_data_driver and map_data_driver['last_clicked']:
            st.session_state.driver_coords = map_data_driver['last_clicked']
            st.rerun()

        if st.button("🔍 Analyze Hotspot", key="hotspot_btn"):
            with st.spinner("Analyzing earning potential..."):
                day_of_week = drv_date.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                input_class_data = pd.DataFrame({'pickup_longitude': [drv_lon], 'pickup_latitude': [drv_lat], 'pickup_hour': [drv_hour], 'pickup_day_of_week': [day_of_week], 'is_weekend': [is_weekend]})
                prediction = models['classification'].predict(input_class_data)[0]
                probability = models['classification'].predict_proba(input_class_data)[0][1]
                
                if prediction == 1:
                    st.balloons()
                    st.markdown(f'<div class="result-box hotspot-box"><h3>💰 High-Earning Hotspot!</h3><h2>This area is likely to be profitable.</h2><p style="font-size: 1.2rem;">Confidence: <strong>{probability:.0%}</strong></p></div>', unsafe_allow_html=True)
                else:
                    st.snow()
                    st.markdown(f'<div class="result-box coldspot-box"><h3>❄️ Cold Spot Detected</h3><h2>This area has lower earning potential right now.</h2><p style="font-size: 1.2rem;">Confidence of it being a hotspot: <strong>{probability:.0%}</strong></p></div>', unsafe_allow_html=True)

else:
    st.error("One or more model files are missing. The application cannot start.")