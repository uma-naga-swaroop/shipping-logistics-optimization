import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page config with custom theme
st.set_page_config(
    page_title="Maritime Route Delivery Time Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stColumn {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        background-color: #f0f2f6;
        border-radius: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_objects():
    """Load the saved model, encoders and mappings"""
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/label_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, mappings, encoders
    except Exception as e:
        st.error(f"Error loading saved files: {str(e)}")
        return None, None, None
def create_gauge_chart(prediction, conf_lower, conf_upper):
    """Create a gauge chart for prediction visualization with dynamic range"""

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(round(prediction)),  # show integer days
        title={
            'text': f"Predicted Delivery Days<br><sub>Range: {int(round(conf_lower))} - {int(round(conf_upper))} days</sub>",
            'font': {'size': 22}
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [conf_lower, conf_upper], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [conf_lower, (conf_lower + conf_upper) / 3], 'color': "lightgreen"},
                {'range': [(conf_lower + conf_upper) / 3, (2 * (conf_lower + conf_upper) / 3)], 'color': "yellow"},
                {'range': [(2 * (conf_lower + conf_upper) / 3), conf_upper], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        },
    ))

    fig.update_layout(height=300)
    return fig

# def create_gauge_chart(prediction, conf_lower, conf_upper):
    """Create a gauge chart for prediction visualization with confidence range"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(round(prediction)),
        domain={'x': [0.3, 0.6], 'y': [0.2, 1]},
        title={
            'text': f"Predicted Delivery Days<br><sub>Range: {int(round(conf_lower))} - {int(round(conf_upper))} days</sub>",
            'font': {'size': 24}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    # Add confidence range as a band
    fig.add_shape(
        type="rect",
        x0=0.2, x1=0.8,
        y0=conf_lower, y1=conf_upper,
        fillcolor="rgba(0,100,255,0.2)",
        line_width=0,
        layer="below"
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_prediction_distribution(predictions):
    """Create histogram of prediction distribution"""
    fig = px.histogram(
        predictions,
        nbins=20,
        title="Distribution of Predictions",
        labels={'value': 'Predicted Days', 'count': 'Frequency'}
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def main():
    st.title("🚢 Maritime Route Delivery Time Predictor")
    
    # Load saved objects
    model, mappings, encoders = load_saved_objects()

    if all([model, mappings, encoders]):
        # Create container for route selection
        with st.container():
            st.markdown("### 📍 Route Selection")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Origin")
                dispatch_options = sorted(mappings['PLACE_OF_DISPATCH'].keys())
                place_dispatch = st.selectbox(
                    "Place of Dispatch",
                    options=dispatch_options,
                    help="Select the starting point"
                )

                loading_options = sorted(mappings['PORT_OF_LOADING'].keys())
                port_loading = st.selectbox(
                    "Port of Loading",
                    options=loading_options,
                    help="Select the loading port"
                )

            with col2:
                st.markdown("#### Destination")
                discharge_options = sorted(mappings['PORT_OF_DISCHARGE'].keys())
                port_discharge = st.selectbox(
                    "Port of Discharge",
                    options=discharge_options,
                    help="Select the discharge port"
                )

                post_discharge_options = sorted(mappings['POST_PORT_OF_DISCHARGE'].keys())
                post_port_discharge = st.selectbox(
                    "Post Port of Discharge",
                    options=post_discharge_options,
                    help="Select the final destination port"
                )

        st.markdown("---")

        # Time information container
        with st.container():
            st.markdown("### 📅 Timing Details")
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("#### Dispatch Timing")
                dispatch_month = st.slider(
                    "Dispatch Month",
                    min_value=1, max_value=12, value=6,
                    help="Month of dispatch (1-12)"
                )
                dispatch_week = st.slider(
                    "Dispatch Day of Week",
                    min_value=0, max_value=6, value=3,
                    help="Day of week for dispatch (0=Monday, 6=Sunday)"
                )

            with col4:
                st.markdown("#### Loading Timing")
                loading_month = st.slider(
                    "Loading Month",
                    min_value=1, max_value=12, value=6,
                    help="Month of loading (1-12)"
                )
                loading_week = st.slider(
                    "Loading Day of Week",
                    min_value=0, max_value=6, value=3,
                    help="Day of week for loading (0=Monday, 6=Sunday)"
                )

        # Prediction button
        if st.button("🚢 Predict Delivery Time", use_container_width=True):
            try:
                # Prepare input data
                input_data = {
                    'PLACE_OF_DISPATCH': mappings['PLACE_OF_DISPATCH'][place_dispatch],
                    'PORT_OF_LOADING': mappings['PORT_OF_LOADING'][port_loading],
                    'PORT_OF_DISCHARGE': mappings['PORT_OF_DISCHARGE'][port_discharge],
                    'POST_PORT_OF_DISCHARGE': mappings['POST_PORT_OF_DISCHARGE'][post_port_discharge],
                    'Dispatch_month': dispatch_month,
                    'Dispatch_weak': dispatch_week,
                    'Loading_month': loading_month,
                    'Loading_weak': loading_week
                }

                features = ['PLACE_OF_DISPATCH', 'PORT_OF_LOADING', 'PORT_OF_DISCHARGE',
                          'POST_PORT_OF_DISCHARGE', 'Dispatch_month', 'Dispatch_weak',
                          'Loading_month', 'Loading_weak']
                input_df = pd.DataFrame([input_data])[features]

                # Get predictions from all trees
                predictions = []
                for estimator in model.estimators_:
                    predictions.append(estimator.predict(input_df)[0])

                # Calculate prediction and confidence intervals
                prediction = model.predict(input_df)[0]
                conf_lower = np.percentile(predictions, 25)
                conf_upper = np.percentile(predictions, 75)

                # Display results
                st.markdown("### 📊 Prediction Results")
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.markdown("""
                    <div class='prediction-box'>
                        <h2 style='color: black'>Estimated Delivery Time</h2>
                        <h1 style='color: black'>{} days</h1>
                    </div>
                    """.format(
                        int(round(prediction)),
                        int(round(conf_lower)),
                        int(round(conf_upper))
                    ), unsafe_allow_html=True)

                with res_col2:
                    # Gauge chart
                    gauge_chart = create_gauge_chart(prediction, conf_lower, conf_upper)
                    st.plotly_chart(gauge_chart, use_container_width=True)


                # Route details
                with st.expander("🗺️ View Detailed Route Information"):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("**Route Information**")
                        st.markdown(f"""
                        * 🏭 **Origin:** {place_dispatch}
                        * 🚢 **Loading Port:** {port_loading}
                        * 🏦 **Discharge Port:** {port_discharge}
                        * 🏁 **Final Destination:** {post_port_discharge}
                        """)
                    
                    with detail_col2:
                        st.markdown("**Timing Details**")
                        st.markdown(f"""
                        * 📅 **Dispatch Month:** {dispatch_month}
                        * 📆 **Dispatch Day:** {dispatch_week} (0=Mon, 6=Sun)
                        * 📅 **Loading Month:** {loading_month}
                        * 📆 **Loading Day:** {loading_week} (0=Mon, 6=Sun)
                        """)

            except Exception as e:
                st.error(f"⚠️ Error making prediction: {str(e)}")
                st.error("Please ensure all inputs are valid")

        # Model information
        with st.expander("ℹ️ About this Predictor"):
            st.markdown("""
            ### 🚢 Maritime Route Delivery Time Predictor
            
            This advanced predictor utilizes a Random Forest model trained on extensive historical shipping data to provide accurate delivery time estimates.
            
            #### 📊 Model Features:
            * 🌍 Origin and destination ports
            * 🚢 Loading and discharge locations
            * 📅 Seasonal patterns
            * 📊 Weekly patterns
            
            #### 🎯 Accuracy Factors:
            * 🌊 Weather conditions
            * 🚧 Port congestion
            * 📋 Customs clearance
            * 🔄 Operational variables
            
            #### 📈 Confidence Intervals:
            The prediction includes a confidence range based on the ensemble of decision trees in the Random Forest model.
            """)

    else:
        st.error("""
        ⚠️ Could not load required model files. Please ensure the following files exist in the 'models' directory:
        - random_forest_model.pkl
        - label_mappings.pkl
        - label_encoders.pkl
        """)

if __name__ == "__main__":
    main()