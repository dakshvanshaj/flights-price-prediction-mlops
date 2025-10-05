import streamlit as st
from enum import Enum
from datetime import date, timedelta
import requests
import os
from dotenv import load_dotenv

# Load environment variables from a .env file( API url)
# add your api url to a .env file or use localhost below.
load_dotenv()


# page tile & layout
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="frontend_streamlit/img/favicon.ico",
    layout="wide",
)
st.logo("frontend_streamlit/img/logo.png")
st.title("✈️ Flight Price Predictor")

st.markdown(
    "Enter the flight details below to get a price prediction from our AI model."
)


# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    "This is a web app to predict flight prices using a machine learning model. "
    "It's built with Streamlit for the frontend and FastAPI for the backend API."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Links")
st.sidebar.markdown(
    """
    - [Project Documentation](https://dakshvanshaj.github.io/flights-price-prediction-mlops/)
    - [GitHub Repository](https://github.com/dakshvanshaj/flights-price-prediction-mlops)
    - [Creator's LinkedIn](https://www.linkedin.com/in/daksh-vanshaj-9a9650344)
    """
)

# Define input options and constraints using enum


class LocationEnum(str, Enum):
    """Allowed locations for flights."""

    ARACAJU_SE = "Aracaju (SE)"
    BRASILIA_DF = "Brasilia (DF)"
    CAMPO_GRANDE_MS = "Campo Grande (MS)"
    FLORIANOPOLIS_SC = "Florianopolis (SC)"
    NATAL_RN = "Natal (RN)"
    RECIFE_PE = "Recife (PE)"
    RIO_DE_JANEIRO_RJ = "Rio de Janeiro (RJ)"
    SALVADOR_BH = "Salvador (BH)"
    SAO_PAULO_SP = "Sao Paulo (SP)"


class FlightTypeEnum(str, Enum):
    """Allowed flight types."""

    ECONOMIC = "economic"
    FIRST_CLASS = "firstClass"
    PREMIUM = "premium"


class AgencyEnum(str, Enum):
    """Allowed travel agencies."""

    CLOUDFY = "CloudFy"
    FLYINGDROPS = "FlyingDrops"
    RAINBOW = "Rainbow"


# Define min/max for numerical inputs based on training data
TIME_MIN = 0.1
TIME_MAX = 3.0
DISTANCE_MIN = 150.3
DISTANCE_MAX = 1300.5

# Get the API URL from environment variables with a fallback for local development
API_URL = os.getenv("API_URL", "http://127.0.0.1:9000/prediction")

# ---------------------------------------------------------------------------- #
#                                  input form                                  #
# ---------------------------------------------------------------------------- #
with st.form(key="prediction_form"):
    # --- Row 1: From and to location ---
    col1, col2 = st.columns(2)
    with col1:
        from_location = st.selectbox(
            "From",
            options=[e.value for e in LocationEnum],
            index=5,  # Default to Florianapolis
            help="Select the departure location.",
        )

    with col2:
        to_location = st.selectbox(
            "To",
            options=[e.value for e in LocationEnum],
            index=3,  # Default to Florianopolis
            help="Select the arrival location.",
        )

    # --- Row 2: Flight Type and Agency ---
    col3, col4 = st.columns(2)
    with col3:
        flight_type = st.selectbox(
            "Flight Type",
            options=[e.value for e in FlightTypeEnum],
            index=1,  # Default to firstClass
            help="Select the class of the flight.",
        )

    with col4:
        agency = st.selectbox(
            "Agency",
            options=[e.value for e in AgencyEnum],
            index=1,  # Default to FlyingDrops
            help="Select the travel agency.",
        )

    # --- Row 3: Time and Distance ---
    col5, col6 = st.columns(2)
    with col5:
        time = st.number_input(
            "Flight Duration (hours)",
            min_value=TIME_MIN,
            max_value=TIME_MAX,
            value=1.5,  # A sensible default
            step=0.1,
            help=f"Enter the flight duration in hours. Must be between {TIME_MIN} and {TIME_MAX}.",
        )
    with col6:
        distance = st.slider(
            "Flight Distance (km)",
            min_value=DISTANCE_MIN,
            max_value=DISTANCE_MAX,
            value=676.53,  # sensible default
            step=10.0,
            help=f"Enter the flight distance in kilometers. Must be between {DISTANCE_MIN} and {DISTANCE_MAX}.",
        )

    # --- Row 4: Date ---
    today = date.today()
    flight_date = st.date_input(
        "Date of Flight",
        value=today,
        min_value=today,
        max_value=today + timedelta(days=150),  # Allow booking up to 150 days from now
        help="Select the date of the flight.",
    )

    # -- Submit Button --
    submit_button = st.form_submit_button(label="Predict Price", type="primary")

# --- Form Submission Logic ---
if submit_button:
    # Basic validation
    if from_location == to_location:
        st.error(
            "Departure and arrival locations cannot be the same. Please select different location."
        )
    else:
        # Construct the payload for the API
        payload = {
            "from_location": from_location,
            "to_location": to_location,
            "flight_type": flight_type,
            "time": time,
            "distance": distance,
            "agency": agency,
            "date": flight_date.isoformat(),  # Format date to "YYYY-MM-DD" string
        }

        try:
            # Send the request to the API
            with st.spinner("LightGBM at the speed of light..."):
                response = requests.post(API_URL, json=payload, timeout=10)

            # Check the response from the server
            if response.status_code == 200:
                prediction = response.json()
                price = prediction.get("predicted_price")
                st.success(f"**Predicted Flight Price: ${price:,.2f}**")
                st.balloons()
            else:
                error_details = response.json().get("detail", "No details provided.")
                st.error(
                    f"Error from API (Status {response.status_code}): {error_details}"
                )

        except requests.exceptions.RequestException as e:
            # Handle network-related errors
            st.error(
                f"Could not connect to the prediction API. Please ensure the server is running. Error: {e}"
            )
