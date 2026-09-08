import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CHURN PREDICTION APP",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# BLACK + BLUE + PINK THEME
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN APP ---------- */

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0, 102, 255, 0.12), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(255, 0, 128, 0.10), transparent 30%),
            #05070D;
        color: #F5F7FF;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0B1220;
        border-right: 1px solid rgba(0, 132, 255, 0.25);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
        color: #F8FAFC !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #CBD5E1 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        color: #FFFFFF !important;
    }


    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 10px;
        margin-bottom: 15px;
    }


    /* ---------- INPUT LABELS ---------- */

    label {
        color: #DDE7FF !important;
        font-weight: 600 !important;
    }


    /* ---------- INPUT BOXES ---------- */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #0B101C !important;
        border: 1px solid #263650 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        height: 52px;
        border: none;
        border-radius: 12px;
        background: linear-gradient(
            90deg,
            #0066FF,
            #008CFF,
            #FF1493
        );
        color: white;
        font-size: 17px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(0, 102, 255, 0.25);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 10px 30px rgba(0, 102, 255, 0.35),
            0 5px 20px rgba(255, 20, 147, 0.20);
    }


    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: linear-gradient(
            135deg,
            #0B111E,
            #080C15
        );
        border: 1px solid #1B3154;
        border-radius: 16px;
        padding: 20px;
        min-height: 110px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.30);
    }

    .info-label {
        color: #8192B5;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-value {
        color: #FFFFFF;
        font-size: 25px;
        font-weight: 700;
        margin-top: 7px;
    }


    /* ---------- RESULT CARDS ---------- */

    .result-card {
        margin-top: 25px;
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        background:
            linear-gradient(
                135deg,
                rgba(0, 105, 255, 0.15),
                rgba(255, 20, 147, 0.10)
            );
        border: 1px solid rgba(0, 153, 255, 0.35);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.40);
    }

    .probability {
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #00BFFF,
            #FF4FA3
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .result-text {
        font-size: 24px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 5px;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #667795;
        font-size: 13px;
        padding: 35px 0 10px 0;
    }


    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #17243A !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_prediction_objects():

    model = load_model(BASE_DIR / "model.h5")
    with open(BASE_DIR / "preprocessor.pkl", "rb") as file:
        preprocessor = pickle.load(file)
    return model, preprocessor


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model, preprocessor = \
        load_prediction_objects()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        "⚠️ Unable to load the model or preprocessing files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### 🧠 Model")

    st.success("ANN Model Loaded")

    st.markdown("""
    **Algorithm**

    Artificial Neural Network

    **Task**

    Customer Churn Prediction

    **Output**

    Churn Probability
    """)

    st.divider()

    st.markdown("### 📊 Input Features")

    st.caption("12 model features")

    st.markdown("""
    - Credit Score
    - Gender
    - Age
    - Tenure
    - Balance
    - Number of Products
    - Credit Card
    - Active Member
    - Estimated Salary
    - Geography
    """)

    st.divider()

    st.caption("Powered by TensorFlow + Streamlit")


# ============================================================
# TOP INFORMATION CARDS
# ============================================================

st.markdown(
    "<h1 style='font-weight:800; margin:0 0 24px 0;'>CHURN PREDICTION APP</h1>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="info-card">
        <div class="info-label">Model</div>
        <div class="info-value">ANN</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="info-card">
        <div class="info-label">Features</div>
        <div class="info-value">12</div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="info-card">
        <div class="info-label">Prediction</div>
        <div class="info-value">Binary</div>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown("""
    <div class="info-card">
        <div class="info-label">Engine</div>
        <div class="info-value">TensorFlow</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# CUSTOMER INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

with col2:

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

with col3:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

with col2:

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )

with col3:

    balance = st.number_input(
        "Account Balance (dataset units)",
        min_value=0.0,
        value=60000.0,
        step=1000.0,
        format="%.2f",
        help="The CSV does not specify a currency or whether this value is monthly or annual. Enter the account balance in the same units as the training data.",
    )


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )

with col2:

    has_credit_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

with col3:

    active_member = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )


# ------------------------------------------------------------
# ROW 4
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    estimated_salary = st.number_input(
        "Estimated Salary (dataset units)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.2f",
        help="The CSV does not specify a currency or whether this value is monthly or annual. Enter the estimated salary in the same units as the training data.",
    )

with col2:

    st.markdown("""
    <div style="
        margin-top:30px;
        color:#8192B5;
        font-size:14px;
    ">
        Prediction threshold
        <br>
        <b style="color:#FFFFFF;">50%</b>
        <br>
        <small>Probability at or above 50% = likely churn</small>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div style="
        margin-top:30px;
        color:#8192B5;
        font-size:14px;
    ">
        Model status
        <br>
        <b style="color:#00D4FF;">● ONLINE</b>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "⚡ ANALYZE CUSTOMER CHURN",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    try:

        # ----------------------------------------------------
        # 1. ENCODE GENDER
        # ----------------------------------------------------

        credit_card_value = 1 if has_credit_card == "Yes" else 0
        active_member_value = 1 if active_member == "Yes" else 0


        # ----------------------------------------------------
        # 4. CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                credit_score,
            geography,
            gender,
                age,
                tenure,
                balance,
                num_products,
                credit_card_value,
                active_member_value,
                estimated_salary,
            ]],
            columns=[
                "CreditScore",
                "Geography",
                "Gender",
                "Age",
                "Tenure",
                "Balance",
                "NumOfProducts",
                "HasCrCard",
                "IsActiveMember",
                "EstimatedSalary",
            ]
        )


        # ----------------------------------------------------
        # 5. SCALE INPUT
        # ----------------------------------------------------

        input_features = preprocessor.transform(input_data)


        # ----------------------------------------------------
        # 6. ANN PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_features,
            verbose=0
        )


        probability = float(prediction[0][0])


        # ----------------------------------------------------
        # 7. CLASSIFICATION
        # ----------------------------------------------------

        if probability >= 0.5:

            result = "Customer is likely to CHURN"
            icon = "⚠️"

        else:

            result = "Customer is likely to STAY"
            icon = "✓"


        # ====================================================
        # RESULT SECTION
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🔮 Prediction Result</div>',
            unsafe_allow_html=True
        )


        result_col1, result_col2 = st.columns([1, 1])


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">

                    <div style="
                        color:#8192B5;
                        text-transform:uppercase;
                        letter-spacing:2px;
                        font-size:13px;
                    ">
                        Churn Probability
                    </div>

                    <div class="probability">
                        {probability:.2%}
                    </div>

                    <div class="result-text">
                        {icon} {result}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # CUSTOMER SUMMARY
        # ----------------------------------------------------

        with result_col2:

            st.markdown(
                f"""
                **Geography:** {geography}

                **Gender:** {gender}

                **Age:** {age}

                **Credit Score:** {credit_score}

                **Balance:** {balance:,.2f}

                **Estimated Salary:** {estimated_salary:,.2f}

                **Active Member:** {active_member}
                """
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.markdown("<br>", unsafe_allow_html=True)

        st.write("### Churn Risk Level")

        st.progress(
            probability,
            text=f"Churn probability: {probability:.2%}"
        )


        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        if probability >= 0.7:

            st.error(
                "🚨 **High Churn Risk:** "
                "This customer has a high predicted probability "
                "of leaving. Consider retention offers, personalized "
                "support, or customer engagement campaigns."
            )

        elif probability >= 0.5:

            st.warning(
                "⚠️ **Medium Churn Risk:** "
                "The customer shows signs of potential churn. "
                "Consider monitoring engagement and providing "
                "targeted retention support."
            )

        else:

            st.success(
                "💙 **Low Churn Risk:** "
                "The model predicts that this customer is more "
                "likely to remain with the company."
            )


        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander("🔍 View processed model input"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    ChurnAI • Artificial Neural Network Customer Prediction

    <br><br>

    Built with TensorFlow • Python • Pandas • Streamlit

</div>
""", unsafe_allow_html=True)