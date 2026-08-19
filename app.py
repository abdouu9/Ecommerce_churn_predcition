
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="E-commerce Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Load model and feature info
@st.cache_resource
def load_model():
    try:
        model = joblib.load('churn_model.pkl')
        feature_info = joblib.load('feature_info.pkl')
        return model, feature_info
    except FileNotFoundError:
        st.error("Model files not found. Please run 'train_model.py' first.")
        st.stop()


model, feature_info = load_model()

# Actions dictionary (solutions)
actions = {
    "Tenure": {
        "positive": [
            "Mettre en avant les avantages liés à l'ancienneté, proposer un programme VIP fidélité et envoyer un message de remerciement personnalisé."
        ],
        "negative": [
            "Tenure faible — activer une campagne d'onboarding, proposer un coupon de première commande et guider le client vers les produits les plus populaires pour augmenter l'engagement précoce."
        ]
    },
    "Complain": {
        "positive": [
            "Communiquer que les plaintes ont été prises en compte, proposer un suivi satisfaction post-résolution et envoyer une offre compensatrice."
        ],
        "negative": [
            "Déclencher une intervention proactive du support, appeler le client pour résoudre ses problèmes urgents et proposer une compensation immédiate."
        ]
    },
    "NumberOfAddress": {
        "positive": [
            "Proposer un service premium multi-adresses et envoyer des recommandations selon les adresses fréquentes."
        ],
        "negative": [
            "Vérifier une éventuelle instabilité logistique, proposer une option de livraison prioritaire et assurer une meilleure précision de localisation."
        ]
    },
    "CashbackAmount": {
        "positive": [
            "Renforcer les offres cashback car elles fonctionnent et envoyer des récompenses supplémentaires."
        ],
        "negative": [
            "Introduire un plan de cashback plus attractif et envoyer des coupons personnalisés."
        ]
    },
    "DaySinceLastOrder": {
        "positive": [
            "Étant donné que le client a commandé récemment, envoyer des offres pertinentes et proposer un programme fidélité pour maintenir l'engagement."
        ],
        "negative": [
            "Lancer une campagne de ré-engagement, proposer une promotion limitée dans le temps et envoyer un rappel automatisé."
        ]
    },
    "WarehouseToHome": {
        "positive": [
            "Mettre en avant la rapidité de livraison actuelle."
        ],
        "negative": [
            "Proposer une option livraison express, réduire l'estimation de délai et optimiser la logistique locale."
        ]
    },
    "SatisfactionScore": {
        "positive": [
            "Encourager l'évaluation positive et proposer un bon de réduction pour compenser l'insatisfaction et proposer une assistance prioritaire."
        ],
        "negative": [
            "Demander un feedback détaillé, offrir un bon de réduction pour compenser l'insatisfaction et proposer une assistance prioritaire."
        ]
    },
    "CityTier": {
        "positive": [
            "Envoyer des offres adaptées au niveau de la ville."
        ],
        "negative": [
            "Adapter les promotions au pouvoir d'achat local et améliorer la disponibilité des produits dans la zone."
        ]
    },
    "OrderAmountHikeFromlastYear": {
        "positive": [
            "Remercier le client pour sa hausse de dépenses, proposer un bonus fidélité et envoyer une offre VIP."
        ],
        "negative": [
            "Analyser pourquoi les dépenses baissent, envoyer une offre de relance incitative et proposer une remise personnalisée."
        ]
    },
    "OrderCount": {
        "positive": [
            "Récompenser les commandes fréquentes et proposer un palier supplémentaire de points fidélité."
        ],
        "negative": [
            "Envoyer une campagne incitative de retour et proposer une remise pour réactiver l'activité."
        ]
    },
    "NumberOfDeviceRegistered": {
        "positive": [
            "Promouvoir les avantages multi-plateformes."
        ],
        "negative": [
            "Encourager l'utilisation multi-device et proposer une synchronisation améliorée."
        ]
    },
    "PreferedOrderCat_Laptop & Accessory": {
        "positive": [
            "Envoyer des recommandations tech personnalisées."
        ],
        "negative": [
            "Offrir des réductions sur les accessoires populaires et proposer des bundles d'achat."
        ]
    },
    "PreferedOrderCat_Fashion": {
        "positive": [
            "Proposer du contenu mode personnalisé."
        ],
        "negative": [
            "Offrir des offres mode ciblées."
        ]
    },
    "PreferredPaymentMode_E wallet": {
        "positive": [
            "Proposer un cashback spécial e-wallet."
        ],
        "negative": [
            "Introduire des avantages exclusifs pour e-wallet."
        ]
    },
    "PreferredPaymentMode_UPI": {
        "positive": [
            "Envoyer des promotions UPI dédiées."
        ],
        "negative": [
            "Proposer un cashback UPI attractif."
        ]
    },
    "CouponUsed": {
        "positive": [
            "Continuer à envoyer des coupons puisque cela fonctionne."
        ],
        "negative": [
            "Proposer des coupons plus incitatifs et envoyer une offre personnalisée basée sur ses achats."
        ]
    }
}

def clean_feature_names(feature_names):
    cleaned = []
    for f in feature_names:
        f = f.replace("num_preprocess__", "")
        f = f.replace("cat_preprocess__", "")
        cleaned.append(f)
    return cleaned


def risk_segment(p):
    """Determine risk segment based on probability"""
    if p < 0.3:
        return "Low", "🟢"
    elif p < 0.7:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


def top_drivers(shap_row, feature_names, n=5):
    """Extract top drivers from SHAP values"""
    idx = np.argsort(np.abs(shap_row.values))[-n:][::-1]
    clean_list = []
    for i in idx:
        raw = feature_names[i]
        clean = raw.split("__")[-1]
        clean_list.append((clean, shap_row.values[i]))
    return clean_list


def recommend_actions(top_driver_list, action_map):
    """Generate recommendations based on SHAP sign"""
    recs = []
    for feature, importance in top_driver_list:
        # Try exact match first
        if feature in action_map:
            if importance > 0:
                recs.extend(action_map[feature]["negative"])
            else:
                recs.extend(action_map[feature]["positive"])
        else:
            # Try partial match for one-hot encoded features
            for key in action_map.keys():
                if key in feature or feature in key:
                    if importance > 0:
                        recs.extend(action_map[key]["negative"])
                    else:
                        recs.extend(action_map[key]["positive"])
                    break
    return recs[:3] if recs else ["Analyse des données en cours..."]


# Title
st.title("📊 E-commerce Churn Prediction Dashboard")
st.markdown("---")

# Sidebar for input
st.sidebar.header("Informations du client")

# Input fields
with st.sidebar.form("customer_form"):
    # Numerical inputs
    tenure = st.number_input("Tenure", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    warehouse_to_home = st.number_input("Warehouse to Home (km)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    hour_spend_on_app = st.number_input("Hours Spent on App", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
    num_device_registered = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=2, step=1)
    satisfaction_score = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5])
    num_address = st.number_input("Number of Addresses", min_value=1, max_value=10, value=2, step=1)
    complain = st.selectbox("Complain", [0, 1])
    order_amount_hike = st.number_input("Order Amount Hike from Last Year (%)", min_value=-100.0, max_value=500.0,
                                        value=0.0, step=1.0)
    coupon_used = st.number_input("Coupon Used", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
    order_count = st.number_input("Order Count", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    day_since_last_order = st.number_input("Days Since Last Order", min_value=0.0, max_value=365.0, value=30.0,
                                           step=1.0)
    cashback_amount = st.number_input("Cashback Amount", min_value=0.0, max_value=500.0, value=0.0, step=1.0)

    # Categorical inputs
    preferred_login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Phone"])
    preferred_payment_mode = st.selectbox("Preferred Payment Mode",
                                          ["Debit Card", "Credit Card", "E wallet", "Cash on Delivery", "UPI"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    preferred_order_cat = st.selectbox("Preferred Order Category",
                                       ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

if submitted:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse_to_home],
        'HourSpendOnApp': [hour_spend_on_app],
        'NumberOfDeviceRegistered': [num_device_registered],
        'SatisfactionScore': [satisfaction_score],
        'NumberOfAddress': [num_address],
        'Complain': [complain],
        'OrderAmountHikeFromlastYear': [order_amount_hike],
        'CouponUsed': [coupon_used],
        'OrderCount': [order_count],
        'DaySinceLastOrder': [day_since_last_order],
        'CashbackAmount': [cashback_amount],
        'PreferredLoginDevice': [preferred_login_device],
        'PreferredPaymentMode': [preferred_payment_mode],
        'Gender': [gender],
        'PreferedOrderCat': [preferred_order_cat],
        'MaritalStatus': [marital_status]
    })

    # Predict
    probability = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]

    # Get risk segment
    risk_level, risk_icon = risk_segment(probability)

    # Calculate SHAP values
    preprocess = model.named_steps["prep"]
    xgb_model = model.named_steps["xgb"]

    X_transformed = preprocess.transform(input_data)
    raw_feature_names = preprocess.get_feature_names_out()
    feature_names = clean_feature_names(raw_feature_names)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    # Create SHAP explanation object
    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_transformed,
        feature_names=feature_names
    )

    # Get top drivers
    top_drivers_list = top_drivers(shap_exp[0], feature_names, n=5)

    # Get recommendations
    recommendations = recommend_actions(top_drivers_list, actions)

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with col2:
        st.metric("Prediction", "Churn" if prediction == 1 else "No Churn")

    with col3:
        st.metric("Risk Segment", f"{risk_icon} {risk_level}")

    st.markdown("---")

    # Risk segment details
    st.subheader("Risk Analysis")
    if risk_level == "Low":
        st.success(
            f"🟢 **Low Risk**: Ce client présente une faible probabilité de churn ({probability:.2%}). Poursuivre avec des stratégies d’engagement standard")
    elif risk_level == "Medium":
        st.warning(
            f"🟡 **Medium Risk**: Ce client présente une probabilité modérée de churn ({probability:.2%}). Envisager des actions de rétention proactives.")
    else:
        st.error(
            f"🔴 **High Risk**: Ce client présente une forte probabilité de churn ({probability:.2%}). Une intervention immédiate est recommandée !")

    st.markdown("---")

    # Top drivers
    st.subheader("Principaux facteurs du churn")
    st.markdown("Voici les 5 principaux facteurs influençant la prédiction du churn:")

    drivers_df = pd.DataFrame(
        top_drivers_list,
        columns=["Feature", "SHAP Value"]
    )
    drivers_df["Impact"] = drivers_df["SHAP Value"].apply(
        lambda x: "Augmente le risque de churn" if x > 0 else "Diminue le risque de churn"
    )
    drivers_df["SHAP Value"] = drivers_df["SHAP Value"].apply(lambda x: f"{x:.4f}")

    st.dataframe(drivers_df, use_container_width=True)


    # Feature importance
    fig, ax = plt.subplots(figsize=(12, 8))

    shap.plots.bar(
        shap_exp[0],
        max_display=8,
        show=False
    )

    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()

    st.subheader("Importance des variables (SHAP)")
    st.pyplot(fig)

    st.markdown("---")

    # Recommendations
    st.subheader("Actions recommandées")
    st.markdown("Sur la base de l’analyse, voici les actions recommandées:")

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

    st.markdown("---")



else:
    st.info("👈 Veuillez renseigner les informations du client dans la barre latérale puis cliquer sur « Predict churn » pour afficher l’analyse.")
