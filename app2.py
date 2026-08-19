import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Prédiction pour un client existant", layout="wide")

# -------------------------
# 🔧 ACTIONS DICTIONARY
# -------------------------
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


# -------------------------
# 🔧 RECOMMENDATION FUNCTION
# -------------------------
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


# -------------------------
# 🎯 RISK & DRIVERS HELPERS
# -------------------------
def risk_segment(p):
    """Determine risk segment based on probability"""
    if p < 0.3:
        return "Low", "🟢"
    if p < 0.7:
        return "Medium", "🟡"
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


# Load model and dataset
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    df = pd.read_excel("E Commerce Dataset.xlsx", sheet_name="E Comm")
    return model, df


model, df = load_model()

st.title("🔍 Prédire le churn pour un client existant")
st.markdown("Saisissez un **CustomerID** afin de récupérer les caractéristiques du client et générer la prédiction.")

customer_id = st.text_input("Customer ID")

if st.button("Predict for this Customer", use_container_width=True):
    if customer_id not in df["CustomerID"].astype(str).values:
        st.error("❌ Identifiant client introuvable dans le jeu de données.")
        st.stop()

    row = df[df["CustomerID"].astype(str) == customer_id]
    input_data = row.drop(columns=["CustomerID", "Churn"], errors="ignore")

    st.subheader("📄 Customer Data")
    st.dataframe(input_data, use_container_width=True)

    probability = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]

    risk_level, risk_icon = risk_segment(probability)

    preprocess = model.named_steps["prep"]
    xgb_model = model.named_steps["xgb"]

    X_transformed = preprocess.transform(input_data)
    raw_feature_names = preprocess.get_feature_names_out()
    feature_names = clean_feature_names(raw_feature_names)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_transformed,
        feature_names=feature_names,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Churn Probability", f"{probability:.2%}")
    col2.metric("Prediction", "Churn" if prediction == 1 else "No Churn")
    col3.metric("Risk Segment", f"{risk_icon} {risk_level}")

    st.markdown("---")

    st.subheader("Risk Analysis")
    if risk_level == "Low":
        st.success(
            f"🟢 **Low Risk**: Ce client présente une faible probabilité de churn ({probability:.2%}). Poursuivre avec des stratégies d’engagement standard."
        )
    elif risk_level == "Medium":
        st.warning(
            f"🟡 **Medium Risk**: Ce client présente une probabilité modérée de churn ({probability:.2%}). Envisager des actions de rétention proactives."
        )
    else:
        st.error(
            f"🔴 **High Risk**: Ce client présente une forte probabilité de churn ({probability:.2%}). Une intervention immédiate est recommandée !"
        )

    st.markdown("---")

    top_drivers_list = top_drivers(shap_exp[0], feature_names, n=5)

    st.subheader("Principaux facteurs du churn")
    st.markdown("Voici les 5 principaux facteurs influençant la prédiction du churn:")

    drivers_df = pd.DataFrame(top_drivers_list, columns=["Feature", "SHAP Value"])
    drivers_df["Impact"] = drivers_df["SHAP Value"].apply(
        lambda x: "Augmente le risque de churn" if x > 0 else "Diminue le risque de churn"
    )
    drivers_df["SHAP Value"] = drivers_df["SHAP Value"].apply(lambda x: f"{x:.4f}")
    st.dataframe(drivers_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 6))

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

    recommendations = recommend_actions(top_drivers_list, actions)

    st.markdown("---")
    st.subheader("Actions recommandées")
    st.markdown("Sur la base de l’analyse, voici les actions recommandées:")

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")

    st.markdown("---")
else:
    st.info("👈 Veuillez saisir un identifiant client et cliquer sur “Prédire” pour afficher l’analyse du churn.")
