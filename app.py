# """
# Plateforme de Prédiction de la Performance des Étudiants
# ==========================================================
# Application Streamlit — projet de portfolio Data Science / Machine Learning.

# Modèle : Ridge Regression (scikit-learn) entraîné sur 10 000 observations.
# Métriques réelles de validation : R² = 0.9892 | RMSE = 2.04
# """

# import streamlit as st
# import joblib as jb
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime
# import os

# # =========================================================================
# # CONFIGURATION DE LA PAGE
# # =========================================================================
# st.set_page_config(
#     page_title="Prédiction de la Performance académique des Étudiants",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =========================================================================
# # CONSTANTES — dérivées des vraies statistiques du dataset d'entraînement
# # (10 000 lignes, cf. quant_data.describe() du notebook d'entraînement)
# # =========================================================================
# FEATURE_RANGES = {
#     "Heures_etude": {"min": 1, "max": 9, "mean": 4.99, "label": "Heures d'étude / jour", "unit": "h"},
#     "Notes_precedentes": {"min": 40, "max": 99, "mean": 69.45, "label": "Notes précédentes", "unit": "/100"},
#     "Heures_sommeil": {"min": 4, "max": 9, "mean": 6.53, "label": "Heures de sommeil / nuit", "unit": "h"},
#     "Sujets_entrainement_pratiques": {"min": 0, "max": 9, "mean": 4.58, "label": "Sujets d'entraînement pratiqués", "unit": ""},
# }
# TARGET_MIN, TARGET_MAX, TARGET_MEAN = 10, 100, 55.22
# MODEL_R2 = 0.9892
# MODEL_RMSE = 2.0394  # RMSE réel mesuré sur le jeu de validation
# CI_MARGIN = 1.96 * MODEL_RMSE  # approx. intervalle de prédiction à 95% (résidus ~ normaux)
# HISTORY_FILE = "historique_predictions.csv"
# MODEL_PATH = "rr_model.joblib"
# ENCODER_PATH = "encoder.joblib"
# SCALER_PATH = "scaler.joblib"

# TIERS = [
#     (85, TARGET_MAX, "Excellent", "#16a34a", "🌟"),
#     (70, 85, "Très bien", "#0ea5e9", "✅"),
#     (55, 70, "Bien", "#6366f1", "👍"),
#     (40, 55, "Moyen", "#f59e0b", "⚠️"),
#     (TARGET_MIN, 40, "À risque", "#dc2626", "🚨"),
# ]


# def get_tier(score: float):
#     for lo, hi, name, color, icon in TIERS:
#         if lo <= score <= hi:
#             return name, color, icon
#     return "Hors plage", "#6b7280", "❓"


# # =========================================================================
# # STYLE — CSS personnalisé (palette lisible, professionnelle)
# # =========================================================================
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

# html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
# h1, h2, h3 { font-family: 'Poppins', sans-serif; }

# .stApp {
#     background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
# }

# /* Bannière héro */
# .hero {
#     background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 45%, #0ea5e9 100%);
#     border-radius: 20px;
#     padding: 2.5rem 2.5rem;
#     margin-bottom: 1.8rem;
#     box-shadow: 0 10px 30px rgba(30, 58, 138, 0.25);
#     animation: fadeIn 0.7s ease-in;
# }
# .hero h1 {
#     color: white;
#     font-size: 2.2rem;
#     font-weight: 800;
#     margin-bottom: 0.4rem;
# }
# .hero p { color: #e0e7ff; font-size: 1.05rem; margin: 0; }
# .badge-row { margin-top: 1rem; }
# .tech-badge {
#     display: inline-block;
#     background: rgba(255,255,255,0.15);
#     color: white;
#     padding: 4px 14px;
#     border-radius: 999px;
#     font-size: 0.8rem;
#     margin-right: 8px;
#     border: 1px solid rgba(255,255,255,0.3);
# }

# @keyframes fadeIn { from {opacity:0; transform: translateY(-8px);} to {opacity:1; transform: translateY(0);} }

# /* Cartes */
# .metric-card {
#     background: white;
#     border-radius: 16px;
#     padding: 1.2rem 1.4rem;
#     box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
#     border: 1px solid #e5e7eb;
#     transition: transform 0.2s ease, box-shadow 0.2s ease;
# }
# .metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(15,23,42,0.10); }

# .result-card {
#     border-radius: 20px;
#     padding: 2rem;
#     color: white;
#     text-align: center;
#     box-shadow: 0 12px 30px rgba(0,0,0,0.15);
#     animation: fadeIn 0.5s ease-in;
# }
# .result-score { font-size: 3.2rem; font-weight: 800; font-family: 'Poppins', sans-serif; margin: 0.2rem 0; }
# .result-tier { font-size: 1.3rem; font-weight: 600; }
# .result-sub { opacity: 0.9; font-size: 0.9rem; margin-top: 0.5rem; }

# section[data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
# }
# section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

# div.stButton > button {
#     background: linear-gradient(135deg, #4338ca, #0ea5e9);
#     color: white;
#     border: none;
#     border-radius: 10px;
#     padding: 0.6rem 1.4rem;
#     font-weight: 600;
#     transition: all 0.2s ease;
# }
# div.stButton > button:hover { transform: scale(1.02); box-shadow: 0 6px 16px rgba(67,56,202,0.35); }

# footer {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)


# # =========================================================================
# # CHARGEMENT DU MODÈLE (mis en cache)
# # =========================================================================
# @st.cache_resource
# def load_artifacts():
#     if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(SCALER_PATH)):
#         return None, None, None
#     encoder = jb.load(ENCODER_PATH)
#     model = jb.load(MODEL_PATH)
#     scaler = jb.load(SCALER_PATH)
#     return encoder, model, scaler


# encoder, model, scaler = load_artifacts()

# # =========================================================================
# # HISTORIQUE (persistant localement + session)
# # =========================================================================
# if "history" not in st.session_state:
#     if os.path.exists(HISTORY_FILE):
#         st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict("records")
#     else:
#         st.session_state.history = []


# def save_history():
#     pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)


# def compute_confidence(heures_etude, notes_prec, heures_sommeil, sujets_pratiques):
#     """
#     Indice de cohérence / confiance (0-100), construit à partir de deux facteurs réels :
#     1) La précision globale du modèle sur le jeu de validation (R² = 0.9892)
#     2) La plausibilité des valeurs saisies par rapport aux plages du jeu d'entraînement
#        (un modèle linéaire est moins fiable en extrapolation hors de son domaine appris)
#     Ce n'est PAS un intervalle de confiance statistique à proprement parler, mais un
#     indicateur pédagogique transparent -- documenté dans l'application.
#     """
#     values = {
#         "Heures_etude": heures_etude,
#         "Notes_precedentes": notes_prec,
#         "Heures_sommeil": heures_sommeil,
#         "Sujets_entrainement_pratiques": sujets_pratiques,
#     }
#     scores = []
#     for key, v in values.items():
#         r = FEATURE_RANGES[key]
#         lo, hi = r["min"], r["max"]
#         width = hi - lo
#         if lo <= v <= hi:
#             scores.append(1.0)
#         else:
#             excess = min(lo - v, v - hi) if v < lo or v > hi else 0
#             excess = (lo - v) if v < lo else (v - hi)
#             scores.append(max(0.0, 1 - excess / width))
#     plausibility = float(np.mean(scores))
#     base = MODEL_R2 * 100  # ~98.9
#     confidence = base * (0.55 + 0.45 * plausibility)  # plancher réaliste même hors plage
#     return round(min(99.5, max(35.0, confidence)), 1)


# def predict_one(heures_etude, notes_prec, activite, heures_sommeil, sujets_pratiques):
#     activite_enc = encoder.transform([activite])[0]
#     x_new = np.array([[heures_etude, notes_prec, activite_enc, heures_sommeil, sujets_pratiques]])
#     x_new = scaler.transform(x_new)
#     y_pred = float(model.predict(x_new)[0])
#     y_pred = round(min(TARGET_MAX, max(TARGET_MIN, y_pred)), 2)
#     confidence = compute_confidence(heures_etude, notes_prec, heures_sommeil, sujets_pratiques)
#     return y_pred, confidence


# def add_to_history(row: dict):
#     row["Horodatage"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     st.session_state.history.append(row)
#     save_history()


# # =========================================================================
# # EN-TÊTE
# # =========================================================================
# st.markdown("""
# <div class="hero">
#     <h1>🎓 Prédiction de la Performance académiquedes  des Étudiants</h1>
#     <p>Plateforme de Machine Learning pour anticiper la réussite académique et cibler
#     l'accompagnement pédagogique — Régression Ridge, R² = 98.9%</p>
#     <div class="badge-row">
#         <span class="tech-badge">🐍 Python</span>
#         <span class="tech-badge">🔬 Scikit-learn</span>
#         <span class="tech-badge">📊 Plotly</span>
#         <span class="tech-badge">⚡ Streamlit</span>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# if model is None or encoder is None or scaler is None:
#     st.error(
#         f"⚠️ Fichiers modèle introuvables. Placez **{MODEL_PATH}**, **{ENCODER_PATH}** "
#         f"et **{SCALER_PATH}** dans le même dossier que app.py, puis relancez l'application."
#     )
#     st.stop()

# # =========================================================================
# # SIDEBAR — Fiche modèle (crédibilité pour recruteurs)
# # =========================================================================
# with st.sidebar:
#     st.markdown("### 📋 Fiche du modèle")
#     st.markdown(f"""
#     - **Algorithme** : Régression Ridge
#     - **R² (validation)** : `{MODEL_R2:.4f}`
#     - **RMSE (validation)** : `{MODEL_RMSE:.2f} pts`
#     - **Données d'entraînement** : 10 000 étudiants
#     - **Variable cible** : Indice de performance (10–100)
#     """)
#     st.divider()
#     st.markdown("### ℹ️ À propos du score de confiance")
#     st.caption(
#         "Combine la précision globale du modèle (R²) et la plausibilité des valeurs "
#         "saisies par rapport aux données d'entraînement. Ce n'est pas un intervalle "
#         "de confiance statistique classique."
#     )
#     st.divider()
#     st.markdown(f"### 🕓 Historique : {len(st.session_state.history)} prédiction(s)")
#     if st.button("🗑️ Réinitialiser l'historique", use_container_width=True):
#         st.session_state.history = []
#         if os.path.exists(HISTORY_FILE):
#             os.remove(HISTORY_FILE)
#         st.rerun()

# # =========================================================================
# # ONGLETS
# # =========================================================================
# tab1, tab2, tab3 = st.tabs([
#     "🎯  Prédiction individuelle",
#     "📁  Prédiction par lot (CSV)",
#     "📊  Tableau de bord"
# ])

# # -------------------------------------------------------------------------
# # TAB 1 — PRÉDICTION INDIVIDUELLE
# # -------------------------------------------------------------------------
# with tab1:
#     col_form, col_result = st.columns([1.1, 1])

#     with col_form:
#         st.markdown("#### 📝 Profil de l'étudiant")
#         with st.form("form_prediction"):
#             c1, c2 = st.columns(2)
#             with c1:
#                 heures_etude = st.slider(
#                     FEATURE_RANGES["Heures_etude"]["label"],
#                     min_value=0, max_value=12,
#                     value=int(FEATURE_RANGES["Heures_etude"]["mean"]), step=1,
#                 )
#                 heures_sommeil = st.slider(
#                     FEATURE_RANGES["Heures_sommeil"]["label"],
#                     min_value=0, max_value=12,
#                     value=int(FEATURE_RANGES["Heures_sommeil"]["mean"]), step=1,
#                 )
#             with c2:
#                 notes_prec = st.slider(
#                     FEATURE_RANGES["Notes_precedentes"]["label"],
#                     min_value=0, max_value=100,
#                     value=int(FEATURE_RANGES["Notes_precedentes"]["mean"]), step=1,
#                 )
#                 sujets_pratiques = st.slider(
#                     FEATURE_RANGES["Sujets_entrainement_pratiques"]["label"],
#                     min_value=0, max_value=12,
#                     value=int(FEATURE_RANGES["Sujets_entrainement_pratiques"]["mean"]), step=1,
#                 )

#             activite = st.radio(
#                 "Activités extrascolaires",
#                 options=list(encoder.classes_),
#                 horizontal=True,
#             )

#             submitted = st.form_submit_button("🚀 Lancer la prédiction", use_container_width=True)

#         if submitted:
#             score, confiance = predict_one(heures_etude, notes_prec, activite, heures_sommeil, sujets_pratiques)
#             st.session_state.last_result = {
#                 "Heures_etude": heures_etude,
#                 "Notes_precedentes": notes_prec,
#                 "Activites_extrascolaires": activite,
#                 "Heures_sommeil": heures_sommeil,
#                 "Sujets_entrainement_pratiques": sujets_pratiques,
#                 "Score_predit": score,
#                 "Confiance": confiance,
#             }
#             add_to_history(dict(st.session_state.last_result))

#     with col_result:
#         st.markdown("#### 🎯 Résultat")
#         if "last_result" in st.session_state:
#             r = st.session_state.last_result
#             tier, color, icon = get_tier(r["Score_predit"])
#             st.markdown(f"""
#             <div class="result-card" style="background: linear-gradient(135deg, {color}dd, {color}99);">
#                 <div>{icon} Indice de performance prédit</div>
#                 <div class="result-score">{r['Score_predit']}</div>
#                 <div class="result-tier">{tier}</div>
#                 <div class="result-sub">
#                     Intervalle approx. (95%) : {max(TARGET_MIN, r['Score_predit']-CI_MARGIN):.1f} – {min(TARGET_MAX, r['Score_predit']+CI_MARGIN):.1f}
#                     &nbsp;|&nbsp; Confiance : {r['Confiance']}%
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#             fig_gauge = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=r["Score_predit"],
#                 number={"suffix": " / 100"},
#                 gauge={
#                     "axis": {"range": [TARGET_MIN, TARGET_MAX]},
#                     "bar": {"color": color},
#                     "steps": [
#                         {"range": [10, 40], "color": "#fee2e2"},
#                         {"range": [40, 55], "color": "#fef3c7"},
#                         {"range": [55, 70], "color": "#e0e7ff"},
#                         {"range": [70, 85], "color": "#dbeafe"},
#                         {"range": [85, 100], "color": "#dcfce7"},
#                     ],
#                 },
#             ))
#             fig_gauge.update_layout(height=260, margin=dict(t=10, b=10, l=20, r=20))
#             st.plotly_chart(fig_gauge, use_container_width=True)
#         else:
#             st.info("Remplissez le formulaire et cliquez sur **Lancer la prédiction** pour voir le résultat.")

# # -------------------------------------------------------------------------
# # TAB 2 — PRÉDICTION PAR LOT (CSV)
# # -------------------------------------------------------------------------
# with tab2:
#     st.markdown("#### 📁 Prédiction pour plusieurs étudiants à la fois")
#     st.caption(
#         "Le fichier CSV doit contenir les colonnes exactes : "
#         "`Heures_etude`, `Notes_precedentes`, `Activites_extrascolaires` (Yes/No), "
#         "`Heures_sommeil`, `Sujets_entrainement_pratiques`."
#     )

#     template_df = pd.DataFrame({
#         "Heures_etude": [5, 8], "Notes_precedentes": [65, 90],
#         "Activites_extrascolaires": ["Yes", "No"],
#         "Heures_sommeil": [7, 6], "Sujets_entrainement_pratiques": [3, 7],
#     })
#     st.download_button(
#         "⬇️ Télécharger un modèle de fichier CSV",
#         template_df.to_csv(index=False).encode("utf-8"),
#         file_name="modele_donnees_etudiants.csv",
#         mime="text/csv",
#     )

#     fichier = st.file_uploader("Déposer un fichier CSV", type=["csv"])

#     if fichier is not None:
#         try:
#             df_csv = pd.read_csv(fichier)
#             colonnes_attendues = list(FEATURE_RANGES.keys())[:1] + ["Notes_precedentes", "Activites_extrascolaires"] + \
#                                   ["Heures_sommeil", "Sujets_entrainement_pratiques"]
#             colonnes_attendues = ["Heures_etude", "Notes_precedentes", "Activites_extrascolaires",
#                                    "Heures_sommeil", "Sujets_entrainement_pratiques"]
#             manquantes = [c for c in colonnes_attendues if c not in df_csv.columns]
#             if manquantes:
#                 st.error(f"Colonnes manquantes dans le fichier : {', '.join(manquantes)}")
#             else:
#                 scores, confiances = [], []
#                 for _, row in df_csv.iterrows():
#                     s, c = predict_one(
#                         row["Heures_etude"], row["Notes_precedentes"],
#                         row["Activites_extrascolaires"], row["Heures_sommeil"],
#                         row["Sujets_entrainement_pratiques"],
#                     )
#                     scores.append(s)
#                     confiances.append(c)
#                     add_to_history({
#                         "Heures_etude": row["Heures_etude"],
#                         "Notes_precedentes": row["Notes_precedentes"],
#                         "Activites_extrascolaires": row["Activites_extrascolaires"],
#                         "Heures_sommeil": row["Heures_sommeil"],
#                         "Sujets_entrainement_pratiques": row["Sujets_entrainement_pratiques"],
#                         "Score_predit": s,
#                         "Confiance": c,
#                     })

#                 df_csv["Indice_performance_predit"] = scores
#                 df_csv["Confiance (%)"] = confiances
#                 df_csv["Niveau"] = [get_tier(s)[0] for s in scores]

#                 st.success(f"✅ {len(df_csv)} prédictions effectuées avec succès.")
#                 st.dataframe(df_csv, use_container_width=True)

#                 fig_batch = px.scatter(
#                     df_csv, x="Heures_etude", y="Indice_performance_predit",
#                     color="Niveau", size="Confiance (%)",
#                     color_discrete_map={t[2]: t[3] for t in TIERS},
#                     hover_data=["Notes_precedentes", "Activites_extrascolaires"],
#                     title="Répartition des prédictions du lot",
#                 )
#                 st.plotly_chart(fig_batch, use_container_width=True)

#                 st.download_button(
#                     "⬇️ Télécharger les résultats (CSV)",
#                     df_csv.to_csv(index=False).encode("utf-8"),
#                     file_name="predictions_resultats.csv",
#                     mime="text/csv",
#                 )
#         except Exception as e:
#             st.error(f"Erreur lors du traitement du fichier : {e}")

# # -------------------------------------------------------------------------
# # TAB 3 — TABLEAU DE BORD
# # -------------------------------------------------------------------------
# with tab3:
#     st.markdown("#### 📊 Tableau de bord des prédictions")

#     if not st.session_state.history:
#         st.info("Aucune prédiction pour le moment. Utilisez l'onglet **Prédiction individuelle** ou **CSV**.")
#     else:
#         hist_df = pd.DataFrame(st.session_state.history)

#         k1, k2, k3, k4 = st.columns(4)
#         k1.metric("Prédictions réalisées", len(hist_df))
#         k2.metric("Score moyen prédit", f"{hist_df['Score_predit'].mean():.1f}")
#         k3.metric("Confiance moyenne", f"{hist_df['Confiance'].mean():.1f}%")
#         best = hist_df["Score_predit"].max()
#         k4.metric("Meilleur score", f"{best:.1f}")

#         st.markdown("##### 🔎 Détail des prédictions")

#         hist_df["Niveau"] = hist_df["Score_predit"].apply(lambda s: get_tier(s)[0])
#         hist_df["Profil"] = hist_df.apply(
#             lambda r: [
#                 r["Heures_etude"] / FEATURE_RANGES["Heures_etude"]["max"],
#                 r["Notes_precedentes"] / FEATURE_RANGES["Notes_precedentes"]["max"],
#                 r["Heures_sommeil"] / FEATURE_RANGES["Heures_sommeil"]["max"],
#                 r["Sujets_entrainement_pratiques"] / FEATURE_RANGES["Sujets_entrainement_pratiques"]["max"],
#             ],
#             axis=1,
#         )

#         display_cols = ["Horodatage", "Heures_etude", "Notes_precedentes", "Activites_extrascolaires",
#                          "Heures_sommeil", "Sujets_entrainement_pratiques", "Score_predit",
#                          "Confiance", "Niveau", "Profil"]
#         display_cols = [c for c in display_cols if c in hist_df.columns]

#         try:
#             st.dataframe(
#                 hist_df[display_cols].iloc[::-1],
#                 use_container_width=True,
#                 hide_index=True,
#                 column_config={
#                     "Score_predit": st.column_config.ProgressColumn(
#                         "Score prédit", min_value=TARGET_MIN, max_value=TARGET_MAX, format="%.1f"
#                     ),
#                     "Confiance": st.column_config.ProgressColumn(
#                         "Confiance", min_value=0, max_value=100, format="%.0f%%"
#                     ),
#                     "Profil": st.column_config.BarChartColumn(
#                         "Profil (normalisé)", y_min=0, y_max=1
#                     ),
#                 },
#             )
#         except Exception:
#             # Repli si version de Streamlit trop ancienne pour column_config avancé
#             st.dataframe(hist_df[display_cols].iloc[::-1], use_container_width=True, hide_index=True)

#         st.markdown("##### 📈 Analyses graphiques")
#         g1, g2 = st.columns(2)

#         with g1:
#             fig_trend = px.line(
#                 hist_df.reset_index(), x="index", y="Score_predit", markers=True,
#                 title="Évolution des scores prédits",
#                 labels={"index": "N° prédiction", "Score_predit": "Indice de performance"},
#             )
#             fig_trend.update_traces(line_color="#4338ca")
#             st.plotly_chart(fig_trend, use_container_width=True)

#         with g2:
#             tier_counts = hist_df["Niveau"].value_counts().reset_index()
#             tier_counts.columns = ["Niveau", "Nombre"]
#             fig_pie = px.pie(
#                 tier_counts, names="Niveau", values="Nombre", hole=0.5,
#                 color="Niveau",
#                 color_discrete_map={t[2]: t[3] for t in TIERS},
#                 title="Répartition par niveau de performance",
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)

#         g3, g4 = st.columns(2)
#         with g3:
#             coefs = pd.DataFrame({
#                 "Variable": ["Heures_etude", "Notes_precedentes", "Activites_extrascolaires",
#                              "Heures_sommeil", "Sujets_entrainement_pratiques"],
#                 "Coefficient": model.coef_,
#             }).sort_values("Coefficient")
#             fig_coef = px.bar(
#                 coefs, x="Coefficient", y="Variable", orientation="h",
#                 title="Importance des variables (coefficients du modèle Ridge)",
#                 color="Coefficient", color_continuous_scale="Blues",
#             )
#             st.plotly_chart(fig_coef, use_container_width=True)

#         with g4:
#             fig_scatter = px.scatter(
#                 hist_df, x="Notes_precedentes", y="Score_predit",
#                 color="Niveau", color_discrete_map={t[2]: t[3] for t in TIERS},
#                 size="Confiance", hover_data=["Heures_etude"],
#                 title="Notes précédentes vs Score prédit",
#             )
#             st.plotly_chart(fig_scatter, use_container_width=True)

# # =========================================================================
# # PIED DE PAGE
# # =========================================================================
# st.divider()
# st.markdown(
#     "<p style='text-align:center; color:#64748b; font-size:0.85rem;'>"
#     "🎓 Projet Data Science & Machine Learning — Régression Ridge (R² = 98.9%) · "
#     "Développé avec Python, scikit-learn, Plotly &amp; Streamlit"
#     "</p>",
#     unsafe_allow_html=True,
# )
"""
Plateforme de Prédiction de la Performance des Étudiants
==========================================================
Application Streamlit — projet de portfolio Data Science / Machine Learning.

Modèle : Ridge Regression
Métriques de validation : R² = 0.9892 | RMSE = 2.04
"""

import streamlit as st
import joblib as jb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os


# =========================================================================
# CONFIGURATION DE LA PAGE
# =========================================================================

st.set_page_config(
    page_title="Prédiction de la Performance Académique",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
# CONSTANTES
# =========================================================================

FEATURE_RANGES = {
    "Heures_etude": {
        "min": 1,
        "max": 9,
        "mean": 4.99,
        "label": "Heures d'étude / jour",
        "unit": "h",
    },

    "Notes_precedentes": {
        "min": 40,
        "max": 99,
        "mean": 69.45,
        "label": "Notes précédentes",
        "unit": "/100",
    },

    "Heures_sommeil": {
        "min": 4,
        "max": 9,
        "mean": 6.53,
        "label": "Heures de sommeil / nuit",
        "unit": "h",
    },

    "Sujets_entrainement_pratiques": {
        "min": 0,
        "max": 9,
        "mean": 4.58,
        "label": "Sujets d'entraînement pratiqués",
        "unit": "",
    },
}


TARGET_MIN = 10
TARGET_MAX = 100
TARGET_MEAN = 55.22

MODEL_R2 = 0.9892
MODEL_RMSE = 2.0394

CI_MARGIN = 1.96 * MODEL_RMSE

HISTORY_FILE = "historique_predictions.csv"

MODEL_PATH = "rr_model.joblib"
ENCODER_PATH = "encoder.joblib"
SCALER_PATH = "scaler.joblib"


# =========================================================================
# ORDRE EXACT DES VARIABLES
# =========================================================================

DEFAULT_FEATURE_ORDER = [
    "Heures_etude",
    "Notes_precedentes",
    "Activites_extrascolaires",
    "Heures_sommeil",
    "Sujets_entrainement_pratiques",
]


# =========================================================================
# NIVEAUX DE PERFORMANCE
# =========================================================================

TIERS = [
    (85, 100, "Excellent", "#16a34a", "🌟"),
    (70, 85, "Très bien", "#0ea5e9", "✅"),
    (55, 70, "Bien", "#6366f1", "👍"),
    (40, 55, "Moyen", "#f59e0b", "⚠️"),
    (10, 40, "À risque", "#dc2626", "🚨"),
]


def get_tier(score: float):

    for lo, hi, name, color, icon in TIERS:

        if lo <= score < hi:
            return name, color, icon

    if score >= 100:
        return "Excellent", "#16a34a", "🌟"

    return "À risque", "#dc2626", "🚨"


# =========================================================================
# STYLE CSS
# =========================================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        180deg,
        #f8fafc 0%,
        #eef2ff 100%
    );
}


/* =========================================================
   HERO
========================================================= */

.hero {

    background: linear-gradient(
        135deg,
        #1e3a8a 0%,
        #4338ca 45%,
        #0ea5e9 100%
    );

    border-radius: 20px;

    padding: 2.5rem;

    margin-bottom: 1.8rem;

    box-shadow:
        0 10px 30px rgba(30, 58, 138, 0.25);

    animation: fadeIn 0.7s ease-in;
}

.hero h1 {

    color: white;

    font-size: 2.2rem;

    font-weight: 800;

    margin-bottom: 0.4rem;
}

.hero p {

    color: #e0e7ff;

    font-size: 1.05rem;

    margin: 0;
}

.badge-row {

    margin-top: 1rem;
}

.tech-badge {

    display: inline-block;

    background: rgba(255,255,255,0.15);

    color: white;

    padding: 4px 14px;

    border-radius: 999px;

    font-size: 0.8rem;

    margin-right: 8px;

    border: 1px solid rgba(255,255,255,0.3);
}


/* =========================================================
   ANIMATION
========================================================= */

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(-8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/* =========================================================
   CARTES
========================================================= */

.metric-card {

    background: white;

    border-radius: 16px;

    padding: 1.2rem 1.4rem;

    box-shadow:
        0 4px 14px rgba(15, 23, 42, 0.06);

    border: 1px solid #e5e7eb;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.metric-card:hover {

    transform: translateY(-3px);

    box-shadow:
        0 10px 24px rgba(15,23,42,0.10);
}


/* =========================================================
   RESULTAT
========================================================= */

.result-card {

    border-radius: 20px;

    padding: 2rem;

    color: white;

    text-align: center;

    box-shadow:
        0 12px 30px rgba(0,0,0,0.15);

    animation: fadeIn 0.5s ease-in;
}

.result-score {

    font-size: 3.2rem;

    font-weight: 800;

    font-family: 'Poppins', sans-serif;

    margin: 0.2rem 0;
}

.result-tier {

    font-size: 1.3rem;

    font-weight: 600;
}

.result-sub {

    opacity: 0.9;

    font-size: 0.9rem;

    margin-top: 0.5rem;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #0f172a 0%,
            #1e293b 100%
        );
}

section[data-testid="stSidebar"] * {

    color: #e2e8f0 !important;
}


/* =========================================================
   BOUTONS
========================================================= */

div.stButton > button {

    background:
        linear-gradient(
            135deg,
            #4338ca,
            #0ea5e9
        );

    color: white;

    border: none;

    border-radius: 10px;

    padding: 0.6rem 1.4rem;

    font-weight: 600;

    transition: all 0.2s ease;
}

div.stButton > button:hover {

    transform: scale(1.02);

    box-shadow:
        0 6px 16px rgba(67,56,202,0.35);
}


/* =========================================================
   KPI
========================================================= */

div[data-testid="stMetric"] {

    background:
        linear-gradient(
            135deg,
            #1e293b,
            #0f172a
        );

    border-radius: 16px;

    padding: 1.1rem 1.2rem;

    box-shadow:
        0 6px 18px rgba(15, 23, 42, 0.18);
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {

    font-family: Arial, Helvetica, sans-serif;

    color: #cbd5e1 !important;

    font-size: 1rem;

    font-weight: 600;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {

    font-family: Arial, Helvetica, sans-serif;

    color: #ffffff !important;

    font-weight: 800;

    font-size: 2.4rem;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================================
# CHARGEMENT DES MODELES
# =========================================================================

@st.cache_resource
def load_artifacts():

    if not os.path.exists(MODEL_PATH):

        return None, None, None

    if not os.path.exists(ENCODER_PATH):

        return None, None, None

    model = jb.load(MODEL_PATH)

    encoder = jb.load(ENCODER_PATH)

    scaler = None

    if os.path.exists(SCALER_PATH):

        scaler = jb.load(SCALER_PATH)

    return encoder, model, scaler


encoder, model, scaler = load_artifacts()


# =========================================================================
# HISTORIQUE
# =========================================================================

if "history" not in st.session_state:

    if os.path.exists(HISTORY_FILE):

        try:

            st.session_state.history = (
                pd.read_csv(HISTORY_FILE)
                .to_dict("records")
            )

        except Exception:

            st.session_state.history = []

    else:

        st.session_state.history = []


def save_history():

    if st.session_state.history:

        pd.DataFrame(
            st.session_state.history
        ).to_csv(
            HISTORY_FILE,
            index=False
        )


# =========================================================================
# CALCUL DE LA CONFIANCE
# =========================================================================

def compute_confidence(
    heures_etude,
    notes_prec,
    heures_sommeil,
    sujets_pratiques
):

    values = {

        "Heures_etude":
            heures_etude,

        "Notes_precedentes":
            notes_prec,

        "Heures_sommeil":
            heures_sommeil,

        "Sujets_entrainement_pratiques":
            sujets_pratiques,
    }

    scores = []

    for key, value in values.items():

        r = FEATURE_RANGES[key]

        lo = r["min"]
        hi = r["max"]

        width = hi - lo

        if lo <= value <= hi:

            scores.append(1.0)

        else:

            if value < lo:

                excess = lo - value

            else:

                excess = value - hi

            scores.append(
                max(
                    0.0,
                    1 - excess / width
                )
            )

    plausibility = float(
        np.mean(scores)
    )

    base = MODEL_R2 * 100

    confidence = (
        base *
        (
            0.55 +
            0.45 * plausibility
        )
    )

    return round(
        min(
            99.5,
            max(
                35.0,
                confidence
            )
        ),
        1
    )


# =========================================================================
# FONCTION DE PREDICTION
# =========================================================================

def predict_one(
    heures_etude,
    notes_prec,
    activite,
    heures_sommeil,
    sujets_pratiques
):

    """
    Effectue une prédiction avec le modèle Ridge.

    IMPORTANT :
    Le modèle doit recevoir les variables dans le même ordre
    que celui utilisé pendant l'entraînement.
    """

    # ---------------------------------------------------------
    # Vérification de l'encodeur
    # ---------------------------------------------------------

    if encoder is None:

        raise ValueError(
            "L'encodeur 'encoder.joblib' est introuvable."
        )


    # ---------------------------------------------------------
    # Encodage de l'activité extrascolaire
    # ---------------------------------------------------------

    try:

        activite_enc = encoder.transform(
            [activite]
        )[0]

    except ValueError:

        raise ValueError(
            f"Valeur inconnue pour "
            f"'Activites_extrascolaires' : {activite}. "
            f"Valeurs acceptées : "
            f"{list(encoder.classes_)}"
        )


    # ---------------------------------------------------------
    # Ordre exact des variables
    # ---------------------------------------------------------

    colonnes_utilisees = [

        "Heures_etude",

        "Notes_precedentes",

        "Activites_extrascolaires",

        "Heures_sommeil",

        "Sujets_entrainement_pratiques",
    ]


    # ---------------------------------------------------------
    # Valeurs brutes
    # ---------------------------------------------------------

    valeurs_brutes = [

        float(heures_etude),

        float(notes_prec),

        float(activite_enc),

        float(heures_sommeil),

        float(sujets_pratiques),
    ]


    # ---------------------------------------------------------
    # Création du tableau
    # ---------------------------------------------------------

    x_new = np.array(
        [valeurs_brutes],
        dtype=float
    )


    # ---------------------------------------------------------
    # Prédiction
    # ---------------------------------------------------------

    try:

        y_pred_raw = float(
            model.predict(x_new)[0]
        )

    except Exception as e:

        raise ValueError(
            f"Erreur lors de la prédiction : {e}"
        )


    # ---------------------------------------------------------
    # Clipping entre 10 et 100
    # ---------------------------------------------------------

    y_pred = round(

        min(
            TARGET_MAX,
            max(
                TARGET_MIN,
                y_pred_raw
            )
        ),

        2
    )


    # ---------------------------------------------------------
    # Confiance
    # ---------------------------------------------------------

    confidence = compute_confidence(

        heures_etude,

        notes_prec,

        heures_sommeil,

        sujets_pratiques
    )


    # ---------------------------------------------------------
    # Informations de diagnostic
    # ---------------------------------------------------------

    debug_info = {

        "prediction_avant_clipping":
            y_pred_raw,

        "colonnes_utilisees":
            colonnes_utilisees,

        "valeurs_brutes":
            valeurs_brutes,
    }


    # ---------------------------------------------------------
    # IMPORTANT :
    # On retourne bien 3 valeurs
    # ---------------------------------------------------------

    return (
        y_pred,
        confidence,
        debug_info
    )


# =========================================================================
# AJOUT HISTORIQUE
# =========================================================================

def add_to_history(row: dict):

    row["Horodatage"] = (
        datetime.now()
        .strftime("%Y-%m-%d %H:%M:%S")
    )

    st.session_state.history.append(row)

    save_history()


# =========================================================================
# =========================================================================
# EN-TÊTE
# =========================================================================

st.markdown(
    """
    <div class="hero">

        <h1>
            🎓 Prédiction de la Performance Académique des Étudiants
        </h1>

        <p>
            Plateforme intelligente basée sur le Machine Learning pour prédire
            la performance académique des étudiants à partir de leurs habitudes
            d'étude, de sommeil et de leurs résultats antérieurs.
            <br>
            <strong>Modèle : Régression Ridge | R² = 98,92 %</strong>
        </p>

        <div class="badge-row">

            <span class="tech-badge">
                🐍 Python
            </span>

            <span class="tech-badge">
                🤖 Machine Learning
            </span>

            <span class="tech-badge">
                🔬 Scikit-learn
            </span>

            <span class="tech-badge">
                📊 Plotly
            </span>

            <span class="tech-badge">
                ⚡ Streamlit
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================================
# VERIFICATION DES FICHIERS
# =========================================================================

if model is None or encoder is None:

    st.error(
        f"""
        ⚠️ Fichiers nécessaires introuvables.

        Vérifiez que les fichiers suivants sont présents
        dans le même dossier que `app.py` :

        - `{MODEL_PATH}`
        - `{ENCODER_PATH}`

        Le fichier `{SCALER_PATH}` est facultatif dans cette
        version car le modèle est utilisé directement sur
        les variables brutes.
        """
    )

    st.stop()


# =========================================================================
# SIDEBAR
# =========================================================================

with st.sidebar:

    st.markdown(
        "### 📋 Fiche du modèle"
    )

    st.markdown(
        f"""
        - **Algorithme** : Régression Ridge
        - **R² validation** : `{MODEL_R2:.4f}`
        - **RMSE validation** : `{MODEL_RMSE:.2f} points`
        - **Données d'entraînement** : 10 000 étudiants
        - **Variable cible** : Indice de performance (10–100)
        """
    )

    st.divider()

    st.markdown(
        "### ℹ️ Score de confiance"
    )

    st.caption(
        """
        Le score de confiance combine la précision
        globale du modèle et la plausibilité des valeurs
        saisies par rapport aux données d'entraînement.

        Il s'agit d'un indicateur pédagogique et non
        d'un intervalle de confiance statistique classique.
        """
    )

    st.divider()

    st.markdown(
        f"### 🕓 Historique : "
        f"{len(st.session_state.history)} prédiction(s)"
    )

    if st.button(
        "🗑️ Réinitialiser l'historique",
        use_container_width=True
    ):

        st.session_state.history = []

        if os.path.exists(HISTORY_FILE):

            os.remove(HISTORY_FILE)

        st.rerun()


# =========================================================================
# ONGLETS
# =========================================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Prédiction individuelle",
        "📁 Prédiction par lot (CSV)",
        "📊 Tableau de bord",
    ]
)


# =========================================================================
# TAB 1 — PREDICTION INDIVIDUELLE
# =========================================================================

with tab1:

    col_form, col_result = st.columns(
        [1.1, 1]
    )


    # ---------------------------------------------------------------------
    # FORMULAIRE
    # ---------------------------------------------------------------------

    with col_form:

        st.markdown(
            "#### 📝 Profil de l'étudiant"
        )

        with st.form(
            "form_prediction"
        ):

            c1, c2 = st.columns(2)


            # -------------------------------------------------------------
            # COLONNE 1
            # -------------------------------------------------------------

            with c1:

                heures_etude = st.slider(

                    FEATURE_RANGES[
                        "Heures_etude"
                    ]["label"],

                    min_value=0,

                    max_value=12,

                    value=int(
                        FEATURE_RANGES[
                            "Heures_etude"
                        ]["mean"]
                    ),

                    step=1,
                )


                heures_sommeil = st.slider(

                    FEATURE_RANGES[
                        "Heures_sommeil"
                    ]["label"],

                    min_value=0,

                    max_value=12,

                    value=int(
                        FEATURE_RANGES[
                            "Heures_sommeil"
                        ]["mean"]
                    ),

                    step=1,
                )


            # -------------------------------------------------------------
            # COLONNE 2
            # -------------------------------------------------------------

            with c2:

                notes_prec = st.slider(

                    FEATURE_RANGES[
                        "Notes_precedentes"
                    ]["label"],

                    min_value=0,

                    max_value=100,

                    value=int(
                        FEATURE_RANGES[
                            "Notes_precedentes"
                        ]["mean"]
                    ),

                    step=1,
                )


                sujets_pratiques = st.slider(

                    FEATURE_RANGES[
                        "Sujets_entrainement_pratiques"
                    ]["label"],

                    min_value=0,

                    max_value=12,

                    value=int(
                        FEATURE_RANGES[
                            "Sujets_entrainement_pratiques"
                        ]["mean"]
                    ),

                    step=1,
                )


            # -------------------------------------------------------------
            # ACTIVITES EXTRASCOLAIRES
            # -------------------------------------------------------------

            activite = st.radio(

                "Activités extrascolaires",

                options=list(
                    encoder.classes_
                ),

                horizontal=True,
            )


            # -------------------------------------------------------------
            # BOUTON
            # -------------------------------------------------------------

            submitted = st.form_submit_button(

                "🚀 Lancer la prédiction",

                use_container_width=True,
            )


        # ---------------------------------------------------------------
        # PREDICTION
        # ---------------------------------------------------------------

        if submitted:

            try:

                score, confiance, debug_info = predict_one(

                    heures_etude,

                    notes_prec,

                    activite,

                    heures_sommeil,

                    sujets_pratiques,
                )


                st.session_state.last_result = {

                    "Heures_etude":
                        heures_etude,

                    "Notes_precedentes":
                        notes_prec,

                    "Activites_extrascolaires":
                        activite,

                    "Heures_sommeil":
                        heures_sommeil,

                    "Sujets_entrainement_pratiques":
                        sujets_pratiques,

                    "Score_predit":
                        score,

                    "Confiance":
                        confiance,
                }


                st.session_state.last_debug = (
                    debug_info
                )


                add_to_history(
                    dict(
                        st.session_state.last_result
                    )
                )


                st.success(
                    "✅ Prédiction réalisée avec succès."
                )


            except Exception as e:

                st.error(
                    f"❌ Erreur lors de la prédiction : {e}"
                )


    # ---------------------------------------------------------------------
    # RESULTAT
    # ---------------------------------------------------------------------

    with col_result:

        st.markdown(
            "#### 🎯 Résultat"
        )

        if "last_result" in st.session_state:

            r = st.session_state.last_result

            tier, color, icon = get_tier(
                r["Score_predit"]
            )


            st.markdown(
                f"""
                <div class="result-card"
                     style="
                     background:
                     linear-gradient(
                         135deg,
                         {color}dd,
                         {color}99
                     );
                     ">

                    <div>
                        {icon}
                        Indice de performance prédit
                    </div>

                    <div class="result-score">
                        {r["Score_predit"]}
                    </div>

                    <div class="result-tier">
                        {tier}
                    </div>

                    <div class="result-sub">

                        Intervalle approximatif (95 %) :
                        {max(
                            TARGET_MIN,
                            r["Score_predit"] - CI_MARGIN
                        ):.1f}

                        –

                        {min(
                            TARGET_MAX,
                            r["Score_predit"] + CI_MARGIN
                        ): .1f}

                        &nbsp; | &nbsp;

                        Confiance :
                        {r["Confiance"]} %

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # -------------------------------------------------------------
            # DIAGNOSTIC
            # -------------------------------------------------------------

            debug = st.session_state.get(
                "last_debug"
            )


            if debug is not None:

                raw = debug[
                    "prediction_avant_clipping"
                ]


                with st.expander(
                    "🛠️ Diagnostic de la prédiction"
                ):

                    st.write(
                        "**Prédiction brute avant clipping :**",
                        raw
                    )

                    st.write(
                        "**Colonnes utilisées :**",
                        debug[
                            "colonnes_utilisees"
                        ]
                    )

                    st.write(
                        "**Valeurs brutes envoyées :**",
                        debug[
                            "valeurs_brutes"
                        ]
                    )


                    if raw <= TARGET_MIN:

                        st.warning(
                            f"""
                            La prédiction brute est de
                            `{raw:.2f}` avant le clipping
                            entre 10 et 100.

                            Vérifiez que l'ordre des variables
                            correspond exactement à celui utilisé
                            lors de l'entraînement du modèle.
                            """
                        )


            # -------------------------------------------------------------
            # JAUGE
            # -------------------------------------------------------------

            fig_gauge = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=r["Score_predit"],

                    number={
                        "suffix": " / 100"
                    },

                    gauge={

                        "axis": {
                            "range": [
                                TARGET_MIN,
                                TARGET_MAX
                            ]
                        },

                        "bar": {
                            "color": color
                        },

                        "steps": [

                            {
                                "range": [10, 40],
                                "color": "#fee2e2"
                            },

                            {
                                "range": [40, 55],
                                "color": "#fef3c7"
                            },

                            {
                                "range": [55, 70],
                                "color": "#e0e7ff"
                            },

                            {
                                "range": [70, 85],
                                "color": "#dbeafe"
                            },

                            {
                                "range": [85, 100],
                                "color": "#dcfce7"
                            },
                        ],
                    },
                )
            )


            fig_gauge.update_layout(

                height=260,

                margin=dict(
                    t=10,
                    b=10,
                    l=20,
                    r=20
                )
            )


            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )


        else:

            st.info(
                """
                Remplissez le formulaire puis cliquez
                sur **Lancer la prédiction** pour voir
                le résultat.
                """
            )


# =========================================================================
# TAB 2 — PREDICTION CSV
# =========================================================================

with tab2:

    st.markdown(
        "#### 📁 Prédiction pour plusieurs étudiants à la fois"
    )

    st.caption(
        """
        Le fichier CSV doit contenir exactement les colonnes :
        `Heures_etude`,
        `Notes_precedentes`,
        `Activites_extrascolaires`,
        `Heures_sommeil`,
        `Sujets_entrainement_pratiques`.
        """
    )


    # ---------------------------------------------------------------------
    # MODELE CSV
    # ---------------------------------------------------------------------

    template_df = pd.DataFrame({

        "Heures_etude": [5, 8],

        "Notes_precedentes": [65, 90],

        "Activites_extrascolaires": [
            "Yes",
            "No"
        ],

        "Heures_sommeil": [7, 6],

        "Sujets_entrainement_pratiques": [
            3,
            7
        ],
    })


    st.download_button(

        "⬇️ Télécharger un modèle CSV",

        template_df
        .to_csv(index=False)
        .encode("utf-8"),

        file_name="modele_donnees_etudiants.csv",

        mime="text/csv",
    )


    fichier = st.file_uploader(

        "Déposer un fichier CSV",

        type=["csv"]
    )


    if fichier is not None:

        try:

            df_csv = pd.read_csv(
                fichier
            )


            colonnes_attendues = [

                "Heures_etude",

                "Notes_precedentes",

                "Activites_extrascolaires",

                "Heures_sommeil",

                "Sujets_entrainement_pratiques",
            ]


            manquantes = [

                c for c in colonnes_attendues

                if c not in df_csv.columns
            ]


            if manquantes:

                st.error(

                    "❌ Colonnes manquantes : "
                    + ", ".join(manquantes)
                )


            else:

                scores = []

                confiances = []

                erreurs = []


                # ---------------------------------------------------------
                # PREDICTION LIGNE PAR LIGNE
                # ---------------------------------------------------------

                for index, row in df_csv.iterrows():

                    try:

                        s, c, debug = predict_one(

                            row["Heures_etude"],

                            row["Notes_precedentes"],

                            row[
                                "Activites_extrascolaires"
                            ],

                            row["Heures_sommeil"],

                            row[
                                "Sujets_entrainement_pratiques"
                            ],
                        )


                        scores.append(s)

                        confiances.append(c)

                        erreurs.append("")


                        add_to_history({

                            "Heures_etude":
                                row["Heures_etude"],

                            "Notes_precedentes":
                                row["Notes_precedentes"],

                            "Activites_extrascolaires":
                                row[
                                    "Activites_extrascolaires"
                                ],

                            "Heures_sommeil":
                                row["Heures_sommeil"],

                            "Sujets_entrainement_pratiques":
                                row[
                                    "Sujets_entrainement_pratiques"
                                ],

                            "Score_predit":
                                s,

                            "Confiance":
                                c,
                        })


                    except Exception as e:

                        scores.append(np.nan)

                        confiances.append(np.nan)

                        erreurs.append(str(e))


                # ---------------------------------------------------------
                # AJOUT DES RESULTATS
                # ---------------------------------------------------------

                df_csv[
                    "Indice_performance_predit"
                ] = scores


                df_csv[
                    "Confiance (%)"
                ] = confiances


                df_csv[
                    "Niveau"
                ] = [

                    get_tier(s)[0]
                    if pd.notna(s)
                    else "Erreur"

                    for s in scores
                ]


                df_csv[
                    "Erreur"
                ] = erreurs


                nb_succes = (
                    df_csv[
                        "Indice_performance_predit"
                    ]
                    .notna()
                    .sum()
                )


                st.success(
                    f"✅ {nb_succes} prédiction(s) "
                    f"effectuée(s) avec succès."
                )


                st.dataframe(
                    df_csv,
                    use_container_width=True,
                    hide_index=True,
                )


                # ---------------------------------------------------------
                # GRAPHIQUE
                # ---------------------------------------------------------

                df_plot = df_csv.dropna(
                    subset=[
                        "Indice_performance_predit"
                    ]
                )


                if not df_plot.empty:

                    fig_batch = px.scatter(

                        df_plot,

                        x="Heures_etude",

                        y="Indice_performance_predit",

                        color="Niveau",

                        size="Confiance (%)",

                        color_discrete_map={
                            t[2]: t[3]
                            for t in TIERS
                        },

                        hover_data=[
                            "Notes_precedentes",
                            "Activites_extrascolaires",
                        ],

                        title=(
                            "Répartition des "
                            "prédictions du lot"
                        ),
                    )


                    st.plotly_chart(
                        fig_batch,
                        use_container_width=True
                    )


                # ---------------------------------------------------------
                # TELECHARGEMENT
                # ---------------------------------------------------------

                st.download_button(

                    "⬇️ Télécharger les résultats CSV",

                    df_csv
                    .to_csv(index=False)
                    .encode("utf-8"),

                    file_name=(
                        "predictions_resultats.csv"
                    ),

                    mime="text/csv",
                )


        except Exception as e:

            st.error(
                f"❌ Erreur lors du traitement du fichier : {e}"
            )


# =========================================================================
# TAB 3 — TABLEAU DE BORD
# =========================================================================

with tab3:

    st.markdown(
        "#### 📊 Tableau de bord des prédictions"
    )


    if not st.session_state.history:

        st.info(
            """
            Aucune prédiction pour le moment.

            Utilisez l'onglet
            **Prédiction individuelle**
            ou **Prédiction par lot CSV**.
            """
        )


    else:

        hist_df = pd.DataFrame(
            st.session_state.history
        )


        # -----------------------------------------------------------------
        # KPI
        # -----------------------------------------------------------------

        k1, k2, k3, k4 = st.columns(4)


        k1.metric(
            "Prédictions réalisées",
            len(hist_df)
        )


        k2.metric(
            "Score moyen prédit",
            f"{hist_df['Score_predit'].mean():.1f}"
        )


        k3.metric(
            "Confiance moyenne",
            f"{hist_df['Confiance'].mean():.1f}%"
        )


        best = hist_df[
            "Score_predit"
        ].max()


        k4.metric(
            "Meilleur score",
            f"{best:.1f}"
        )


        # -----------------------------------------------------------------
        # NIVEAU
        # -----------------------------------------------------------------

        hist_df["Niveau"] = (
            hist_df["Score_predit"]
            .apply(
                lambda s: get_tier(s)[0]
            )
        )


        # -----------------------------------------------------------------
        # PROFIL NORMALISE
        # -----------------------------------------------------------------

        hist_df["Profil"] = hist_df.apply(

            lambda r: [

                r["Heures_etude"]
                / FEATURE_RANGES[
                    "Heures_etude"
                ]["max"],

                r["Notes_precedentes"]
                / FEATURE_RANGES[
                    "Notes_precedentes"
                ]["max"],

                r["Heures_sommeil"]
                / FEATURE_RANGES[
                    "Heures_sommeil"
                ]["max"],

                r[
                    "Sujets_entrainement_pratiques"
                ]
                / FEATURE_RANGES[
                    "Sujets_entrainement_pratiques"
                ]["max"],
            ],

            axis=1,
        )


        # -----------------------------------------------------------------
        # TABLEAU
        # -----------------------------------------------------------------

        st.markdown(
            "##### 🔎 Détail des prédictions"
        )


        display_cols = [

            "Horodatage",

            "Heures_etude",

            "Notes_precedentes",

            "Activites_extrascolaires",

            "Heures_sommeil",

            "Sujets_entrainement_pratiques",

            "Score_predit",

            "Confiance",

            "Niveau",

            "Profil",
        ]


        display_cols = [

            c for c in display_cols

            if c in hist_df.columns
        ]


        try:

            st.dataframe(

                hist_df[
                    display_cols
                ].iloc[::-1],

                use_container_width=True,

                hide_index=True,

                column_config={

                    "Score_predit":
                        st.column_config.ProgressColumn(

                            "Score prédit",

                            min_value=TARGET_MIN,

                            max_value=TARGET_MAX,

                            format="%.1f"
                        ),

                    "Confiance":
                        st.column_config.ProgressColumn(

                            "Confiance",

                            min_value=0,

                            max_value=100,

                            format="%.0f%%"
                        ),

                    "Profil":
                        st.column_config.BarChartColumn(

                            "Profil normalisé",

                            y_min=0,

                            y_max=1
                        ),
                },
            )


        except Exception:

            st.dataframe(

                hist_df[
                    display_cols
                ].iloc[::-1],

                use_container_width=True,

                hide_index=True,
            )


        # -----------------------------------------------------------------
        # ANALYSES GRAPHIQUES
        # -----------------------------------------------------------------

        st.markdown(
            "##### 📈 Analyses graphiques"
        )


        g1, g2 = st.columns(2)


        # -----------------------------------------------------------------
        # EVOLUTION DES SCORES
        # -----------------------------------------------------------------

        with g1:

            fig_trend = px.line(

                hist_df.reset_index(),

                x="index",

                y="Score_predit",

                markers=True,

                title=(
                    "Évolution des scores prédits"
                ),

                labels={

                    "index":
                        "N° prédiction",

                    "Score_predit":
                        "Indice de performance",
                },
            )


            fig_trend.update_traces(

                line_color="#4338ca"
            )


            st.plotly_chart(

                fig_trend,

                use_container_width=True
            )


        # -----------------------------------------------------------------
        # CAMEMBERT
        # -----------------------------------------------------------------

        with g2:

            tier_counts = (
                hist_df[
                    "Niveau"
                ]
                .value_counts()
                .reset_index()
            )


            tier_counts.columns = [
                "Niveau",
                "Nombre"
            ]


            fig_pie = px.pie(

                tier_counts,

                names="Niveau",

                values="Nombre",

                hole=0.5,

                color="Niveau",

                color_discrete_map={
                    t[2]: t[3]
                    for t in TIERS
                },

                title=(
                    "Répartition par niveau "
                    "de performance"
                ),
            )


            st.plotly_chart(

                fig_pie,

                use_container_width=True
            )


        # -----------------------------------------------------------------
        # COEFFICIENTS + NOTES
        # -----------------------------------------------------------------

        g3, g4 = st.columns(2)


        # -----------------------------------------------------------------
        # COEFFICIENTS RIDGE
        # -----------------------------------------------------------------

        with g3:

            try:

                coefs = pd.DataFrame({

                    "Variable": [
                        "Heures_etude",
                        "Notes_precedentes",
                        "Activites_extrascolaires",
                        "Heures_sommeil",
                        "Sujets_entrainement_pratiques",
                    ],

                    "Coefficient":
                        model.coef_,
                })


                coefs = coefs.sort_values(
                    "Coefficient"
                )


                fig_coef = px.bar(

                    coefs,

                    x="Coefficient",

                    y="Variable",

                    orientation="h",

                    title=(
                        "Coefficients du modèle Ridge"
                    ),

                    color="Coefficient",

                    color_continuous_scale="Blues",
                )


                st.plotly_chart(

                    fig_coef,

                    use_container_width=True
                )


            except Exception as e:

                st.warning(
                    "Impossible d'afficher les "
                    f"coefficients : {e}"
                )


        # -----------------------------------------------------------------
        # NOTES PRECEDENTES
        # -----------------------------------------------------------------

        with g4:

            fig_scatter = px.scatter(

                hist_df,

                x="Notes_precedentes",

                y="Score_predit",

                color="Niveau",

                color_discrete_map={
                    t[2]: t[3]
                    for t in TIERS
                },

                size="Confiance",

                hover_data=[
                    "Heures_etude"
                ],

                title=(
                    "Notes précédentes "
                    "vs Score prédit"
                ),
            )


            st.plotly_chart(

                fig_scatter,

                use_container_width=True
            )


# =========================================================================
# PIED DE PAGE
# =========================================================================

st.divider()


st.markdown(

    """
    <p style="
        text-align:center;
        color:#64748b;
        font-size:0.85rem;
    ">

        🎓 Projet Data Science & Machine Learning —
        Régression Ridge (R² = 98,9 %)

        · Développé avec Python,
        scikit-learn, Plotly & Streamlit

    </p>
    """,

    unsafe_allow_html=True
)
