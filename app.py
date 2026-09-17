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
import os
from datetime import datetime

import joblib as jb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Prédiction de la Performance des Étudiants",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VARIABLES DU MODÈLE
# ============================================================

FEATURE_RANGES = {
    "Heures_etude": (1, 9),
    "Notes_precedentes": (40, 99),
    "Activites_extrascolaires": (
        "Aucune",
        "Sport",
        "Musique",
        "Club",
        "Bénévolat",
    ),
    "Heures_sommeil": (4, 9),
    "Sujets_entrainement_pratiques": (0, 9),
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

DEFAULT_FEATURE_ORDER = [
    "Heures_etude",
    "Notes_precedentes",
    "Activites_extrascolaires",
    "Heures_sommeil",
    "Sujets_entrainement_pratiques",
]


# ============================================================
# NIVEAUX DE PERFORMANCE
# ============================================================

TIERS = [
    (85, 100, "Excellent", "#16a34a", "🌟"),
    (70, 85, "Très bien", "#0ea5e9", "✅"),
    (55, 70, "Bien", "#6366f1", "👍"),
    (40, 55, "Moyen", "#f59e0b", "⚠️"),
    (10, 40, "À risque", "#dc2626", "🚨"),
]


def get_tier(score):
    """Retourne le niveau correspondant au score."""
    score = float(score)

    for minimum, maximum, niveau, couleur, icone in TIERS:
        if minimum <= score < maximum:
            return niveau, couleur, icone

    if score >= 100:
        return "Excellent", "#16a34a", "🌟"

    return "À risque", "#dc2626", "🚨"


# ============================================================
# STYLE CSS
# ============================================================

st.markdown(
    """
<style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 35px;
        border-radius: 20px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 17px;
        opacity: 0.92;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        text-align: center;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }

    .result-card {
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.15);
    }

    .result-score {
        font-size: 65px;
        font-weight: 900;
        line-height: 1.1;
        margin: 10px 0;
    }

    .result-tier {
        font-size: 25px;
        font-weight: 800;
    }

    .result-sub {
        font-size: 15px;
        margin-top: 15px;
        opacity: 0.95;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 15px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

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


# ============================================================
# HISTORIQUE
# ============================================================

def load_history():

    if os.path.exists(HISTORY_FILE):

        try:
            return pd.read_csv(HISTORY_FILE)

        except Exception:
            pass

    return pd.DataFrame(
        columns=[
            "Date",
            "Heures_etude",
            "Notes_precedentes",
            "Activites_extrascolaires",
            "Heures_sommeil",
            "Sujets_entrainement_pratiques",
            "Score_predit",
            "Confiance",
            "Niveau",
        ]
    )


history = load_history()


def save_history(df):

    try:
        df.to_csv(HISTORY_FILE, index=False)

    except Exception as e:
        st.warning(f"Impossible d'enregistrer l'historique : {e}")


# ============================================================
# CALCUL DE LA CONFIANCE
# ============================================================

def compute_confidence(values):

    confidence = MODEL_R2 * 100

    penalties = 0

    for feature, value in values.items():

        if feature not in FEATURE_RANGES:
            continue

        feature_range = FEATURE_RANGES[feature]

        if isinstance(feature_range[0], (int, float)):

            minimum = feature_range[0]
            maximum = feature_range[1]

            if value < minimum or value > maximum:
                penalties += 5

    confidence -= penalties

    confidence = max(0, min(99.9, confidence))

    return confidence


# ============================================================
# ENCODAGE DE L'ACTIVITÉ
# ============================================================

def encode_activity(activite):

    try:

        value = encoder.transform([activite])[0]

        return float(value)

    except Exception:

        try:

            value = encoder.transform([[activite]])

            return float(np.asarray(value).ravel()[0])

        except Exception as e:

            raise ValueError(
                f"Impossible d'encoder l'activité extrascolaire : {e}"
            )


# ============================================================
# PRÉDICTION
# ============================================================

def predict_one(
    heures_etude,
    notes_precedentes,
    activite,
    heures_sommeil,
    sujets_pratiques,
):

    if encoder is None:

        raise ValueError(
            "L'encodeur encoder.joblib est introuvable."
        )

    activite_enc = encode_activity(activite)

    values = {
        "Heures_etude": heures_etude,
        "Notes_precedentes": notes_precedentes,
        "Activites_extrascolaires": activite,
        "Heures_sommeil": heures_sommeil,
        "Sujets_entrainement_pratiques": sujets_pratiques,
    }

    x_new = np.array(
        [[
            heures_etude,
            notes_precedentes,
            activite_enc,
            heures_sommeil,
            sujets_pratiques,
        ]],
        dtype=float,
    )

    # IMPORTANT :
    # Le scaler n'est utilisé que si le modèle a été entraîné
    # avec exactement ce même scaler.
    #
    # Si ton modèle Ridge a été entraîné sur les variables brutes,
    # on ne transforme pas ici les données.

    prediction = model.predict(x_new)[0]

    prediction_before_clipping = float(prediction)

    prediction = np.clip(
        prediction,
        TARGET_MIN,
        TARGET_MAX,
    )

    confidence = compute_confidence(values)

    debug_info = {
        "prediction_avant_clipping": prediction_before_clipping,
        "variables_entree": values,
        "vecteur_modele": x_new.tolist(),
    }

    return (
        float(prediction),
        float(confidence),
        debug_info,
    )


# ============================================================
# AJOUT À L'HISTORIQUE
# ============================================================

def add_to_history(
    heures_etude,
    notes_precedentes,
    activite,
    heures_sommeil,
    sujets_pratiques,
    score,
    confidence,
):

    global history

    niveau = get_tier(score)[0]

    new_row = pd.DataFrame(
        [
            {
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Heures_etude": heures_etude,
                "Notes_precedentes": notes_precedentes,
                "Activites_extrascolaires": activite,
                "Heures_sommeil": heures_sommeil,
                "Sujets_entrainement_pratiques": sujets_pratiques,
                "Score_predit": round(score, 2),
                "Confiance": round(confidence, 2),
                "Niveau": niveau,
            }
        ]
    )

    history = pd.concat(
        [history, new_row],
        ignore_index=True,
    )

    save_history(history)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

    <h1>🎓 Prédiction de la Performance des Étudiants</h1>

    <p>
        Une plateforme basée sur le Machine Learning permettant
        de prédire l'indice de performance académique d'un étudiant.
    </p>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# VÉRIFICATION DES FICHIERS
# ============================================================

if model is None or encoder is None:

    st.error(
        """
        ❌ Les fichiers nécessaires au modèle sont introuvables.

        Vérifiez que les fichiers suivants sont bien présents
        dans le même dossier que app.py :

        • rr_model.joblib
        • encoder.joblib
        • scaler.joblib
        """
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Paramètres du modèle")

    st.metric(
        "R² du modèle",
        f"{MODEL_R2:.4f}",
    )

    st.metric(
        "RMSE",
        f"{MODEL_RMSE:.2f}",
    )

    st.markdown("---")

    st.markdown(
        """
### 📊 Variables utilisées

- 📚 Heures d'étude
- 📝 Notes précédentes
- 🎯 Activités extrascolaires
- 😴 Heures de sommeil
- 📖 Sujets d'entraînement pratiques

### 🤖 Modèle

**Ridge Regression**
"""
    )


# ============================================================
# ONGLETS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Prédiction individuelle",
        "📂 Prédiction CSV",
        "📊 Tableau de bord",
    ]
)


# ============================================================
# TAB 1 — PRÉDICTION INDIVIDUELLE
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">🔎 Informations de l’étudiant</div>',
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns(
        [1.1, 0.9],
        gap="large",
    )

    with col_form:

        with st.form("prediction_form"):

            heures_etude = st.slider(
                "📚 Heures d'étude",
                min_value=1,
                max_value=9,
                value=5,
                step=1,
            )

            notes_precedentes = st.slider(
                "📝 Notes précédentes",
                min_value=40,
                max_value=99,
                value=70,
                step=1,
            )

            activite = st.selectbox(
                "🎯 Activités extrascolaires",
                [
                    "Aucune",
                    "Sport",
                    "Musique",
                    "Club",
                    "Bénévolat",
                ],
            )

            heures_sommeil = st.slider(
                "😴 Heures de sommeil",
                min_value=4,
                max_value=9,
                value=7,
                step=1,
            )

            sujets_pratiques = st.slider(
                "📖 Sujets d'entraînement pratiques",
                min_value=0,
                max_value=9,
                value=5,
                step=1,
            )

            submitted = st.form_submit_button(
                "🚀 PRÉDIRE LA PERFORMANCE",
                use_container_width=True,
            )

        if submitted:

            try:

                score, confidence, debug_info = predict_one(
                    heures_etude,
                    notes_precedentes,
                    activite,
                    heures_sommeil,
                    sujets_pratiques,
                )

                st.session_state.last_result = {
                    "Score_predit": score,
                    "Confiance": confidence,
                }

                st.session_state.last_debug = debug_info

                add_to_history(
                    heures_etude,
                    notes_precedentes,
                    activite,
                    heures_sommeil,
                    sujets_pratiques,
                    score,
                    confidence,
                )

                st.success(
                    "✅ Prédiction effectuée avec succès."
                )

            except Exception as e:

                st.error(
                    f"❌ Erreur pendant la prédiction : {e}"
                )

    # ========================================================
    # RÉSULTAT
    # ========================================================

    with col_result:

        st.markdown("#### 🎯 Résultat")

        if "last_result" in st.session_state:

            result = st.session_state.last_result

            score = result["Score_predit"]
            confidence = result["Confiance"]

            niveau, couleur, icone = get_tier(score)

            borne_inf = max(
                TARGET_MIN,
                score - CI_MARGIN,
            )

            borne_sup = min(
                TARGET_MAX,
                score + CI_MARGIN,
            )

            st.markdown(
                f"""
                <div class="result-card"
                     style="background: linear-gradient(135deg, {couleur}dd, {couleur}99);">

                    <div>
                        {icone} Indice de performance prédit
                    </div>

                    <div class="result-score">
                        {score:.2f}
                    </div>

                    <div class="result-tier">
                        {niveau}
                    </div>

                    <div class="result-sub">

                        Intervalle approximatif (95 %) :
                        <strong>
                            {borne_inf:.1f} – {borne_sup:.1f}
                        </strong>

                        &nbsp; | &nbsp;

                        Confiance :
                        <strong>
                            {confidence:.1f} %
                        </strong>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.info(
                "👈 Saisissez les informations de l'étudiant puis cliquez sur « PRÉDIRE LA PERFORMANCE »."
            )


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

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
                "Valeur prédite avant limitation :",
                round(raw, 4),
            )

            st.write(
                "Variables transmises au modèle :"
            )

            st.json(
                debug["variables_entree"]
            )

            st.write(
                "Vecteur utilisé par le modèle :"
            )

            st.write(
                debug["vecteur_modele"]
            )


    # ========================================================
    # JAUGE
    # ========================================================

    if "last_result" in st.session_state:

        score = st.session_state.last_result[
            "Score_predit"
        ]

        st.markdown(
            "### 📈 Niveau de performance"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={
                    "text": "Indice de performance"
                },
                gauge={
                    "axis": {
                        "range": [
                            TARGET_MIN,
                            TARGET_MAX,
                        ]
                    },
                    "bar": {
                        "color": "#1e3a8a"
                    },
                    "steps": [
                        {
                            "range": [10, 40],
                            "color": "#fee2e2",
                        },
                        {
                            "range": [40, 55],
                            "color": "#fef3c7",
                        },
                        {
                            "range": [55, 70],
                            "color": "#e0e7ff",
                        },
                        {
                            "range": [70, 85],
                            "color": "#e0f2fe",
                        },
                        {
                            "range": [85, 100],
                            "color": "#dcfce7",
                        },
                    ],
                },
            )
        )

        gauge.update_layout(
            height=350,
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )


# ============================================================
# TAB 2 — CSV
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">📂 Prédiction à partir d’un fichier CSV</div>',
        unsafe_allow_html=True,
    )

    st.info(
        """
        Le fichier CSV doit contenir les colonnes suivantes :

        `Heures_etude`, `Notes_precedentes`,
        `Activites_extrascolaires`, `Heures_sommeil`,
        `Sujets_entrainement_pratiques`
        """
    )

    uploaded_file = st.file_uploader(
        "📎 Importer votre fichier CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            df_csv = pd.read_csv(
                uploaded_file
            )

            st.markdown(
                "### 👀 Aperçu des données"
            )

            st.dataframe(
                df_csv,
                use_container_width=True,
            )

            missing_columns = [
                col
                for col in DEFAULT_FEATURE_ORDER
                if col not in df_csv.columns
            ]

            if missing_columns:

                st.error(
                    "❌ Colonnes manquantes : "
                    + ", ".join(missing_columns)
                )

            else:

                if st.button(
                    "🚀 Lancer les prédictions",
                    key="csv_prediction",
                ):

                    scores = []
                    confidences = []
                    niveaux = []

                    for _, row in df_csv.iterrows():

                        try:

                            score, confidence, _ = predict_one(
                                row["Heures_etude"],
                                row["Notes_precedentes"],
                                row["Activites_extrascolaires"],
                                row["Heures_sommeil"],
                                row["Sujets_entrainement_pratiques"],
                            )

                            scores.append(score)
                            confidences.append(
                                confidence
                            )

                            niveaux.append(
                                get_tier(score)[0]
                            )

                        except Exception:

                            scores.append(np.nan)
                            confidences.append(np.nan)
                            niveaux.append("Erreur")

                    df_result = df_csv.copy()

                    df_result[
                        "Score_predit"
                    ] = scores

                    df_result[
                        "Confiance"
                    ] = confidences

                    df_result[
                        "Niveau"
                    ] = niveaux

                    st.markdown(
                        "### 📊 Résultats des prédictions"
                    )

                    st.dataframe(
                        df_result,
                        use_container_width=True,
                    )

                    csv_download = (
                        df_result.to_csv(
                            index=False
                        ).encode("utf-8")
                    )

                    st.download_button(
                        "⬇️ Télécharger les résultats",
                        data=csv_download,
                        file_name="predictions_etudiants.csv",
                        mime="text/csv",
                    )

        except Exception as e:

            st.error(
                f"❌ Impossible de lire le fichier : {e}"
            )


# ============================================================
# TAB 3 — TABLEAU DE BORD
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">📊 Tableau de bord des prédictions</div>',
        unsafe_allow_html=True,
    )

    if history.empty:

        st.info(
            "📌 Aucune prédiction n'a encore été enregistrée."
        )

    else:

        total_predictions = len(
            history
        )

        average_score = history[
            "Score_predit"
        ].mean()

        average_confidence = history[
            "Confiance"
        ].mean()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Total des prédictions
                    </div>

                    <div class="metric-value">
                        {total_predictions}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Score moyen
                    </div>

                    <div class="metric-value">
                        {average_score:.2f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        Confiance moyenne
                    </div>

                    <div class="metric-value">
                        {average_confidence:.1f} %
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ====================================================
        # HISTOGRAMME
        # ====================================================

        st.markdown(
            "### 📈 Distribution des performances"
        )

        fig_hist = px.histogram(
            history,
            x="Score_predit",
            nbins=15,
            title="Distribution des scores prédits",
        )

        fig_hist.update_layout(
            xaxis_title="Score prédit",
            yaxis_title="Nombre d'étudiants",
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )

        # ====================================================
        # RELATION HEURES D'ÉTUDE / PERFORMANCE
        # ====================================================

        if "Heures_etude" in history.columns:

            st.markdown(
                "### 📚 Heures d'étude et performance"
            )

            fig_scatter = px.scatter(
                history,
                x="Heures_etude",
                y="Score_predit",
                color="Niveau",
                title="Relation entre les heures d'étude et la performance",
            )

            fig_scatter.update_layout(
                xaxis_title="Heures d'étude",
                yaxis_title="Score prédit",
            )

            st.plotly_chart(
                fig_scatter,
                use_container_width=True,
            )

        # ====================================================
        # HISTORIQUE
        # ====================================================

        st.markdown(
            "### 🧾 Historique des prédictions"
        )

        st.dataframe(
            history,
            use_container_width=True,
        )
