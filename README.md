# 🎓 Prédiction de la Performance des Étudiants

Plateforme Streamlit de Machine Learning (régression Ridge, R² = 98.9%, RMSE = 2.04)
prédisant l'indice de performance d'un étudiant à partir de 5 variables académiques
et personnelles.

## 📦 Contenu du dossier

```
app.py               → application Streamlit
requirements.txt      → dépendances Python
encoder.joblib         → LabelEncoder (variable "Activités extrascolaires")
rr_model.joblib         → modèle Ridge entraîné
scaler.joblib          → RobustScaler (reconstruit à partir des statistiques
README.md                         réelles du dataset : médiane/IQR de chaque variable,
                          car le fichier scaler original fourni était corrompu)
```

⚠️ **Ces 4 fichiers doivent rester dans le même dossier** que `app.py`.

## 🚀 Installation et lancement

```bash
# 1. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse
`http://localhost:8501`.

## 🧠 À propos du modèle

| Métrique | Valeur (validation) |
|---|---|
| R² | 0.9892 |
| RMSE | 2.04 points |
| Algorithme | Ridge Regression |
| Données d'entraînement | 10 000 étudiants |

**Variables d'entrée** : Heures d'étude, Notes précédentes, Activités extrascolaires
(Oui/Non), Heures de sommeil, Sujets d'entraînement pratiqués.

**Variable cible** : Indice de performance (échelle 10–100).

### Score de confiance

Le score affiché combine deux éléments réels :
1. La précision globale du modèle sur le jeu de validation (R² = 98.9%)
2. La plausibilité des valeurs saisies par rapport aux plages du jeu d'entraînement
   (un modèle linéaire perd en fiabilité lorsqu'on extrapole en dehors des données
   sur lesquelles il a appris)

Ce n'est **pas** un intervalle de confiance statistique au sens strict — c'est un
indicateur pédagogique, documenté comme tel dans l'application elle-même.

## 📁 Prédiction par lot (CSV)

Le fichier CSV doit contenir exactement ces colonnes :
`Heures_etude`, `Notes_precedentes`, `Activites_extrascolaires` (valeurs `Yes`/`No`),
`Heures_sommeil`, `Sujets_entrainement_pratiques`.

Un exemple de fichier est téléchargeable directement depuis l'application
(onglet "Prédiction par lot").

## 🎯 Pistes d'amélioration pour aller plus loin

- Ajouter l'authentification multi-utilisateurs (ex: `streamlit-authenticator`)
- Déployer sur Streamlit Community Cloud, Render ou Hugging Face Spaces pour un lien public
- Ré-entraîner le modèle avec un vrai `scaler.joblib` sauvegardé proprement dès
  l'entraînement (voir remarque ci-dessous)
- Ajouter un vrai intervalle de prédiction (ex: `MAPIE`, quantile regression) plutôt
  que l'heuristique actuelle

## ⚠️ Remarque technique importante

Le fichier `scaler.joblib` original fourni était corrompu (`center_ = [0,0,0,0,0]`,
`scale_ = [1,1,1,1,1]` — l'effet d'une double transformation lors de l'entraînement,
un piège classique en notebook Jupyter/Colab). Il a été **reconstruit** ici à partir
des statistiques réelles (médiane et IQR) mesurées sur les 10 000 lignes du dataset
d'entraînement, ce qui a été vérifié empiriquement : les prédictions redeviennent
cohérentes (dans la plage 10–100) sur l'ensemble du domaine de données.

**Pour un usage en production réel**, il est recommandé de ré-entraîner le modèle et
de sauvegarder un `scaler.joblib` proprement fitté (voir la correction faite plus tôt
sur `x_scaled = scaler.fit_transform(x)` sans écraser la variable `x` d'origine).
