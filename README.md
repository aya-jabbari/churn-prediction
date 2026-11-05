# Prédiction du churn client – opérateur télécom

Ce projet part d'une question simple : parmi les clients d'un opérateur télécom,
lesquels vont résilier dans les prochains mois, et pourquoi ?

Le but n'est pas seulement d'avoir un bon score, mais un modèle utilisable par une équipe
marketing : une liste de clients à contacter en priorité, avec les raisons du risque pour
chacun.

**Démo :** https://XX.streamlit.app

![Aperçu de l'application](images/app.png)

---

## Les données

J'ai utilisé le dataset *Telco Customer Churn* (IBM, disponible sur Kaggle) : 7 043 clients,
19 variables (profil, services souscrits, type de contrat, facturation) et une cible `Churn`.

Points à noter :

- Environ 26 % des clients sont partis. Les classes sont déséquilibrées, donc l'accuracy
  n'est pas une bonne métrique ici : un modèle qui prédit « personne ne part » aurait déjà 74 %.
- La colonne `TotalCharges` était lue comme du texte à cause de 11 valeurs vides. Ces lignes
  correspondent toutes à des clients avec `tenure = 0` (inscrits ce mois-ci), je les ai donc
  remplacées par 0 plutôt que de les supprimer.

## Ce que montre l'analyse exploratoire

- **Le type de contrat est le facteur le plus marqué** : environ 40 % de départs chez les
  clients au mois, contre moins de 5 % pour les contrats de deux ans.
- **Les départs se concentrent dans les premiers mois.** Passé la première année, le risque
  baisse nettement.
- Les clients qui paient entre 70 et 110 $/mois partent plus souvent. Ce sont surtout des
  abonnés à la fibre, ce qui pose la question du rapport qualité/prix de cette offre.
- L'absence de support technique et le paiement par chèque électronique sont aussi associés
  à plus de départs.

Le détail est dans `notebooks/01_eda.ipynb`.

## Modélisation

Tout le prétraitement (standardisation des variables numériques, one-hot encoding des
catégorielles) est dans un `Pipeline` scikit-learn, appris uniquement sur le jeu
d'entraînement. Le même objet sert pour l'évaluation et dans l'application, ce qui évite
le data leakage et les écarts entre notebook et production.

J'ai comparé trois modèles en validation croisée stratifiée (5 folds) :

| Modèle                 | ROC-AUC (CV)   |
|------------------------|----------------|
| Régression logistique  | 0.XX ± 0.0X    |
| Random Forest          | 0.XX ± 0.0X    |
| XGBoost                | 0.XX ± 0.0X    |

Les écarts sont faibles : sur ce type de données tabulaires assez simples, un modèle
linéaire bien préparé fait presque aussi bien qu'un modèle de boosting. J'ai retenu
**XGBoost**, qui obtient un ROC-AUC de **0.XX** sur le jeu de test (20 % des données,
jamais vu pendant l'entraînement).

Le déséquilibre des classes est géré par pondération (`class_weight`, `scale_pos_weight`)
plutôt que par suréchantillonnage, ce qui garde les données d'origine intactes.

## Choisir le seuil selon le coût, pas à 0.5

Un modèle de churn sert à décider qui contacter. Les deux erreurs n'ont pas le même coût :

- rater un client qui part, c'est perdre son chiffre d'affaires (estimé à 500 $) ;
- contacter un client qui serait resté, c'est le coût d'une offre de rétention (50 $).

J'ai donc calculé le coût total pour chaque seuil entre 0.05 et 0.95. Le seuil optimal est
**0.XX**, ce qui réduit le coût d'environ **XX %** par rapport au seuil par défaut.

En pratique, **cibler les 20 % de clients les plus à risque permet de toucher XX % des
départs réels.**

Ces coûts sont des hypothèses. Dans un vrai contexte, ils viendraient des équipes finance
et marketing, et le seuil se recalcule en une ligne.

## Comprendre les prédictions avec SHAP

![Valeurs SHAP](images/shap_summary.png)

SHAP confirme l'analyse exploratoire : contrat mensuel, faible ancienneté et fibre optique
sont les variables qui poussent le plus vers le départ. L'intérêt est surtout au niveau
individuel : pour chaque client, on peut dire quelles caractéristiques expliquent son score,
ce qui aide à choisir la bonne action (proposer un engagement, offrir le support technique…).

## Pistes d'action

À partir des résultats :

1. Proposer aux clients au mois une remise pour passer à un engagement d'un an.
2. Renforcer le suivi des nouveaux clients pendant les six premiers mois.
3. Inclure le support technique pour les clients fibre identifiés comme à risque.

## Limites

- Le dataset est une photo à un instant donné : il n'y a pas d'historique d'usage
  (appels au service client, incidents réseau…), qui serait sans doute très prédictif.
- Les coûts utilisés pour le seuil sont des hypothèses.
- Le modèle n'a pas été testé dans le temps. En production, il faudrait surveiller la
  dérive des données et le réentraîner régulièrement.

## Structure du projet

    churn-prediction/
    ├── data/                     données brutes et nettoyées
    ├── notebooks/
    │   ├── 01_eda.ipynb          analyse exploratoire
    │   ├── 02_modeling.ipynb     pipeline, comparaison des modèles, seuil
    │   └── 03_explainability.ipynb   SHAP
    ├── models/model.pkl          pipeline entraîné + seuil
    ├── app/streamlit_app.py      application
    ├── images/
    └── requirements.txt

## Lancer le projet en local

    git clone https://github.com/XX/churn-prediction.git
    cd churn-prediction
    python -m venv venv
    venv\Scripts\activate        # sous Linux/Mac : source venv/bin/activate
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

## Outils

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib, seaborn.

## Suite possible

- Exposer le modèle via une API (FastAPI) et le conteneuriser avec Docker.
- Ajouter un suivi de dérive des données (Evidently) et un réentraînement planifié.
- Tester un modèle de survie pour prédire *quand* un client va partir, pas seulement *si*.