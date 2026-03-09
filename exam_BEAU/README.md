# Cas d'étude sur le flux de travail de surveillance

## 1. Créer le projet
mkdir drift_monitoring
cd drift_monitoring

## 2. Créer l'environnement
uv venv --python 3.10
source .venv/bin/activate

## 3. Installer depuis requirements.txt
uv pip install -r requirements.txt

## 4. On crée les reports
uv run analyse_data_bike_v4.py

# 1. Les réponses aux questions suivantes :

## Après l'étape 4, expliquez ce qui a changé au cours des semaines 1, 2 et 3.

w1 : degradation des prevision par sous estimation des besoins - fort
w2 : degradation des prevision par sous estimation des besoins - moyen
w3 : degradation des prevision par sous estimation des besoins - fort

## Après l'étape 5, expliquez ce qui semble être la cause première de la dérive (uniquement à l'aide de données).

1 hr
2 temp
3 atemp
4 hum

## Après l'étape 6, expliquez quelle stratégie appliquer.
variables avec drift
2 temp
3 atemp
4 hum

Mise à jour des données : Réentraîner le modèle en incluant les données récentes (celles de février) pour qu'il apprenne les nouvelles tendances.

# 2. La commande unique pour exécuter votre script.

# 3. OPTIONNEL : Informations supplémentaires qu'il vous sera utile de partager.