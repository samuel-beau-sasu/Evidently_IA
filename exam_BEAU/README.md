# Cas d'étude sur le flux de travail de surveillance

## 1. Créer le projet
mkdir drift_monitoring
cd drift_monitoring

### 2. Créer l'environnement
uv venv --python 3.10
source .venv/bin/activate

### 3. Installer depuis requirements.txt
uv pip install -r requirements.txt

## 1. Les réponses aux questions suivantes :

### Après l'étape 4, expliquez ce qui a changé au cours des semaines 1, 2 et 3.

On observe au cours des semaines 1,2 et 3 une dégradation des prévisions par rapport à la référence. 
La ME (Mean Error) devient de plus en plus négative. 
Le modèle sous-estime de plus en plus la valeur réelle.

Sur cette période, le modele a tendance à sous éstimer le nombre de vélos louer. 

Explosion des cas de sous-estimation

Semaine 1
Sous-estimation moyenne : -65 

Semaine 2 
Sous-estimation moyenne : -62 

Semaine 3
Sous-estimation moyenne : -110 

Le modele commence à perdre en précision semaine 1 et semaine 2 puis devient fortement biaisé semaine 3. 

### Après l'étape 5, expliquez ce qui semble être la cause première de la dérive (uniquement à l'aide de données).

On détecte un drift extrêmement fort pour la semaine 3 sur 'cnt' (Score K-S = 0.0). Cela indique que La distribution réelle des valeurs 'cnt' a complètement changé en semaine 3.

Il n'y a par contre pas de drift sur 'prediction'. Le modèle continue à sortir exactement les mêmes prédictions qu’avant (distribution des predictions stable)

Ce n'est pas le modele en lui meme qui a un probleme mais c'est changement de la cible réelle 'cnt'  qui semble être la cause première de la dérive. 

### Après l'étape 6, expliquez quelle stratégie appliquer.

On a un Data Drift massif sur les variables d’entrée ce qui donne donc un Target Drift très fort sur 'cnt'. 

8 colonnes / 12 ont dépassé leur seuil de drift. 

Le modèle est stable dans ses sorties, il continue à appliquer les mêmes règles qu’avant alors que les données ont changées (la météo). 

Il faut faire une mise à jour des données : Réentraîner le modèle en incluant les données récentes (celles de février) pour qu'il apprenne les nouvelles tendances.

## 2. La commande unique pour exécuter votre script.

uv run analyse_data_bike_v4.py

## 3. OPTIONNEL : Informations supplémentaires qu'il vous sera utile de partager.