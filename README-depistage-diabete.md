# Dépistage du diabète

Estimer le risque de diabète de type 2 à partir de mesures simples comme la glycémie, l'IMC et l'âge.

Les données viennent du jeu Pima Indians Diabetes du NIDDK, soit 768 patientes.

J'ai comparé une régression logistique, une forêt aléatoire et XGBoost. J'ai gardé la régression logistique, plus simple à expliquer.

Sur 154 patientes de test, le modèle détecte 87 % des malades, contre 46 % pour une règle simple sur la glycémie.

Les patientes viennent toutes de la même population, donc le modèle ne marche pas forcément ailleurs. Ce n'est pas un outil médical.

Pour le lancer, installer les dépendances avec `pip install -r requirements.txt` puis ouvrir `notebooks/depistage_diabete.ipynb`.

Issa Aziz Karambiri
