<center><h2>Détection de thème sur une question environnement du grand débat</h2></center>

Fork du projet TALAGRAND disponible ici : https://github.com/Quantmetry/grand-debat. 
Dans mon implémentation, j'ai utilisé le même algo de LDA de scikit-learn, modifié la préparation de données, et déterminé le nombre de thèmes grâce un score de cohérence.

Mon but est d'essayer des techniques d'extraction de thème (topic detection) sur des questions du grand débat.
Pour cet essai j'ai pris : "de quelle manière votre vie quotidienne est-elle touchée par le changement climatique ?").  
Je l'ai comparé ensuite à la synthèse "officielle" réalisée par OpinionWay : 
https://granddebat.fr/media/default/0001/01/b88758e8caa2733bec607a74b3b5371cc0a3b420.pdf

### installation

> pip install -r requirements.txt
> python -m spacy download fr_core_news_md