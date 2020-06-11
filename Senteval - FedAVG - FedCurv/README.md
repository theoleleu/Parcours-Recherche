# Senteval

Senteval  est utilisé sous Unix et utilise le moteur Cuda de la carte Graphique pour faire l'apprentissage. Il est nécessaire d'installer tout les toolkits ici par contrainte de taille j'ai mis uniquement les dossiers senteval et examples de Senteval (sans les données fasttext ou Glove qui doivent être placés dans le dossier examples pour fonctionner)
J'ai utilisé la méthode bow du dossier examples.
J'ai modifié les programmes classifier.py et validation.py du sous dossier de senteval nommé tools pour permettre un apprentissage FedAVG sur les trois tâches MR, CR et MPQA.
