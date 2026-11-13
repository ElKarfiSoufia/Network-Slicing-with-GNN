# Discours de soutenance (30 diapositives, durée totale ≈ 23 min 35)

Rythme visé : ≈ 140 mots par minute. Le texte est aussi dans les notes de l'orateur de chaque diapositive.

Répartition suggérée entre les quatre orateurs (à adapter) : diapositives 1 à 8 (introduction et état de l'art), 9 à 17 (données et méthode), 18 à 23 (GNN et comparaison), 24 à 30 (SLA, ablation, décalage, conclusion).

## 1. Titre  ·  ≈ 30 s  ·  cumul 0 min 30

Bonjour à toutes et à tous. Nous allons vous présenter notre projet SMARTICT : l'allocation dynamique des ressources pour le network slicing en B5G, à l'aide de l'intelligence artificielle. Concrètement, nous avons cherché à prédire la qualité de service des slices avec un réseau de neurones sur graphes, et à vérifier si cette prédiction reste fiable quand le réseau change brusquement. Le projet a été réalisé par notre groupe de quatre, sous l'encadrement de M. El Ghazi.

## 2. Une question, une réponse courte  ·  ≈ 60 s  ·  cumul 1 min 30

Le guide nous posait une question centrale : peut-on apprendre, à partir de données, une prédiction utile pour gérer les ressources des slices, et cette prédiction reste-t-elle fiable lors de changements brusques du réseau ? Nous vous donnons la réponse tout de suite, en trois points, puis nous la justifierons. Premièrement, oui pour la prédiction : notre GNN estime le délai de chaque flux avec une erreur moyenne de 0,175 décade, soit un facteur 1,5 environ, alors qu'une référence naïve se trompe d'un facteur 7. Deuxièmement, la fiabilité n'est que partielle : pendant les changements brusques, l'erreur de tous les modèles augmente. Mais celle du GNN n'est multipliée que par 1,55, contre 2,4 pour le meilleur modèle classique. Troisièmement, le graphe apporte un gain réel et mesuré : 37 % d'erreur en moins que le meilleur modèle sans graphe, et 59 % de moins pendant les événements.

## 3. Démarche  ·  ≈ 45 s  ·  cumul 2 min 15

Pour arriver à ces conclusions, nous avons suivi les sept livrables du guide. D'abord l'exploration du dataset, puis la formulation du problème comme un jumeau numérique de la qualité de service. Ensuite des modèles de référence, une définition rigoureuse des événements, la construction du graphe et du GNN, et enfin une comparaison complète : globale, pendant les événements, par ablation et sous décalage de distribution. Tout est reproductible : un notebook exécuté, un code commenté avec son guide, le rapport et cette présentation. Commençons par le contexte scientifique.

## 4. Section — État de l'art  ·  ≈ 10 s  ·  cumul 2 min 25

Première partie : l'état de l'art, pour poser les notions de network slicing et de réseaux de neurones sur graphes.

## 5. Le network slicing  ·  ≈ 75 s  ·  cumul 3 min 40

La 5G doit servir trois familles d'usages, définies par l'Union internationale des télécommunications, aux exigences opposées. L'eMBB veut du débit : jusqu'à 20 gigabits par seconde en crête. L'URLLC veut une latence d'une milliseconde et une fiabilité de 99,999 %. Le mMTC veut connecter jusqu'à un million d'objets par kilomètre carré. Dimensionner un réseau unique pour toutes ces contraintes coûterait trop cher. Le network slicing répond à ce problème : on découpe une même infrastructure physique en réseaux logiques isolés, les slices, chacune adaptée à un usage. Le 3GPP l'a normalisé dans la spécification TS 23.501 : chaque slice porte un type de service, et les valeurs 1, 2 et 3 correspondent exactement à eMBB, URLLC et MIoT, les trois slices de notre dataset. Techniquement, le slicing repose sur la virtualisation des fonctions réseau et sur les réseaux définis par logiciel. Mais le point central pour nous est en bas de la figure : toutes les slices partagent les mêmes antennes et les mêmes liens. Il faut donc décider quelle part des ressources chacune reçoit.

## 6. Allouer les ressources : WFQ et saturation  ·  ≈ 75 s  ·  cumul 4 min 55

Comment se fait ce partage en pratique ? Sur chaque port de sortie, un ordonnanceur à plusieurs files. Le modèle de référence est le GPS de Parekh et Gallager, dont WFQ est la version par paquets : chaque file reçoit un poids et se voit garantir une part de la capacité proportionnelle à ce poids. La capacité non utilisée par une file vide est redistribuée aux autres. Dans notre dataset, chaque antenne a une file par slice, et le poids de la file, c'est la réservation de la slice : c'est le levier de l'opérateur. Pourquoi est-ce difficile à prédire ? La théorie des files d'attente donne la réponse, illustrée par la courbe de droite. Dans une file M/M/1, le délai vaut un sur mu moins lambda : il explose quand la charge approche 1. Au-delà, la file est pleine et des paquets sont perdus. Une petite variation de réservation peut donc faire passer une slice de quelques microsecondes à plusieurs millisecondes. Et dans un réseau, ces files sont couplées entre elles, ce qui rend les formules analytiques insuffisantes.

## 7. Les réseaux de neurones sur graphes  ·  ≈ 75 s  ·  cumul 6 min 10

Pour modéliser ces couplages, nous utilisons des réseaux de neurones sur graphes. Un graphe n'a ni taille fixe ni ordre naturel de ses nœuds : un modèle classique, qui attend un vecteur, n'est pas adapté. Le principe commun des GNN, formalisé par Gilmer et ses coauteurs, est le passage de messages. À chaque itération, chaque nœud reçoit des messages de ses voisins, les agrège avec une opération insensible à l'ordre, puis met à jour son état. Après T itérations, un nœud connaît son voisinage à T sauts : le modèle apprend des interactions. Les principales familles diffèrent par leur agrégation. Le GCN fait une moyenne normalisée. GraphSAGE agrège un ensemble de voisins et s'applique à des graphes nouveaux. GAT pondère les voisins par une attention. GIN utilise une somme, plus expressive. Ces modèles ont des limites connues, comme le sur-lissage quand on empile trop de couches, ou la difficulté à généraliser à des graphes plus grands, que nous testerons.

## 8. Les GNN pour les réseaux et notre positionnement  ·  ≈ 60 s  ·  cumul 7 min 10

Pour les réseaux de communication, le travail de référence est RouteNet. Son idée clé : ne pas utiliser le graphe physique des routeurs, mais un graphe qui relie chaque flux aux liens qu'il traverse. Un réseau récurrent parcourt le chemin de chaque flux dans l'ordre, et les liens agrègent les flux qui les traversent. RouteNet-Fermi ajoute des nœuds de files pour modéliser des ordonnancements comme WFQ. Ces travaux visent surtout la précision moyenne. Deux questions restent peu étudiées : le rôle des réservations de slices comme variable de décision, et la fiabilité quand le réseau sort brusquement de son régime habituel. C'est exactement là que se place notre travail : un graphe qui représente explicitement les files WFQ par slice, une évaluation spécifique pendant des événements, et une comparaison avec des références solides.

## 9. Section — Données  ·  ≈ 10 s  ·  cumul 7 min 20

Deuxième partie : les données, que nous avons dû explorer en profondeur avant toute modélisation.

## 10. Le dataset  ·  ≈ 60 s  ·  cumul 8 min 20

Nous utilisons le dataset publié par Farreras et ses coauteurs en 2024, généré avec un simulateur paquet basé sur OMNeT++. Il compte 7 127 simulations stables, 1,58 million de flux et 162 topologies réelles issues de l'Internet Topology Zoo, de 16 à 765 nœuds, avec les trois types de slices. Chaque simulation suit le même schéma. Des terminaux envoient leur trafic vers une antenne, qui ordonnance les slices avec WFQ, une file par slice. Le trafic traverse ensuite le réseau de transport, en FIFO avec des files de 32 paquets, jusqu'à la destination de sa slice. La réservation de chaque slice est fixée par un paramètre delta entre 0 et 1, qui place le débit réservé entre 90 et 120 % du débit nominal. Nous avons écrit notre propre parseur, car celui fourni pouvait exécuter du code.

## 11. Quatre pièges dans les données  ·  ≈ 75 s  ·  cumul 9 min 35

L'exploration a révélé quatre pièges, et c'est sans doute l'apport le plus important de cette phase. Le premier : 42 % des chemins indiqués dans les fichiers de slices ne correspondent pas au routage réellement simulé. Nous l'avons détecté en comparant la charge que nous calculions avec l'utilisation mesurée par le simulateur : un quart des files semblaient porter du trafic sans débit mesuré. Après reconstruction des chemins depuis le fichier de routage, la corrélation est de 0,98. Sans cette vérification, tous nos modèles auraient appris sur des entrées fausses, sans aucun message d'erreur. Deuxième piège, conséquence du premier : 8 % des flux passent par un port où leur slice n'a aucune réservation. Troisième : le débit réel vaut 1,25 fois le nominal, à cause d'un champ incohérent ; 91 % des flux dépassent donc leur réservation. Quatrième : 93 % des flux ont un jumeau aux entrées identiques. Nous les avons regroupés en classes pondérées : les métriques restent exactes et le calcul est six fois plus léger.

## 12. La QoS change de régime  ·  ≈ 50 s  ·  cumul 10 min 25

Pour relier entrées et qualité de service, nous définissons pour chaque flux la charge de son goulot, b : le rapport entre le débit offert et la capacité de la ressource la plus chargée qu'il traverse. Les courbes montrent exactement ce que prévoyait la théorie des files d'attente : un coude net autour de b égal à 1. En dessous, aucune perte et un délai de quelques dizaines de microsecondes. Au-dessus, les pertes montent et le délai gagne deux décades. Le délai couvre au total huit ordres de grandeur, d'où notre choix de le prédire en logarithme. Et aucune variable isolée ne suffit : la QoS d'un flux dépend des autres flux.

## 13. Section — Méthode  ·  ≈ 10 s  ·  cumul 10 min 35

Troisième partie : la méthode, c'est-à-dire ce que le modèle prédit et comment nous l'évaluons sans tricher.

## 14. La cible : un jumeau numérique  ·  ≈ 60 s  ·  cumul 11 min 35

Notre cible : pour chaque flux, prédire le logarithme de son délai moyen et son taux de pertes, à partir de la configuration, c'est-à-dire la topologie, le routage, les demandes et les poids WFQ. Pourquoi est-ce un problème de gestion des slices ? Parce que l'opérateur doit savoir, avant de déployer une allocation, si elle respecte les contrats de service. Or une simulation paquet prend de quelques minutes à quinze heures, alors que notre modèle répond en millisecondes. C'est un jumeau numérique, qui permet de tester des allocations candidates, de contrôler l'admission d'une nouvelle slice ou de détecter une violation de SLA. Nous n'avons pas cherché à prédire directement l'allocation optimale : le dataset ne contient que des allocations aléatoires, aucune n'est étiquetée comme optimale. Enfin, le logarithme pénalise de façon symétrique la sur- et la sous-estimation.

## 15. Un protocole sans fuite  ·  ≈ 50 s  ·  cumul 12 min 25

Le protocole repose sur quatre règles. Un : chaque topologie est simulée environ 44 fois, donc nous découpons par topologies entières. Le test ne contient que des réseaux jamais vus, comme le recommande la littérature sur les données groupées. Deux : seules les entrées servent de features. L'utilisation mesurée des liens serait très prédictive, mais elle n'est connue qu'après la simulation : ce serait une fuite d'information. Trois : les métriques, MAE et MSE, sont pondérées par flux, et le test n'est lu qu'une seule fois. Quatre : un second découpage entraîne sur les petits réseaux et teste sur les plus grands, pour mesurer le décalage de distribution.

## 16. Les événements  ·  ≈ 60 s  ·  cumul 13 min 25

Le guide demandait de définir des événements, c'est-à-dire des changements brusques. Difficulté : le dataset n'a pas d'axe temporel, chaque simulation est un régime stationnaire. Nous utilisons donc le fait que chaque topologie est simulée des dizaines de fois : ces simulations définissent son régime habituel. Un événement est un flux dont la charge du goulot s'écarte fortement de ce régime. Comme b couvre plus de dix ordres de grandeur, nous mesurons l'écart en logarithme, avec la médiane et l'écart absolu médian, qui résistent aux valeurs extrêmes. La règle est celle de Hampel : un écart d'au moins trois. Nous distinguons les surcharges, 10,6 % des flux, avec des pertes médianes de 85 %, et les délestages, 2,9 % des flux, sans pertes. Le masque ne sert qu'à l'évaluation, jamais à l'entraînement.

## 17. Vérification des événements  ·  ≈ 40 s  ·  cumul 14 min 05

Nous avons vérifié visuellement ces événements sur une topologie simulée 48 fois. À gauche, chaque point est une classe de flux : les points orange et violets sortent bien de la bande habituelle. Au centre et à droite, la qualité de service mesurée confirme le changement de régime : les flux en surcharge perdent beaucoup plus de paquets et attendent plus longtemps, les flux en délestage n'ont presque aucune perte. Le masque, calculé sans regarder la qualité de service, isole donc de vrais changements de régime.

## 18. Section — GNN  ·  ≈ 10 s  ·  cumul 14 min 15

Quatrième partie : la représentation en graphe et notre modèle, SliceGNN.

## 19. Le graphe hétérogène  ·  ≈ 55 s  ·  cumul 15 min 10

Chaque simulation devient un graphe hétérogène à trois types de nœuds. Les flux, avec leur slice, leur débit et leur paramètre delta. Les files WFQ, avec leur poids et leur charge : c'est là que s'applique la réservation. Et les liens, avec leur capacité et leur charge. Les arêtes relient chaque flux aux liens de son chemin, dans l'ordre, et à sa file WFQ ; chaque file est reliée au port de son antenne. Pourquoi pas un simple vecteur de features ? Parce qu'un vecteur résume le chemin par quelques statistiques, et ignore qui partage chaque lien et combien de trafic arrive vraiment après les pertes en amont. C'est un problème de point fixe, que le passage de messages approche par itérations.

## 20. L'architecture SliceGNN  ·  ≈ 70 s  ·  cumul 16 min 20

Le modèle fonctionne en trois temps. D'abord, trois petits réseaux encodent les attributs de chaque type de nœud en vecteurs de dimension 64. Ensuite, quatre itérations de passage de messages. Dans chacune, chaque flux parcourt son chemin avec un GRU, un réseau récurrent : à chaque saut, il lit l'état du lien et, au premier saut, l'état de sa file WFQ. Puis chaque lien fait la moyenne des flux qui le traversent, pondérée par leur nombre, et chaque file fait de même. Enfin, une lecture transforme l'état du flux en délai et en pertes. Pourquoi ce choix plutôt qu'un GCN ou un GAT ? Le graphe est hétérogène. L'ordre du chemin compte, car un flux qui perd des paquets en amont charge moins l'aval. Et les dépendances sont circulaires. Le surapprentissage est contrôlé par une validation sur des topologies inédites et par un arrêt précoce.

## 21. Section — Résultats  ·  ≈ 10 s  ·  cumul 16 min 30

Cinquième partie : les résultats, tous mesurés sur des topologies jamais vues pendant l'entraînement.

## 22. Comparaison des modèles  ·  ≈ 70 s  ·  cumul 17 min 40

Voici la comparaison principale : en bleu l'erreur globale, en orange l'erreur pendant les événements. Sur le délai, le modèle naïf se trompe de 0,83 décade et Ridge de 0,52 : une relation linéaire ne capture pas le coude de saturation. Le Gradient Boosting, notre meilleur modèle sans graphe, atteint 0,28, et le MLP 0,34. Le GNN atteint 0,175, soit 37 % d'erreur en moins que le Gradient Boosting. Le classement est le même pour les pertes : 0,045 contre 0,067. Mais la différence la plus nette est en orange : pendant les événements, le Gradient Boosting monte à 0,66, alors que le GNN reste à 0,27. Nous avons répété l'entraînement du GNN avec trois initialisations : l'écart-type n'est que de 0,004, vingt-cinq fois plus petit que l'écart avec le Gradient Boosting. Le gain n'est donc pas un effet du hasard.

## 23. Erreur selon l'écart au régime habituel  ·  ≈ 45 s  ·  cumul 18 min 25

Cette figure montre l'erreur en fonction de l'écart au régime habituel. Toutes les courbes montent : plus la situation est inhabituelle, plus la prédiction est difficile, ce qui confirme notre deuxième hypothèse. Mais la pente du GNN, en bleu, est bien plus faible. À droite, dans la zone de surcharge, il se trompe environ deux fois moins que le Gradient Boosting. À gauche, en délestage, son erreur reste au niveau du régime normal, alors que celle des modèles classiques remonte.

## 24. Détection des violations de SLA  ·  ≈ 45 s  ·  cumul 19 min 10

Un opérateur ne raisonne pas en MAE : il veut savoir si une slice va violer son contrat. Nous avons donc transformé le délai prédit en décision, délai au-dessus du SLA ou non, pour eMBB et URLLC. Le GNN atteint une exactitude équilibrée de 0,926, contre 0,907 pour le Gradient Boosting, et 0,858 contre 0,836 pendant les événements. L'écart est plus faible que sur la MAE : loin du seuil, tous les bons modèles décident correctement. Le GNN se distingue près du seuil, et sur l'ampleur des dégradations, utile pour savoir de combien augmenter une réservation.

## 25. Ablation  ·  ≈ 70 s  ·  cumul 20 min 20

L'ablation explique d'où vient le gain. Premièrement, si l'on supprime les interactions entre flux, avec T égal à zéro, l'erreur du GNN passe de 0,175 à 0,24, et de 0,27 à 0,50 pendant les événements : environ 60 % du gain vient de ces interactions. Deuxièmement, sans les nœuds de files WFQ, l'erreur monte à 0,43, pire que le Gradient Boosting : représenter explicitement la réservation par slice est indispensable. Troisièmement, sans aucune information d'allocation, le GNN monte à 0,40 et le Gradient Boosting à 0,45 : le levier de l'opérateur est bien l'information clé. Dernier résultat intéressant : notre feature physique, le délai de file pleine, aide beaucoup les modèles simples, mais presque pas le GNN, qui retrouve seul cette relation.

## 26. Décalage de distribution  ·  ≈ 40 s  ·  cumul 21 min 00

Enfin, le décalage de distribution. Entraînés sur les petits et moyens réseaux puis testés sur les plus grands, au-delà de 212 nœuds, tous les modèles se dégradent un peu. Le GNN se dégrade de 16 %, plus que le Gradient Boosting, mais il reste nettement le meilleur : 0,20 contre 0,29, et pendant les événements de ces grands réseaux, 0,28 contre 0,57.

## 27. Section — Conclusion  ·  ≈ 10 s  ·  cumul 21 min 10

Pour conclure, revenons à la question finale du guide.

## 28. Quand le GNN apporte de la valeur  ·  ≈ 70 s  ·  cumul 22 min 20

Dans quelles conditions un GNN apporte-t-il une valeur ajoutée pour la gestion dynamique des slices ? Nos expériences en identifient quatre. Un : quand la qualité de service dépend de ressources partagées et d'effets en cascade, ce que montre l'ablation T égal zéro. Deux : quand le graphe porte explicitement le levier de gestion, ici les files WFQ par slice. Trois : quand le réseau sort de son régime habituel, c'est-à-dire dans le cas d'usage même de la gestion dynamique. Quatre : quand la topologie change ou grandit. À l'inverse, le GNN n'est pas nécessaire pour une décision grossière, loin des seuils, ni quand le coût compte : il demande seize minutes de GPU contre deux minutes pour le Gradient Boosting. Et il perd tout son intérêt si son graphe ne représente pas l'allocation. Le meilleur score n'est pas automatiquement la meilleure solution : tout dépend de l'usage.

## 29. Limites et suites  ·  ≈ 50 s  ·  cumul 23 min 10

Notre travail a des limites claires. Les données viennent d'un seul simulateur, en régime stationnaire, sans la partie radio. Les événements sont reconstruits comme des écarts au régime habituel, et non observés dans le temps. Nous n'avons estimé la variabilité que sur l'initialisation du GNN, pas sur le découpage. Enfin, notre modèle évalue une allocation, il ne la choisit pas encore. La suite naturelle est donc de boucler ce jumeau numérique avec un optimiseur des réservations, puis de le valider sur des données temporelles et des mesures réelles.

## 30. Merci  ·  ≈ 25 s  ·  cumul 23 min 35

En résumé : un GNN qui voit les files WFQ et les interactions entre flux prédit la qualité de service des slices avec 37 % d'erreur en moins, et 59 % de moins pendant les changements brusques. Merci pour votre attention ; nous sommes prêts à répondre à vos questions.
