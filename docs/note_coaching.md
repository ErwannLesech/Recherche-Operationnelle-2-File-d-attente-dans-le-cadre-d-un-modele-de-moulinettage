# Notes synthétiques du coaching de mi-projet

faire heatmap pour le comportement des hyperparamètres
objectif minimiser l'esperence / moyenne
min [ alpha * E[T](K,mu, tb) + beta * Cout(K, prix_serveur, mu, ...)] avec alpha + beta = 1 pour pondérer l'intéret du temps de service/attente par rapport au cout. 
Discuter convexité des optimisations des paramètres

peux rajouter de l'intéret de faire du M/D/¹ pour la fin de sortie pour comparer avec un traitement déterministe
Relancer plusieurs fois les simulations pour diminuer l'aléatoire et calculer les moyennes & variances/écart type
Il faut tracer les différentes simulations, et modéliser l'esperance dans le temps et l'ecart type
On peut seconder ces graphiques via un benchmark/heatmap en fct des lambda et mu 
personna : Etudiant (sup / ing), Admin

Parler de notre approche pour le coût (serveurs déjà possédés par l'école...)

il apprécierai que l'on demande à l'admin des données terrains sur le nb  de tag/etudiant/unité de temps
--> On pourra justifier ces données via ce site qui donne ces infos
https://grafa?na.ops.k8s.cri.epita.fr/k8s/clusters//api/v1/namespaces/cattle-monitoring-system/services/http:rancher-monitoring-grafana:80/proxy/d/128b8190f34adb4e356df4a324acef05/maas-overvieworgId=1&refresh=30s

faudra étudier le flux qui passe de la première file à la seconde, comment assurer la stabilité