# Chatbot with NLP and Deep Learning

Ce projet consiste à construire un chatbot simple en utilisant des techniques de traitement du langage naturel (NLP) et un modèle de deep learning entraîné avec **TensorFlow** et **Keras**. Le chatbot est capable de classer les entrées de l'utilisateur en différentes intentions et de fournir des réponses basées sur ces intentions.

#### Source du projet : 
[ProjectPro: Python Chatbot Project - Learn to Build a Chatbot from Scratch](https://www.projectpro.io/article/python-chatbot-project-learn-to-build-a-chatbot-from-scratch/429#mcetoc_1fofi43626)

---

## Concepts Clés Appris
- **Lemmatization**
- **Bag-of-Words (BoW)**
- **Tokenization**
- **Architecture du Réseau Neuronal**
- **Optimiseur (Adam)**
- **Entropie Croisée Catégorique**

### 1. Lemmatization
**Definition**: Lemmatization is the process of reducing a word to its base or dictionary form (lemma). Unlike stemming, lemmatization considers the context of the word to ensure that the base form is meaningful.

**Why it's important**: Lemmatization helps in reducing inflected words to their base forms, which reduces the vocabulary size and improves the consistency of text data used in training. This is crucial for making the input more uniform before feeding it into the model.

**Example**:
- "running" becomes "run"
- "better" becomes "good"

In this project, the `WordNetLemmatizer` from the `nltk` library is used to lemmatize words.

### 2. Bag-of-Words (BoW)
**Definition**: Bag-of-Words is a text representation technique that converts text data into a set of features based on word occurrence. It creates a vector where each dimension corresponds to a word in the vocabulary, and the values indicate the presence or absence (or frequency) of words in a given sentence.

**Why it's important**: BoW allows us to convert textual data into numerical form, which is required for training a machine learning model. It provides a simple and effective way to represent text while preserving the presence of important words in different patterns.

**Example**:
For the vocabulary `["hello", "how", "are", "you"]`, the sentence "hello you" would be represented as `[1, 0, 0, 1]`.

### 3. Tokenization
**Definition**: Tokenization is the process of breaking down a text into smaller units, called tokens (typically words or sentences).

**Why it's important**: Tokenization helps in analyzing the structure of the text by breaking it down into manageable parts, which is the first step in converting text data into a form that can be processed by a model.

**Example**:
The sentence "Hello, how are you?" can be tokenized into `["Hello", ",", "how", "are", "you", "?"]`.

### 4. Neural Network Architecture
**Definition**: A neural network is a sequence of layers (nodes) that learn to map input features to desired outputs through a series of computations. In this project, a simple feedforward neural network with `Dense` and `Dropout` layers is used.

**Model Architecture**:
- **Input Layer**: Takes in the bag-of-words representation of a sentence.
- **Hidden Layers**: Two layers with ReLU activation and dropout for regularization.
- **Output Layer**: Uses a softmax function for multi-class classification.

### 5. Optimizers (Adam)
**Definition**: Optimizers are algorithms that adjust the weights of a neural network to minimize the loss during training. `Adam` (Adaptive Moment Estimation) is an optimizer that combines the advantages of two other optimizers, AdaGrad and RMSProp.

**Why it's important**: Optimizers play a critical role in the convergence speed and overall performance of a model. `Adam` is widely used due to its efficiency and effectiveness.

### 6. Categorical Cross-Entropy
**Definition**: Categorical cross-entropy is a loss function commonly used in classification tasks. It measures the difference between the predicted probability distribution and the true distribution (labels).

**Why it's important**: Using the right loss function ensures that the model is properly penalized for incorrect predictions, guiding the learning process in the right direction.

---

## Structure du Projet

Voici la structure du projet :

```
├── Chatbot
│   ├── Dockerfile            # Dockerfile pour la création de l'image
│   ├── .dockerignore         # Fichiers à ignorer lors de la création de l'image
│   ├── intents.json          # Fichier de données d'intention
│   ├── main.py               # Script Python du chatbot
│   ├── README.md             # Documentation du projet
│   └── requirements.txt      # Liste des dépendances Python
```

---

## Comment Exécuter le Projet

### Option 1 : Lancer avec Docker (Recommandé)

L'utilisation de Docker est recommandée pour garantir que le projet fonctionne de manière cohérente, indépendamment de votre environnement local.

#### Étapes pour exécuter le chatbot avec Docker :

1. **Assurez-vous d'avoir Docker installé** sur votre machine. Vous pouvez télécharger et installer Docker depuis [ici](https://www.docker.com/products/docker-desktop).

2. **Construire l'image Docker** :
   Dans le répertoire du projet, ouvrez un terminal et exécutez la commande suivante pour construire l'image Docker :

   ```bash
   docker build -t chatbot-image .
   ```

   Cela crée une image Docker avec le nom `chatbot-image` en utilisant le `Dockerfile` présent dans le répertoire.

3. **Exécuter le conteneur Docker** :
   Lancez un conteneur basé sur l'image Docker que vous venez de construire :

   ```bash
   docker run -it chatbot-image
   ```

   Le chatbot devrait maintenant être lancé à l'intérieur du conteneur Docker. Vous pourrez interagir avec lui directement dans le terminal.

4. **Vérifiez l'interaction avec le chatbot** :
   Après avoir démarré le conteneur, vous pouvez saisir des questions ou des phrases et le chatbot vous répondra en fonction des intentions définies dans `intents.json`.

#### Dockerfile

Le `Dockerfile` contient les instructions pour créer l'image Docker. Il inclut les étapes suivantes :

- Installation de Python et des dépendances.
- Copie du code et des fichiers nécessaires dans l'image Docker.
- Définition de la commande par défaut pour exécuter le chatbot (`main.py`).

#### .dockerignore

Le fichier `.dockerignore` spécifie les fichiers et répertoires à ignorer lors de la construction de l'image Docker. Cela inclut généralement les fichiers inutiles comme les fichiers temporaires ou les fichiers de configuration spécifiques à votre environnement local.

---

### Option 2 : Lancer sans Docker (Manuellement)

Si vous ne souhaitez pas utiliser Docker, vous pouvez exécuter le projet directement sur votre machine. Voici les étapes :

1. **Installer les dépendances** : Assurez-vous que Python est installé sur votre machine, puis créez un environnement virtuel et installez les dépendances nécessaires :

   ```bash
   python -m venv chatbot-env
   source chatbot-env/bin/activate  # Sur macOS/Linux
   chatbot-env\Scripts\activate  # Sur Windows
   pip install -r requirements.txt
   ```

2. **Télécharger le fichier `intents.json`** : Assurez-vous que le fichier `intents.json` contenant les intentions est bien dans le même répertoire que le script `main.py`.

3. **Exécuter le script** : Lancez le chatbot avec la commande suivante :

   ```bash
   python main.py
   ```

4. **Interagir avec le chatbot** : Après avoir lancé le script, vous pouvez saisir des messages et obtenir des réponses du chatbot dans le terminal.

---

## Bibliothèques Utilisées

- **nltk** : Pour le prétraitement du texte (tokenisation et lemmatisation).
- **tensorflow** : Pour construire et entraîner le réseau neuronal.
- **json** : Pour analyser et charger les données des intentions.
- **random** : Pour fournir des réponses variées à chaque intention.

---

## Améliorations Futures

Voici quelques pistes d'amélioration pour rendre ce projet plus puissant :

- **Utiliser des techniques NLP avancées**, comme les **Word Embeddings** (par exemple, **Word2Vec** ou **GloVe**) pour mieux comprendre le contexte des mots et améliorer la précision du chatbot.
- **Implémenter un modèle plus complexe**, comme un **LSTM** (Long Short-Term Memory) ou un **Transformer**, pour une meilleure gestion des dépendances à long terme dans le texte.
- **Déployer le chatbot** via un framework comme **Flask** ou **FastAPI** pour créer une interface web.
- **Améliorer la gestion des intentions** en permettant au chatbot d'apprendre de nouvelles intentions au fil du temps.

---

## Notes Importantes

- Si vous utilisez Docker, vous n'avez pas à vous soucier des problèmes de dépendances ou de configuration de l'environnement Python. Docker s'occupe de tout.
- N'oubliez pas de vérifier le fichier `intents.json` pour voir quelles intentions sont actuellement prises en charge par le chatbot et de les personnaliser si nécessaire.


