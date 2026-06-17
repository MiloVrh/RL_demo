
#Egyszerű Python snippet mély és biztonságos megerősítéses tanulásra
#Adathalmaz: Training Dataset.arff (Phishing Websites)
#link az adathalmazhoz: https://archive.ics.uci.edu/dataset/327/phishing+websites

#Ötlet:
#- Minden weboldal egy állapot (state): 30 jellemzőből álló vektor.
#- Az ágens két akció közül választ:
#    0 = phishing oldalnak jelöli
#    1 = legitim oldalnak jelöli
#- Jutalom:
#    +1, ha jól dönt
#    -1, ha rosszul dönt
#- Biztonságos tanulás változat:
#    hamis negatív esetben (phishing oldalt legitimnek mond) nagyobb büntetést kap,
#    mert ez a felhasználó számára veszélyesebb hiba.
#    Ezen kívül egy egyszerű safety shield felülírhatja a döntést, ha sok jellemző gyanús.

import random
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim


DATA_PATH = "Training Dataset.arff"
#seed használata rekonstruálhatóságért
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 1. Adatok betöltése
def load_arff(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # bytes -> egész érték átalakítás, mert az .arff file beolvasás után b:-1 formában jönnek elő az értékek
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x.decode("utf-8")) if isinstance(x, bytes) else int(x))

    X = df.drop("Result", axis=1).values.astype(np.float32)

    # Eredeti címkék: -1 = phishing, 1 = legitim
    # A modell számára így módosítva: 0 = phishing, 1 = legitim
    y_original = df["Result"].values
    y = np.where(y_original == -1, 0, 1).astype(np.int64)

    return X, y, df


# 2. Egyszerű környezet: phishing klasszifikáció mint egylépéses megerősítéses tanulás feladat
class PhishingEnvironment:
    def __init__(self, X, y, safe_mode=False):
        self.X = X
        self.y = y
        self.safe_mode = safe_mode

    def sample(self):
        idx = np.random.randint(0, len(self.X))
        return self.X[idx], self.y[idx]

    def reward(self, true_label, action):
        # action: 0 = phishing, 1 = legitimate
        if action == true_label:
            return 1.0

        if self.safe_mode:
            # Veszélyesebb hiba: phishing oldalt legitimnek jelölünk.
            false_negative = (true_label == 0 and action == 1)
            if false_negative:
                return -5.0

        return -1.0


# 3. Mély Q-netwrok: kis komplexitású neurális háló, dimenzió-szám input függé
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# 4. Tanítás: egyszerű epsilon-greedy DQN inspirált megoldás - gemini segítségével írva
def train_agent(X_train, y_train, safe_mode=False, episodes=5000, lr=0.001):
    env = PhishingEnvironment(X_train, y_train, safe_mode=safe_mode)
    model = QNetwork(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    for ep in range(episodes):
        state, true_label = env.sample()
        state_tensor = torch.tensor(state).unsqueeze(0)

        # Felfedezés vagy kihasználás
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = int(torch.argmax(q_values).item())

        r = env.reward(true_label, action)

        q_values = model(state_tensor)
        target = q_values.clone().detach()
        target[0, action] = r

        loss = loss_fn(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return model


# 5. Safety shield: nagyon egyszerű biztonsági szabály
def safety_shield(x, proposed_action, threshold=12):
    #Ha sok feature értéke -1, akkor az oldal erősen gyanús.
    #Ilyenkor nem engedjük, hogy az ágens legitim oldalnak jelölje.
    #Ez egy egyszerű biztonságos megerősítéses tanulás jellegű védőréteg.
    suspicious_count = np.sum(x == -1)

    if suspicious_count >= threshold and proposed_action == 1:
        return 0  # felülírás: phishing

    return proposed_action


# 6. Kiértékelés
def predict(model, X, use_shield=False):
    model.eval()
    preds = []

    with torch.no_grad():
        for x in X:
            x_tensor = torch.tensor(x).unsqueeze(0)
            q_values = model(x_tensor)
            action = int(torch.argmax(q_values).item())

            if use_shield:
                action = safety_shield(x, action)

            preds.append(action)

    return np.array(preds)


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm felépítés:
    # [[phishing helyesen, phishing tévesen legitimnek],
    #  [legitim tévesen phishingnek, legitim helyesen]]
    false_negatives = cm[0, 1]

    print("\n" + "=" * 60)
    print(name)
    print("Pontosság:", round(acc, 4))
    print("Konfúziós mátrix [0=phishing, 1=legitim]:")
    print(cm)
    print("Veszélyes hibák száma, amikor phishing oldalt legitimnek mondott:", false_negatives)
    print("\nRészletes riport:")
    print(classification_report(y_true, y_pred, target_names=["phishing", "legitim"]))


# 7. Main
if __name__ == "__main__":
    X, y, df = load_arff(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # Skálázás: neurális hálóknál stabilabb tanítást ad
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print("Adathalmaz mérete:", X.shape)
    print("Tanító minták:", X_train.shape[0])
    print("Teszt minták:", X_test.shape[0])
    print("Célváltozó: 0 = phishing, 1 = legitim")

    # 1) Deep RL: sima jutalmazással
    deep_rl_model = train_agent(X_train, y_train, safe_mode=False, episodes=5000)
    deep_preds = predict(deep_rl_model, X_test, use_shield=False)
    evaluate("1. Mély megerősítéses tanulás - alap DQN-szerű ágens", y_test, deep_preds)

    # 2) Safe Deep RL: veszélyes hibák nagyobb büntetése + safety shield
    safe_rl_model = train_agent(X_train, y_train, safe_mode=True, episodes=5000)
    safe_preds = predict(safe_rl_model, X_test, use_shield=True)
    evaluate("2. Biztonságos mély megerősítéses tanulás - büntetés + safety shield", y_test, safe_preds)
