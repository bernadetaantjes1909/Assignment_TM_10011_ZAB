#%%
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Liver_radiomicFeatures.csv'), index_col=0)

    return data


df = load_data()

if df is not None:
    print("Data succesvol geladen!")
    print(f"Aantal rijen en kolommen: {df.shape}")
    #print(df.head())  # Toon de eerste 5 rijen
else:
    print("Dataframe is leeg of laden is mislukt.")
#%%
# 1. Laad je dataset (vervang 'jouw_data.csv' door je eigen bestand)
# Stel dat je kolom met de diagnose 'label' of 'diagnose' heet
df = pd.read_csv('worcliver\Liver_radiomicFeatures.csv')

# Stel we hebben kolommen 'X' (kenmerken) en 'y' (de uitkomst: maligne/benigne)
X = df.drop('label', axis=1) # Alle kolommen behalve de diagnose
y = df['label']              # Alleen de kolom met maligne/benigne

# 2. Opsplitsen in 80% training en 20% test
# 'stratify=y' is essentieel om de verhouding gelijk te houden!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y #door 42 is een standaard random dan hou je de zelfde manier 
)

# 3. Functie om de ratio's en aantallen weer te geven
def toon_verdeling(label, naam): 
    aantallen = label.value_counts()
    ratios = label.value_counts(normalize=True) * 100
    print(f"--- Verdeling in {naam} ---")
    for categorie in aantallen.index:
        print(f"{categorie}: {aantallen[categorie]} patiënten ({ratios[categorie]:.2f}%)")
    print()

# Toon de resultaten
toon_verdeling(y, "Originele Dataset")
toon_verdeling(y_train, "Training Set (80%)")
toon_verdeling(y_test, "Test Set (20%)")
# %%
# 1. Verwijder spaties uit kolomnamen voor de zekerheid
df.columns = df.columns.str.strip()

# 2. Toon de kolommen zodat je zeker weet hoe de diagnose-kolom heet
#print("Jouw kolommen zijn:", df.columns.tolist())

# 3. Pak alleen de kolommen met getallen voor X
X = df.select_dtypes(include=['number'])

# 4. Als je diagnose-kolom (bijv. 'labels') toevallig ook een getal is (0 of 1), 
# dan moet je die handmatig uit X halen, anders leert het model de uitkomst uit de data!
if 'label' in X.columns:
    X = X.drop('label', axis=1)

y = df['label'] # Zorg dat dit de juiste naam is van je maligne/benigne kolom
# %%
