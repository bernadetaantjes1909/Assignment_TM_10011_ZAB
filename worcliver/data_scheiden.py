#%%
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

#z score berekenen
from scipy import stats
import numpy as np

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Liver_radiomicFeatures.csv'), index_col=0)

    return data


df = load_data()
df.columns = df.columns.str.strip()



if df is not None:
    print("Data succesvol geladen!")
    print(f"Aantal rijen en kolommen: {df.shape}")
    #print(df.head())  # Toon de eerste 5 rijen
else:
    print("Dataframe is leeg of laden is mislukt.")
#%%
# 1. Laad je dataset (vervang 'jouw_data.csv' door je eigen bestand)
# Stel dat je kolom met de diagnose 'label' of 'diagnose' heet
#df = pd.read_csv('worcliver\Liver_radiomicFeatures.csv')

# Stel we hebben kolommen 'X' (kenmerken) en 'y' (de uitkomst: maligne/benigne)
#X = df.drop('label', axis=1) # Alle kolommen behalve de diagnose
y = df['label']              # Alleen de kolom met maligne/benigne

# 2. Opsplitsen in 80% training en 20% test
# 'stratify=y' is essentieel om de verhouding gelijk te houden!
#data_train, X_test, y_train, y_test = train_test_split(
   # X, y, test_size=0.20, random_state=42, stratify=y #door 42 is een standaard random dan hou je de zelfde manier 
def preprocessing_data(data):
    # Let op: Geen hekjes voor de code die moet werken!
    data_train, data_test, classification_train, classification_test = model_selection.train_test_split(
        data.iloc[:,2:], 
        data['label'], 
        test_size=0.20, 
        random_state=42, 
        stratify=data['label']
    )
    # De return is essentieel!
    return data_train, data_test, classification_train, classification_test
    #data_train, data_test, classification_train, classification_test = preprocessing_data(df)
    #return data_train, data_test, classification_train, classification_test


# 3. Functie om de ratio's en aantallen weer te geven
def toon_verdeling(label, naam): 
    aantallen = label.value_counts()
    ratios = label.value_counts(normalize=True) * 100
    print(f"--- Verdeling in {naam} ---")
    for categorie in aantallen.index:
        print(f"{categorie}: {aantallen[categorie]} patiënten ({ratios[categorie]:.2f}%)")
    print()

#alles laden
df = load_data()
df.columns = df.columns.str.strip() # Spaties verwijderen
y = df['label']

#functie uitvoeren 
data_train, data_test, classification_train, classification_test = preprocessing_data(df)

# Toon de resultaten
toon_verdeling(y, "Originele Dataset")
toon_verdeling(classification_train, "Training Set (80%)")
toon_verdeling(classification_test, "Test Set (20%)")

#%%
# De eerste 5 rijen van de trainings-features
print(classification_train.head())
# De eerste 5 labels van de test-set
print(classification_test.head())

#%%



#%% z score berekenen weet niet of het nodig is 

# 1. Bereken de Z-scores voor alle numerieke kolommen in data_train
z_scores = stats.zscore(data_train)
# 2. Maak er een overzichtelijke DataFrame van
z_scores_df = pd.DataFrame(z_scores, columns=data_train.columns, index=data_train.index)
# 3. Bepaal wat een outlier is (waarde > 3 of < -3)
outliers_mask = (np.abs(z_scores_df) > 3)
#print("Z-scores zijn berekend.")

# Tel het aantal outliers per feature (kolom)
outliers_per_feature = outliers_mask.sum().sort_values(ascending=False)
#print("Aantal outliers per feature (top 10):")
#print(outliers_per_feature.head(10))

# Zoek de locaties waar de mask 'True' is
outlier_locations = np.where(outliers_mask)

#print("\nLijst van specifieke outliers:")
for row_idx, col_idx in zip(outlier_locations[0], outlier_locations[1]):
    patient_id = data_train.index[row_idx]
    feature_naam = data_train.columns[col_idx]
    z_waarde = z_scores_df.iloc[row_idx, col_idx]
    
    #print(f"Patiënt {patient_id} heeft een outlier in '{feature_naam}' (Z-score: {z_waarde:.2f})")

# Behoud alleen de rijen die GEEN enkele outlier hebben (over alle kolommen)
data_train_clean = data_train[~(outliers_mask).any(axis=1)]
classification_train_clean = classification_train[~(outliers_mask).any(axis=1)]

print(f"Oude aantal rijen: {len(data_train)}")
print(f"Nieuwe aantal rijen: {len(data_train_clean)}")
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
