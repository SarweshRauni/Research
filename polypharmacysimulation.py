!pip install hmmlearn
import numpy as np
import pandas as pd
!pip install mca
import mca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
!pip install -U scikit-fuzzy
import skfuzzy as fuzz
from hmmlearn import hmm
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(123)

# Number of individuals and time points
n_individuals = 50
n_years = 5

# Initialize an empty list to store data for each year
data_list = []

for year in range(1, n_years + 1):
    # Simulate demographics
    age = np.random.normal(loc=78.8, scale=7.4, size=n_individuals)
    age = np.clip(age, 65, 99)

    sex = np.random.choice(['Male', 'Female'], size=n_individuals, p=[0.596, 0.404])

    socioeconomic_status = np.random.choice([1, 2, 3, 4, 5], size=n_individuals)

    # Simulate diseases
    disease_prevalence = np.random.uniform(0.02, 0.50, size=41)
    diseases = np.array([np.random.binomial(1, p, n_individuals) for p in disease_prevalence]).T
    disease_cols = [f'Disease_{i+1}' for i in range(41)]

    # Adjust disease prevalence due to on age
    prob = 1 / (1 + np.exp(-(-5 + 0.05 * age)))
    diseases = np.array([np.random.binomial(1, p) for p in prob]).reshape(-1, 1) * np.ones((1, 41))

    # Simulate medications
    medication_prevalence = np.random.uniform(0.01, 0.50, size=89)
    medications = np.array([np.random.binomial(1, p, n_individuals) for p in medication_prevalence]).T
    medication_cols = [f'Medication_{i+1}' for i in range(89)]

    # Adjust medication due to disease
    for i in range(89):
        medication_name = medication_cols[i]
        associated_disease = np.random.choice(disease_cols)
        prob = np.where(diseases[:, disease_cols.index(associated_disease)] == 1, 0.7, 0.1)
        medications[:, i] = np.random.binomial(1, prob)

    # Simulate lab results
    creatinine = np.random.normal(loc=1, scale=0.3, size=n_individuals)

    # Combine all data
    data_year = pd.DataFrame({
        'ID': np.arange(1, n_individuals + 1),
        'Year': year,
        'Age': age,
        'Sex': sex,
        'Socioeconomic_Status': socioeconomic_status,
        'Creatinine': creatinine
    })

    diseases_df = pd.DataFrame(diseases, columns=disease_cols)
    medications_df = pd.DataFrame(medications, columns=medication_cols)

    data_year = pd.concat([data_year, diseases_df, medications_df], axis=1)

    data_list.append(data_year)

# Combine data for all years
data_all_years = pd.concat(data_list, ignore_index=True)

# Data
data_all_years['Sex'] = data_all_years['Sex'].astype('category')
data_all_years['Socioeconomic_Status'] = data_all_years['Socioeconomic_Status'].astype('category')

disease_cols = [col for col in data_all_years.columns if 'Disease_' in col]
medication_cols = [col for col in data_all_years.columns if 'Medication_' in col]

data_all_years[disease_cols] = data_all_years[disease_cols].astype(int)
data_all_years[medication_cols] = data_all_years[medication_cols].astype(int)

# Calculate polypharmacy and multimorbidity
data_all_years['Polypharmacy'] = (data_all_years[medication_cols].sum(axis=1) >= 5).astype(int)
data_all_years['Multimorbidity'] = (data_all_years[disease_cols].sum(axis=1) >= 2).astype(int)

# Dimensionality reduction using mca
categorical_vars = disease_cols + medication_cols + ['Sex', 'Socioeconomic_Status']

data_categorical = pd.get_dummies(data_all_years[categorical_vars],
                                 columns=categorical_vars,
                                 drop_first=True)

# Perform MCA
mca_ben = mca.MCA(data_categorical)

# Get MCA components
mca_components = pd.DataFrame(mca_ben.fs_r(N=5), columns=[f'MCA_{i+1}' for i in range(5)])

# Perform PCA on continuous variables
continuous_vars = ['Age', 'Creatinine']
scaler = StandardScaler()
scaled_continuous = scaler.fit_transform(data_all_years[continuous_vars])

pca = PCA(n_components=2, random_state=123)
pca_components = pca.fit_transform(scaled_continuous)

# Combine components
combined_components = pd.concat([pd.DataFrame(pca_components, columns=['PCA_1', 'PCA_2']), mca_components.reset_index(drop=True)], axis=1)

# Fuzzy c-means clustering
data_array = combined_components.values.T
n_clusters = 8

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_array, n_clusters, m=2, error=0.005, maxiter=1000, init=None, seed=123)

cluster_labels = np.argmax(u, axis=0)
data_all_years['Cluster'] = cluster_labels

print(data_all_years)

# Hidden Markov Model
data_all_years.sort_values(by=['ID', 'Year'], inplace=True)
cluster_pivot = data_all_years.pivot(index='ID', columns='Year', values='Cluster')
cluster_pivot.dropna(inplace=True)
X = cluster_pivot.values.astype(int)

model = hmm.MultinomialHMM(n_components=n_clusters, n_iter=100, random_state=123, verbose=True)
model.fit(X)
hidden_states = model.predict(X)

hidden_states_expanded = np.repeat(hidden_states, n_years)
data_all_years = data_all_years[data_all_years['ID'].isin(cluster_pivot.index)]
data_all_years['State'] = hidden_states_expanded


# Results
state_grouped = data_all_years.groupby('State')
state_prevalence = state_grouped[disease_cols + medication_cols].mean()
overall_prevalence = data_all_years[disease_cols + medication_cols].mean()

O_E_ratio = state_prevalence.div(overall_prevalence)
state_sizes = state_grouped.size()
total_counts = data_all_years[disease_cols + medication_cols].sum()

exclusivity = (state_prevalence.mul(state_sizes, axis=0)).div(total_counts)



# Visualizations
demographics = state_grouped['Age'].mean()
sex_distribution = state_grouped['Sex'].value_counts(normalize=True).unstack()

print("Average Age by State:")
print(demographics)
print("\nSex Distribution by State:")
print(sex_distribution)

# Transition matrix
transitions = np.zeros((n_clusters, n_clusters))

for i in range(X.shape[0]):
    for t in range(n_years - 1):
        transitions[X[i, t], X[i, t + 1]] += 1

transition_matrix = transitions / transitions.sum(axis=1, keepdims=True)

plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues')
plt.title('State Transition Matrix')
plt.xlabel('To State')
plt.ylabel('From State')
plt.show()
