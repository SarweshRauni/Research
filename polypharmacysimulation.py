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

#The loop simulates data for each year from 1 to n_years 
#For each year, it simulates demographics, diseases, medications, lab results, and combines them into a DataFrame
for year in range(1, n_years + 1):
    # Simulate demographics
    # Simulated age using a normal distribution with a mean of 78.8 and a standard deviation of 7.4. np.clip limits the ages between 65 and 99
    age = np.random.normal(loc=78.8, scale=7.4, size=n_individuals)
    age = np.clip(age, 65, 99)

    #Randomly assigned 'Male' or 'Female' based on probabilities (59.6% Male, 40.4% Female)
    sex = np.random.choice(['Male', 'Female'], size=n_individuals, p=[0.596, 0.404])

    #Randomly assigned integers from 1 to 5
    socioeconomic_status = np.random.choice([1, 2, 3, 4, 5], size=n_individuals)

    # Simulate diseases
    # Randomly generated prevalence rates between 2% and 50% for 41 diseases
    disease_prevalence = np.random.uniform(0.02, 0.50, size=41)

    # For each disease, a binary outcome (0 or 1) is generated for each individual based on the disease's prevalence
    diseases = np.array([np.random.binomial(1, p, n_individuals) for p in disease_prevalence]).T

    # A list of column names for the diseases
    disease_cols = [f'Disease_{i+1}' for i in range(41)]

    # Adjust disease prevalence due to on age
    prob = 1 / (1 + np.exp(-(-5 + 0.05 * age)))
    diseases = np.array([np.random.binomial(1, p) for p in prob]).reshape(-1, 1) * np.ones((1, 41))

    # Simulate medications
    medication_prevalence = np.random.uniform(0.01, 0.50, size=89)
    medications = np.array([np.random.binomial(1, p, n_individuals) for p in medication_prevalence]).T
    medication_cols = [f'Medication_{i+1}' for i in range(89)]

    # Adjust medication due to disease
    # For each medication:
        # An associated disease is randomly selected
        # If the individual has the associated disease, the probability of taking the medication is set to 70%; otherwise, it's 10%
        # Medications are reassigned based on these probabilities
    for i in range(89):
        medication_name = medication_cols[i]
        associated_disease = np.random.choice(disease_cols)
        prob = np.where(diseases[:, disease_cols.index(associated_disease)] == 1, 0.7, 0.1)
        medications[:, i] = np.random.binomial(1, prob)

    # Simulates creatinine levels for each individual, representing kidney function.
    creatinine = np.random.normal(loc=1, scale=0.3, size=n_individuals)

    # Combine all data
    # Creates a DataFrame for the current year with all simulated data
    # Combines demographics, diseases, medications, and lab results
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

    # Adds the current year's data to data_list
    data_list.append(data_year)

# Combine data for all years
# Concatenates the data from all years into a single DataFrame data_all_years
data_all_years = pd.concat(data_list, ignore_index=True)

# Converts 'Sex' and 'Socioeconomic_Status' columns to categorical data types
data_all_years['Sex'] = data_all_years['Sex'].astype('category')
data_all_years['Socioeconomic_Status'] = data_all_years['Socioeconomic_Status'].astype('category')

# Creates lists of column names for diseases and medications
disease_cols = [col for col in data_all_years.columns if 'Disease_' in col]
medication_cols = [col for col in data_all_years.columns if 'Medication_' in col]

# Converts disease and medication data to integers (0 or 1)
data_all_years[disease_cols] = data_all_years[disease_cols].astype(int)
data_all_years[medication_cols] = data_all_years[medication_cols].astype(int)

# Polypharmacy:
    # Individuals taking 5 or more medications are assigned a value of 1; others are 0
# Multimorbidity:
    # Individuals with 2 or more diseases are assigned a value of 1; others are 0
data_all_years['Polypharmacy'] = (data_all_years[medication_cols].sum(axis=1) >= 5).astype(int)
data_all_years['Multimorbidity'] = (data_all_years[disease_cols].sum(axis=1) >= 2).astype(int)

# Dimensionality reduction using mca
# Categorical Variables:
    # Combines disease columns, medication columns, 'Sex', and 'Socioeconomic_Status'
# One-Hot Encoding:
    # Converts categorical variables into binary variables using pd.get_dummies
    # drop_first=True avoids multicollinearity by dropping first category
categorical_vars = disease_cols + medication_cols + ['Sex', 'Socioeconomic_Status']

data_categorical = pd.get_dummies(data_all_years[categorical_vars],
                                 columns=categorical_vars,
                                 drop_first=True)

# Perform MCA
# MCA:
    # Dimensionality  reduction technique for categorical data
    # Extracts factors that has the most variance in the data
# Extracting Components:
    # Retrieves the top 5 MCA components and stores them in mca_components
mca_ben = mca.MCA(data_categorical)
mca_components = pd.DataFrame(mca_ben.fs_r(N=5), columns=[f'MCA_{i+1}' for i in range(5)])

# Perform PCA on continuous variables
# Continuous Variables:
    # 'Age' and 'Creatinine'
# Standardization:
    # Scales the continuous variables to have a mean of 0 and standard deviation of 1
# PCA:
    # Reduces continuous variables into principal components
    # Extracts the top 2 components
continuous_vars = ['Age', 'Creatinine']
scaler = StandardScaler()
scaled_continuous = scaler.fit_transform(data_all_years[continuous_vars])

pca = PCA(n_components=2, random_state=123)
pca_components = pca.fit_transform(scaled_continuous)

# Combine components
# Combines PCA and MCA components into a single DataFrame combined_components for clustering
combined_components = pd.concat([pd.DataFrame(pca_components, columns=['PCA_1', 'PCA_2']), mca_components.reset_index(drop=True)], axis=1)

# Fuzzy c-means clustering
# Data Preparation:
    # Transposes the combined_components to match the input format of the clustering function
# Parameters:
    # n_clusters = 8: Number of clusters to form
    # m=2: Fuzziness parameter; controls the degree of fuzziness
# Clustering:
    # fuzz.cluster.cmeans performs fuzzy c-means clustering
    # cntr: Cluster centers
    # u: Membership degrees for each data point
# Assigning Clusters:
    # np.argmax(u, axis=0): Assigns each data point to the cluster where it has the highest membership degree
    # Adds the cluster labels to data_all_years

data_array = combined_components.values.T
n_clusters = 8

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_array, n_clusters, m=2, error=0.005, maxiter=1000, init=None, seed=123)

cluster_labels = np.argmax(u, axis=0)
data_all_years['Cluster'] = cluster_labels

print(data_all_years)

# Hidden Markov Model
# Sorting Data:
    # Ensures that data is ordered by patient ID and year
# Pivoting Data:
    # Creates a matrix where rows represent patients and columns represent years
    # Each cell contains the cluster label for a patient at a given year
# Missing Data:
    # Drops any rows with missing values
# Preparing for HMM:
    # Converts the pivot table to a NumPy array X with integer type
data_all_years.sort_values(by=['ID', 'Year'], inplace=True)
cluster_pivot = data_all_years.pivot(index='ID', columns='Year', values='Cluster')
cluster_pivot.dropna(inplace=True)
X = cluster_pivot.values.astype(int)

# Model Initialization:
    # n_components=n_clusters: Number of hidden states equals the number of clusters
    # n_iter=100: Maximum number of iterations
# Model Training:
    # Fits the HMM to the data X
# Predicting Hidden States:
    # Uses the trained model to predict the most likely sequence of hidden states for each patient
model = hmm.MultinomialHMM(n_components=n_clusters, n_iter=100, random_state=123, verbose=True)
model.fit(X)
hidden_states = model.predict(X)

# Expanding Hidden States:
    # Repeats the hidden state for each patient across all years
# Filtering:
    # Makes sure that only patients included in cluster_pivot are in data_all_years
# Assigning States:
    # Adds the hidden state labels to data_all_years
hidden_states_expanded = np.repeat(hidden_states, n_years)
data_all_years = data_all_years[data_all_years['ID'].isin(cluster_pivot.index)]
data_all_years['State'] = hidden_states_expanded

# Results
# Groups the data by the hidden states assigned by the HMM
state_grouped = data_all_years.groupby('State')

# State Prevalence:
    # Calculates the mean occurrence (prevalence) of each disease and medication within each state
# Overall Prevalence:
    # Calculates the mean occurrence across all data
state_prevalence = state_grouped[disease_cols + medication_cols].mean()
overall_prevalence = data_all_years[disease_cols + medication_cols].mean()

# O/E Ratio:
    # Divides the state prevalence by the overall prevalence to find overrepresented diseases/medications in each state
O_E_ratio = state_prevalence.div(overall_prevalence)

# State Sizes:
    # Counts the number of records (patient-years) in each state
# Total Counts:
    # Sums the occurrences of each disease and medication across all data
state_sizes = state_grouped.size()
total_counts = data_all_years[disease_cols + medication_cols].sum()

# Exclusivity:
    # Measures how unique a disease or medication is to a state, adjusted for state size and total counts
exclusivity = (state_prevalence.mul(state_sizes, axis=0)).div(total_counts)

# Visualizations
# Average Age by State:
    # Calculates the mean age of individuals in each state
# Sex Distribution by State:
    # Calculates the proportion of 'Male' and 'Female' in each state
# Printing Results:
    # Outputs the demographic information
demographics = state_grouped['Age'].mean()
sex_distribution = state_grouped['Sex'].value_counts(normalize=True).unstack()

print("Average Age by State:")
print(demographics)
print("\nSex Distribution by State:")
print(sex_distribution)

# Transition matrix
# Initializing Transition Counts:
    # Creates a matrix to count transitions from one state to another
# Counting Transitions:
    # For each patient, increments the count for transitions between consecutive years
# Calculating Probabilities:
    # Normalizes the counts to probabilities by dividing by the total transitions from each state
transitions = np.zeros((n_clusters, n_clusters))

for i in range(X.shape[0]):
    for t in range(n_years - 1):
        transitions[X[i, t], X[i, t + 1]] += 1

transition_matrix = transitions / transitions.sum(axis=1, keepdims=True)

# Visualization:
    # Uses a heatmap to visualize the transition matrix
    # annot=True: Displays the probability values on heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues')
plt.title('State Transition Matrix')
plt.xlabel('To State')
plt.ylabel('From State')
plt.show()
