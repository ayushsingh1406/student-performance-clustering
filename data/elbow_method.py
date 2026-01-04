import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_excel("data/cleaned_student_dataset_random.xlsx")

# Rename for clarity
df = df.rename(columns={"Regd No.": "Student_ID"})

# Select features
features = ["Cgpa", "PASS", "Total_Courses", "Current_Attendance"]
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- ELBOW METHOD ----------
wcss = []
K = range(2, 11)  # start from 2 (silhouette not valid for k=1)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# ---------- SILHOUETTE METHOD ----------
silhouette_scores = {}

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = score

# Find optimal K
optimal_k = max(silhouette_scores, key=silhouette_scores.get)

# ---------- PRINT RESULTS ----------
print("\nSilhouette Scores:")
for k, score in silhouette_scores.items():
    print(f"K = {k} → Silhouette Score = {score:.4f}")

print(f"\nOptimal number of clusters based on Silhouette Score: K = {optimal_k}")

# ---------- PLOT ELBOW ----------
plt.figure(figsize=(8, 5))
plt.plot(K, wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.show()
