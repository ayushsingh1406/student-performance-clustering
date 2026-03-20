import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Student Performance Analytics",
    layout="wide"
)

st.title("🎓 Student Performance Pattern Clustering")
st.write("Data-driven academic risk analysis and intervention system")

# =====================================================
# TRAIN MODEL (CACHED)
# =====================================================
@st.cache_resource
def train_model(df):

    df = df.copy()

    # Rename ID column for clarity
    if "Regd No." in df.columns:
        df = df.rename(columns={"Regd No.": "Student_ID"})

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    df["attendance_pct"] = df["Current_Attendance"]
    df["fail_count"] = (df["Total_Courses"] - df["PASS"]).clip(lower=0)
    df["pass_ratio"] = df["PASS"] / df["Total_Courses"]

    # -------------------------------
    # Features for clustering
    # -------------------------------
    feature_cols = [
        "Cgpa",
        "pass_ratio",
        "attendance_pct",
        "fail_count"
    ]

    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------
    # KMeans with K = 3 (JUSTIFIED)
    # -------------------------------
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_raw"] = kmeans.fit_predict(X_scaled)

    # -------------------------------
    # Rank clusters by academic risk
    # -------------------------------
    cluster_profile = df.groupby("cluster_raw")[[
        "Cgpa", "pass_ratio", "attendance_pct"
    ]].mean()

    cluster_profile["risk_score"] = (
        (10 - cluster_profile["Cgpa"]) +
        (1 - cluster_profile["pass_ratio"]) * 10 +
        (100 - cluster_profile["attendance_pct"]) / 10
    )

    cluster_rank_map = {
        cluster: rank
        for rank, cluster in enumerate(
            cluster_profile.sort_values("risk_score", ascending=False).index
        )
    }

    df["cluster"] = df["cluster_raw"].map(cluster_rank_map)

    return df, scaler, kmeans, cluster_rank_map, feature_cols


# =====================================================
# CLUSTER INTERPRETATION
# =====================================================
cluster_reviews = {
    0: "High academic risk. Students require immediate support.",
    1: "Moderate academic risk. Students need monitoring and mentoring.",
    2: "Low academic risk. Students are performing well."
}

cluster_suggestions = {
    0: "Remedial classes, academic counselling, attendance monitoring.",
    1: "Periodic mentoring, study-skills workshops, performance tracking.",
    2: "Advanced coursework, skill development, research opportunities."
}

# =====================================================
# DATA UPLOAD
# =====================================================
st.info("""
📋 **Required Column Names:**
- Regd No.
- Cgpa
- Total_Courses
- PASS
- Current_Attendance
- Delivered
- Attended

Make sure your Excel file contains these columns.
""")

uploaded_file = st.file_uploader(
    "Upload student dataset (.xlsx)",
    type=["xlsx"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df, scaler, kmeans, cluster_rank_map, feature_cols = train_model(df)

    # =================================================
    # DATASET PREVIEW
    # =================================================
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # =================================================
    # CLUSTER DISTRIBUTION
    # =================================================
    st.subheader("📊 Cluster Distribution")

    cluster_counts = df["cluster"].value_counts().sort_index()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(cluster_counts)

    with col2:
        fig, ax = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
        ax.set_xlabel("Cluster (Risk Level)")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)

    # =================================================
    # STUDENT VIEW
    # =================================================
    st.subheader("🧍 Student-wise Risk Analysis")
    student_id = st.selectbox("Select Student ID", df["Student_ID"].unique())
    student = df[df["Student_ID"] == student_id].iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("CGPA", f"{student['Cgpa']:.2f}")
        st.metric("Attendance %", f"{student['attendance_pct']:.1f}")

    with c2:
        st.metric("Pass Ratio", f"{student['pass_ratio']:.2f}")
        st.metric("Fail Count", int(student["fail_count"]))

    with c3:
        st.metric("Risk Cluster", int(student["cluster"]))

    st.info(cluster_reviews[int(student["cluster"])])
    st.warning(cluster_suggestions[int(student["cluster"])])

    # =================================================
    # NEW STUDENT PREDICTION
    # =================================================
    st.subheader("🆕 Predict Risk for New Student")

    with st.form("predict_form"):
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
        total = st.number_input("Total Courses", 1, 30, 16)
        passed = st.number_input("Passed Courses", 0, total, 14)
        attendance = st.number_input("Attendance %", 0, 100, 75)

        submit = st.form_submit_button("Predict Risk")

    if submit:
        new_student = pd.DataFrame([{
            "Cgpa": cgpa,
            "pass_ratio": passed / total,
            "attendance_pct": attendance,
            "fail_count": total - passed
        }])

        X_new = scaler.transform(new_student[feature_cols])
        raw_cluster = kmeans.predict(X_new)[0]
        final_cluster = cluster_rank_map[raw_cluster]

        st.success(f"Predicted Risk Cluster: {final_cluster}")
        st.info(cluster_reviews[final_cluster])
        st.warning(cluster_suggestions[final_cluster])

else:
    st.info("👆 Upload the dataset to start analysis")
