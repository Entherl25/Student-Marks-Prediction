"""
Students Marks Prediction — Interactive Dashboard
Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, os

st.set_page_config(page_title="Students Marks Prediction", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

# ---------- Styling ----------
st.markdown("""
<style>
.main { background-color: #0e1117; }
.metric-card {
    background: linear-gradient(135deg,#1f2a48 0%,#2d3a63 100%);
    padding: 18px 20px; border-radius: 14px; border:1px solid #2e3b5e;
    box-shadow:0 4px 14px rgba(0,0,0,.25);
}
.metric-card h3 { margin:0; color:#9fb3ff; font-size:.85rem; font-weight:600; letter-spacing:.04em;}
.metric-card p  { margin:4px 0 0; color:#fff; font-size:1.7rem; font-weight:700;}
.big-title { font-size:2.1rem; font-weight:800; background:linear-gradient(90deg,#7aa2ff,#c08cff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.pred-box { background:linear-gradient(135deg,#13315c,#1d4e89); padding:24px; border-radius:16px;
    text-align:center; border:1px solid #2e6bb0;}
.pred-box .val { font-size:3rem; font-weight:800; color:#fff;}
.pred-box .lbl { color:#bcd4ff; font-size:1rem; letter-spacing:.05em;}
</style>
""", unsafe_allow_html=True)

# ---------- Load data & models ----------
@st.cache_data
def load_data():
    if os.path.exists("data/cleaned_student_data.csv"):
        path = "data/cleaned_student_data.csv"
    elif os.path.exists("cleaned_student_data.csv"):
        path = "cleaned_student_data.csv"
    elif os.path.exists("data/StudentPerformanceFactors.csv"):
        path = "data/StudentPerformanceFactors.csv"
    else:
        path = "StudentPerformanceFactors.csv"
    return pd.read_csv(path)

@st.cache_resource
def load_bundle():
    if os.path.exists("models/model_bundle.pkl"):
        return joblib.load("models/model_bundle.pkl")
    return joblib.load("model_bundle.pkl")

try:
    df = load_data()
    bundle = load_bundle()
    feature_cols = bundle["feature_cols"]
    ordinal_maps = bundle["ordinal_maps"]
    binary_maps  = bundle["binary_maps"]
except Exception as e:
    st.error(f"Could not load data/models. Run the notebook first to generate them.\n\n{e}")
    st.stop()



def grade_color(g):
    return {"A":"#2ecc71","B":"#27ae60","C":"#f1c40f","D":"#e67e22","F":"#e74c3c"}.get(g,"#888")

# ---------- Sidebar ----------
st.sidebar.markdown("## 🎓 Marks Prediction")
page = st.sidebar.radio("Navigate", ["📊 Overview", "📈 EDA Dashboard",
                                     "🤖 Model Comparison", "🔮 Live Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("CS402 — Data Science Semester Project")
st.sidebar.caption(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ============================================================ OVERVIEW
if page == "📊 Overview":
    st.markdown('<div class="big-title">Students Marks Prediction</div>', unsafe_allow_html=True)
    st.write("An end-to-end data science pipeline predicting student exam scores and grades "
             "from study habits, attendance and background factors.")
    c = st.columns(4)
    cards = [("STUDENTS", f"{len(df):,}"),
             ("AVG SCORE", f"{df['Exam_Score'].mean():.1f}"),
             ("PASS RATE", f"{(df['Exam_Score']>=60).mean()*100:.0f}%"),
             ("FEATURES", f"{len(feature_cols)}")]
    for col,(t,v) in zip(c, cards):
        col.markdown(f'<div class="metric-card"><h3>{t}</h3><p>{v}</p></div>', unsafe_allow_html=True)
    st.markdown("### Sample of the data")
    st.dataframe(df.head(12), use_container_width=True)
    st.markdown("### Statistical summary")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

# ============================================================ EDA
elif page == "📈 EDA Dashboard":
    st.markdown('<div class="big-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Exam_Score", nbins=30, marginal="box",
                           title="Distribution of Exam Scores", color_discrete_sequence=["#4C72B0"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        gc = df["Grade"].value_counts().reindex(["A","B","C","D","F"]).fillna(0).reset_index()
        gc.columns = ["Grade","Count"]
        fig = px.bar(gc, x="Grade", y="Count", title="Students per Grade",
                     color="Grade", color_discrete_map={g:grade_color(g) for g in ["A","B","C","D","F"]})
        st.plotly_chart(fig, use_container_width=True)

    num_cols = ["Hours_Studied","Attendance","Sleep_Hours","Previous_Scores",
                "Tutoring_Sessions","Physical_Activity","Exam_Score"]
    c3, c4 = st.columns(2)
    with c3:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        xcol = st.selectbox("Scatter X-axis", ["Hours_Studied","Attendance","Previous_Scores","Tutoring_Sessions"])
        fig = px.scatter(df, x=xcol, y="Exam_Score", trendline="ols", opacity=0.3,
                         title=f"{xcol} vs Exam Score", color_discrete_sequence=["#55A868"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Exam score by a categorical factor")
    cat = st.selectbox("Choose factor", ["Motivation_Level","Access_to_Resources","Parental_Involvement",
                                         "Teacher_Quality","Peer_Influence","School_Type","Gender"])
    fig = px.box(df, x=cat, y="Exam_Score", color=cat, title=f"Exam Score by {cat}")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================ MODELS
elif page == "🤖 Model Comparison":
    st.markdown('<div class="big-title">Model Comparison</div>', unsafe_allow_html=True)

    st.markdown(f"### Regression — predict exact score  ·  best: **{bundle['regressor_name']}**")
    reg = pd.DataFrame(bundle["reg_metrics"]).round(3)
    st.dataframe(reg, use_container_width=True)
    fig = px.bar(reg, x="Model", y="R2", color="R2", color_continuous_scale="Greens",
                 title="Regression R² (higher = better)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Classification — predict grade  ·  best: **{bundle['classifier_name']}**")
    clf = pd.DataFrame(bundle["clf_metrics"]).round(3)
    st.dataframe(clf, use_container_width=True)
    m = clf.melt(id_vars="Model", value_vars=["Accuracy","Precision","Recall","F1"],
                 var_name="Metric", value_name="Score")
    fig = px.bar(m, x="Model", y="Score", color="Metric", barmode="group",
                 title="Classification metrics by model")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================ PREDICTION
elif page == "🔮 Live Prediction":
    st.markdown('<div class="big-title">Live Prediction</div>', unsafe_allow_html=True)
    st.write("Enter a student's details to predict their **exam score** and **grade**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        hours = st.slider("Hours Studied / week", 1, 44, 20)
        attend = st.slider("Attendance (%)", 60, 100, 85)
        prev = st.slider("Previous Scores", 50, 100, 75)
        sleep = st.slider("Sleep Hours", 4, 10, 7)
        tut = st.slider("Tutoring Sessions", 0, 8, 2)
        phys = st.slider("Physical Activity", 0, 6, 3)
    with c2:
        par_inv = st.selectbox("Parental Involvement", ["Low","Medium","High"], 1)
        access = st.selectbox("Access to Resources", ["Low","Medium","High"], 1)
        motiv = st.selectbox("Motivation Level", ["Low","Medium","High"], 1)
        income = st.selectbox("Family Income", ["Low","Medium","High"], 1)
        teacher = st.selectbox("Teacher Quality", ["Low","Medium","High"], 1)
        peer = st.selectbox("Peer Influence", ["Negative","Neutral","Positive"], 2)
    with c3:
        par_edu = st.selectbox("Parental Education", ["High School","College","Postgraduate"], 1)
        dist = st.selectbox("Distance from Home", ["Near","Moderate","Far"], 0)
        extra = st.selectbox("Extracurricular Activities", ["No","Yes"], 1)
        net = st.selectbox("Internet Access", ["No","Yes"], 1)
        ld = st.selectbox("Learning Disabilities", ["No","Yes"], 0)
        school = st.selectbox("School Type", ["Public","Private"], 0)
        gender = st.selectbox("Gender", ["Male","Female"], 0)

    raw = {
        "Hours_Studied":hours,"Attendance":attend,"Parental_Involvement":par_inv,
        "Access_to_Resources":access,"Extracurricular_Activities":extra,"Sleep_Hours":sleep,
        "Previous_Scores":prev,"Motivation_Level":motiv,"Internet_Access":net,
        "Tutoring_Sessions":tut,"Family_Income":income,"Teacher_Quality":teacher,
        "School_Type":school,"Peer_Influence":peer,"Physical_Activity":phys,
        "Learning_Disabilities":ld,"Parental_Education_Level":par_edu,
        "Distance_from_Home":dist,"Gender":gender,
    }
    enc = {}
    for k,v in raw.items():
        if k in ordinal_maps: enc[k] = ordinal_maps[k][v]
        elif k in binary_maps: enc[k] = binary_maps[k][v]
        else: enc[k] = v
    Xrow = np.array([[enc[c] for c in feature_cols]], dtype=float)

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        score = float(bundle["regressor"].predict(bundle["reg_scaler"].transform(Xrow))[0])
        score = max(0, min(100, score))
        grade = bundle["classifier"].predict(bundle["clf_scaler"].transform(Xrow))[0]
        result = "PASS ✅" if score >= 60 else "FAIL ❌"

        a,b,c = st.columns(3)
        a.markdown(f'<div class="pred-box"><div class="lbl">PREDICTED SCORE</div>'
                   f'<div class="val">{score:.1f}</div></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="pred-box"><div class="lbl">PREDICTED GRADE</div>'
                   f'<div class="val" style="color:{grade_color(grade)}">{grade}</div></div>', unsafe_allow_html=True)
        c.markdown(f'<div class="pred-box"><div class="lbl">RESULT</div>'
                   f'<div class="val" style="font-size:2rem">{result}</div></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, title={"text":"Exam Score"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":grade_color(grade)},
                   "steps":[{"range":[0,60],"color":"#3a1f1f"},
                            {"range":[60,80],"color":"#3a341f"},
                            {"range":[80,100],"color":"#1f3a25"}]}))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
