import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 특성을 바탕으로 비슷한 영화끼리 세 유형으로 묶어 봅니다.")

FEATURES = {
    "스크린 수 (상용로그)": "log_first_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}
RAW_LABELS = {
    "log_first_scrn": "스크린 수 (상용로그)",
    "log_total_audi": "누적 관객 (상용로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}
CLUSTER_NAMES = {0: "㉮", 1: "㉯", 2: "㉰"}


@st.cache_data(show_spinner="영화 데이터를 불러오는 중입니다...")
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(missing))

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 네 가지 분류 속성에 필요한 값이 없거나 첫 주 관객이 0인 영화 제외
    valid = (
        df["first_scrn"].notna()
        & df["total_audi"].notna()
        & df["days_in_top10"].notna()
        & df["first_week_audi"].notna()
        & (df["first_week_audi"] != 0)
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
    )
    df = df.loc[valid].copy()

    df["log_first_scrn"] = df["first_scrn"].apply(lambda x: __import__("math").log10(x))
    df["log_total_audi"] = df["total_audi"].apply(lambda x: __import__("math").log10(x))
    df["long_run_index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=list(FEATURES.values()))
    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오거나 처리하는 중 문제가 발생했습니다.")
    st.exception(e)
    st.stop()

total_count = len(pd.read_csv(DATA_URL, encoding="utf-8"))
valid_count = len(df)

# 사용자가 분류에 쓸 속성을 고름
st.subheader("1. 분류에 사용할 속성")
selected_labels = st.multiselect(
    "두 개 이상 선택하세요.",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
)
if len(selected_labels) < 2:
    st.warning("K-평균 분류를 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()

selected_features = [FEATURES[label] for label in selected_labels]

# 선택된 속성을 표준화하고, K-평균으로 3개 군집 생성
scaler = StandardScaler()
scaled = scaler.fit_transform(df[selected_features])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df = df.copy()
df["_cluster_raw"] = model.fit_predict(scaled)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("_cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
name_map = {cluster_id: ["㉮", "㉯", "㉰"][i] for i, cluster_id in enumerate(cluster_order)}
df["영화 유형"] = df["_cluster_raw"].map(name_map)

st.markdown(
    f"**전체 영화 {total_count:,}편 · 조건에 맞아 분류한 영화 {valid_count:,}편**"
)

st.divider()
st.subheader("2. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_label = st.selectbox("가로축 속성", selected_labels, index=0, key="scatter_x")
with col2:
    y_default = 1 if len(selected_labels) > 1 else 0
    y_label = st.selectbox("세로축 속성", selected_labels, index=y_default, key="scatter_y")

scatter_df = df.copy()
scatter_df["가로축 값"] = scatter_df[FEATURES[x_label]]
scatter_df["세로축 값"] = scatter_df[FEATURES[y_label]]
fig2 = px.scatter(
    scatter_df,
    x="가로축 값",
    y="세로축 값",
    color="영화 유형",
    hover_name="movieNm",
    category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
    labels={"가로축 값": x_label, "세로축 값": y_label, "영화 유형": "영화 유형"},
    title=f"{x_label} × {y_label}",
)
fig2.update_traces(marker={"size": 8, "opacity": 0.78})
fig2.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("3. 3차원 산점도")
if len(selected_labels) < 3:
    st.info("3차원 산점도를 보려면 분류 속성을 세 개 이상 선택해 주세요.")
else:
    axis_cols = st.columns(3)
    with axis_cols[0]:
        x3_label = st.selectbox("x축 속성", selected_labels, index=0, key="x3")
    with axis_cols[1]:
        y3_label = st.selectbox("y축 속성", selected_labels, index=1, key="y3")
    with axis_cols[2]:
        z3_label = st.selectbox("z축 속성", selected_labels, index=2, key="z3")

    plot3 = df.copy()
    plot3["x값"] = plot3[FEATURES[x3_label]]
    plot3["y값"] = plot3[FEATURES[y3_label]]
    plot3["z값"] = plot3[FEATURES[z3_label]]
    fig3 = px.scatter_3d(
        plot3,
        x="x값",
        y="y값",
        z="z값",
        color="영화 유형",
        hover_name="movieNm",
        category_orders={"영화 유형": ["㉮", "㉯", "㉰"]},
        labels={
            "x값": x3_label, "y값": y3_label, "z값": z3_label,
            "영화 유형": "영화 유형"
        },
        title="마우스로 드래그해 각도와 시점을 바꿔 보세요.",
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형", height=650)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("4. 유형별 영화 수와 속성 평균")
st.caption("평균은 로그 변환 전의 원래 단위로 계산했습니다. 롱런 지수는 20으로 제한한 값의 평균입니다.")

summary_rows = []
raw_mean_columns = {
    "스크린 수 평균": "first_scrn",
    "누적 관객 평균": "total_audi",
    "10위권 일수 평균": "days_in_top10",
    "롱런 지수 평균": "long_run_index",
}
for group_name in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == group_name]
    row = {"영화 유형": group_name, "편수": len(group)}
    for display_name, col in raw_mean_columns.items():
        row[display_name] = group[col].mean()
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary_display = summary.copy()
for col in ["스크린 수 평균", "누적 관객 평균", "10위권 일수 평균", "롱런 지수 평균"]:
    summary_display[col] = summary_display[col].round(2)
summary_display["편수"] = summary_display["편수"].map(lambda x: f"{x:,}")
summary_display["누적 관객 평균"] = summary_display["누적 관객 평균"].map(
    lambda x: f"{float(x):,.0f}"
)
summary_display["스크린 수 평균"] = summary_display["스크린 수 평균"].map(
    lambda x: f"{float(x):,.1f}"
)
st.dataframe(summary_display, use_container_width=True, hide_index=True)

st.divider()
st.subheader("5. 유형별 누적 관객 상위 5편")
for group_name in ["㉮", "㉯", "㉰"]:
    group = df[df["영화 유형"] == group_name].sort_values(
        "total_audi", ascending=False
    ).head(5)
    with st.expander(f"{group_name} 유형 — 누적 관객 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            top = group[["movieNm", "total_audi"]].rename(
                columns={"movieNm": "영화 제목", "total_audi": "누적 관객"}
            ).copy()
            top["누적 관객"] = top["누적 관객"].map(lambda x: f"{x:,.0f}")
            top.insert(0, "순위", range(1, len(top) + 1))
            st.dataframe(top, use_container_width=True, hide_index=True)

st.caption("참고: K-평균은 선택한 속성에 따라 영화 간 거리와 유형이 달라질 수 있습니다. 유형 번호는 누적 관객 평균 순서에 따라 표시합니다.")
