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
# main.py — 영화 유형 나누기: 속성을 골라 묶고, 묶음 수도 바꿔 가며 정한다
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")
st.caption("정답 열 없이, 내가 고른 속성이 비슷한 영화끼리 묶습니다.")

MOVIES = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
기호 = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]          # 묶음 번호 대신 사용하는 표시
속성 = ["스크린 수(로그)", "누적 관객(로그)", "10위권 일수", "롱런 지수"]
원래단위 = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]


@st.cache_data
def load_data():
    df = pd.read_csv(MOVIES, dtype={"movieCd": str})
    df = df.sort_values("movieCd").reset_index(drop=True)
    전체편수 = len(df)
    # 네 속성을 만들 수 없는 영화는 뺀다: 값이 없거나 첫 주 관객이 0인 행
    쓸열 = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
    df = df.dropna(subset=쓸열 + ["movieNm"])
    df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0) & (df["first_week_audi"] > 0)]
    df = df.reset_index(drop=True)
    return df, 전체편수


df, 전체편수 = load_data()

# 네 가지 속성을 만든다. 숫자 차이가 너무 큰 둘은 상용로그를 취한다
df["스크린 수"] = df["first_scrn"]
df["누적 관객"] = df["total_audi"]
df["10위권 일수"] = df["days_in_top10"]
df["롱런 지수"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)   # 20을 넘으면 20으로
df["스크린 수(로그)"] = np.log10(df["first_scrn"])
df["누적 관객(로그)"] = np.log10(df["total_audi"])

st.info(f"전체 {전체편수:,}편 가운데 네 속성을 모두 만들 수 있는 {len(df):,}편으로 묶었습니다.")

고른속성 = st.multiselect("묶는 데 사용할 속성 (둘 이상)", 속성, default=속성)
if len(고른속성) < 2:
    st.warning("속성을 둘 이상 골라 주세요.")
    st.stop()

Xs = StandardScaler().fit_transform(df[고른속성])          # 고른 속성을 표준화한다
k = st.slider("몇 묶음으로 나눌까요", 2, 7, 3)


def 나누기(개수):
    """표준화한 값을 k-평균으로 나눈다. 난수를 고정해 다시 실행해도 결과가 같다."""
    return KMeans(n_clusters=개수, n_init=10, random_state=42).fit(Xs)


모델 = 나누기(k)
번호 = 모델.labels_

# 누적 관객 평균이 큰 묶음부터 ㉮·㉯·㉰ 순으로 표시한다
순서 = pd.Series(df["누적 관객"].to_numpy()).groupby(번호).mean().sort_values(ascending=False).index
자리 = {g: i for i, g in enumerate(순서)}
df["묶음"] = [기호[자리[g]] for g in 번호]

st.subheader("묶음 지도 · 2차원")
c1, c2 = st.columns(2)
x축 = c1.selectbox("가로축", 고른속성, index=0)
y축 = c2.selectbox("세로축", 고른속성, index=min(1, len(고른속성) - 1))
fig = px.scatter(df, x=x축, y=y축, color="묶음", hover_name="movieNm",
                 category_orders={"묶음": 기호[:k]})
fig.update_traces(marker=dict(size=9, opacity=0.8))
fig.update_layout(height=460)
st.plotly_chart(fig, width="stretch")
st.caption("점 하나가 영화 한 편입니다. 축을 바꿔 보면 묶음이 나뉘는 축과 섞이는 축이 보입니다.")

st.subheader("묶음 지도 · 3차원")
if len(고른속성) < 3:
    st.info("속성을 셋 이상 고르면 3차원 그림이 표시됩니다.")
else:
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("x축", 고른속성, index=0, key="x3")
    y3 = d2.selectbox("y축", 고른속성, index=1, key="y3")
    z3 = d3.selectbox("z축", 고른속성, index=2, key="z3")
    fig3 = px.scatter_3d(df, x=x3, y=y3, z=z3, color="묶음", hover_name="movieNm",
                         category_orders={"묶음": 기호[:k]})
    fig3.update_traces(marker=dict(size=2, opacity=0.7))
    fig3.update_layout(height=560, legend=dict(orientation="h"))
    st.plotly_chart(fig3, width="stretch")
    st.caption("마우스로 끌면 돌아갑니다. 점에 마우스를 올리면 제목이 보입니다.")

st.subheader("묶음별 편수와 네 속성의 평균")
요약 = df.groupby("묶음")[원래단위].mean().round(2)
요약.insert(0, "편수", df.groupby("묶음").size())
요약 = 요약.reindex(기호[:k])
st.dataframe(요약, width="stretch", column_config={
    "편수": st.column_config.NumberColumn("편수", format="%,d편"),
    "스크린 수": st.column_config.NumberColumn("스크린 수", format="%.1f개"),
    "누적 관객": st.column_config.NumberColumn("누적 관객", format="%,.0f명"),
    "10위권 일수": st.column_config.NumberColumn("10위권 일수", format="%.1f일"),
    "롱런 지수": st.column_config.NumberColumn("롱런 지수", format="%.2f배"),
})
st.caption("평균은 로그를 취하기 전의 원래 단위입니다. 롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이고 20에서 잘랐습니다.")

st.subheader("묶음마다 관객이 많은 다섯 편")
열 = st.columns(k)
for i, 이름 in enumerate(기호[:k]):
    묶음 = df[df["묶음"] == 이름].sort_values("누적 관객", ascending=False).head(5)
    열[i].markdown(f"**{이름} 묶음 ({len(df[df['묶음'] == 이름]):,}편)**")
    열[i].dataframe(pd.DataFrame({"영화": 묶음["movieNm"].to_numpy(),
                                "누적 관객": 묶음["누적 관객"].to_numpy()}),
                   width="stretch", hide_index=True,
                   column_config={"누적 관객": st.column_config.NumberColumn("누적 관객", format="%,d명")})

# 여기부터 ③ — 묶음 수를 몇으로 할지 정하는 화면
st.divider()
st.subheader("묶음 수는 몇이 좋을까")

# 고른 속성으로 다시 계산한다. 묶음 안에서 점들이 중심에서 떨어진 거리의 제곱을 모두 더한 값(k=1~7)
합 = pd.DataFrame({"묶음 수": list(range(1, 8)),
                   "거리 제곱의 합": [round(float(나누기(개수).inertia_), 1) for 개수 in range(1, 8)]})
# 바로 앞 묶음 수에서 줄어든 크기. 맨 첫 줄은 비교할 앞 값이 없다
합["앞보다 줄어든 값"] = ["—" if pd.isna(v) else f"{v:,.1f}" for v in -합["거리 제곱의 합"].diff()]

꺾은선 = px.line(합, x="묶음 수", y="거리 제곱의 합", markers=True)
꺾은선.add_vline(x=k, line_dash="dash", annotation_text=f"지금 고른 묶음 수 {k}")
꺾은선.update_layout(height=380, xaxis_title="묶음 수(k)", yaxis_title="묶음 안 거리 제곱의 합")
st.plotly_chart(꺾은선, width="stretch")
st.caption("묶음 수를 늘리면 값은 반드시 줄어듭니다. 줄어드는 폭이 크게 꺾이는 자리를 찾습니다.")

st.dataframe(합, width="stretch", hide_index=True)

점수 = silhouette_score(Xs, 번호)
st.metric(f"묶음 {k}개일 때의 실루엣 점수", f"{점수:.3f}")
st.caption("실루엣 점수는 -1에서 1 사이이고, 1에 가까울수록 묶음이 뚜렷하다는 뜻입니다. "
           "점수가 가장 높은 묶음 수와 사람이 이해하기 좋은 묶음 수가 늘 같지는 않습니다.")
