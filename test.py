import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="지수·로그 교점 찾기", layout="centered")
st.title("지수·로그 함수 교점 찾기 (자동 확대)")

st.markdown(
    r"""
### 함수식
- **지수** \(y = a^{\,x+b} + c\)  
- **로그** \(y = p\log_{d}(x+e) + f\)

전 실수 축에서 교점을 검색한 뒤, **교점이 포함된 구간을 자동 확대**해 그립니다.
""")

# ────────────────────────── 1️⃣  매개변수 입력
st.subheader("1️⃣  매개변수 입력")
col_exp, col_log = st.columns(2, gap="large")

with col_exp:
    st.markdown("**지수함수**  \(a^{x+b}+c\)")
    a = st.number_input("a (밑, >0 & ≠1)", value=2.0, format="%.6f")
    b = st.number_input("b (수평 이동)", value=0.0, format="%.6f")
    c = st.number_input("c (수직 이동)", value=0.0, format="%.6f")

with col_log:
    st.markdown("**로그함수**  \(p·log_d(x+e)+f\)")
    p = st.number_input("p (계수)", value=1.0, format="%.6f")

    use_ln = st.checkbox("밑 d 를 자연상수 e 로 사용", value=False)
    if use_ln:
        d_val = np.e
        st.markdown(f"**d = {d_val:.5f} (고정)**")
    else:
        d_val = st.number_input("d (밑, >0 & ≠1)", value=10.0, format="%.6f")

    e = st.number_input("e (수평 이동)", value=0.0, format="%.6f")
    f = st.number_input("f (수직 이동)", value=0.0, format="%.6f")

btn = st.button("🔍 교점 찾기")

# ────────────────────────── 2️⃣  교점·그래프
def g(x: float) -> float:
    """지수 - 로그 (NaN 반환 → brentq 탐색에서 제외)."""
    if x + e <= 0:
        return np.nan
    return a ** (x + b) + c - (p * np.log(x + e) / np.log(d_val) + f)

if btn:
    st.subheader("2️⃣  교점 계산 결과")

    # 넉넉한 탐색 구간: 로그 정의역 시작점보다 살짝 오른쪽부터 + 오른쪽 50
    left_search  = max(-50.0, -e + 1e-4)
    right_search = 50.0
    xs = np.linspace(left_search, right_search, 10001)
    ys = np.array([g(x) for x in xs])

    roots = []
    for i in range(len(xs) - 1):
        y1, y2 = ys[i], ys[i+1]
        if np.isnan(y1) or np.isnan(y2):
            continue
        if y1 == 0:
            roots.append(xs[i])
        elif y1 * y2 < 0:
            try:
                r = brentq(g, xs[i], xs[i+1])
                if all(abs(r - r0) > 1e-7 for r0 in roots):
                    roots.append(r)
            except ValueError:
                pass

    # ---------- 결과 표 ----------
    if not roots:
        st.info("탐색 범위 내 교점을 찾지 못했습니다.")
    else:
        roots.sort()
        st.table([{"#": i+1, "x": r, "y": a**(r+b)+c} for i, r in enumerate(roots)])

    # ---------- 그래프 ----------
    st.subheader("3️⃣  그래프")

    if roots:
        if len(roots) == 1:
            x_center = roots[0]
            x_pad = 5.0                       # 단일 root 기본 패딩
            x_min_plot, x_max_plot = x_center - x_pad, x_center + x_pad
        else:
            span = roots[-1] - roots[0]
            pad = max(1.0, 0.3 * span)        # 루트 간격의 30% 또는 최소 1
            x_min_plot, x_max_plot = roots[0] - pad, roots[-1] + pad
    else:
        # 교점이 없으면 탐색 구간 그대로
        x_min_plot, x_max_plot = left_search, right_search

    # 고해상도 데이터
    x_dense = np.linspace(x_min_plot, x_max_plot, 4001)
    y_exp = a ** (x_dense + b) + c
    mask = x_dense + e > 0
    x_log = x_dense[mask]
    y_log = p * np.log(x_log + e) / np.log(d_val) + f

    # y 범위 자동 확대 (10% 여유)
    y_all = np.concatenate([y_exp[mask], y_log]) if mask.any() else y_log
    y_min, y_max = np.min(y_all), np.max(y_all)
    y_pad = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
    y_min_plot, y_max_plot = y_min - y_pad, y_max + y_pad

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_dense, y=y_exp,
                             mode="lines", name="지수함수"))
    fig.add_trace(go.Scatter(x=x_log, y=y_log,
                             mode="lines", name="로그함수"))
    if roots:
        fig.add_trace(go.Scatter(
            x=roots, y=[a**(r+b)+c for r in roots],
            mode="markers", marker=dict(size=10, symbol="x"),
            name="교점"))

    fig.update_layout(xaxis_title="x", yaxis_title="y",
                      hovermode="closest", template="plotly_white")
    fig.update_xaxes(range=[x_min_plot, x_max_plot])
    fig.update_yaxes(range=[y_min_plot, y_max_plot])
    st.plotly_chart(fig, use_container_width=True)
