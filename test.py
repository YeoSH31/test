import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="지수·로그 교점 찾기", layout="centered")
st.title("지수·로그 함수 교점 찾기 (자동 확대 + 축 표시)")

# ────────────────────────── 설명
st.markdown(
    r"""
### 함수식
* **지수** \(y = a^{\,x+b} + c\)  
* **로그** \(y = p\log_{d}(x+e) + f\)

전 실수 축에서 교점을 구한 뒤  
**교점이 포함된 구간만 자동 확대**하고,  
그래프에 **x축·y축**(0, 0)을 점선으로 표시합니다.
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

# ────────────────────────── 2️⃣  근을 위한 함수
def g(x: float) -> float:
    """지수 - 로그 (정의역 밖은 NaN)."""
    if x + e <= 0:
        return np.nan
    return a ** (x + b) + c - (p * np.log(x + e) / np.log(d_val) + f)

# ────────────────────────── 3️⃣  계산 & 그래프
if btn:
    st.subheader("2️⃣  교점 결과 표")

    # (1) 넉넉한 탐색 범위
    left_search  = max(-50.0, -e + 1e-4)
    right_search = 50.0
    xs = np.linspace(left_search, right_search, 10001)
    ys = np.array([g(x) for x in xs])

    # (2) 부호 변화 스캔 + Brent
    roots = []
    for i in range(len(xs) - 1):
        y1, y2 = ys[i], ys[i + 1]
        if np.isnan(y1) or np.isnan(y2):
            continue
        if y1 == 0:
            roots.append(xs[i])
        elif y1 * y2 < 0:
            try:
                r = brentq(g, xs[i], xs[i + 1])
                if all(abs(r - r0) > 1e-7 for r0 in roots):
                    roots.append(r)
            except ValueError:
                pass

    if not roots:
        st.info("탐색 범위 내 교점을 찾지 못했습니다.")
    else:
        roots.sort()
        st.table(
            [{"#": i + 1, "x": r, "y": a ** (r + b) + c} for i, r in enumerate(roots)]
        )

    # (3) 그래프 범위 자동 설정
    if roots:
        if len(roots) == 1:
            x_c = roots[0]
            pad = 5.0
            x_min_plot, x_max_plot = x_c - pad, x_c + pad
        else:
            span = roots[-1] - roots[0]
            pad = max(1.0, 0.3 * span)
            x_min_plot, x_max_plot = roots[0] - pad, roots[-1] + pad
    else:
        x_min_plot, x_max_plot = left_search, right_search

    # (4) 그리기용 데이터
    x_dense = np.linspace(x_min_plot, x_max_plot, 4001)
    y_exp = a ** (x_dense + b) + c
    mask = x_dense + e > 0
    x_log = x_dense[mask]
    y_log = p * np.log(x_log + e) / np.log(d_val) + f

    # y축 범위 확보
    y_vals = np.concatenate([y_exp[mask], y_log]) if mask.any() else y_log
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    y_pad = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
    y_min_plot, y_max_plot = y_min - y_pad, y_max + y_pad

    # (5) Plotly 그래프
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_dense, y=y_exp, mode="lines", name="지수함수"))
    fig.add_trace(go.Scatter(x=x_log, y=y_log, mode="lines", name="로그함수"))
    if roots:
        fig.add_trace(
            go.Scatter(
                x=roots,
                y=[a ** (r + b) + c for r in roots],
                mode="markers",
                marker=dict(size=10, symbol="x"),
                name="교점",
            )
        )

    # ★ x·y 축 (0,0) 점선 표시
    fig.add_shape(
        type="line",
        x0=x_min_plot,
        x1=x_max_plot,
        y0=0,
        y1=0,
        line=dict(color="gray", width=1, dash="dash"),
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=y_min_plot,
        y1=y_max_plot,
        line=dict(color="gray", width=1, dash="dash"),
        layer="below",
    )

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(range=[x_min_plot, x_max_plot])
    fig.update_yaxes(range=[y_min_plot, y_max_plot])

    st.subheader("3️⃣  그래프")
    st.plotly_chart(fig, use_container_width=True)

    # (6) 계산 과정 상세 설명
    st.subheader("4️⃣  교점 계산 과정")
    with st.expander("상세 보기 / 접기", expanded=False):
        st.markdown(
            rf"""
* **차이 함수**  
  \[
    g(x) = a^{{\,x+b}} + c \;-\;\bigl(p\log_{{d}}(x+e)+f\bigr)
  \]

* **탐색 구간** : [{left_search:.4g}, {right_search:.4g}] 을 10 001개로 분할  
* **NaN 처리** : \(x+e≤0\) 인 지점은 로그 정의역이 아니므로 무시  
* **부호 변화 탐지** → 각 구간을 `scipy.optimize.brentq` 에 전달  
  (브렌트 기법: 이분법 + 보간 결합, 수렴 속도 빠름)  
* **중복 제거** : 근 사이가 |Δx| < 1 e-7 이면 동일 근으로 간주  
* **총 스캔 구간 수** : {len(xs)-1:,}  
* **발견한 교점 개수** : {len(roots)}
"""
        )
