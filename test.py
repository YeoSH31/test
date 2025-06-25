import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="지수·로그 교점 찾기", layout="centered")
st.title("지수·로그 함수 교점 찾기 (전 구간)")

st.markdown(
    r"""
### 함수식
- **지수** \(y = a^{\,x+b} + c\)  
- **로그** \(y = p\log_{d}(x+e) + f\)

모든 실수 \(x\) 에 대해 두 함수를 고려합니다  
(\(\log\) 정의역 조건 \(x+e>0\) 은 내부에서 자동 처리)
""")

# ────────────────────────── 1️⃣ 매개변수 입력
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

# ────────────────────────── 2️⃣ 교점 계산
def g(x: float) -> float:
    """지수 - 로그 (g(x)=0 ⇒ 교점). 로그 쪽 정의역 자동 체킹."""
    if x + e <= 0:
        return np.nan   # 정의역 밖 -> NaN
    return a ** (x + b) + c - (p * np.log(x + e) / np.log(d_val) + f)

if btn:
    st.subheader("2️⃣  교점 계산 결과")

    # ❶ 탐색 구간 자동 설정: 로그 정의역 시작점을 기준으로 넉넉히 좌‧우 50씩
    left  = -50.0 if ( -50.0 + e > 0 ) else (-e + 1e-4)
    right = 50.0
    xs = np.linspace(left, right, 10001)
    ys = np.array([g(x) for x in xs])

    # NaN 구간은 건너뛰고 부호 변화 탐색
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

    if not roots:
        st.info("탐색 범위 내 교점을 찾지 못했습니다.")
    else:
        roots.sort()
        st.table([{"#": i+1, "x": r, "y": a**(r+b)+c} for i, r in enumerate(roots)])

        st.markdown(
            r"""
- 로그 정의역 \((x+e>0)\) 를 제외한 부분은 자동으로 무시했습니다.  
- `scipy.optimize.brentq` 로 근을 찾고, |Δx| < 1 × 10⁻⁷ 중복은 제거했습니다.
""")

        # ────────────────── 3️⃣ 그래프
        st.subheader("3️⃣  그래프")
        # 더 넓은 해상도용 X축
        x_dense = np.linspace(left, right, 4001)
        y_exp = a ** (x_dense + b) + c

        mask = x_dense + e > 0      # 로그 정의역
        x_log, y_log = x_dense[mask], p * np.log(x_dense[mask] + e) / np.log(d_val) + f

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
        st.plotly_chart(fig, use_container_width=True)
