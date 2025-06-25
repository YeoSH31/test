import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="지수·로그 교점 찾기", layout="centered")
st.title("지수·로그 함수 교점 찾기")

st.markdown(
    r"""
### 사용 방법
1. **지수함수**   \(y = a^{\,x+b} + c\)  
2. **로그함수**   \(y = p\log_{d}(x+e) + f\)

아래 입력창에 **a, b, c, p, d (or e), e, f** 값을 넣고  
**🔍 교점 찾기**를 누르세요.  
제약 : \(a>0,\;a\neq1,\;d>0,\;d\neq1,\;x+e>0\)
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

    # ★ 자연로그 옵션 추가 ★
    use_ln = st.checkbox("밑 d 를 자연상수 e (≈ 2.71828) 로 사용", value=False)
    if use_ln:
        d_val = np.e
        st.markdown(f"**d = {d_val:.5f} (고정)**")
    else:
        d_val = st.number_input("d (밑, >0 & ≠1)", value=10.0, format="%.6f")

    e = st.number_input("e (수평 이동)", value=0.0, format="%.6f")
    f = st.number_input("f (수직 이동)", value=0.0, format="%.6f")

st.divider()
xmin, xmax = st.columns(2)
x_min = xmin.number_input("그래프 x 최소", value=-5.0, format="%.2f")
x_max = xmax.number_input("그래프 x 최대", value=5.0, format="%.2f")

if x_min >= x_max:
    st.error("x 최소는 최대보다 작아야 합니다.")
    st.stop()

btn = st.button("🔍 교점 찾기")

# ────────────────────────── 2️⃣ 교점 계산
def g(x: float) -> float:
    """지수 - 로그 (g(x)=0 ⇒ 교점)"""
    return a ** (x + b) + c - (p * np.log(x + e) / np.log(d_val) + f)

if btn:
    if x_min + e <= 0:
        st.warning("선택 구간 일부에서 로그 정의역 조건 x+e>0 가 위반됩니다.")

    st.subheader("2️⃣  교점 계산 결과")
    xs = np.linspace(x_min, x_max, 4001)
    ys = g(xs)
    roots = []

    for xl, xr, yl, yr in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if yl == 0:
            roots.append(xl)
        elif yl * yr < 0:
            try:
                r = brentq(g, xl, xr)
                if all(abs(r - r0) > 1e-7 for r0 in roots):
                    roots.append(r)
            except ValueError:
                pass

    if not roots:
        st.info("주어진 구간에서 교점을 찾지 못했습니다.")
    else:
        roots.sort()
        st.table([{"#": i+1, "x": r, "y": a**(r+b)+c} for i, r in enumerate(roots)])

        st.markdown(
            r"""
- 차이 함수 \(g(x)=a^{x+b}+c -\bigl(p\log_{d}(x+e)+f\bigr)\) 를 정의,  
  **부호 변화 구간**을 찾아 `scipy.optimize.brentq` 로 근을 구했습니다.  
- 근 중복은 |Δx| < 1 × 10⁻⁷ 범위에서 제거했습니다.
""")

        # ────────────────── 3️⃣ 그래프
        st.subheader("3️⃣  그래프")
        x_dense = np.linspace(x_min, x_max, 1500)
        y_exp = a ** (x_dense + b) + c

        mask = x_dense + e > 0
        x_log = x_dense[mask]
        y_log = p * np.log(x_log + e) / np.log(d_val) + f

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
