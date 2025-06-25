import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import brentq

st.set_page_config(page_title="지수‧로그 함수 교점 찾기", layout="centered")

st.title("지수함수 y = a·x^ + b 와 로그함수 y = c·log_d(x+e) + f 의 교점 찾기")

st.markdown(
    """
**사용 방법**

1. 지수함수  \n
   \\(y = a^{x} + b\\)

2. 로그함수  \n
   \\(y = c\\;\\log_{d}(x+e) + f\\)

각 **상수 a, b, c, d, e, f** 값을 입력하세요.  
(제한:  \\(a>0,\\;a\\neq1,\\;d>0,\\;d\\neq1,\\;x+e>0\\) )

입력 후 *교점 찾기* 버튼을 누르면  
- 방정식 \\(a^{x}+b = c\\log_{d}(x+e)+f\\) 을  
  **브렌트 이분법**으로 수치적으로 풀어 교점을 찾습니다.  
- 찾은 x좌표를 이용해 y좌표를 구해 표‧그래프에 표시합니다.
"""
)

st.subheader("1️⃣  매개변수 입력")

col1, col2 = st.columns(2)
with col1:
    a = st.number_input("지수 a (>0, ≠1)", value=2.0, format="%.5f")
    b = st.number_input("지수 b", value=0.0, format="%.5f")
with col2:
    c = st.number_input("로그 c", value=1.0, format="%.5f")
    d = st.number_input("로그 밑 d (>0, ≠1)", value=10.0, format="%.5f")
    e = st.number_input("로그 (x+e) 의 e", value=0.0, format="%.5f")
    f = st.number_input("로그 f", value=0.0, format="%.5f")

st.write(" ")

# 영역 설정
xmin = st.number_input("그래프 x 최소값", value=-5.0, format="%.2f")
xmax = st.number_input("그래프 x 최대값", value=5.0, format="%.2f")

if xmin >= xmax:
    st.error("x 최소값은 최대값보다 작아야 합니다.")
    st.stop()

btn = st.button("🔍 교점 찾기")

# --------- 계산 로직 ---------
def diff(x):
    """ f(x)=지수 - 로그  (root=교점) """
    return a ** x + b - (c * np.log(x + e) / np.log(d) + f)

if btn:
    # 로그 정의역 체크
    if xmin + e <= 0:
        st.warning("선택한 x 구간에서 로그 정의역 x+e > 0 을 만족하지 않는 구간이 포함됩니다.")
    st.subheader("2️⃣  교점 계산 과정")

    # 1. 스캔하면서 부호변화 구간 찾기
    xs = np.linspace(xmin, xmax, 4001)
    ys = diff(xs)
    roots = []
    for x_left, x_right, y_left, y_right in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if y_left == 0:
            roots.append(x_left)
        elif y_left * y_right < 0:  # 부호 변화 → 근 존재
            try:
                root = brentq(diff, x_left, x_right)
                # 중복 제거용 근사 비교
                if all(abs(root - r) > 1e-6 for r in roots):
                    roots.append(root)
            except ValueError:
                pass

    if not roots:
        st.info("이 구간에서 교점을 찾지 못했습니다.")
    else:
        # 2. 결과 표
        roots_sorted = sorted(roots)
        result_rows = [{"#": i + 1, "x": r, "y": a ** r + b} for i, r in enumerate(roots_sorted)]
        st.table(result_rows)

        # 3. 과정 설명
        st.markdown(
            """
- 차이 함수  \\(g(x)=a^{x}+b - \\{ c\\log_{d}(x+e)+f \\}\\) 를 정의했습니다.  
- 선택한 x 범위를 4 000개로 세분화해 \\(g(x)\\) 를 계산했고,   
  **인접 샘플 간 부호 변화**가 있는 구간에서 scipy `brentq`(이분 + 보간)으로 근을 찾았습니다.
- 동일 근 오차 허용범위 1 e-6 으로 중복을 제거했습니다.
"""
        )

        # ---------------- 그래프 ----------------
        st.subheader("3️⃣  함수 그래프 및 교점")

        # 지수‧로그 데이터 (정의역 조건 포함)
        x_exp = xs
        y_exp = a ** x_exp + b

        mask = xs + e > 0
        x_log = xs[mask]
        y_log = c * np.log(x_log + e) / np.log(d) + f

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_exp, y=y_exp, mode="lines", name="지수함수"))
        fig.add_trace(go.Scatter(x=x_log, y=y_log, mode="lines", name="로그함수"))
        if roots:
            fig.add_trace(
                go.Scatter(
                    x=roots_sorted,
                    y=[a ** r + b for r in roots_sorted],
                    mode="markers",
                    marker=dict(size=10, symbol="x"),
                    name="교점",
                )
            )
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            hovermode="closest",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

