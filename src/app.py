import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="xG Model | La Liga",
    page_icon="⚽",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xg_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Pitch drawing function ────────────────────────────────────
def draw_pitch(shot_x=None, shot_y=None, xg=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0d2b0f')
    ax.set_facecolor('#163d19')

    # Pitch outline
    ax.plot([0,120],[0,0], [0,120],[80,80],
            [0,0],[0,80], [120,120],[0,80], color='white', linewidth=2)

    # Centre line & circle
    ax.plot([60,60],[0,80], color='white', linewidth=1.5)
    centre = plt.Circle((60,40), 9.15, color='white', fill=False, linewidth=1.5)
    ax.add_patch(centre)
    ax.plot(60, 40, 'o', color='white', markersize=3)

    # Penalty areas
    ax.plot([102,120],[62,62],[102,102],[62,18],[102,120],[18,18], color='white', linewidth=1.5)
    ax.plot([0,18],[62,62],[18,18],[62,18],[0,18],[18,18], color='white', linewidth=1.5)

    # Six yard boxes
    ax.plot([114,120],[54,54],[114,114],[54,26],[114,120],[26,26], color='white', linewidth=1.5)
    ax.plot([0,6],[54,54],[6,6],[54,26],[0,6],[26,26], color='white', linewidth=1.5)

    # Goals
    ax.plot([120,122],[36,36],[122,122],[36,44],[120,122],[44,44], color='white', linewidth=2)
    ax.plot([0,-2],[36,36],[-2,-2],[36,44],[0,-2],[44,44], color='white', linewidth=2)

    # Penalty spots
    ax.plot(108, 40, 'o', color='white', markersize=3)
    ax.plot(12, 40, 'o', color='white', markersize=3)

    # Penalty arcs
    left_arc = Arc((12,40), height=18.3, width=18.3, angle=0, theta1=310, theta2=50, color='white', linewidth=1.5)
    right_arc = Arc((108,40), height=18.3, width=18.3, angle=0, theta1=130, theta2=230, color='white', linewidth=1.5)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # Shot marker
    if shot_x and shot_y:
        color = '#b5f530' if xg and xg >= 0.3 else '#ffb020' if xg and xg >= 0.1 else '#ff4444'
        ax.plot(shot_x, shot_y, 'o', color=color, markersize=14,
                markeredgecolor='white', markeredgewidth=2, zorder=5)
        ax.annotate(f'xG: {xg:.3f}', (shot_x, shot_y),
                    textcoords='offset points', xytext=(0, 16),
                    ha='center', fontsize=12, fontweight='bold',
                    color=color)

    ax.set_xlim(-3, 123)
    ax.set_ylim(-3, 83)
    ax.axis('off')
    return fig

# ── App layout ────────────────────────────────────────────────
st.markdown("# ⚽ xG Prediction Model")
st.markdown("#### Built on StatsBomb La Liga data · Logistic Regression")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Shot Parameters")

    x = st.slider("Distance from goal (metres)", 1.0, 40.0, 16.0, 0.5)
    angle = st.slider("Shot angle (radians)", 0.0, 1.6, 0.8, 0.05)
    is_header = st.toggle("Header?", value=False)
    is_open_play = st.toggle("Open play?", value=True)
    is_late_game = st.toggle("After 75 mins?", value=False)
    assisted_by_cross = st.toggle("Assisted by cross?", value=False)
    first_time = st.toggle("First time shot?", value=False)

    # Build feature vector
    features = pd.DataFrame([[
        x, angle,
        int(is_header), int(is_open_play), int(is_late_game),
        int(assisted_by_cross), int(first_time)
    ]], columns=['distance','angle','is_header','is_open_play',
                 'is_late_game','assisted_by_cross','first_time'])

    xg_prob = model.predict_proba(features)[0][1]

    st.markdown("---")
    st.markdown("### Predicted xG")

    color = "green" if xg_prob >= 0.3 else "orange" if xg_prob >= 0.1 else "red"
    st.markdown(f"<h1 style='color:{color}; font-size:64px'>{xg_prob:.3f}</h1>",
                unsafe_allow_html=True)

    quality = "🟢 High quality chance" if xg_prob >= 0.3 else "🟡 Moderate chance" if xg_prob >= 0.1 else "🔴 Low quality chance"
    st.markdown(quality)

with col2:
    st.markdown("### Shot Location on Pitch")
    # Convert distance/angle back to pitch coords for visualisation
    shot_x = 120 - x * np.cos(angle)
    shot_y = 40 + x * np.sin(angle)
    fig = draw_pitch(shot_x, shot_y, xg_prob)
    st.pyplot(fig)

st.markdown("---")
st.markdown("*Model trained on StatsBomb open data · La Liga 2018-2021 · Built by Paul*")