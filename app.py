# app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pandas as pd
import numpy as np
import textwrap

from itertools import cycle
from matplotlib import cm

from PIL import Image
from io import BytesIO

from pelo_functions import (
    get_ride_id,
    get_complete_workout,
    get_recent_workouts,
    generate_share_card
)

import config

st.set_page_config(layout="wide")
st.title("#ChaseTheHare Ride Visualiser")


def get_power_zones(ftp):
    return {
        "Zone 1": (0, 0.55 * ftp),
        "Zone 2": (0.56 * ftp, 0.75 * ftp),
        "Zone 3": (0.76 * ftp, 0.90 * ftp),
        "Zone 4": (0.91 * ftp, 1.05 * ftp),
        "Zone 5": (1.06 * ftp, 1.20 * ftp),
        "Zone 6": (1.21 * ftp, 1.50 * ftp),
        "Zone 7": (1.51 * ftp, 10 * ftp),
    }


def peloton_login(session):
    login_payload = {
        "username_or_email": config.USERNAME,
        "password": config.PASSWORD
    }
    login_response = session.post("https://api.onepeloton.com/auth/login", json=login_payload)
    if login_response.status_code != 200:
        raise Exception("Peloton login failed. Check credentials.")
    return session


def plot_metric_with_overlay(workout_df, metric, overlay):
    fig, ax = plt.subplots(figsize=(20, 8))

    # Dark theme setup
    fig.patch.set_facecolor('#111111')  # full background
    ax.set_facecolor('#111111')         # chart area background

    # Set dark theme colours
    plt.rcParams.update({
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "figure.facecolor": "#111111",
        "legend.facecolor": "#222222",
        "legend.edgecolor": "white",
    })
    sns.lineplot(data=workout_df, x="time (s)", y=metric, ax=ax, label="Your " + metric)

    # --- PowerZone colours
    zone_colours = {
        "Zone 1": "#a57dbb",
        "Zone 2": "#50c3c8",
        "Zone 3": "#4fc08d",
        "Zone 4": "#f3df6c",
        "Zone 5": "#f29e4c",
        "Zone 6": "#e4572e",
        "Zone 7": "#d62828"
    }

    if metric == "Output (watts)":
        from config import FTP
        power_zones = {
            "Zone 1": (0, 0.55 * FTP),
            "Zone 2": (0.56 * FTP, 0.75 * FTP),
            "Zone 3": (0.76 * FTP, 0.90 * FTP),
            "Zone 4": (0.91 * FTP, 1.05 * FTP),
            "Zone 5": (1.06 * FTP, 1.20 * FTP),
            "Zone 6": (1.21 * FTP, 1.50 * FTP),
            "Zone 7": (1.51 * FTP, min(FTP * 2, 600))  # Cap Zone 7 realistically
        }

        for zone, (low, high) in power_zones.items():
            ax.fill_between(
                workout_df["time (s)"].astype(float),
                float(low), float(high),
                color=zone_colours.get(zone, "grey"),
                alpha=0.3,
                label=zone
            )
            ax.text(
                workout_df["time (s)"].iloc[5],
                (low + high) / 2,
                zone,
                va='center',
                fontsize=9,
                color="white",
                fontweight="bold",
                bbox=dict(facecolor=zone_colours[zone], alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2")
            )

    # --- Cadence/Resistance overlays
    metric_dict = {
        "Output (watts)": ("lower_output_est", "upper_output_est"),
        "Cadence (rpm)": ("lower_cadence", "upper_cadence"),
        "Resistance (%)": ("lower_resistance", "upper_resistance")
    }

    if metric in metric_dict:
        lower_col, upper_col = metric_dict[metric]
        if lower_col in workout_df.columns and upper_col in workout_df.columns:
            try:
                lower_vals = pd.to_numeric(workout_df[lower_col], errors="coerce")
                upper_vals = pd.to_numeric(workout_df[upper_col], errors="coerce")

                ax.fill_between(
                    workout_df["time (s)"],
                    lower_vals,
                    upper_vals,
                    color="blue", alpha=0.2
                )
                above = workout_df[metric] > upper_vals
                below = workout_df[metric] < lower_vals

                ax.fill_between(
                    workout_df["time (s)"],
                    workout_df[metric],
                    upper_vals,
                    where=above,
                    color="green", alpha=0.2
                )
                ax.fill_between(
                    workout_df["time (s)"],
                    workout_df[metric],
                    lower_vals,
                    where=below,
                    color="red", alpha=0.2
                )
            except Exception as e:
                print("Target metric fill failed:", e)

    # --- Music overlay
    if overlay == "music" and "title" in workout_df.columns and "artist" in workout_df.columns:
        songs = workout_df.loc[workout_df["title"].shift(-1) != workout_df["title"]].reset_index()[["index", "title", "artist"]]
        songs["start"] = songs["index"].shift(1).fillna(0).astype(int)
        colours = cycle(cm.tab20.colors)

        for _, row in songs.iterrows():
            color = next(colours)
            ax.axvspan(row["start"], row["index"], alpha=0.15, color=color)
            ax.text((row["start"] + row["index"]) / 2, ax.get_ylim()[1] * 0.95,
                    f"{textwrap.fill(row['title'], 20)}\n{textwrap.fill(row['artist'], 20)}",
                    ha='center', va='top', fontsize=8)

    elif overlay == "segments" and "display_name" in workout_df.columns:
        segments = workout_df.loc[workout_df["display_name"].shift(-1) != workout_df["display_name"]].reset_index()
        segments["start"] = segments["index"].shift(1).fillna(0).astype(int)
        for _, row in segments.iterrows():
            ax.axvspan(row["start"], row["index"], alpha=0.15, color="lightgreen")
            ax.text((row["start"] + row["index"]) / 2, ax.get_ylim()[1] * 0.95,
                    textwrap.fill(str(row["display_name"]), 20),
                    ha='center', va='top', fontsize=8)

    ax.set_ylabel(metric, color='white')
    ax.set_xlabel("Time (s)", color='white')

    title = workout_df.get("class_title", ["Workout"])[0]
    instructor = workout_df.get("instructor", ["Unknown"])[0]
    main_title = f"{title} — {instructor}"

    raw_start = workout_df.get("start_time", [None])[0]
    start_time = pd.to_datetime(raw_start, unit="s") if raw_start else None

    username = workout_df.get("display_name", [None])[0] or workout_df.get("username", ["User"])[0]

    if start_time:
        subtitle = f"{start_time.strftime('%d %b %Y')} | {username}"
    else:
        subtitle = f"{username}"

    ax.set_title(f"{main_title}\n{subtitle} — {metric} with {overlay.capitalize()} Overlay", loc="left")

    # --- Final layout tweaks
    ax.legend().remove()
    fig.subplots_adjust(top=0.92, right=0.95)
    return fig



# Session setup
if "workout_df" not in st.session_state:
    st.session_state.workout_df = None
if "session" not in st.session_state:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    st.session_state.session = peloton_login(s)

# Sidebar: Load workouts
with st.sidebar:
    st.header("Recent Workouts")
    workouts = get_recent_workouts(st.session_state.session)
    workout_map = {
        f"{w['start']} — {w['name']} ({w['instructor']})": w["id"]
        for w in workouts if "name" in w and "instructor" in w
    }

    if workout_map:
        selected_display = st.selectbox("Select Workout", list(workout_map.keys()))
        selected_workout_id = workout_map.get(selected_display)

        if st.button("Load Selected Workout") and selected_workout_id:
            try:
                ride_id = get_ride_id(selected_workout_id, st.session_state.session)
                st.session_state.workout_df = get_complete_workout(selected_workout_id, ride_id, st.session_state.session)
                st.success("Workout loaded successfully.")
            except Exception as e:
                st.error(f"Error loading workout: {e}")
    else:
        st.warning("No workouts found.")

# Sidebar: Chart options
if st.session_state.workout_df is not None:
    with st.sidebar:
        st.header("Chart Options")
        metric = st.selectbox("Metric", [
            "Output (watts)",
            "Cadence (rpm)",
            "Resistance (%)"
        ])
        overlay = st.selectbox("Overlay", ["music", "segments"])

    fig = plot_metric_with_overlay(st.session_state.workout_df, metric, overlay)
    st.pyplot(fig)
    # Generate and show shareable ride card
    st.subheader("📸 Shareable Ride Card")
    share_card = generate_share_card(st.session_state.workout_df)
    st.image(share_card, caption="Shareable Ride Card", use_container_width=True)

