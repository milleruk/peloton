import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap
import numpy as np
from itertools import cycle
from matplotlib import cm
import config

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

def plot_metric_with_overlay(workout_df, metric, overlay):
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.lineplot(data=workout_df, x="time (s)", y=metric, ax=ax, label=f"Your {metric}")

    class_type_ids = workout_df.get("class_type_ids", [[""]])[0]
    is_powerzone = "665395ff3abf4081bf315686227d1a51" in class_type_ids

    if metric == "Output (watts)":
        ftp = getattr(config, "FTP", 250)
        zones = get_power_zones(ftp)

        for zone, (low, high) in zones.items():
            ax.axhspan(low, high, alpha=0.1, label=zone)

    elif not is_powerzone:
        metric_dict = {
            "Cadence (rpm)": ("lower_cadence", "upper_cadence"),
            "Resistance (%)": ("lower_resistance", "upper_resistance")
        }

        if metric in metric_dict:
            lower_col, upper_col = metric_dict[metric]
            if lower_col in workout_df.columns and upper_col in workout_df.columns:
                workout_df[lower_col] = pd.to_numeric(workout_df[lower_col], errors='coerce').ffill().bfill()
                workout_df[upper_col] = pd.to_numeric(workout_df[upper_col], errors='coerce').ffill().bfill()

                ax.fill_between(
                    workout_df["time (s)"].astype(int),
                    workout_df[lower_col],
                    workout_df[upper_col],
                    color="blue", alpha=0.2, label="Target Range"
                )

                above = workout_df[metric] > workout_df[upper_col]
                below = workout_df[metric] < workout_df[lower_col]

                ax.fill_between(
                    workout_df["time (s)"],
                    workout_df[metric],
                    workout_df[upper_col],
                    where=above,
                    color="green", alpha=0.2, label="Above Range"
                )
                ax.fill_between(
                    workout_df["time (s)"],
                    workout_df[metric],
                    workout_df[lower_col],
                    where=below,
                    color="red", alpha=0.2, label="Below Range"
                )

    if overlay == "music":
        if "title" in workout_df.columns and "artist" in workout_df.columns:
            songs = workout_df.loc[workout_df["title"].shift(-1) != workout_df["title"]].reset_index()[["index", "title", "artist"]]
            songs["start"] = songs["index"].shift(1).fillna(0).astype(int)
            colours = cycle(cm.Set3.colors)
            for _, row in songs.iterrows():
                ax.axvspan(row["start"], row["index"], alpha=0.2, color=next(colours))
                ax.text((row["start"] + row["index"]) / 2, ax.get_ylim()[1]*0.95,
                        f"{textwrap.fill(row['title'], 20)}\n{textwrap.fill(row['artist'], 20)}",
                        ha='center', va='top', fontsize=8)
    elif overlay == "segments":
        if "display_name" in workout_df.columns:
            segments = workout_df.loc[workout_df["display_name"].shift(-1) != workout_df["display_name"]].reset_index()
            segments["start"] = segments["index"].shift(1).fillna(0).astype(int)
            for _, row in segments.iterrows():
                ax.axvspan(row["start"], row["index"], alpha=0.15, color="lightgreen")
                ax.text((row["start"] + row["index"]) / 2, ax.get_ylim()[1]*0.95,
                        textwrap.fill(str(row["display_name"]), 20),
                        ha='center', va='top', fontsize=8)

    ax.set_ylabel(metric)
    ax.set_xlabel("Time (s)")
    class_title = workout_df.get("class_title", ["Workout"])[0]
    instructor = workout_df.get("instructor", ["Unknown"])[0]

def plot_tile_preview(workout_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(4, 1.5), dpi=150)

    if "Output (watts)" in workout_df.columns:
        sns.lineplot(
            x=workout_df["time (s)"],
            y=workout_df["Output (watts)"],
            ax=ax,
            color="#B5179E",
            linewidth=1.25
        )

    ax.axis("off")
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig