import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
import config
import textwrap
import json

from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO


def get_ride_id(workout_id, s):
    workout = s.get("https://api.onepeloton.com/api/workout/" + workout_id)
    workout = json.loads(workout.content)
    return workout["ride"]["id"]

def get_recent_workouts(session, limit=50):
    from time import sleep
    r = session.get(f"https://api.onepeloton.com/api/user/{config.USER_ID}/workouts?limit={limit}&page=0")
    workouts = json.loads(r.content)["data"]

    results = []
    for w in workouts:
        workout_id = w["id"]
        details = session.get(f"https://api.onepeloton.com/api/workout/{workout_id}")
        if details.status_code != 200:
            continue
        data = json.loads(details.content)

        # 🚫 Skip if no pedaling metrics
        if not data.get("has_pedaling_metrics", False):
            continue


        ride = data.get("ride", {})
        title = ride.get("title", "Unnamed Class")

        instructor = "Unknown"
        if ride.get("instructor") and isinstance(ride["instructor"], dict):
            instructor = ride["instructor"].get("name", "Unknown")

        start = pd.to_datetime(data.get("start_time", 0), unit="s").strftime("%Y-%m-%d")

        results.append({
            "id": workout_id,
            "name": title,
            "instructor": instructor,
            "start": start,
        })

        sleep(0.1)

    return results


def get_ride_playlist(ride_id, s):
    ride = s.get(f"https://api.onepeloton.com/api/ride/{ride_id}/details")
    ride = json.loads(ride.content)
    ride = pd.DataFrame.from_dict(ride["playlist"]["songs"])
    ride["artist"] = ride["artists"].apply(lambda x: ", ".join(pd.DataFrame.from_dict(x)["artist_name"].values))
    ride = ride[["title", "artist", "start_time_offset", "explicit_rating"]]
    ride.insert(0, "ride_id", ride_id)
    return ride

def get_target_metrics(target_metrics_data, s):
    target_metrics = pd.DataFrame.from_dict(target_metrics_data["target_metrics"])
    offsets = target_metrics["offsets"].apply(pd.Series)
    offsets = offsets - 60  # offset fix

    # Handle resistance
    resistance_metrics = target_metrics["metrics"].apply(
        lambda x: pd.Series(x[0]) if len(x) > 0 and isinstance(x[0], dict) else pd.Series()
    ).drop(columns=["name"], errors="ignore").add_suffix("_resistance")

    # Handle cadence
    cadence_metrics = target_metrics["metrics"].apply(
        lambda x: pd.Series(x[1]) if len(x) > 1 and isinstance(x[1], dict) else pd.Series()
    ).drop(columns=["name"], errors="ignore").add_suffix("_cadence")

    # Combine with proper safety checks
    components = [offsets, target_metrics["segment_type"]]
    if not resistance_metrics.empty:
        components.append(resistance_metrics)
    if not cadence_metrics.empty:
        components.append(cadence_metrics)

    if not components:
        return pd.DataFrame(columns=["time (s)"])  # Return safe fallback

    target_metrics = pd.concat(components, axis=1)
    if "start" not in target_metrics or "end" not in target_metrics:
        return pd.DataFrame(columns=["time (s)"])  # Defensive fallback

    try:
        target_metrics_df = pd.DataFrame(columns=target_metrics.drop(["start", "end"], axis=1).columns,
                                         index=np.arange(np.min(target_metrics["start"]), np.max(target_metrics["end"])))
        for i in np.arange(len(target_metrics)):
            target_metrics_df.loc[
                target_metrics.loc[i, "start"]:target_metrics.loc[i, "end"],
                target_metrics_df.columns
            ] = target_metrics.iloc[i, 2:].values
        target_metrics_df = target_metrics_df.reset_index().rename(columns={"index": "time (s)"})
        return target_metrics_df
    except Exception as e:
        print("Error in target_metrics parsing:", e)
        return pd.DataFrame(columns=["time (s)"])



def unpack_segments(segments, s):
    segments_df = pd.DataFrame.from_dict(segments)
    length_of_ride = np.sum(segments_df["length"])
    
    complete_segments = pd.DataFrame(columns=["time (s)", "intensity_in_mets", "name"])
    complete_segments["time (s)"] = np.arange(length_of_ride)

    for i in np.arange(len(segments_df)):
        this_segment = segments_df.iloc[i]
        segment_duration = np.arange(this_segment["start_time_offset"],
                                     this_segment["start_time_offset"] + this_segment["length"])
        complete_segments.loc[segment_duration, "time (s)"] = segment_duration
        complete_segments.loc[segment_duration, "intensity_in_mets"] = this_segment["intensity_in_mets"]
        complete_segments.loc[segment_duration, "name"] = this_segment["name"]
        try:
            subsegments = pd.DataFrame.from_dict(this_segment["subsegments_v2"])
            subsegments = subsegments.loc[subsegments.index.repeat(subsegments.length)].reset_index(drop=True)
            complete_segments.loc[segment_duration, "display_name"] = subsegments["display_name"].values
        except:
            complete_segments.loc[segment_duration, "display_name"] = this_segment["name"]
    return complete_segments

def get_ride_details(ride_id, s):
    ride = s.get(f"https://api.onepeloton.com/api/ride/{ride_id}/details")
    ride = json.loads(ride.content)

    # Segment + Target Metrics
    ride_metrics = pd.concat([
        unpack_segments(ride["segments"]["segment_list"], s),
        get_target_metrics(ride["target_metrics_data"], s)
    ], axis=1)

    # Playlist (music)
    ride_playlist = get_ride_playlist(ride_id, s)
    ride_playlist["start_time_offset"] = ride_playlist["start_time_offset"] - 60
    ride_playlist["end_time"] = ride_playlist["start_time_offset"].shift(-1).fillna(len(ride_metrics)).astype(int)
    ride_playlist["length"] = ride_playlist["end_time"] - ride_playlist["start_time_offset"]
    ride_playlist = ride_playlist[ride_playlist["length"] > 0]
    ride_playlist = ride_playlist.loc[ride_playlist.index.repeat(ride_playlist.length)].reset_index(drop=True)
    ride_metrics = pd.concat([ride_metrics, ride_playlist[["title", "artist", "explicit_rating"]]], axis=1)

    # Expected Output Ranges
    total_expected_output = ride["target_metrics_data"].get("total_expected_output", {})
    ride_metrics["expected_upper_output"] = total_expected_output.get("expected_upper_output")
    ride_metrics["expected_lower_output"] = total_expected_output.get("expected_lower_output")

    # Averages across all rows
    avg_df = pd.DataFrame([ride["averages"]] * len(ride_metrics)).reset_index(drop=True)
    ride_metrics = pd.concat([ride_metrics, avg_df], axis=1)

    # Broadcast scalar metadata
    instructor_name = ride["ride"]["instructor"]["name"]
    class_title = ride["ride"]["title"]
    class_type_id = ride["ride"]["id"]  # or class_type_ids if you prefer

    ride_metrics["instructor"] = instructor_name
    ride_metrics["class_title"] = class_title
    ride_metrics["class_type_ids"] = [class_type_id] * len(ride_metrics)

    # Final clean-up
    ride_metrics = ride_metrics.loc[:, ~ride_metrics.columns.duplicated()]

    return ride_metrics


def get_complete_workout(workout_id, ride_id, s):
    # Fetch and structure the user's performance graph
    workout_response = s.get(f"https://api.onepeloton.com/api/workout/{workout_id}/performance_graph?every_n=1")
    workout = json.loads(workout_response.content)
    metrics = pd.DataFrame.from_dict(workout["metrics"])

    workout_df = pd.DataFrame()
    metrics["metric"] = metrics["display_name"] + " (" + metrics["display_unit"] + ")"

    for _, row in metrics.iterrows():
        raw_vals = row["values"]
        # Flatten any nested lists in raw_vals
        cleaned_vals = [
            np.nanmean(v) if isinstance(v, list) or isinstance(v, np.ndarray) else v
            for v in raw_vals
        ]
        workout_df[row["metric"]] = cleaned_vals

    # Fetch ride details (segments, targets, music, etc.)
    ride_metrics = get_ride_details(ride_id, s)

    # Combine performance data with ride metadata
    complete_df = pd.concat([workout_df, ride_metrics], axis=1)
    return complete_df



def generate_shareable_card(workout_df):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor('#1c1e26')  # Dark background

    # Output plot
    ax.plot(workout_df["time (s)"], workout_df["Output (watts)"], color="#E91E63", linewidth=1.5)
    ax.set_facecolor('#2d2f3a')
    ax.set_xlim(0, workout_df["time (s)"].max())
    ax.set_ylim(0, workout_df["Output (watts)"].max() * 1.1)
    ax.axis('off')

    # Overlay text elements
    title = workout_df["class_title"].iloc[0]
    instructor = workout_df["instructor"].iloc[0]
    avg_output = int(workout_df["Output (watts)"].mean())
    total_output = int(workout_df["total_work"]["mean()"].iloc[0]) if "total_work" in workout_df else 0
    date = pd.to_datetime(workout_df["time (s)"].index[0], unit='s').strftime('%Y-%m-%d')

    # Annotation box
    ax.text(0.01, 1.15, f"{title}", transform=ax.transAxes, fontsize=10, fontweight='bold', color='white')
    ax.text(0.01, 1.05, f"Instructor: {instructor}", transform=ax.transAxes, fontsize=8, color='lightgray')
    ax.text(0.01, -0.15, f"Avg Output: {avg_output} W     Total Output: {total_output} kj", transform=ax.transAxes, fontsize=9, color='white')
    ax.text(0.99, -0.15, f"{date}", transform=ax.transAxes, fontsize=8, color='gray', ha='right')

    plt.subplots_adjust(top=0.85, bottom=0.25, left=0.05, right=0.95)

    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)