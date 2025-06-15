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
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO


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

def generate_share_card(workout_df):
    # Set up base image
    card_width, card_height = 1600, 900
    card = Image.new("RGB", (card_width, card_height), "white")
    draw = ImageDraw.Draw(card)

    # Plot the chart
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.lineplot(data=workout_df, x="time (s)", y="Output (watts)", ax=ax, color="black", label="Your Output")

    # Zone shading
    power_zones = get_power_zones(config.FTP)
    zone_colours = {
        "Zone 1": "#E0F7FA",
        "Zone 2": "#B2EBF2",
        "Zone 3": "#4DD0E1",
        "Zone 4": "#26C6DA",
        "Zone 5": "#00BCD4",
        "Zone 6": "#00ACC1",
        "Zone 7": "#0097A7"
    }

    if "target_zone" in workout_df.columns:
        for zone, (low, high) in power_zones.items():
            mask = workout_df["target_zone"] == zone
            ax.fill_between(workout_df["time (s)"], low, high, where=mask, 
                            color=zone_colours.get(zone, "grey"), alpha=0.3)

    # Music overlays
    if "title" in workout_df.columns and "artist" in workout_df.columns:
        songs = workout_df.loc[workout_df["title"].shift(-1) != workout_df["title"]].reset_index()[["index", "title", "artist"]]
        songs["start"] = songs["index"].shift(1).fillna(0).astype(int)
        alt_colours = ["#fce4ec", "#f3e5f5"]
        for i, (_, row) in enumerate(songs.iterrows()):
            ax.axvspan(row["start"], row["index"], alpha=0.1, color=alt_colours[i % 2])

    ax.set_ylim(0, min(1.2 * config.FTP, 600))
    ax.set_title("")
    ax.set_ylabel("Output (watts)")
    ax.set_xlabel("")

    ax.get_legend().remove()
    fig.subplots_adjust(top=0.85, right=0.95, bottom=0.2)

    buf = BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)

    chart_img = Image.open(buf)
    card.paste(chart_img, (100, 200))

    # Draw class title
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
    title_text = workout_df["class_title"].iloc[0]
    draw.text((100, 50), textwrap.fill(title_text, 40), fill="black", font=font)

    # Draw average/total output
    avg_output = round(
        workout_df["avg_watts"].mean() if "avg_watts" in workout_df.columns else
        workout_df["Average Output (watts)"].mean() if "Average Output (watts)" in workout_df.columns else
        0, 1
    )

    total_output = round(
        workout_df["total_work"].mean() if "total_work" in workout_df.columns else
        workout_df["Total Output (kJ)"].mean() if "Total Output (kJ)" in workout_df.columns else
        0, 1
    )

    sub_font = ImageFont.truetype("DejaVuSans.ttf", 40)
    draw.text((100, card_height - 120), f"Avg Output: {avg_output} watts", fill="black", font=sub_font)
    draw.text((100, card_height - 60), f"Total Output: {total_output} kJ", fill="black", font=sub_font)

    return card