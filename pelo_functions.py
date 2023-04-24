import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
import config
import textwrap
import json

def get_ride_id(workout_id, s):
    workout = s.get("https://api.onepeloton.com/api/workout/" + workout_id)
    workout = json.loads(workout.content)
    return workout["ride"]["id"]

def get_ride_playlist(ride_id, s):
    ride = s.get("https://api.onepeloton.com/api/ride/{}/details".format(ride_id))
    ride = json.loads(ride.content)
    ride = pd.DataFrame.from_dict(ride["playlist"]["songs"])
    ride["artist"] = ride["artists"].apply(lambda x: ", ".join(pd.DataFrame.from_dict(x)["artist_name"].values))
    ride = ride[["title", "artist", "start_time_offset", "explicit_rating"]]
    ride.insert(0, "ride_id", ride_id)
    return ride

def get_target_metrics(target_metrics_data, s):
    target_metrics = pd.DataFrame.from_dict(target_metrics_data["target_metrics"])
    offsets = target_metrics["offsets"].apply(pd.Series)
    offsets = offsets - 60

    resistance_metrics = target_metrics["metrics"].apply(lambda x: pd.Series(x[0]))
    resistance_metrics = resistance_metrics.drop("name", axis=1)
    resistance_metrics.columns = [col + "_resistance" for col in resistance_metrics.columns]

    cadence_metrics = target_metrics["metrics"].apply(lambda x: pd.Series(x[1]))
    cadence_metrics = cadence_metrics.drop("name", axis=1)
    cadence_metrics.columns = [col + "_cadence" for col in cadence_metrics.columns]

    target_metrics = pd.concat([offsets, target_metrics["segment_type"], resistance_metrics, cadence_metrics], axis=1)
    target_metrics_df = pd.DataFrame(columns=target_metrics.drop(["start", "end"], axis=1).columns, 
                                     index=np.arange(np.min(target_metrics["start"]), np.max(target_metrics["end"])))

    for i in np.arange(len(target_metrics)):
        target_metrics_df.loc[target_metrics.loc[i, "start"]:target_metrics.loc[i, "end"],
                              target_metrics_df.columns] = target_metrics.iloc[i, 2:].values

    target_metrics_df = target_metrics_df.reset_index().rename(columns={"index": "time (s)"})
    return target_metrics_df

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
        #pd.DataFrame.from_dict(this_segment["subsegments_v2"][0]["movements"][0]["muscle_groups"])
    return complete_segments

def get_ride_details(ride_id, s):
    ride = s.get("https://api.onepeloton.com/api/ride/{}/details".format(ride_id))
    ride = json.loads(ride.content)
    ride_metrics = pd.concat([unpack_segments(ride["segments"]["segment_list"], s),
                              get_target_metrics(ride["target_metrics_data"], s)], axis=1)
    ride_playlist = get_ride_playlist(ride_id, s)
    ride_playlist["start_time_offset"] = ride_playlist["start_time_offset"] - 60
    ride_playlist["end_time"] = ride_playlist["start_time_offset"].shift(-1).fillna(len(ride_metrics)).astype(int)
    ride_playlist["length"] = ride_playlist["end_time"] - ride_playlist["start_time_offset"]
    ride_playlist = ride_playlist[ride_playlist["length"] > 0]
    ride_playlist = ride_playlist.loc[ride_playlist.index.repeat(ride_playlist.length)].reset_index(drop=True)
    ride_metrics = pd.concat([ride_metrics, ride_playlist[["title", "artist", "explicit_rating"]]], axis=1)
    ride_metrics["difficulty_estimate"] = ride["ride"]["difficulty_estimate"]
    total_expected_output = ride["target_metrics_data"]["total_expected_output"]
    ride_metrics["expected_upper_output"] = total_expected_output["expected_upper_output"]
    ride_metrics["expected_lower_output"] = total_expected_output["expected_lower_output"]
    avg = pd.concat([pd.DataFrame(ride["averages"],index=[0])] * len(ride_metrics)).reset_index(drop=True)
    ride_metrics = pd.concat([ride_metrics, avg], axis=1)
    ride_metrics = ride_metrics.loc[:,~ride_metrics.columns.duplicated()]
    ride_metrics["instructor"] = ride["ride"]["instructor"]["name"]
    ride_metrics["class_title"] = ride["ride"]["title"]
    return ride_metrics

def get_complete_workout(workout_id, ride_id, s):
    workout_df = pd.DataFrame()
    workout = s.get("https://api.onepeloton.com/api/workout/{}/performance_graph?every_n=1".format(workout_id))
    workout = json.loads(workout.content)
    workout_metrics = pd.DataFrame.from_dict(workout["metrics"])
    workout_metrics["metric"] = workout_metrics["display_name"] + " (" + workout_metrics["display_unit"] + ")"
    for i in np.arange(len(workout_metrics)):
        workout_df[workout_metrics["metric"][i]] = workout_metrics["values"][i]
    ride_metrics = get_ride_details(ride_id, s)
    return pd.concat([workout_df, ride_metrics], axis=1)

def plot_workout(workout_df, metric, overlay):
    metric_dict = {"Output (watts)": ("lower_output_est", "upper_output_est"),
                   "Resistance (%)": ("lower_resistance", "upper_resistance"),
                   "Cadence (rpm)": ("lower_cadence", "upper_cadence")}
    
    lower_metric, upper_metric = metric_dict[metric]
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    sns.lineplot(data=workout_df, x="time (s)", y=metric)
    sns.lineplot(data=workout_df, x="time (s)", y=upper_metric, color="blue", alpha=0.2)
    sns.lineplot(data=workout_df, x="time (s)", y=lower_metric, color="blue", alpha=0.2)

    ax.fill_between(workout_df["time (s)"].astype(int),
                    workout_df[upper_metric].astype(int),
                    workout_df[lower_metric].astype(int), alpha=0.1, color="blue");

    above = workout_df[workout_df[metric] > workout_df[upper_metric]]
    below = workout_df[workout_df[metric] < workout_df[lower_metric]]
    above_mask = np.zeros(len(workout_df))
    above_mask[above.index] = 1
    below_mask = np.zeros(len(workout_df))
    below_mask[below.index] = 1
    ax.fill_between(workout_df["time (s)"].astype(int),
                    workout_df[metric].astype(int),
                    workout_df[upper_metric].astype(int), where=above_mask, alpha=0.2, color="green");
    ax.fill_between(workout_df["time (s)"].astype(int),
                    workout_df[metric].astype(int),
                    workout_df[lower_metric].astype(int), where=below_mask, alpha=0.2, color="red");
    ax.set_title("{}: {}".format(workout_df["detailed_class_title"].values[0], metric));

    if overlay == "music":
        songs = workout_df.loc[workout_df["title"].shift(-1) != workout_df["title"]].reset_index()[["index", "title", "artist"]]
        songs["title"] = songs["title"].apply(lambda x: textwrap.fill(x,20)).str.replace("$", "S")
        songs["artist"] = songs["artist"].apply(lambda x: textwrap.fill(x,20)).str.replace("$", "S")
        songs["start"] = songs["index"].shift(1).fillna(0).astype(int)
        song_colors = dict(zip(np.unique(songs["title"]), 
                                  sns.color_palette("Set1", n_colors=len(np.unique(songs["title"])))))
        for i, row in songs.iterrows():
            ax.axvspan(xmin=row["start"], xmax=row["index"], facecolor=song_colors[row["title"]], alpha=0.15,
                       label=row["title"], zorder =-100, lw=0)
            plt.text((row["start"] + row["index"]) / 2, ax.get_ylim()[1] * 0.99, row["title"], ha='center', va='top',
                     size="small",wrap=True)
            plt.text((row["start"] + row["index"]) / 2, ax.get_ylim()[0] + ((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10) , row["artist"], ha='center', va='top',
                     size="small",wrap=True)
    if overlay == "segments":
        segments = workout_df["display_name"].loc[workout_df["display_name"].shift(-1) != workout_df["display_name"]].reset_index()
        segments["start"] = segments["index"].shift(1).fillna(0).astype(int)
        segment_colors = dict(zip(np.unique(segments["display_name"]), sns.color_palette("Set2")))
        for i, row in segments.iterrows():
            ax.axvspan(xmin=row["start"], xmax=row["index"], facecolor=segment_colors[row["display_name"]], alpha=0.15,
                       label=row["display_name"], zorder =-100, lw=0)
            plt.text((row["start"] + row["index"]) / 2, ax.get_ylim()[1] * 0.99, row["display_name"], ha='center', va='top')  