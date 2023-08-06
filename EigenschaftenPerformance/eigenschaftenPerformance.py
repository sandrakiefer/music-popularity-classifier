from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(path):
    df = pd.read_csv(path, index_col=0)
    df.dropna(axis=0, inplace=True)
    df.duplicated().sum()
    return df


df = read_data("data/Spotify_Youtube.csv")

POPULARITY_LABELS = [
    "Low Popularity",
    "Moderate Popularity",
    "Good Popularity",
    "High Popularity",
    "Very High Popularity",
]


def calculate_popularity(df):
    # Berechne normalisierten Popularitätsscore
    normalized_likes = (df["Likes"] - df["Likes"].min()) / (
        df["Likes"].max() - df["Likes"].min()
    )
    normalized_views = (df["Views"] - df["Views"].min()) / (
        df["Views"].max() - df["Views"].min()
    )
    normalized_comments = (df["Comments"] - df["Comments"].min()) / (
        df["Comments"].max() - df["Comments"].min()
    )
    normalized_streams = (df["Stream"] - df["Stream"].min()) / (
        df["Stream"].max() - df["Stream"].min()
    )

    # Zuweisung der Gewichte
    popularity_score = (
        (normalized_views * 0.3)
        + (normalized_streams * 0.3)
        + (normalized_likes * 0.2)
        + (normalized_comments * 0.2)
    )

    # Definieren des Thresholds
    popularity_thresholds = np.percentile(popularity_score, [0, 30, 50, 80, 90])

    # Popularitätsklassen anhand des Scores zuweisen
    popularity = np.select(
        [
            popularity_score <= popularity_thresholds[1],
            popularity_score <= popularity_thresholds[2],
            popularity_score <= popularity_thresholds[3],
            popularity_score <= popularity_thresholds[4],
            popularity_score > popularity_thresholds[4],
        ],
        POPULARITY_LABELS,
        default=POPULARITY_LABELS[0],
    )

    df["Popularity Score"] = popularity_score
    df["Popularity"] = popularity
    return df


# Rufe die Funktion calculate_popularity(df) auf und speichere das zurückgegebene DataFrame
result_df = calculate_popularity(df)

# Loop über die Eigenschaften und Erstellung der Box-Whisker-Plots pro Eigenschaft
properties = [
    "Energy",
    "Danceability",
    "Speechiness",
    "Valence",
    "Tempo",
    "Loudness",
    "Acousticness",
    "Liveness",
    "Instrumentalness",
]


def create_box_plots(df, popularity_labels, property_name):
    fig, axes = plt.subplots(ncols=len(popularity_labels), figsize=(15, 8))
    fig.suptitle(
        f"Box-Whisker plots for '{property_name}' ordered by popularity", fontsize=16
    )

    # Finde Minimum und Maximum der Eigenschaften in allen Popularitätsklassen
    min_val = df[property_name].min()
    max_val = df[property_name].max()

    for i, popularity_label in enumerate(popularity_labels):
        property_values = df[df["Popularity"] == popularity_label][property_name]
        median_val = property_values.median()
        axes[i].boxplot(x=property_values)
        axes[i].set_xlabel(property_name)
        axes[i].set_xticklabels([])  # Entferne x-Achsenticklabels
        axes[i].set_ylabel("")  # Entferne y-Achsentitel
        axes[i].set_title(popularity_label, pad=20)  # Setze Titel oberhalb des Plots
        axes[i].axhline(
            y=median_val, color="red", linestyle="--"
        )  # Verbinde Medianwerte mit roter Linie

        # Setze die gleiche y-Achsen-Skala für alle Diagramme
        axes[i].set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.show()


for property_name in properties:
    create_box_plots(result_df, POPULARITY_LABELS, property_name)
