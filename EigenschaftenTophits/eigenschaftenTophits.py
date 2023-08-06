from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import textwrap

df = pd.read_csv("data/Spotify_Youtube.csv")

# Entferne die URL-Spalten
url_cols = ["Url_spotify", "Uri", "Url_youtube", "Description"]
df.drop(url_cols, axis=1, inplace=True)

# Fülle fehlende Werte und entferne NaNs
df["Likes"] = df["Likes"].fillna(0)
df["Comments"] = df["Comments"].fillna(0)
df.dropna(inplace=True)

# Entferne Duplikate basierend auf Track
df.drop_duplicates(subset="Track", keep="first", inplace=True)

# Top 10 Songs basierend auf Views
top5_views = df.nlargest(5, "Views")

# Top 10 Songs basierend auf Streams
top5_streams = df.nlargest(5, "Stream")

# Erstes Fenster für das Diagramm der Top 10 Songs mit den meisten Views
plt.figure()
plt.barh(top5_views["Track"], top5_views["Views"])
plt.title("Top 5 songs on YouTube")
plt.xlabel("Views in billions")
plt.gca().invert_yaxis()

# Mehr Platz links in beiden Fenstern
plt.subplots_adjust(left=0.4)

# Zweites Fenster für das Diagramm der Top 10 Songs mit den meisten Streams
plt.figure()
# Y-Achsenbeschriftung auf zwei Zeilen aufteilen
wrapped_labels = [textwrap.fill(label, 20) for label in top5_streams["Track"]]
plt.barh(wrapped_labels, top5_streams["Stream"])
plt.title("Top 5 songs on Spotify")
plt.xlabel("Streams in billions")
plt.gca().invert_yaxis()

# Mehr Platz links in beiden Fenstern
plt.subplots_adjust(left=0.3)

# Daten der Radar Charts sammeln
top5_streams_prop = top5_streams[
    [
        "Track",
        "Energy",
        "Danceability",
        "Speechiness",
        "Valence",
        "Tempo",
        "Acousticness",
        "Liveness",
    ]
]
top5_views_prop = top5_views[
    [
        "Track",
        "Energy",
        "Danceability",
        "Speechiness",
        "Valence",
        "Tempo",
        "Acousticness",
        "Liveness",
    ]
]

# Skalieren des Tempo auf eine Skala von 0 bis 1
scaler = MinMaxScaler(feature_range=(0, 1))
top5_streams_prop["Tempo"] = scaler.fit_transform(top5_streams_prop[["Tempo"]])
top5_views_prop["Tempo"] = scaler.fit_transform(top5_views_prop[["Tempo"]])

# Erstellen des Radar Charts der Top 5 Spotify Streams
fig1 = plt.figure(figsize=(6, 8))
ax1 = fig1.add_subplot(111, polar=True)
theta = [
    "Energy",
    "Danceability",
    "Speechiness",
    "Valence",
    "Tempo",
    "Acousticness",
    "Liveness",
]

# Gleichmäßig verteilte Winkel für die Achsen
n = len(theta)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
values_streams = top5_streams_prop.values.tolist()  # Konvertiere zu einer Python-Liste
values_streams = [
    v[1:] for v in values_streams
]  # Extrahiere Werte, ohne den Tracknamen
values_streams += values_streams[:1]  # Schließe den Kreis

ax1.plot(angles, values_streams[0], label=top5_streams_prop.iloc[0]["Track"])
ax1.plot(angles, values_streams[1], label=top5_streams_prop.iloc[1]["Track"])
ax1.plot(angles, values_streams[2], label=top5_streams_prop.iloc[2]["Track"])
ax1.plot(angles, values_streams[3], label=top5_streams_prop.iloc[3]["Track"])
ax1.plot(angles, values_streams[4], label=top5_streams_prop.iloc[4]["Track"])
ax1.fill(angles, values_streams[0], alpha=0.3)
ax1.fill(angles, values_streams[1], alpha=0.3)
ax1.fill(angles, values_streams[2], alpha=0.3)
ax1.fill(angles, values_streams[3], alpha=0.3)
ax1.fill(angles, values_streams[4], alpha=0.3)

ax1.set_xticks(angles)
ax1.set_xticklabels(theta)

ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
ax1.set_ylim(0, 1)  # Festlegen der y-Achsen-Skala auf 0 bis 1
plt.title("Different properties for top 5 songs based on Spotify streams", y=1.1)
plt.tight_layout()

# Erstellen des Radar Charts der Top 5 YouTube Views
fig2 = plt.figure(figsize=(6, 8))
ax2 = fig2.add_subplot(111, polar=True)

# Gleichmäßig verteilte Winkel für die Achsen
values_views = top5_views_prop.values.tolist()  # Konvertiere zu einer Python-Liste
values_views = [v[1:] for v in values_views]  # Extrahiere Werte, ohne den Tracknamen
values_views += values_views[:1]  # Schließe den Kreis

ax2.plot(angles, values_views[0], label=top5_views_prop.iloc[0]["Track"])
ax2.plot(angles, values_views[1], label=top5_views_prop.iloc[1]["Track"])
ax2.plot(angles, values_views[2], label=top5_views_prop.iloc[2]["Track"])
ax2.plot(angles, values_views[3], label=top5_views_prop.iloc[3]["Track"])
ax2.plot(angles, values_views[4], label=top5_views_prop.iloc[4]["Track"])
ax2.fill(angles, values_views[0], alpha=0.3)
ax2.fill(angles, values_views[1], alpha=0.3)
ax2.fill(angles, values_views[2], alpha=0.3)
ax2.fill(angles, values_views[3], alpha=0.3)
ax2.fill(angles, values_views[4], alpha=0.3)

ax2.set_xticks(angles)
ax2.set_xticklabels(theta)

ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
ax2.set_ylim(0, 1)  # Festlegen der y-Achsen-Skala auf 0 bis 1
plt.title("Different properties for top 5 songs based on YouTube views", y=1.1)

plt.tight_layout()
plt.show()
