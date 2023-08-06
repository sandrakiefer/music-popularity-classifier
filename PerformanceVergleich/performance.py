from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Breite der Balken
bar_width = 0.35

# Erstelle den Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Position der Balken auf der x-Achse für top5_views
r1_views = np.arange(len(top5_views))
# Position der Balken auf der x-Achse für top5_streams
r1_streams = [x + bar_width for x in r1_views]

# Plot für top5_views
ax1.bar(r1_views, top5_views["Views"], color="blue", width=bar_width, label="YouTube")
ax1.bar(
    r1_streams, top5_views["Stream"], color="green", width=bar_width, label="Spotify"
)

# Zahlenwerte über den Balken anzeigen (in Millionen)
for i, (views, streams) in enumerate(zip(top5_views["Views"], top5_views["Stream"])):
    ax1.text(i, views, f"{views/1000000:.1f}M", ha="center", va="bottom", fontsize=8)
    ax1.text(
        i + bar_width,
        streams,
        f"{streams/1000000:.1f}M",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# X-Achse anpassen
ax1.set_xticks(r1_views + bar_width / 2)
ax1.set_xticklabels(top5_views["Track"])
ax1.tick_params(axis="x", rotation=45, labelsize=8)

# Y-Achse anpassen
ax1.set_ylabel("Streams in millions")
ax1.set_title("Top 5 YouTube Songs: Views vs. Streams")
ax1.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: "{:,.1f}".format(x / 1000000))
)

# Legende anzeigen
ax1.legend()

# Position der Balken auf der x-Achse für top5_streams
r2_streams = np.arange(len(top5_streams))
# Position der Balken auf der x-Achse für top5_views
r2_views = [x + bar_width for x in r2_streams]

# Plot für top5_streams
ax2.bar(
    r2_streams, top5_streams["Stream"], color="green", width=bar_width, label="Spotify"
)
ax2.bar(r2_views, top5_streams["Views"], color="blue", width=bar_width, label="YouTube")

# Zahlenwerte über den Balken anzeigen (in Millionen)
for i, (streams, views) in enumerate(
    zip(top5_streams["Stream"], top5_streams["Views"])
):
    ax2.text(
        i, streams, f"{streams/1000000:.1f}M", ha="center", va="bottom", fontsize=8
    )
    ax2.text(
        i + bar_width,
        views,
        f"{views/1000000:.1f}M",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# X-Achse anpassen
ax2.set_xticks(r2_streams + bar_width / 2)
ax2.set_xticklabels(top5_streams["Track"])
ax2.tick_params(axis="x", rotation=45, labelsize=8)

# Y-Achse anpassen
ax2.set_ylabel("Streams in millions")
ax2.set_title("Top 5 Spotify Songs: Streams vs. Views")
ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: "{:,.1f}".format(x / 1000000))
)

# Legende anzeigen
ax2.legend()

# Layout anpassen und Plot anzeigen
plt.tight_layout()
plt.show()
