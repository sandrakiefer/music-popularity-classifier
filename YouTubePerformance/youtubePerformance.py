from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Daten aus CSV-Datei laden
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

# Daten für den Scatter Plot extrahieren
likes = df["Likes"]
comments = df["Comments"]
views = df["Views"]

# Normalisiere die Größe der Punkte, damit sie besser sichtbar sind
max_views = views.max()
min_views = views.min()
normalized_sizes = (views - min_views) / (max_views - min_views) * 100

# Scatter Plot erstellen
plt.scatter(likes, comments, s=normalized_sizes, alpha=0.5, c=views, cmap="plasma")

# Achsenbeschriftungen
plt.xlabel("Likes in millions")
plt.ylabel("Comments in millions")

# Titel und Farblegende (der Titel wird um 1.05 Einheiten nach oben verschoben)
plt.title("Relation between likes, comments and views on YouTube", y=1.05)

# Anpassung der Skala für bessere Lesbarkeit (optional)
plt.xscale("linear")  # Lineare Skala für die x-Achse (Likes)
plt.yscale("linear")  # Lineare Skala für die y-Achse (Kommentare)

# Ganzzahlige Ticker-Formatierung für die Achsen
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))


# Achsenticker-Formatierung für die Achsen in Millionen
def millions_formatter(x, pos):
    return "{:.1f}".format(x / 1e6)


# Achsenticker-Formatierung für die Achsen in Millionen
def billions_formatter(x, pos):
    return "{:.1f}".format(x / 1e9)


# Farblegende in Milliarden formatieren
colorbar = plt.colorbar(label="Views in billions")
colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(billions_formatter))

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

# Zeige den Plot
plt.tight_layout()
plt.show()
