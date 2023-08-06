# Performance verschiedener Lieder auf Spotify und YouTube

Hochschule RheinMain - Informatik (Master of Science) - Collective Intelligence - Sommersemester 2023 <br>
Gruppenmitglieder: **Sebastian Braun und Sandra Kiefer**

## Installation und Startanweisungen

```sh
# Installation der benötigten Packete
$ pip install -r requirements.txt

# Ausführen des Skriptes zur Betrachtung der Eigenschaften der Performance
$ python3 EigenschaftenPerformance/eigenschaftenPerformance.py

# Ausführen des Skriptes zur Betrachtung der Eigenschaften der Tophits
$ python3 EigenschaftenTophits/eigenschaftenTophits.py

# Ausführen des Skriptes zur Betrachtung des Vergleichs der Performance
$ python3 PerformanceVergleich/performance.py

# Ausführen des Skriptes zur Anwendung verschiedener Modelle zur Vorhersage der Performance
$ python3 PredictPerformance/PredictPerformance.py
```

Das Projekt wurde im Rahmen der Lehrveranstaltung Collective Intelligence realisiert. Es umfasst verschiedene Skripte mit unterschiedlichen Themenschwerpunkten in Unterordnern. Die Skripte sind in Python geschrieben und können auch damit ausgeführt werden. Die Skripte sind in der Lage, die benötigten Daten aus der CSV-Datei zu laden und die Ergebnisse in geeigneten Visualisierungen darzustellen. Diese Ergebnisse sind in den entsprechenden Unterordnern "/plots" abgelegt.

Der behandelte Datensatz umfasst die jeweils 10 populärsten Lieder eines Künstlers auf Spotify und YouTube. Des Weiteren beinhaltet der Datensatz verschiedene Eigenschaften der Lieder, wie zum Beispiel die Tonart, die Geschwindigkeit, die Tanzbarkeit, die Energie, die Lautstärke, die Instrumentalität, die Redeanteile und die Akustik. Die entsprechende CSV-Datei ist zu finden unter dem Pfad "/data/Spotify_YouTube.csv".

Inhaltlich wird sich mit der Explorative Datenanalyse (EDA) und dem Trainieren verschiedener Modelle zum Vorhersagen der Performance (bezogen auf Views, Stream, Likes und Kommentare) eines Liedes.

Die ausführliche Dokumentation des Projektes ist zu finden unter dem Pfad "/documentation/Projektbericht.pdf".
