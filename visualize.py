import pandas as pd
import matplotlib.pyplot as plt

messungen = {
    ("512", "Slow"): [2800, 2750, 2820],
    ("512", "Fast"): [12, 11, 13],
    ("1024", "Slow"): [11000, 11250, 10980],
    ("1024", "Fast"): [45, 47, 44],
    ("2048", "Slow"): [44000, 43800, 44500],
    ("2048", "Fast"): [180, 175, 182],
}

df = pd.DataFrame(
    [(bild, methode, zeit) for (bild, methode), werte in messungen.items() for zeit in werte],
    columns=["Bild", "Methode", "Zeit"]
)

stats = df.groupby(["Bild", "Methode"])["Zeit"].mean().unstack()
stats.plot.bar()
plt.ylabel("Laufzeit [ms]")

print(df.groupby(["Bild", "Methode"])["Zeit"].agg(["mean", "std"]))

plt.show()


# Die schnelle Variante ist deutlich schneller, weil sie vektorisierte NumPy-Operationen auf dem gesamten Bild nutzt, während die langsame Variante mit np.frompyfunc und einer Python-Funktion mehr Overhead erzeugt.
# Mit zunehmender Bildgröße steigt die Laufzeit beider Methoden, da mehr Pixel verarbeitet werden müssen und auf jedes Pixel dieselbe Rechenvorschrift angewendet wird.
# Kleine Unterschiede zwischen den Messungen sind normal und lassen sich durch Hintergrundprozesse, Cache-Effekte und allgemeine Systemschwankungen erklären.
