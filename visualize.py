import pandas as pd
import matplotlib.pyplot as plt

messungen = {
    ("512", "Slow"): [2369.3, 3168.1, 2457.1],
    ("512", "Fast"): [4.0, 4.1, 4.7],
    ("1024", "Slow"): [13032.0, 12352.8, 14602.7],
    ("1024", "Fast"): [15.7, 17.6, 17.2],
    ("2088", "Slow"): [51667.2, 60986.1, 71132.4],
    ("2088", "Fast"): [64.9, 63.1, 68.1],
}

df = pd.DataFrame(
    [(bild, methode, zeit) for (bild, methode), werte in messungen.items() for zeit in werte],
    columns=["Bild", "Methode", "Zeit"]
)

stats = df.groupby(["Bild", "Methode"])["Zeit"].mean().unstack()
stats.plot.bar()
plt.ylabel("Laufzeit [ms] (log scale)")
plt.yscale("log") 

print(df.groupby(["Bild", "Methode"])["Zeit"].agg(["mean", "std"]))

plt.show()
