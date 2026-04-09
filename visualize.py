import pandas as pd
import matplotlib.pyplot as plt

messungen = {
    ("512", "Slow"): [24223.7, 31447.2, 21910.3],
    ("512", "Fast"): [16.1, 13.8, 5.0],
    #("1024", "Slow"): [13032.0, 12352.8, 14602.7],
    #("1024", "Fast"): [15.7, 17.6, 17.2],
    #("2088", "Slow"): [51667.2, 60986.1, 71132.4],
    #("2088", "Fast"): [64.9, 63.1, 68.1],
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
