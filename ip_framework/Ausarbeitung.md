# Aufgabe 1.1

![Screenshot Aufgabe 1.1](./Screenshot_20260325_162749.png)

# Aufgabe 1.2

Die schnelle Variante ist deutlich schneller, weil sie vektorisierte NumPy-Operationen auf dem gesamten Bild nutzt, während die langsame Variante mit np.frompyfunc und einer Python-Funktion mehr Overhead erzeugt.
Größere Bilder haben mehr Pixel, weswegen mehr Berechnungen erforderlich sind, was die Laufzeit erhöht.
Kleine Unterschiede zwischen den Messungen sind normal und lassen sich durch Hintergrundprozesse, Cache-Effekte und allgemeine Systemschwankungen erklären.

Bild1: 512x512
Bild2: 1024x1024
Bild3: 2088x2088

