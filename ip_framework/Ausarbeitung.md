# Aufgabe 1.1

![Screenshot Aufgabe 1.1](./Screenshot_20260325_162749.png)

# Aufgabe 1.2

Die schnelle Variante ist deutlich schneller, weil sie vektorisierte NumPy-Operationen auf dem gesamten Bild nutzt, während die langsame Variante mit np.frompyfunc und einer Python-Funktion mehr Overhead erzeugt.
Größere Bilder haben mehr Pixel, weswegen mehr Berechnungen erforderlich sind, was die Laufzeit erhöht.
Kleine Unterschiede zwischen den Messungen sind normal und lassen sich durch Hintergrundprozesse, Cache-Effekte und allgemeine Systemschwankungen erklären.

Es wurden die folgenden drei Bildgrößen verwendet:
- `512 × 512 Pixel`
- `1024 × 1024 Pixel`
- `2088 × 2088 Pixel`

## Balkendiagramm der Laufzeiten:

![Screenshot Aufgabe 1.2](./Diagramm_1.2.png)

## Tabelle mit Mittelwerten und Standardabweichungen:

| Bild | Methode  | mean         | std          |
|------|----------|--------------|--------------|
| 1024 | Fast     | 16.833333    | 1.001665     |
| 1024 | Slow     | 13329.166667 | 1154.011925  |
| 2088 | Fast     | 65.366667    | 2.532456     |
| 2088 | Slow     | 61261.900000 | 9735.530391  |
| 512  | Fast     | 4.266667     | 0.378594     |
| 512  | Slow     | 2664.833333  | 438.047045   |

# Aufgabe 3

## Balkendiagramm der Laufzeiten

![Screenshot Aufgabe 1.3](Diagramm_1.3.png)

## Tabelle mit Mittelwerten und Standardabweichungen

| Methode | mean         | std         |
|---------|--------------|-------------|
| Fast    | 11.633333    | 5.858612    |
| Slow    | 25860.400000 | 4974.656327 |

# Aufgabe 4

## Morphologische Operatoren

- ('kernel1', 'erosion'): 7.8,
- ('kernel1', 'dilation'): 8.0,
- ('kernel1', 'closing'): 15.8,
- ('kernel1', 'opening'): 13.8,
- ('kernel2', 'erosion'): 9.8,
- ('kernel2', 'dilation'): 8.0,
- ('kernel2', 'closing'): 13.8,
- ('kernel2', 'opening'): 13.8,
- ('kernel3', 'erosion'): 7.5,
- ('kernel3', 'dilation'): 8.0,
- ('kernel3', 'closing'): 13.8,
- ('kernel3', 'opening'): 15.8,

## Median-Filter

Resultat des Medianfilters auf Salt.jpg: Das Rauschen verschwindet, jedoch ist das Bild danach verschwommener.

# Aufgabe 1.5 – Semantic Segmentation

Für die semantische Segmentierung wurden zwei vortrainierte Modelle aus `torchvision` eingebunden:

- `fcn_resnet101`
- `deeplabv3_resnet101`

Beide Modelle sind auf Basis von ResNet101 trainiert und liefern für jedes Pixel eine Klassenwahrscheinlichkeit. Die Ausgabe wird anschließend per `argmax` in eine Klassenkarte umgewandelt. Aus dieser Klassenkarte wird eine farbige Maske erzeugt, sodass jede Klasse eine eigene RGB-Farbe erhält. Klasse `0` ist dabei schwarz und steht für den Hintergrund.

## Implementierung

In `SemanticSegmentationStep.load_model(...)` wird das gewünschte Modell anhand des Strings `model` geladen und direkt mit `model.eval()` in den Evaluationsmodus gesetzt. Das Modell wird in `self.models` zwischengespeichert, damit es beim nächsten Aufruf nicht erneut geladen werden muss. Dieses Verhalten nennt man *lazy loading*.

In `apply(...)` wird das Eingabebild zuerst auf RGB reduziert, als `uint8` kopiert und dann mit `transforms.ToTensor()` in einen PyTorch-Tensor umgewandelt. Danach wird noch eine Batch-Dimension ergänzt, weil PyTorch-Modelle Bilder typischerweise in der Form `(Batch, Channels, Height, Width)` erwarten.

Die Segmentierungsmaske wird anschließend mit einer festen VOC-Farbpalette visualisiert. Dadurch sind die Klassen direkt sichtbar, ohne dass man die Zahlenwerte der Maske interpretieren muss.

## Pipelines

Es wurden zwei neue Pipelines angelegt:

1. `Semantic Segmentation (Load File)` – lädt ein Bild von der Festplatte und segmentiert es.
2. `Semantic Segmentation (Camera)` – liest mehrere Kamerabilder kurz hintereinander ein und segmentiert das letzte verfügbare Bild.

Die Kamera-Klasse wurde dafür angepasst, dass nicht nur ein einzelner Frame verwendet wird. Stattdessen werden mehrere Frames mit kurzer Pause aufgenommen. Das ist hilfreich, wenn das erste Kamerabild noch dunkel oder unscharf ist.

## Test und Beobachtung

### 1) Pipeline `Semantic Segmentation (Load File)`

Es wurden zwei Bilder getestet:

- `img/street.png`
- `img/lake.png`

Zusätzlich wurden beide Modellvarianten getestet (`fcn` und `deeplab`).

Beobachtung:

- Bei `street.png` wurden in beiden Modellen mehrere Klassenbereiche erkannt (in der Stichprobe mehrere Farben sichtbar, typischerweise Hintergrund + Objektklassen).
- Bei `lake.png` war die Maske in der Stichprobe fast einfarbig (vorwiegend Hintergrund-/eine dominante Klasse), was bei solchen Szenen ohne klar abgegrenzte VOC-Objekte plausibel ist.

### 2) Pipeline `Semantic Segmentation (Camera)`

Die Kamera-Pipeline wurde zweimal hintereinander ausgeführt (jeweils neues Kamerabild):

- Lauf 1: Eingabebild `480x640x3`, Ausgabemaske `480x640x3`
- Lauf 2: Eingabebild `480x640x3`, Ausgabemaske `480x640x3`

In beiden Läufen waren mehrere Klassenfarben in der Maske sichtbar (in der Stichprobe jeweils zwei dominante Farben). Durch `num_frames` und `delay_seconds` wird nicht nur ein einzelner Frame genutzt, sondern ein späterer, stabilerer Frame zurückgegeben. Dadurch werden dunkle oder instabile Erstframes reduziert.

## Erklärung in einem Satz

Die Pipeline lädt ein vortrainiertes Segmentierungsmodell, wandelt das Bild in einen Tensor um, berechnet für jedes Pixel eine Klasse und färbt diese Klassen anschließend sichtbar ein.

