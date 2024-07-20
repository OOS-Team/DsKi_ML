# Dokumentation für heart_attack und heart_failure

## Info
Für  die Präsentation wurde hauptsächlich das heart_failure Datenset verwendet.

## Links zu Daten
- Heart Failure Prediction Dataset  <br>
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
<br>
- Heart Attack Analysis & Prediction Dataset (mit eindeutigem Selection Bias) <br>

https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

## Erklärung der Features

### Auflistung der Attribute heart_failure

1. **Alter (Age)**: Alter der Person in Jahren.
2. **Geschlecht (Sex)**: Geschlecht der Person. [M: Männlich, F: Weiblich]
3. **Brustschmerztyp (ChestPainType)**: Art des Brustschmerzes. [TA: Typische Angina, ATA: Atypische Angina, NAP: Nicht-anginöser Schmerz, ASY: Asymptomatisch]
4. **Ruhender Blutdruck (RestingBP)**: Ruhender Blutdruck in mm Hg.
5. **Cholesterin (Cholesterol)**: Cholesterinspiegel in mg/dl.
6. **Nüchternblutzucker (FastingBS)**: Nüchternblutzucker über 120 mg/dl [1: Ja, 0: Nein].
7. **Ruhe-EKG-Ergebnisse (RestingECG)**: Ergebnisse des Ruhe-EKGs. [Normal: Normal, ST: ST-T-Wellen-Anomalie, LVH: Linksventrikuläre Hypertrophie]
8. **Maximale Herzfrequenz (MaxHR)**: Höchste erreichte Herzfrequenz.
9. **Belastungsinduzierte Angina (ExerciseAngina)**: Angina ausgelöst durch körperliche Anstrengung [Y: Ja, N: Nein].
10. **Oldpeak**: ST-Depression gemessen nach Anstrengung.
11. **ST-Steigung (ST_Slope)**: Steigung des ST-Segments bei Belastungs-EKG. [Up: Aufsteigend, Flat: Flach, Down: Absteigend]
12. **Herzerkrankung (HeartDisease)**: Vorhandensein einer Herzerkrankung [1: Herzerkrankung, 0: Normal].


### Auflistung der Attribute heart_attack


1. **Alter (age)**: Das Alter der Person in Jahren.

2. **Geschlecht (sex)**: Das biologische Geschlecht der Person. Typischerweise 0 für weiblich und 1 für männlich.

3. **Brustschmerztyp (cp)**: Art des Brustschmerzes, den die Person erlebt. Werte reichen von 0 bis 3, wobei 0 für typische Angina, 1 für atypische Angina, 2 für nicht-anginösen Schmerz und 3 für asymptomatische Fälle steht.

4. **Ruhender Blutdruck (trtbps)**: Der Blutdruck der Person in Ruhe, gemessen in Millimeter Quecksilbersäule (mm Hg).

5. **Cholesterin (chol)**: Der Cholesterinspiegel der Person in mg/dl, ermittelt durch einen BMI-Sensor.

6. **Nüchternblutzucker (fbs)**: Zeigt an, ob der Nüchternblutzucker der Person über 120 mg/dl liegt. 1 steht für Ja (wahr), 0 für Nein (falsch).

7. **Ruhe-EKG-Ergebnisse (restecg)**: Ergebnisse des Ruhe-Elektrokardiogramms. 0 steht für normal, 1 für ST-T-Wellen-Anomalien, 2 für linksventrikuläre Hypertrophie.

8. **Maximale Herzfrequenz (thalachh)**: Die höchste Herzfrequenz, die die Person erreicht hat.

9. **Oldpeak**: Der ST-Depressionswert im Vergleich zur Ruhe, gemessen nach körperlicher Anstrengung.

10. **Steigung (slp)**: Die Steigung des ST-Segments beim Belastungs-EKG.

11. **Anzahl der großen Gefäße (caa)**: Anzahl der Hauptblutgefäße, die durch Fluoroskopie sichtbar sind. Die Werte reichen typischerweise von 0 bis 3.

12. **Thallium-Stress-Test (thall)**: Ergebnisse des Thallium-Stress-Tests, der zur Diagnose von Herzerkrankungen verwendet wird. Werte liegen normalerweise zwischen 0 und 3.

13. **Belastungsinduzierte Angina (exng)**: Zeigt an, ob bei der Person Angina (Brustschmerzen) durch körperliche Anstrengung ausgelöst wird. 1 steht für Ja, 0 für Nein.

14. **Zielvariable (output)**: Die Zielvariable, die angibt, ob die Person eine Herzerkrankung hat oder nicht. Oft ist 1 ein Indikator für das Vorhandensein einer Herzerkrankung und 0 für deren Abwesenheit.
