# Predator-Prey Reinforcement Learning Simulation

Dieses Projekt simuliert ein dynamisches Predator-Prey-Szenario zur Untersuchung lernfähiger Agenten mit Hilfe von Reinforcement Learning. Die Agenten operieren in einer physikalisch plausiblen, 2D-Umgebung mit Hindernissen, Futterquellen und einem jagenden Gegenspieler. Die Lernarchitektur basiert auf PPO (Proximal Policy Optimization) mit LSTM zur Verarbeitung sequentieller Beobachtungen.

Ziel ist es, die Lernfähigkeit und Generalisierbarkeit von Agentenverhalten in verschiedenen Subtasks wie Navigation, Flucht und Nahrungssuche zu analysieren.

---

##  Installation

1. **Virtuelle Umgebung erstellen**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Für Linux/macOS
   venv\Scripts\activate         # Für Windows

2. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt

## Training
   1. **Konfiguration anpassen**
      Config.json enthält alle Parameter zum Aufbau des Environments und zum Szenario. Trainingsparameter müssen in algorithms.py angepasst werden

   2. **Training starten**
      Das Training kann mit folgenden Befehl gestartet werden. Trainingsverläufe werden unter logs/<timestamp>/ gespeichert.
      ```bash
      python main.py
   3. **Trainingsverlauf analysieren**
      Der Verlauf des Trainings kann mithilfe von analyze_run.py analysiert werden. In /graphs werden hierbei verschiedene Metriken des aktuellsten Trainingslaufs dargestellt.
      ```bash
      python analyze_run.py

   4. **Trainingsverlauf visualisieren**
      Der Trainingsverlauf kann ebenfalls visuell dargestellt werden. In visualize_run.py kann hier der gewünschte Zeitraum ausgewählt werden. Als Ouptut wird eine Videodatei im AVI-Format generiert, welche dne gewünschten Zeitraum abbildet.
       ```bash
      python visualize_run.py

## Evaluation
   1. **Performance analysieren**
      Um die Performance eines Modelles zu evaluieren, kann der Pfad zum Modell in evaluate.py angegeben werden. Ebenfalls kann hier das zu testende Szenario konfiguriert werden. Als Output werden die gewünschten Metriken generiert und unter /evaluate_log die besten 5 Anläufe gespeichert.
       ```bash
      python evaluate.py





