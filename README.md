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
3. **Konfiguration anpassen*
   Config.json enthält alle Parameter zum Aufbau des Environments und zum Szenario. Trainingsparameter müssen in algorithms.py angepasst      werden

## Training


