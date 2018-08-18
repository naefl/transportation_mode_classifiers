# transportation_mode_classifiers
ETH Zurich Master's Thesis "Deep vs. Classical Machine Learning Approaches for Transportation Mode Recognition in a Digital  Health Application" for Classifying Bus/Walk/Cycle/Car/Tram/Train with up to 94% accuracy based on Android Sensors and GTFS information from Zurich.

Compares SVMs, Boosted Trees and others with RNNs (GRU/LSTM).
Includes Dataparser and feature calculation script. Includes param optimization and feature selection. Includes KDA/LDA based dimensionality reduction.

Sensors used: GPS, Accel, Gyro, Sound Pressure.

Main frameworks: sk-learn, TensorFlow 1.7
