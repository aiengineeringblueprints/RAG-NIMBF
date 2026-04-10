# Kompakte Vergleichsansicht (empfohlen)                                      
  python3 view_mlflow_runs.py --compare                                         
                                                                                
  # Volle Tabelle mit allen Details                                             
  python3 view_mlflow_runs.py                                                   
                                                                                
  # Nur ein bestimmtes Experiment                                               
  python3 view_mlflow_runs.py -e 1

  # Alle Metriken anzeigen (inkl. min/max/mean/std)
  python3 view_mlflow_runs.py --all-metrics

  # Als CSV exportieren                                                         
  python3 view_mlflow_runs.py --csv results.csv
                                                                                
  clear_mlflow.py — Datenbank leeren                                          
                                      
  # Alle Runs/Metriken/Params löschen (mit Bestätigung)
  python3 clear_mlflow.py

  # Ohne Bestätigung
  python3 clear_mlflow.py -y

  # Auch Experiments löschen
  python3 clear_mlflow.py --reset-experiments


Test change