@ECHO OFF

time /t
ECHO ---- Script Horizontal FL ----

ECHO Lanzando clientes distribuidos...

START python client.py empresa_1 8080
START python client.py empresa_2 8080
START python client.py empresa_3 8080

ECHO Todos los clientes en ejecucion

ECHO ---- Fin del script ----
time /t

::PAUSE