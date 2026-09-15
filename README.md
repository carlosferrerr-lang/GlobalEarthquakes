Acceso publica a la aplicacion
https://globalearthquakes.onrender.com/

Análisis Global de Terremotos (1995–2023)
Este proyecto nace con el objetivo de demostrar el potencial de la ciencia de datos en el procesamiento, análisis y visualización de fenómenos naturales complejos. A través del estudio de más de 25 años de registros sísmicos globales, la aplicación transforma datos crudos en conocimiento accionable, permitiendo explorar patrones espaciales, distribuciones de magnitud y variables de impacto en una interfaz totalmente interactiva.

Propósito del Proyecto
El proyecto busca servir como plantilla de referencia para proyectos end-to-end en Python, destacando cómo el análisis cuantitativo y las herramientas modernas de visualización permiten:Democratizar el acceso a datos complejos: Presentar información geofísica mediante tableros dinámicos e intuitivos.Validar leyes empíricas: Contrastar modelos teóricos (como la Ley de Gutenberg-Richter y la regresión OLS) con evidencia empírica.Proporcionar soporte multi-idioma: Integrar internacionalización dinámica (Español / Inglés) para ampliar el alcance del análisis.

Arquitectura y Tecnologías
El desarrollo se encuentra modularizado en Python y utiliza un stack de librerías especializadas:
Procesamiento de Datos: pandas, numpy
Visualización Interactiva: plotly (graph_objects, express, subplots)
Modelado Estadístico: statsmodels
Aplicación Web Interactiva: streamlit

Módulos y Visualizaciones IncluidasEl dashboard web (app.py) incluye las siguientes herramientas de análisis:
Distribución Geográfica Global: Mapa interactivo con proyección Natural Earth que escala el tamaño de los marcadores según la magnitud y colorea según la profundidad del hipocentro.
Scatter Plot (Profundidad vs. Magnitud): Gráfica de dispersión con línea de tendencia OLS (Ordinary Least Squares) para evaluar la correlación entre variables sísmicas.
Histograma de Frecuencia de Magnitudes: Representación gráfica de la distribución de eventos que ilustra el comportamiento exponencial de los sismos.
Análisis Temporal por Nivel de Alerta: Gráfico de barras anuales con filtro independiente por categoría de alerta (incluyendo registros sin clasificar).

#Modelo Estadístico OLS: Resumen matemático impreso en pantalla que detalla coeficientes, errores estándar y métricas de ajuste ($R^2$).

Instrucciones de EjecuciónClonar el repositorio e instalar dependencias:
pip install pandas numpy plotly statsmodels streamlit

Asegurar el archivo de datos:
Coloca el archivo earthquake_1995-2023.csv en la raíz del proyecto.

Lanzar la aplicación:
treamlit run app.py
