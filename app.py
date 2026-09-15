import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)

# Diccionario de traducciones
TRANSLATIONS = {
    "Español": {
        "page_title": "Análisis Global de Terremotos (1995 - 2023)",
        "title": "Análisis Global de Datos de Terremotos (1995 - 2023)",
        "datasource": "Fuente de datos: kaggle https://www.kaggle.com/datasets/warcoder/earthquake-dataset",
        "lang_selector": "Idioma / Language",
        "file_not_found": "El archivo '{path}' no existe en el directorio actual.",
        "file_empty": "El archivo de datos está vacío.",
        "missing_cols": "Faltan columnas necesarias: {cols}",
        "slider_label": "Filtrar por Rango de Magnitud:",
        "caption": "Mostrando **{filtered}** eventos de **{total}** sismos registrados.",
        "no_data": "No hay eventos sísmicos en el rango de magnitud seleccionado.",
        "sub_geo": "Distribución Geográfica (Magnitud: {min_m} - {max_m})",
        "sub_scatter": "Relación entre Profundidad y Magnitud",
        "sub_hist": "Distribución de Frecuencia de Magnitudes",
        "sub_yearly": "Total de Sismos por Año según Nivel de Alerta",
        "alert_filter_label": "Filtrar por Nivel de Alerta (Solo para este gráfico):",
        "alert_all": "Todos",
        "alert_unknown": "Desconocido",
        "depth_colorbar": "Profundidad (km)",
        "depth_axis": "Profundidad (km)",
        "mag_axis": "Magnitud",
        "count_axis": "Frecuencia (Cantidad de Sismos)",
        "year_axis": "Año",
        "trendline": "Línea de Tendencia",
        "ols_header": "Resumen del Modelo Estadístico (OLS)",
        "geo_info": """
<b>Cinturón de Fuego del Pacífico:</b><br>
La mayoría de los sismos de alta magnitud (6.5 - 9.1) se alinean a lo largo de las fronteras de placas tectónicas alrededor del Océano Pacífico
(costas de América, Japón, Filipinas e Indonesia).<br><br>
<b>Profundidad del Hipocentro:</b><br>
Los eventos superficiales (color amarillo, <100 km) predominan globalmente, mientras que los eventos profundos (tonos morados/azules, hasta 600 km) se concentran en zonas específicas de subducción, como la fosa de las Aleutianas, la costa oeste de Sudamérica (Andes) y el Pacífico occidental.<br><br>
<b>Cinturón Alpino-Himalayo:</b><br>
Se observa una franja secundaria de sismicidad moderadamente superficial que atraviesa el sur de Europa, Medio Oriente y el sur de Asia.
""",
        "depth_vs_mag_info": """
<b>Relación Profundidad vs Magnitud:</b><br>
Se observa una ligera tendencia positiva, indicando que los sismos más profundos tienden a tener magnitudes ligeramente mayores. Sin embargo, la dispersión de los datos sugiere que otros factores también influyen en la magnitud de los sismos.
""",
        "mag_distribution_info": """
<b>Distribución de Frecuencia de Magnitud:</b><br>
La distribución de las magnitudes de los sismos muestra una escala logarítmica, con menos eventos de alta magnitud y más eventos de baja magnitud. Este patrón es consistente con la ley de Gutenberg-Richter, que describe la relación entre la frecuencia y la magnitud de los sismos.
""",
        "yearly_info": """<b>Distribución Anual de Sismos por Nivel de Alerta:</b><br>
La cantidad anual de sismos registrados oscila de forma consistente entre 20 y 55 eventos por año a lo largo del periodo 1995 a 2023. Se observa un incremento notable en el periodo 2013 a 2015, alcanzando los valores máximos del registro (superando los 50 eventos anuales en 2013 y 2015)."""
    },
    "English": {
        "page_title": "Global Earthquakes Analysis (1995 - 2023)",
        "title": "Global Earthquakes Data Analysis (1995 - 2023)",
        "datasource": "Data source: kaggle https://www.kaggle.com/datasets/warcoder/earthquake-dataset",
        "lang_selector": "Language / Idioma",
        "file_not_found": "The file '{path}' does not exist in the current directory.",
        "file_empty": "The data file is empty.",
        "missing_cols": "Missing required columns: {cols}",
        "slider_label": "Filter by Magnitude Range:",
        "caption": "Showing **{filtered}** events out of **{total}** recorded earthquakes.",
        "no_data": "No seismic events found in the selected magnitude range.",
        "sub_geo": "Geographical Distribution (Magnitude: {min_m} - {max_m})",
        "sub_scatter": "Relationship between Depth and Magnitude",
        "sub_hist": "Magnitude Frequency Distribution",
        "sub_yearly": "Total Earthquakes per Year by Alert Level",
        "alert_filter_label": "Filter by Alert Level (Only for this chart):",
        "alert_all": "All",
        "alert_unknown": "Unknown",
        "depth_colorbar": "Depth (km)",
        "depth_axis": "Depth (km)",
        "mag_axis": "Magnitude",
        "count_axis": "Frequency (Count of Earthquakes)",
        "year_axis": "Year",
        "trendline": "Trendline",
        "ols_header": "Statistical Model Summary (OLS)",
        "geo_info": """
        <b>Pacific Ring of Fire: </b><br >
        The majority of high-magnitude earthquakes(6.5 - 9.1) align along the boundaries of tectonic plates around the Pacific Ocean
        (coastlines of America, Japan, the Philippines, and Indonesia). <br > <br >
        <b>Focus Depth: </b><br >
        Shallow events(yellow color, < 100 km) dominate globally, while deep events(purple/blue tones, up to 600 km) are concentrated in specific subduction zones, such as the Aleutian Trench, the western coast of South America(Andes), and the western Pacific. <br > <br >
        <b> Alpine-Himalayan Belt: </b><br >
        A secondary band of moderately shallow seismicity is visible stretching across Southern Europe, the Middle East, and Southern Asia.
        """,
        "depth_vs_mag_info": """
        <b> Depth vs Magnitude: </b><br >
        A slight positive trend is observed, indicating that deeper earthquakes tend to have slightly higher magnitudes. However, the data dispersion suggests that other factors also influence the magnitude of earthquakes.
        """,
        "mag_distribution_info": """
        <b> Magnitude Frequency Distribution: </b><br >
        The distribution of earthquake magnitudes shows a logarithmic scale, with fewer high-magnitude events and more low-magnitude ones. This pattern is consistent with the Gutenberg-Richter law, which describes the relationship between the frequency and magnitude of earthquakes.
        """,
        "yearly_info": """
        <b> Yearly Earthquake Distribution by Alert Level: </b><br>
        The bar chart shows the total number of earthquakes recorded per year, filterable by alert level.
        The annual count of recorded earthquakes consistently ranges between 20 and 55 events per year throughout the period 1995 to 2023.
        A notable increase is observed during the period 2013 to 2015, reaching the maximum values of the record(exceeding 50 annual events in 2013 and 2015).""",
    }
}

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Global Earthquakes Analysis (1995 - 2023)", layout="wide")

# Selector de idioma en la barra lateral
lang_choice = st.sidebar.selectbox("Idioma / Language", ["Español", "English"])
t = TRANSLATIONS[lang_choice]

st.title(t["title"])

st.text(t["datasource"])


@st.cache_data
def load_and_clean_data(file_path: str = "earthquake_1995-2023.csv") -> pd.DataFrame:
    """Carga, limpia y enriquece los datos del archivo CSV."""
    if not os.path.exists(file_path):
        st.error(t["file_not_found"].format(path=file_path))
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    if df.empty:
        st.error(t["file_empty"])
        return pd.DataFrame()

    if "date_time" in df.columns:
        logging.info(
            "Creando columna 'date_time' a partir de 'date' y 'time' si es necesario.")
        df["date_time"] = pd.to_datetime(
            df['date_time'], format='%d-%m-%Y %H:%M', errors='coerce')
        # df["date_time"] = pd.to_datetime(
        #    df["date_time"], errors="coerce", format="%d-%m-%Y %H:%M:%S")
        df["year"] = df["date_time"].dt.year
        df["month"] = df["date_time"].dt.month
        df["day"] = df["date_time"].dt.day
        df["hour"] = df["date_time"].dt.hour

    if "alert" in df.columns:
        df["alert"] = df["alert"].fillna("desconocido")

    if "country" in df.columns and "location" in df.columns:
        extracted_country = df["location"].astype(
            str).str.split(",").str[-1].str.strip()
        df["country"] = df["country"].fillna(extracted_country)

    if "continent" in df.columns:
        df["continent"] = df["continent"].fillna("Desconocido")

    return df


def render_dashboard(df: pd.DataFrame, t_dict: dict):
    """Genera y despliega las visualizaciones principales."""
    required_cols = {"latitude", "longitude", "magnitude", "depth"}
    if not required_cols.issubset(df.columns):
        st.warning(t_dict["missing_cols"].format(
            cols=required_cols - set(df.columns)))
        return

    # Slider interactivo de magnitud
    min_mag = float(df["magnitude"].min())
    max_mag = float(df["magnitude"].max())

    selected_mag_range = st.slider(
        t_dict["slider_label"],
        min_value=min_mag,
        max_value=max_mag,
        value=(min_mag, max_mag),
        step=0.1
    )

    df_filtered = df[
        (df["magnitude"] >= selected_mag_range[0]) &
        (df["magnitude"] <= selected_mag_range[1])
    ]

    st.caption(t_dict["caption"].format(
        filtered=len(df_filtered), total=len(df)))

    if df_filtered.empty:
        st.info(t_dict["no_data"])
        return

    # 1. Mapa interactivo
    fig_geo = go.Figure()
    fig_geo.add_trace(
        go.Scattergeo(
            lat=df_filtered["latitude"],
            lon=df_filtered["longitude"],
            mode="markers",
            text=df_filtered["title"] if "title" in df_filtered.columns else df_filtered["location"],
            hoverinfo="text+lat+lon",
            marker=dict(
                size=df_filtered["magnitude"] * 2.5,
                color=df_filtered["depth"],
                colorscale="Viridis_r",
                colorbar=dict(title=t_dict["depth_colorbar"]),
                showscale=True
            )
        )
    )
    fig_geo.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)"
    )
    fig_geo.update_layout(
        title=dict(
            text=f"<b>{t_dict['sub_geo'].format(min_m=selected_mag_range[0], max_m=selected_mag_range[1])}</b>",
            x=0.5,
            xanchor="center"
        ),
        height=500,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_geo, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 2. Scatter Profundidad vs Magnitud + Línea de Tendencia
    fig_scatter = go.Figure()

    st.html(t["depth_vs_mag_info"])

    fig_scatter.add_trace(
        go.Scatter(
            x=df_filtered["depth"],
            y=df_filtered["magnitude"],
            mode="markers",
            name="Sismos",
            text=df_filtered["title"] if "title" in df_filtered.columns else df_filtered["location"],
            marker=dict(
                size=8,
                color=df_filtered["magnitude"],
                colorscale="OrRd",
                opacity=0.7,
                showscale=False
            )
        )
    )

    clean_scatter = df_filtered[["depth", "magnitude"]].dropna()
    if len(clean_scatter) > 1:
        x_data = clean_scatter["depth"]
        y_data = clean_scatter["magnitude"]
        m, b = np.polyfit(x_data, y_data, 1)

        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
        y_trend = m * x_trend + b

        fig_scatter.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name=t_dict["trendline"],
                line=dict(color="#1F77B4", width=3, dash="dash")
            )
        )

    fig_scatter.update_xaxes(title_text=t_dict["depth_axis"])
    fig_scatter.update_yaxes(title_text=t_dict["mag_axis"])
    fig_scatter.update_layout(
        title=dict(
            text=f"<b>{t_dict['sub_scatter']}</b>",
            x=0.5,
            xanchor="center"
        ),
        height=400,
        showlegend=True,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.html(t["mag_distribution_info"])

    # 3. Histograma de Magnitudes
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=df_filtered["magnitude"],
            xbins=dict(size=0.2),
            marker_color="#E74C3C",
            opacity=0.8
        )
    )
    fig_hist.update_xaxes(title_text=t_dict["mag_axis"])
    fig_hist.update_yaxes(title_text=t_dict["count_axis"])
    fig_hist.update_layout(
        title=dict(
            text=f"<b>{t_dict['sub_hist']}</b>",
            x=0.5,
            xanchor="center"
        ),
        height=400,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)


def render_yearly_alert_chart(df: pd.DataFrame, t_dict: dict):
    """Muestra un gráfico de barras por año filtrable exclusivamente por tipo de alerta."""

    st.html(t["yearly_info"])

    if "year" not in df.columns or "alert" not in df.columns:
        return

    df_yearly = df.dropna(subset=["year"]).copy()
    df_yearly["year"] = df_yearly["year"].astype(int)

    # Obtener opciones únicas de alerta
    raw_alerts = sorted([str(a) for a in df_yearly["alert"].unique()])

    # Mapeo para nombres amigables en la interfaz
    alert_options = [t_dict["alert_all"]] + [
        t_dict["alert_unknown"] if a == "desconocido" else a.capitalize()
        for a in raw_alerts
    ]

    selected_alert_display = st.selectbox(
        t_dict["alert_filter_label"],
        options=alert_options
    )

    # Aplicar el filtro según la selección
    if selected_alert_display == t_dict["alert_all"]:
        df_chart_data = df_yearly
    elif selected_alert_display == t_dict["alert_unknown"]:
        df_chart_data = df_yearly[df_yearly["alert"] == "desconocido"]
    else:
        df_chart_data = df_yearly[df_yearly["alert"]
                                  == selected_alert_display.lower()]

    # Agrupar conteo por año
    counts_by_year = df_chart_data.groupby(
        "year").size().reset_index(name="count")

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=counts_by_year["year"],
            y=counts_by_year["count"],
            marker_color="#2980B9",
            opacity=0.85
        )
    )

    fig_bar.update_xaxes(title_text=t_dict["year_axis"], dtick=1)
    fig_bar.update_yaxes(title_text=t_dict["count_axis"])
    fig_bar.update_layout(
        title=dict(
            text=f"<b>{t_dict['sub_yearly']}</b>",
            x=0.5,
            xanchor="center"
        ),
        height=450,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("<br><br>", unsafe_allow_html=True)


def run_ols_regression(df: pd.DataFrame, t_dict: dict):
    """Ejecuta el modelo OLS estadístico en pantalla."""
    required_cols = {"sig", "magnitude", "depth"}
    if not required_cols.issubset(df.columns):
        return

    data_model = df[list(required_cols)].dropna()
    if data_model.empty:
        return

    model = smf.ols("sig ~ magnitude + depth", data=data_model).fit()

    st.subheader(t_dict["ols_header"])
    st.text(str(model.summary()))


# Flujo principal
FILE_PATH = "earthquake_1995-2023.csv"
data = load_and_clean_data(FILE_PATH)

st.html(t["geo_info"])

if not data.empty:
    logging.info("Datos cargados y limpios correctamente.")
    render_dashboard(data, t)
    logging.info("Dashboard renderizado correctamente.")
    render_yearly_alert_chart(data, t)
    logging.info("Gráfico anual de alertas renderizado correctamente.")
    # run_ols_regression(data, t)
