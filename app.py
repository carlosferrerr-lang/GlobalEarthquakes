import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import streamlit as st

# Diccionario de traducciones
TRANSLATIONS = {
    "Español": {
        "page_title": "Análisis Global de Terremotos",
        "title": "Análisis Global de Datos de Terremotos",
        "lang_selector": "Idioma / Language",
        "file_not_found": "El archivo '{path}' no existe en el directorio actual.",
        "file_empty": "El archivo de datos está vacío.",
        "missing_cols": "Faltan columnas necesarias: {cols}",
        "slider_label": "Filtrar por Rango de Magnitud:",
        "caption": "Mostrando **{filtered}** eventos de **{total}** sismos registrados.",
        "no_data": "No hay eventos sísmicos en el rango de magnitud seleccionado.",
        "sub_geo": "Distribución Geográfica (Magnitud: {min_m} - {max_m})",
        "sub_scatter": "Relación entre Profundidad y Magnitud",
        "depth_colorbar": "Profundidad (km)",
        "depth_axis": "Profundidad (km)",
        "mag_axis": "Magnitud",
        "ols_header": "Resumen del Modelo Estadístico (OLS)"
    },
    "English": {
        "page_title": "Global Earthquakes Analysis",
        "title": "Global Earthquakes Data Analysis",
        "lang_selector": "Language / Idioma",
        "file_not_found": "The file '{path}' does not exist in the current directory.",
        "file_empty": "The data file is empty.",
        "missing_cols": "Missing required columns: {cols}",
        "slider_label": "Filter by Magnitude Range:",
        "caption": "Showing **{filtered}** events out of **{total}** recorded earthquakes.",
        "no_data": "No seismic events found in the selected magnitude range.",
        "sub_geo": "Geographical Distribution (Magnitude: {min_m} - {max_m})",
        "sub_scatter": "Relationship between Depth and Magnitude",
        "depth_colorbar": "Depth (km)",
        "depth_axis": "Depth (km)",
        "mag_axis": "Magnitude",
        "ols_header": "Statistical Model Summary (OLS)"
    }
}

# Configuración inicial de Streamlit
st.set_page_config(page_title="Global Earthquakes Analysis", layout="wide")

# Selector de idioma en la barra lateral
lang_choice = st.sidebar.selectbox("Idioma / Language", ["Español", "English"])
t = TRANSLATIONS[lang_choice]

st.title(t["title"])


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
        df["date_time"] = pd.to_datetime(
            df["date_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
        df["year"] = df["date_time"].dt.year
        df["month"] = df["date_time"].dt.month
        df["day"] = df["date_time"].dt.day
        df["hour"] = df["date_time"].dt.hour

    if "alert" in df.columns:
        df["alert"] = df["alert"].fillna("sin_alerta")

    if "country" in df.columns and "location" in df.columns:
        extracted_country = df["location"].astype(
            str).str.split(",").str[-1].str.strip()
        df["country"] = df["country"].fillna(extracted_country)

    if "continent" in df.columns:
        df["continent"] = df["continent"].fillna("Desconocido")

    return df


def render_dashboard(df: pd.DataFrame, t_dict: dict):
    """Genera y despliega las visualizaciones dinámicas con textos traducidos."""
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

    # Construcción de subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            t_dict["sub_geo"].format(
                min_m=selected_mag_range[0], max_m=selected_mag_range[1]),
            t_dict["sub_scatter"]
        ),
        specs=[[{"type": "geo"}], [{"type": "xy"}]],
        vertical_spacing=0.12
    )

    # Mapa interactivo
    fig.add_trace(
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
                colorbar=dict(
                    title=t_dict["depth_colorbar"], len=0.45, y=0.78),
                showscale=True
            )
        ),
        row=1, col=1
    )

    # Scatter Profundidad vs Magnitud
    fig.add_trace(
        go.Scatter(
            x=df_filtered["depth"],
            y=df_filtered["magnitude"],
            mode="markers",
            text=df_filtered["title"] if "title" in df_filtered.columns else df_filtered["location"],
            marker=dict(
                size=8,
                color=df_filtered["magnitude"],
                colorscale="OrRd",
                opacity=0.7,
                showscale=False
            )
        ),
        row=2, col=1
    )

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)"
    )
    fig.update_xaxes(title_text=t_dict["depth_axis"], row=2, col=1)
    fig.update_yaxes(title_text=t_dict["mag_axis"], row=2, col=1)

    fig.update_layout(
        height=850,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


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

if not data.empty:
    render_dashboard(data, t)
    run_ols_regression(data, t)
