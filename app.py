import os
import pandas as pd
import plotly.graph_objects as go
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
        "sub_hist": "Distribución de Frecuencia de Magnitudes",
        "depth_colorbar": "Profundidad (km)",
        "depth_axis": "Profundidad (km)",
        "mag_axis": "Magnitud",
        "count_axis": "Frecuencia (Cantidad de Sismos)",
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
        "sub_hist": "Magnitude Frequency Distribution",
        "depth_colorbar": "Depth (km)",
        "depth_axis": "Depth (km)",
        "mag_axis": "Magnitude",
        "count_axis": "Frequency (Count of Earthquakes)",
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
    """Genera y despliega las visualizaciones separadas con saltos HTML y títulos en negrilla."""
    required_cols = {"latitude", "longitude", "magnitude", "depth"}
    if not required_cols.issubset(df.columns):
        st.warning(t_dict["missing_cols"].format(cols=required_cols - set(df.columns)))
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

    st.caption(t_dict["caption"].format(filtered=len(df_filtered), total=len(df)))

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

    # Espaciado con 2 saltos de línea HTML
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 2. Scatter Profundidad vs Magnitud
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
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
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Espaciado con 2 saltos de línea HTML
    st.markdown("<br><br>", unsafe_allow_html=True)

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

    # Espaciado antes de la sección final
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

if not data.empty:
    render_dashboard(data, t)
    run_ols_regression(data, t)