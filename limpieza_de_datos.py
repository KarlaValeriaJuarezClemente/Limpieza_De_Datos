import os
import logging 
import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from difflib import SequenceMatcher
from functools import lru_cache
import io

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
carpeta_datos = 'Datos'
os.makedirs('KPIs', exist_ok=True)

ruta_sueno = os.path.join(carpeta_datos, 'calidad_de_sueño.xlsx')
ruta_salud = os.path.join(carpeta_datos, 'salud_mental.xlsx')

try:
    if not os.path.exists(ruta_sueno):
        raise FileNotFoundError(f"Archivo no encontrado en la ruta: {ruta_sueno}")
    
    if ruta_sueno.endswith('.csv'):
        test_sueno = pd.read_csv(ruta_sueno, nrows=1)
    else:
        test_sueno = pd.read_excel(ruta_sueno, nrows=1)
    print("  Archivo de sueño es legible.")
except Exception as e:
    print(f"Error fatal leyendo archivo de sueño: {e}")
    raise

try:
    if not os.path.exists(ruta_salud):
        raise FileNotFoundError(f"Archivo no encontrado en la ruta: {ruta_salud}")

    if ruta_salud.endswith('.csv'):
        test_salud = pd.read_csv(ruta_salud, nrows=1)
    else:
        test_salud = pd.read_excel(ruta_salud, nrows=1)
    print("  Archivo de salud es legible.")
except Exception as e:
    print(f"Error fatal leyendo archivo de salud: {e}")
    raise

ruta_salida_xlsx = 'Datos/Datos_Limpios_Unificados.xlsx'
ruta_salida_csv = 'Datos/Datos_Limpios_Unificados.csv'

class Limpiador:
    def __init__(self):
        # mapeo plano para respuestas
        self.mapeo_si_no_flat = {
            'sí': 'Sí', 'si': 'Sí', 's': 'Sí', 'yes': 'Sí', 'y': 'Sí', '1': 'Sí','sip': 'Sí','sipi': 'Sí','afirmativo': 'Sí', 'correcto': 'Sí', 'claro': 'Sí', 'si tengo': 'Sí',
                   'cuento con': 'Sí', 'tengo': 'Sí', 'por supuesto': 'Sí', 'obvio': 'Sí', 'siempre': 'Sí', 'definitivamente': 'Sí',
                   'Si, la verdad.': 'Sí', 'si, la verdad': 'Sí', 'concierto que si': 'Sí', 'afirmativo': 'Sí','todo el maldito tiempo': 'Sí', 'mucho': 'Sí', 'cada día': 'Sí', 'diario': 'Sí', 'todo el tiempo': 'Sí',
                   'si la verdad': 'Sí',
            'no': 'No', 'n': 'No', '0': 'No', 'nope': 'No', 'negativo': 'No', 'para nada': 'No', 'ninguno': 'No', 'nada': 'No',
                  'jamás': 'No', 'no tengo': 'No', 'no cuento': 'No', 'concierto que no': 'No','considero que no': 'No', 'no según yo': 'No', 'no según lo': 'No'
        }
        # patrones más flexibles para detectar columnas por tipo
        self.patrones_horas = ['hora', 'horas', 'duerm', 'sueño', 'sueno', 'dormir', 'sleep', 'descanso', 'descansa']
        self.patrones_desvelo = ['desvel', 'trasnoch', 'noche', 'insomnio', 'vela', 'no dormir']
        self.patrones_estres = ['estres', 'estrés', 'stress', 'abrum', 'presión', 'presion', 'tensión', 'tension']
        self.patrones_ansiedad = ['ansiedad', 'ansios', 'nervios', 'preocup', 'inquiet', 'angustia']
        # posibles nombres ID
        self.patrones_id = ['numero', 'número', 'num', 'num_cuenta', 'cuenta', 'no_cuenta', 'matricula', 'matrícula', 'id', 'código', 'codigo', 'identificacion']

    @staticmethod
    @lru_cache(maxsize=10000)
    def similitud_texto(a: str, b: str) -> float:
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    def normalizar_si_no(self, valor):
        if pd.isna(valor):
            return np.nan
        s = str(valor).strip().lower()
        # quitar tildes simples y espacios
        s_clean = s.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        if s_clean in self.mapeo_si_no_flat:
            return self.mapeo_si_no_flat[s_clean]
        # heurísticos más robustos
        if any(k in s_clean for k in ['si', 'sí', 'yes', 'siempre', 'tengo', 'afirmativo', 'claro', 'correcto']):
            return 'Sí'
        if any(k in s_clean for k in ['no', 'nunca', 'ningun', 'ninguno', 'nada', 'negativo', 'nope']):
            return 'No'
        return valor  # deja el original si no se puede normalizar

    def extraer_numero(self, valor, rango: Tuple[float,float] = (0, 100)):
        if pd.isna(valor):
            return np.nan
        s = str(valor).lower().strip()
        # mapear números escritos
        mapping = {
            'cero':0,'uno':1,'dos':2,'tres':3,'cuatro':4,'cinco':5,'seis':6,'siete':7,'ocho':8,'nueve':9,'diez':10,
            'once':11,'doce':12,'trece':13,'catorce':14,'quince':15
        }
        for k,v in mapping.items():
            if re.search(rf'\b{k}\b', s):
                if rango[0] <= v <= rango[1]:
                    return float(v)
        # buscar número - patrones más flexibles
        m = re.search(r'(\d+[\.,]?\d*)', s)
        if m:
            try:
                num = float(m.group(1).replace(',', '.'))
                if rango[0] <= num <= rango[1]:
                    return num
            except:
                return np.nan
        # buscar rangos (ej: "6-7 horas")
        m_rango = re.search(r'(\d+)[\s\-]+(\d+)', s)
        if m_rango:
            try:
                num1 = float(m_rango.group(1))
                num2 = float(m_rango.group(2))
                promedio = (num1 + num2) / 2
                if rango[0] <= promedio <= rango[1]:
                    return promedio
            except:
                pass
        return np.nan

    def detectar_columna_id(self, df: pd.DataFrame) -> Optional[str]:
        # prioridad: columnas que contienen patrones exactos
        for c in df.columns:
            lc = c.lower()
            for p in self.patrones_id:
                if p in lc:
                    return c
        # fallback: buscar por similitud con 'numero cuenta'
        referencia = 'numero de cuenta'
        mejor = (None, 0.0)
        for c in df.columns:
            sim = self.similitud_texto(c, referencia)
            if sim > 0.6:  # Umbral más alto para evitar falsos positivos
                return c
        return None

    def detectar_columnas_por_tipo(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        cols = {'horas': [], 'desvelo': [], 'estres': [], 'ansiedad': []}
        for c in df.columns:
            lc = c.lower()
            if any(p in lc for p in self.patrones_horas):
                cols['horas'].append(c)
            if any(p in lc for p in self.patrones_desvelo):
                cols['desvelo'].append(c)
            if any(p in lc for p in self.patrones_estres):
                cols['estres'].append(c)
            if any(p in lc for p in self.patrones_ansiedad):
                cols['ansiedad'].append(c)
        return cols

    def aplicar_limpieza(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalizar strings y tratar valores vacíos
        for c in df.select_dtypes(include=['object', 'string']).columns:
            df[c] = df[c].astype('string').str.strip()
            df[c] = df[c].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'null': pd.NA})

        # detectar columnas por tipo
        tipos = self.detectar_columnas_por_tipo(df)
        print(f"Columnas detectadas por tipo: {tipos}")

        # para horas: crear columna estandarizada
        for c in tipos['horas']:
            nueva_col = f'{c}_estandarizada'
            df[nueva_col] = df[c].apply(lambda x: self.extraer_numero(x, (0,24)))
            print(f"Columna {c} -> {nueva_col}: {df[nueva_col].notna().sum()} valores extraídos")

        # para desvelos: 0-7 días por semana
        for c in tipos['desvelo']:
            nueva_col = f'{c}_estandarizada'
            df[nueva_col] = df[c].apply(lambda x: self.extraer_numero(x, (0,7)))
            print(f"Columna {c} -> {nueva_col}: {df[nueva_col].notna().sum()} valores extraídos")

        # normalizar si/no en estres y ansiedad
        for c in tipos['estres']:
            df[c] = df[c].apply(self.normalizar_si_no)
            print(f"Columna {c} normalizada: {df[c].value_counts().to_dict()}")

        for c in tipos['ansiedad']:
            df[c] = df[c].apply(self.normalizar_si_no)
            print(f"Columna {c} normalizada: {df[c].value_counts().to_dict()}")

        return df

limpiador = Limpiador()

print("\n Limpiador inicializado.\n")

try:
    if ruta_sueno.endswith('.csv'):
        df_sueno = pd.read_csv(ruta_sueno)
    else:
        df_sueno = pd.read_excel(ruta_sueno)
        
    if ruta_salud.endswith('.csv'):
        df_salud = pd.read_csv(ruta_salud)
    else:
        df_salud = pd.read_excel(ruta_salud)

    logging.info("Archivos cargados: sueno %s, salud %s", df_sueno.shape, df_salud.shape)
    print(f"Columnas en sueño: {list(df_sueno.columns)}")
    print(f"Columnas en salud: {list(df_salud.columns)}")
except Exception as e:
    logging.error("Error al leer archivos: %s", e)
    raise

# Aplicar limpieza
df_sueno_clean = limpiador.aplicar_limpieza(df_sueno)
df_salud_clean = limpiador.aplicar_limpieza(df_salud)

# Mostrar detecciones rápidas
id_sueno = limpiador.detectar_columna_id(df_sueno_clean)
id_salud = limpiador.detectar_columna_id(df_salud_clean)
print("ID detectado en Sueño:", id_sueno)
print("ID detectado en Salud:", id_salud)

"""#Homologación"""

def elegir_id_col(df1, df2, limpiador_obj: Limpiador) -> Tuple[Optional[str], Optional[str], str]:
    id1 = limpiador_obj.detectar_columna_id(df1)
    id2 = limpiador_obj.detectar_columna_id(df2)
    # estandarizar nombre final
    nombre_id_final = 'numero_cuenta'
    # renombrar si existen
    if id1:
        df1 = df1.rename(columns={id1: nombre_id_final})
    if id2:
        df2 = df2.rename(columns={id2: nombre_id_final})
    return df1, df2, nombre_id_final

df_sueno_clean, df_salud_clean, id_col = elegir_id_col(df_sueno_clean, df_salud_clean, limpiador)
print("ID usado para merge:", id_col)
# Asegurar tipo string y strip
for df in [df_sueno_clean, df_salud_clean]:
    if id_col in df.columns:
        df[id_col] = df[id_col].astype('string').str.strip()

"""#Unificación"""

def combinar_columnas_similares(df, base_name):
    candidates = [c for c in df.columns if base_name in c.lower()]
    if len(candidates) <= 1:
        return df
    # crear columna combinada priorizando columnas no nulas en orden
    df[base_name] = df[candidates].bfill(axis=1).iloc[:,0]
    return df

if id_col and id_col in df_sueno_clean.columns and id_col in df_salud_clean.columns:
    df_unificado = pd.merge(df_sueno_clean, df_salud_clean, on=id_col, how='outer', suffixes=('_sueno','_salud'))
    logging.info("Merge outer realizado por ID. Resultado: %s", df_unificado.shape)
else:
    print("No se pudo hacer merge por ID, concatenando por filas...")
    df_sueno_clean['_fuente'] = 'sueno'
    df_salud_clean['_fuente'] = 'salud'
    df_unificado = pd.concat([df_sueno_clean, df_salud_clean], ignore_index=True, sort=False)
    logging.info("Concatenacion por filas. Resultado: %s", df_unificado.shape)

# Combinar columnas similares
df_unificado = combinar_columnas_similares(df_unificado, 'estres')
df_unificado = combinar_columnas_similares(df_unificado, 'ansiedad')

# combina horas estandarizadas si hay más de una
horas_cols = [c for c in df_unificado.columns if c.endswith('_estandarizada') and any(p in c.lower() for p in ['hora','sueno','sueño','duerm','sleep'])]
if horas_cols:
    df_unificado['horas_sueno_estandarizadas'] = df_unificado[horas_cols].bfill(axis=1).iloc[:,0]
    print(f"Columnas de horas combinadas: {horas_cols}")

print("Columnas resultantes:", list(df_unificado.columns)[:15])
print("\nPrimeras filas del dataset unificado:")
print(df_unificado.head()) 

try:
    df_unificado.to_excel(ruta_salida_xlsx, index=False, engine='openpyxl')
    df_unificado.to_csv(ruta_salida_csv, index=False)
    logging.info("Archivo unificado guardado: %s  (y %s)", ruta_salida_xlsx, ruta_salida_csv)
except Exception as e:
    logging.error("Error guardando archivo final: %s", e)
    raise

def calcular_kpis(df: pd.DataFrame) -> Dict[str, float]:
    kpis = {}

    # Horas de sueño - buscar cualquier columna con horas estandarizadas
    horas_cols = [c for c in df.columns if 'hora' in c.lower() and 'estandarizada' in c.lower()]
    if not horas_cols:
        # Buscar columnas numéricas que podrían contener horas
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dropna().between(0, 24).any():
                horas_cols = [col]
                break

    if horas_cols:
        col_horas = horas_cols[0]
        s = df[col_horas].dropna()
        s = pd.to_numeric(s, errors='coerce').dropna()
        if len(s) > 0:
            kpis['horas_promedio'] = float(s.mean())
            kpis['horas_mediana'] = float(s.median())
            kpis['horas_std'] = float(s.std())
            kpis['pct_insuficientes'] = float((s < 6).mean() * 100)
            kpis['pct_adecuado_7_9'] = float(((s >= 7) & (s <= 9)).mean() * 100)
            print(f"KPIs de sueño calculados usando columna: {col_horas}")
            print(f"  Muestra: {len(s)} registros, Rango: {s.min():.1f}-{s.max():.1f}h")

    # Estres y ansiedad (buscar columnas que contenghan estos términos)
    for name in ['estres', 'ansiedad']:
        posibles_cols = [c for c in df.columns if name in c.lower()]
        for col in posibles_cols:
            ser = df[col].dropna()
            if len(ser) > 0:
                # Contar respuestas afirmativas
                si_count = ser.astype(str).str.lower().str.contains('sí|si|yes|1', na=False).sum()
                kpis[f'pct_{name}_si'] = float((si_count / len(ser)) * 100)
                print(f"KPI {name} calculado usando columna: {col}")
                print(f"  Muestra: {len(ser)}, Sí: {si_count}, No: {len(ser) - si_count}")
                break  # Usar la primera columna encontrada

    logging.info("KPIs calculados: %s", kpis)
    return kpis

kpis = calcular_kpis(df_unificado)
print("\nKPIs calculados:")
for k, v in kpis.items():
    print(f"  {k}: {v:.2f}")

"""Visualización KPIs"""

def crear_dashboard_kpis(kpis, df_unificado):
    # Determinar cuántos subplots necesitamos
    num_plots = 0
    tiene_sueno = 'horas_promedio' in kpis and not pd.isna(kpis['horas_promedio'])
    tiene_estres = 'pct_estres_si' in kpis and not pd.isna(kpis['pct_estres_si'])
    tiene_ansiedad = 'pct_ansiedad_si' in kpis and not pd.isna(kpis['pct_ansiedad_si'])

    if tiene_sueno:
        num_plots += 2
    if tiene_estres:
        num_plots += 1
    if tiene_ansiedad:
        num_plots += 1

    if num_plots == 0:
        print("No hay suficientes datos para crear el dashboard.")
        return

    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    plot_idx = 0

    # Distribución de horas de sueño
    if tiene_sueno:
        horas_cols = [c for c in df_unificado.columns if 'hora' in c.lower() and 'estandarizada' in c.lower()]
        if horas_cols:
            data_sueno = df_unificado[horas_cols[0]].dropna()
            data_sueno = pd.to_numeric(data_sueno, errors='coerce').dropna()

            if len(data_sueno) > 0:
                axes[plot_idx].hist(data_sueno, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                axes[plot_idx].axvline(kpis['horas_promedio'], color='red', linestyle='--',
                                     label=f'Promedio: {kpis["horas_promedio"]:.1f}h')
                axes[plot_idx].axvline(kpis['horas_mediana'], color='green', linestyle='--',
                                     label=f'Mediana: {kpis["horas_mediana"]:.1f}h')
                axes[plot_idx].set_title('Distribución de Horas de Sueño', fontweight='bold')
                axes[plot_idx].set_xlabel('Horas de sueño')
                axes[plot_idx].set_ylabel('Frecuencia')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

    # KPIs de Sueño
    if tiene_sueno:
        kpis_sueno = {k: v for k, v in kpis.items() if any(x in k for x in ['horas', 'pct_insuficientes', 'pct_adecuado'])}
        kpis_sueno = {k: v for k, v in kpis_sueno.items() if not pd.isna(v)}

        if kpis_sueno:
            labels = [k.replace('_', ' ').title() for k in kpis_sueno.keys()]
            values = list(kpis_sueno.values())
            bars = axes[plot_idx].bar(labels, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(values)])
            axes[plot_idx].set_title('KPIs de Calidad de Sueño', fontweight='bold')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, values):
                axes[plot_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                  f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            plot_idx += 1

    # Estrés
    if tiene_estres:
        estres_si = kpis['pct_estres_si']
        estres_no = 100 - estres_si
        sizes_estres = [estres_si, estres_no]
        labels_pie = ['Sí', 'No']
        colors = ['#ff6b6b', '#c8d6e5']

        axes[plot_idx].pie(sizes_estres, labels=labels_pie, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[plot_idx].set_title('Prevalencia de Estrés', fontweight='bold')
        plot_idx += 1

    # Ansiedad
    if tiene_ansiedad:
        ansiedad_si = kpis['pct_ansiedad_si']
        ansiedad_no = 100 - ansiedad_si
        sizes_ansiedad = [ansiedad_si, ansiedad_no]
        labels_pie = ['Sí', 'No']
        colors = ['#ff6b6b', '#c8d6e5']

        axes[plot_idx].pie(sizes_ansiedad, labels=labels_pie, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[plot_idx].set_title('Prevalencia de Ansiedad', fontweight='bold')
        plot_idx += 1

    # Ocultar ejes no utilizados
    for i in range(plot_idx, 4):
        axes[i].axis('off')

    plt.tight_layout()
    
    save_path = 'KPIs/General_KPIs.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"KPIs guardado en: {save_path}")

print("KPIs Completo")
crear_dashboard_kpis(kpis, df_unificado)

"""
EDA Univariado
"""
num_cols = df_unificado.select_dtypes(include=[np.number]).columns.tolist()
print("Columnas numéricas detectadas:", num_cols)

if num_cols:
    # Filtrar columnas que realmente tienen datos
    num_cols_validas = []
    for col in num_cols:
        non_na_count = df_unificado[col].notna().sum()
        if non_na_count > 0:
            num_cols_validas.append(col)
            print(f"  {col}: {non_na_count} valores no nulos")

    if num_cols_validas:
        resumen = df_unificado[num_cols_validas].describe().T
        print(resumen) # cambiado display por print

        # Histogramas y boxplots solo para columnas con datos
        for c in num_cols_validas[:4]:  # Mostrar máximo 4 para no saturar
        
            nombre_limpio = c.replace('_estandarizada', '').replace('_', ' ').capitalize()

            # Crear figura con subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Histograma
            data_clean = df_unificado[c].dropna()
            if len(data_clean) > 0:
                # Convertir a numérico por si acaso
                data_clean = pd.to_numeric(data_clean, errors='coerce').dropna()

                if len(data_clean) > 0:
                    # Calcular estadísticas antes de graficar
                    mean_val = data_clean.mean()
                    median_val = data_clean.median()
                    std_val = data_clean.std()
                    min_val = data_clean.min()
                    max_val = data_clean.max()

                    # Crear histograma
                    n, bins, patches = ax1.hist(data_clean, bins=15, alpha=0.7, color='skyblue',
                                              edgecolor='black', density=False)

                    # Evitar superposición
                    max_freq = max(n)
                    bin_width = bins[1] - bins[0]

                    # Posicionar el texto en un área menos congestionada
                    text_x = min_val + (max_val - min_val) * 0.02  # 2% desde el inicio
                    text_y = max_freq * 0.85  # 85% de la altura máxima

                    stats_text = f'Media: {mean_val:.2f}\nMediana: {median_val:.2f}\nDesv: {std_val:.2f}\nMín: {min_val:.2f}\nMáx: {max_val:.2f}\nN: {len(data_clean)}'

                    # Añadir estadísticas en una posición fija que no interfiera
                    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                            fontsize=9, family='monospace')

                    # Añadir líneas de media y mediana
                    ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Media: {mean_val:.2f}')
                    ax1.axvline(median_val, color='green', linestyle='--', alpha=0.8, linewidth=2, label=f'Mediana: {median_val:.2f}')

                    ax1.set_title(f'Distribución: {nombre_limpio}', fontweight='bold', pad=20)
                    ax1.set_xlabel(nombre_limpio)
                    ax1.set_ylabel('Frecuencia')
                    ax1.legend(loc='upper right')
                    ax1.grid(True, alpha=0.3)

                    # Ajustar límites para mejor visualización
                    ax1.set_xlim(min_val - bin_width, max_val + bin_width)

            # Boxplot en el segundo subplot
            if len(data_clean) > 0:
                # Crear boxplot
                boxplot = ax2.boxplot(data_clean, vert=True, patch_artist=True,
                                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                                    medianprops=dict(color='darkred', linewidth=2),
                                    flierprops=dict(marker='o', markerfacecolor='red', markersize=4))

                # Añadir puntos de datos para mostrar distribución real
                y = data_clean.values
                x = np.random.normal(1, 0.04, size=len(y))
                ax2.scatter(x, y, alpha=0.4, color='blue', s=20)

                # Estadísticas para el boxplot
                q1 = data_clean.quantile(0.25)
                q3 = data_clean.quantile(0.75)
                iqr = q3 - q1

                box_stats = f'Q1: {q1:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}'
                ax2.text(0.95, 0.95, box_stats, transform=ax2.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontsize=9, family='monospace')

                ax2.set_title(f'Boxplot: {nombre_limpio}', fontweight='bold', pad=20)
                ax2.set_ylabel(nombre_limpio)
                ax2.set_xlabel('Distribución')
                ax2.grid(True, alpha=0.3)

                # Ocultar el eje x ya que solo tenemos una variable
                ax2.set_xticks([])

            else:
                # Si no hay datos, mostrar mensaje
                ax1.text(0.5, 0.5, 'No hay datos\nnuméricos',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=12, color='red')
                ax1.set_title(f'Distribución: {nombre_limpio}', fontweight='bold')

                ax2.text(0.5, 0.5, 'No hay datos\nnuméricos',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12, color='red')
                ax2.set_title(f'Boxplot: {nombre_limpio}', fontweight='bold')

            plt.tight_layout()

            filename_c = nombre_limpio.replace(' ', '_').replace(':', '') \
                                      .replace('?', '').replace('¿', '') \
                                      .replace('.', '').replace('/', '') \
                                      .replace('\\', '')
            save_path = f'KPIs/{filename_c}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 
            print(f"Gráfico univariado guardado en: {save_path}")

            if len(data_clean) > 0:
                print(f"\nResumen estadístico para '{nombre_limpio}':")
                print(f"   Rango: {min_val:.2f} - {max_val:.2f}")
                print(f"   Media ± Desv: {mean_val:.2f} ± {std_val:.2f}")
                print(f"   Mediana: {median_val:.2f}")
                print(f"   Asimetría: {data_clean.skew():.2f}")
                print(f"   Curtosis: {data_clean.kurtosis():.2f}")
                print(f"   Valores únicos: {data_clean.nunique()}")
                print("-" * 50)

    else:
        print("No se detectaron columnas numéricas con datos para EDA univariado.")
else:
    print("No se detectaron columnas numéricas para EDA univariado.")

"""EDA Bivariado"""

if len(num_cols) >= 2:
    # Filtrar solo columnas verdaderamente numéricas
    numeric_cols_clean = []
    for col in num_cols:
        try:
            # Intentar convertir a numérico
            pd.to_numeric(df_unificado[col].dropna(), errors='raise')
            numeric_cols_clean.append(col)
        except:
            print(f"Advertencia: La columna '{col}' contiene valores no numéricos y será excluida del análisis de correlación")

    if len(numeric_cols_clean) >= 2:
        corr = df_unificado[numeric_cols_clean].corr()

        # Heatmap
    if len(num_cols) >= 2:
        corr = df_unificado[num_cols].corr()

        # Crea una lista de nombres limpios
        clean_names = [col.replace('_estandarizada', '').replace('_', ' ').capitalize() 
                       for col in corr.columns]
        # Asigna los nombres limpios al índice y columnas del DataFrame de correlación
        corr.columns = clean_names
        corr.index = clean_names

        fig_corr = plt.figure(figsize=(10,8))
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag')
        plt.title('Matriz de correlaciones (numéricas)')

        save_path = 'KPIs/Correlacion.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_corr) 
        print(f"Heatmap de correlación guardado en: {save_path}")
        
        print("Top correlaciones absolutas (excluye 1.0):")
        corr_vals = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        print(corr_vals[(corr_vals < 1)].head(10)) 
    else:
        print("No hay suficientes columnas numéricas para correlación bivariada.")

    # Si hay horas y desvelo, scatter plots con manejo de errores
    if 'horas_sueno_estandarizadas' in df_unificado.columns:
        posibles = [c for c in df_unificado.columns if 'desvelo' in c.lower() or 'desvel' in c.lower() or 'dias' in c.lower()]

    if posibles:
        for p in posibles:
            try:
                nombre_limpio_p = p.replace('_estandarizada', '').replace('_', ' ').capitalize()

                # Crear copia temporal y convertir a numérico
                tmp = df_unificado[['horas_sueno_estandarizadas', p]].copy()

                # Convertir ambas columnas a numérico, forzando errores a NaN
                tmp['horas_sueno_estandarizadas'] = pd.to_numeric(tmp['horas_sueno_estandarizadas'], errors='coerce')
                tmp[p] = pd.to_numeric(tmp[p], errors='coerce')

                # Eliminar filas con NaN
                tmp = tmp.dropna()

                if len(tmp) > 5:
                    # --- MODIFICACIÓN ---
                    # Asignar la figura a una variable
                    fig_scatter = plt.figure(figsize=(8, 6))
                    # --- FIN DE MODIFICACIÓN ---

                    # Scatter plot
                    scatter = sns.scatterplot(data=tmp, x=p, y='horas_sueno_estandarizadas',
                                             alpha=0.6, s=60)

                    # Calcular correlación
                    corr_val = tmp['horas_sueno_estandarizadas'].corr(tmp[p])

                    # Añadir línea de tendencia solo si hay suficientes puntos
                    if len(tmp) > 1:
                        try:
                            z = np.polyfit(tmp[p], tmp['horas_sueno_estandarizadas'], 1)
                            p_line = np.poly1d(z)
                            plt.plot(tmp[p], p_line(tmp[p]), "r--", alpha=0.8, label='Línea de tendencia')
                        except:
                            print(f"No se pudo calcular la línea de tendencia para {p}")

                    # Añadir correlación en recuadro separado
                    plt.text(0.05, 0.95, f'Correlación: {corr_val:.2f}',
                            transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            verticalalignment='top', fontsize=12)

                    plt.title(f'Horas sueño vs {nombre_limpio_p}', fontweight='bold')
                    plt.xlabel(nombre_limpio_p)
                    plt.ylabel('Horas de sueño') # También limpiamos esta etiqueta

                    # Añadir leyenda si hay línea de tendencia
                    if len(tmp) > 1:
                        plt.legend()

                    plt.tight_layout()

                    filename_p = nombre_limpio_p.replace(' ', '_').replace(':', '') \
                                                .replace('?', '').replace('¿', '') \
                                                .replace('.', '').replace('/', '') \
                                                .replace('\\', '')
                    save_path = f'KPIs/Sueño_vs_Desvelo.png'
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig_scatter) 
                    print(f"Gráfico scatter guardado en: {save_path} ({len(tmp)} puntos válidos)")


                else:
                    print(f"No hay suficientes datos válidos para {p} (solo {len(tmp)} puntos)")

            except Exception as e:
                print(f"Error al procesar {p}: {str(e)}")
                continue