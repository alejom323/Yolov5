import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5 de manera compatible con versiones anteriores de PyTorch
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        # Importar yolov5
        import yolov5
        
        # Para versiones de PyTorch anteriores a 2.0, cargar directamente con weights_only=False
        # o usar el parámetro map_location para asegurar compatibilidad
        try:
            # Primer método: cargar con weights_only=False si la versión lo soporta
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            # Segundo método: si el primer método falla, intentar un enfoque más básico
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                # Si todo falla, intentar cargar el modelo con torch directamente
                st.warning(f"Intentando método alternativo de carga...")
                
                # Modificar sys.path temporalmente para poder importar torch correctamente
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                
                # Cargar el modelo con torch directamente
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar una versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo del modelo en la ubicación correcta
        3. Si el problema persiste, intenta descargar el modelo directamente de torch hub
        """)
        return None

# Título y descripción de la aplicación
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

# Cargar el modelo
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# Si el modelo se cargó correctamente, configuramos los parámetros
if model:
    # Sidebar para los parámetros de configuración
    st.sidebar.title("Parámetros")
    
    # Ajustar parámetros del modelo
    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        
        # Opciones adicionales
        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")
    
    # Contenedor principal para la cámara y resultados
    main_container = st.container()
    
    with main_container:
        # Capturar foto con la cámara
        picture = st.camera_input("Capturar imagen", key="camera")
        
        if picture:
            # Procesar la imagen capturada
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Realizar la detección
            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            
            # Parsear resultados
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                
                # Mostrar resultados
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Imagen con detecciones")
                    # Renderizar las detecciones
                    results.render()
                    # Mostrar imagen con las detecciones
                    st.image(cv2_img, channels='BGR', use_container_width=True)
                
                with col2:
                    st.subheader("Objetos detectados")
                    
                    # Obtener nombres de etiquetas
                    label_names = model.names
                    
                    # Contar categorías
                    category_count = {}
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        if category_idx in category_count:
                            category_count[category_idx] += 1
                        else:
                            category_count[category_idx] = 1
                    
                    # Crear dataframe para mostrar resultados
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Mostrar gráfico de barras
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# Información adicional y pie de página
st.markdown("---")
st.caption("""
**Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real.
Desarrollada con Streamlit y PyTorch.
""")
