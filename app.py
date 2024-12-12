# Importamos librerías
import cv2
import easyocr
from io import BytesIO
import numpy as np
import os
import streamlit as st
import zipfile

# Crear el lector de EasyOCR
reader = easyocr.Reader(['en', 'es'])  # Idiomas: inglés y español

# Configuración
alto = 600
ancho = round(350 / 220 * alto)

# Función para detectar si una imagen tiene texto
# Si tiene texto, devuelve la coordenada Y inferior del bounding box. Si no, devuelve 0
def detect_text(image_path):
    # Detectar texto y coordenadas
    results = reader.readtext(image_path)

    # Si no se detecta texto, devolver 0
    if not results:
        return 0

    # Cargar la imagen con OpenCV en escala de grises
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape  # Altura (h) y ancho (w) de la imagen

    # Encontrar la coordenada Y más cercana al borde inferior, solo para texto negro
    # Esto se hace para evitar que detecte texto que aparezca en las marcas de calzado
    lowest_y = 0
    for (coords, text, prob) in results:
        # Extraer las coordenadas del bounding box
        x_min = int(min(coord[0] for coord in coords))
        y_min = int(min(coord[1] for coord in coords))
        x_max = int(max(coord[0] for coord in coords))
        y_max = int(max(coord[1] for coord in coords))

        # Recortar el área del bounding box
        roi = gray[y_min:y_max, x_min:x_max]

        # Comprobar si el texto es negro sobre fondo blanco
        text_mean = np.mean(roi)
        if text_mean < 128:  # El texto es oscuro (promedio bajo)
            bg_mean = np.mean(gray[y_min-10:y_min, x_min:x_max])  # Supuesto fondo blanco (fuera del bounding box)
            if bg_mean > 200:  # El fondo es claro (promedio alto)
                max_y = max(coord[1] for coord in coords)  # Máximo valor de Y
                if max_y > lowest_y:
                    lowest_y = max_y

    return lowest_y

# Función que detecta el zapato en la imagen y cambia su tamaño
def resize_and_detect_shoe(image_path, output_path, target_width=ancho, target_height=alto):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)
        shoe = image[y_min:y_max, x_min:x_max]
    else:
        raise ValueError("No se detectó ningún contorno en la imagen")

    shoe_h, shoe_w, _ = shoe.shape
    aspect_ratio = shoe_w / shoe_h
    target_width -= 2 * margen_horizontal
    target_height -= 2 * margen_vertical

    if aspect_ratio > (target_width / target_height):
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_shoe = cv2.resize(shoe, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.ones((target_height + 2 * margen_vertical, target_width + 2 * margen_horizontal, 3), dtype=np.uint8) * 255
    x_offset = (target_width + 2 * margen_horizontal - new_width) // 2
    y_offset = (target_height + 2 * margen_vertical - new_height) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_shoe

    cv2.imwrite(output_path, canvas)

# Procesa las imágenes en el fichero comprimido
def process_images_in_zip(zip_file, target_width=ancho, target_height=alto):
    temp_input_dir = "temp_input_images"
    temp_output_dir = "temp_output_images"
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_input_dir)

    image_files = [f for f in os.listdir(temp_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(image_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(image_files):
        input_path = os.path.join(temp_input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(temp_output_dir, output_filename)
        try:
            resize_and_detect_shoe(input_path, output_path, target_width, target_height)
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
        
        # Actualizar barra de progreso
        progress = (i + 1) / total_images
        progress_bar.progress(progress)
        status_text.text(f"Procesando {i + 1}/{total_images} imágenes...\nArchivo actual: {filename}")
    
    output_zip_path = BytesIO()
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for root, _, files in os.walk(temp_output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    output_zip_path.seek(0)

    for folder in [temp_input_dir, temp_output_dir]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)
    
    progress_bar.empty()
    status_text.text("Procesamiento completado.")
    
    return output_zip_path

# Interfaz de Streamlit
st.title("¡Hola ojos de cuquillo!")
st.write("Redimensionamos tus imágenes de zapatos. Sube un archivo ZIP con imágenes, las proceso y te las envío en un nuevo ZIP.")

# Agregar controles deslizantes para los márgenes
margen_horizontal = st.slider("Margen horizontal", min_value=0, max_value=100, value=20)
margen_vertical = st.slider("Margen vertical", min_value=0, max_value=100, value=20)

uploaded_file = st.file_uploader("Sube tu archivo ZIP con imágenes", type=["zip"])
if uploaded_file is not None:
    st.write("Procesando tu archivo...")
    result_zip = process_images_in_zip(uploaded_file)
    st.success("Procesamiento completado. Descarga el archivo abajo:")
    st.download_button(
        label="Descargar imágenes",
        data=result_zip,
        file_name="processed_images.zip",
        mime="application/zip"
    )
