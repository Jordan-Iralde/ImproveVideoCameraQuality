import cv2
import numpy as np
import itertools
import concurrent.futures
import time
import logging

# Configuración del logging
logging.basicConfig(filename='video_optimization.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_frame_quality(frame):
    """ Evalúa la calidad del frame basado en la suma de todos los valores de los píxeles. """
    return np.sum(frame)

def process_configuration(cap, resolution, d, sigmaColor, sigmaSpace):
    """ Procesa una configuración específica de parámetros y devuelve el frame filtrado y su calidad. """
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        time.sleep(0.1)  # Pausa para permitir que la cámara se ajuste

        ret, frame = cap.read()
        if not ret or frame is None:
            return None, 0

        filtered_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
        quality = evaluate_frame_quality(filtered_frame)

        return filtered_frame, quality

    except cv2.error as e:
        logging.error(f"Error de OpenCV procesando la configuración {resolution}, d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}: {e}")
        return None, 0
    except Exception as e:
        logging.error(f"Error inesperado procesando la configuración {resolution}, d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}: {e}")
        return None, 0

def optimize_video_parameters():
    """ Optimiza los parámetros de video buscando la mejor configuración. """
    print("Iniciando optimización de parámetros de video...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise ValueError("No se pudo abrir la cámara. Asegúrate de que está conectada y no está en uso por otra aplicación.")
        
        print("Cámara abierta correctamente.")

        # Parámetros de prueba
        d_values = [5, 9, 15]  # Valores seguros para evitar errores
        sigmaColor_values = [50, 75, 100]
        sigmaSpace_values = [50, 75, 100]
        resolutions = [(640, 480), (1280, 720)]

        best_quality = 0
        best_params = {}
        best_frame = None

        while True:
            configurations = itertools.product(resolutions, d_values, sigmaColor_values, sigmaSpace_values)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_configuration, cap, *config): config for config in configurations}

                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    config = futures[future]
                    try:
                        frame, quality = future.result()
                        print(f"Configuración {i}: Resolución={config[0]}, d={config[1]}, sigmaColor={config[2]}, sigmaSpace={config[3]}, Calidad={quality}")

                        if frame is not None and quality > best_quality:
                            best_quality = quality
                            best_params = {
                                "resolution": config[0],
                                "d": config[1],
                                "sigmaColor": config[2],
                                "sigmaSpace": config[3]
                            }
                            best_frame = frame
                            print(f"** Nueva mejor configuración encontrada: {best_params} con calidad: {best_quality} **")

                            # Mostrar comparación entre original y mejorado
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            ret, original_frame = cap.read()
                            if ret and original_frame is not None:
                                combined_frame = np.hstack((original_frame, best_frame))
                                cv2.imshow('Comparación: Original (Izquierda) vs Mejorado (Derecha)', combined_frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
                                    break
                    except Exception as e:
                        logging.error(f"Error procesando el futuro para configuración {config}: {e}")

            # Pausa para evitar sobrecargar recursos
            print("Esperando antes de continuar con la próxima iteración...")
            time.sleep(10)

    except ValueError as ve:
        logging.error(f"Error de valor: {ve}")
    except cv2.error as e:
        logging.error(f"Error con OpenCV: {e}")
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    optimize_video_parameters()
