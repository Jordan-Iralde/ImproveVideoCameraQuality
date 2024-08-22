import cv2
import numpy as np
import itertools
import concurrent.futures
import time

def evaluate_frame_quality(frame):
    return np.sum(frame)

def process_configuration(cap, resolution, d, sigmaColor, sigmaSpace):
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        ret, frame = cap.read()
        if not ret or frame is None:
            return None, 0

        filtered_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
        quality = evaluate_frame_quality(filtered_frame)

        return filtered_frame, quality
    
    except cv2.error as e:
        print(f"Error procesando la configuración {resolution}, d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}: {e}")
        return None, 0
    except Exception as e:
        print(f"Error inesperado procesando la configuración {resolution}, d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}: {e}")
        return None, 0

def optimize_video_parameters():
    print("Iniciando optimización de parámetros de video...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise ValueError("No se pudo abrir la cámara. Asegúrate de que está conectada y no está en uso por otra aplicación.")
        
        print("Cámara abierta correctamente.")

        # Valores de prueba
        d_values = [5, 9, 15, 21, 25, 30]
        sigmaColor_values = [50, 75, 100, 125, 150, 200]
        sigmaSpace_values = [50, 75, 100, 125, 150, 200]
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]

        # Inicializar variables para la mejor calidad
        best_quality = 0
        best_params = {}
        best_frame = None

        while True:
            # Generar configuraciones en cada iteración
            configurations = itertools.product(resolutions, d_values, sigmaColor_values, sigmaSpace_values)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(process_configuration, cap, *config): config for config in configurations}

                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    config = futures[future]
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
                        
                        # Mostrar la mejor configuración en comparación con el video original
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        ret, original_frame = cap.read()
                        if ret and original_frame is not None:
                            combined_frame = np.hstack((original_frame, best_frame))
                            cv2.imshow('Comparación: Original (Izquierda) vs Mejorado (Derecha)', combined_frame)
                            cv2.waitKey(1)  # Mostrar la ventana y actualizarla

            # Esperar un tiempo antes de continuar con la siguiente iteración
            print("Esperando antes de continuar con la próxima iteración...")
            time.sleep(10)

    except ValueError as ve:
        print(f"Error de valor: {ve}")
    except cv2.error as e:
        print(f"Error con OpenCV: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    optimize_video_parameters()
