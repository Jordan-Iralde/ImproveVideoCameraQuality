import cv2
import numpy as np
import concurrent.futures
import itertools

def evaluate_frame_quality(frame):
    # Evaluar la calidad del frame basándose en la suma de todos los valores de píxeles.
    return np.sum(frame)

def process_configuration(cap, resolution, d, sigmaColor, sigmaSpace):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    for _ in range(10):  # Intentar capturar el frame hasta 10 veces
        ret, frame = cap.read()
        if ret and frame is not None:
            break

    if not ret or frame is None:
        return None, 0

    filtered_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
    quality = evaluate_frame_quality(filtered_frame)
    
    return filtered_frame, quality

def optimize_video_parameters():
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Definir los valores para las pruebas
        d_values = [5, 9, 15, 21, 25, 30]
        sigmaColor_values = [50, 75, 100, 125, 150, 200]
        sigmaSpace_values = [50, 75, 100, 125, 150, 200]
        resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
        
        configurations = list(itertools.product(resolutions, d_values, sigmaColor_values, sigmaSpace_values))
        
        # Crear una lista de configuraciones para las pruebas
        configurations = configurations[:200]  # Limitar a las primeras 200 configuraciones

        best_quality = 0
        best_params = {}
        best_frame = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for config in configurations:
                futures.append(executor.submit(process_configuration, cap, *config))

            for future in concurrent.futures.as_completed(futures):
                frame, quality = future.result()
                if quality > best_quality:
                    best_quality = quality
                    best_params = {
                        "resolution": config[0],
                        "d": config[1],
                        "sigmaColor": config[2],
                        "sigmaSpace": config[3]
                    }
                    best_frame = frame

        cap.release()
        cv2.destroyAllWindows()

        print(f"Mejor configuración encontrada: {best_params} con calidad: {best_quality}")

        if best_frame is not None:
            cv2.imshow('Mejor Video', best_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except cv2.error as e:
        print(f"Error con OpenCV: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    optimize_video_parameters()
