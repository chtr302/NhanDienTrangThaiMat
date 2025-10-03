"""
Main entry point cho hệ thống nhận diện ngủ gật
"""

import argparse
import sys
import os

# Ensure Keras 3 uses TensorFlow backend before any keras import
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def main():
    parser = argparse.ArgumentParser(description="Hệ Thống Nhận Diện Ngủ Gật")
    parser.add_argument("--cli", action="store_true", help="Chạy chế độ CLI với OpenCV window")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    args = parser.parse_args()

    if args.cli:
        # Chế độ CLI - sử dụng OpenCV window
        run_cli_mode(args.camera)
    else:
        # Chế độ GUI - sử dụng PyQt5
        run_gui_mode()


def run_cli_mode(camera_id=0):
    """Chạy chế độ CLI với OpenCV"""
    try:
        import cv2
        from DriverDetection.driver_monitor import DriverMonitor

        # Code CLI mode
        monitor = DriverMonitor()
        enable_yawn_detection = True

        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            try:
                monitor.update_config(enable_yawn_detection=enable_yawn_detection)
                results = monitor.process_frame(frame)
                controls = [
                    "q=quit, t=tesselation, c=contours, i=iris, p=preprocessing, r=reset, s=stop alarm, x=enable/disable yawn",
                    f"T: {monitor.detection_config['tesselation']}, "
                    f"C: {monitor.detection_config['contours']}, "
                    f"I: {monitor.detection_config['irises']}, "
                    f"P: {monitor.detection_config['use_preprocessing']}"
                ]
                yawn_status = "Yawn Detection: Enabled" if monitor.detection_config['enable_yawn_detection'] else "Yawn Detection: Disabled"
                yawn_color = (0, 255, 0) if monitor.detection_config['enable_yawn_detection'] else (0, 0, 255)
                cv2.putText(results['frame'], yawn_status,
                            (10, results['frame'].shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)

                for i, text in enumerate(controls):
                    cv2.putText(results['frame'], text,
                               (10, results['frame'].shape[0] - 60 + i*20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow('Driver Monitoring System', results['frame'])
            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.imshow('Driver Monitoring System', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('t'):
                monitor.detection_config['tesselation'] = not monitor.detection_config['tesselation']
            elif key == ord('c'):
                monitor.detection_config['contours'] = not monitor.detection_config['contours']
            elif key == ord('i'):
                monitor.detection_config['irises'] = not monitor.detection_config['irises']
            elif key == ord('p'):
                monitor.detection_config['use_preprocessing'] = not monitor.detection_config['use_preprocessing']
            elif key == ord('r'):
                monitor.detection_config.update({
                    'tesselation': True,
                    'contours': True,
                    'irises': True
                })
            elif key == ord('s'):
                # Stop alarm - không có method này trong DriverMonitor
                pass
            elif key == ord('x'):
                enable_yawn_detection = not enable_yawn_detection

        cap.release()
        cv2.destroyAllWindows()

    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("Hãy cài đặt dependencies bằng: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Lỗi CLI mode: {e}")


def run_gui_mode():
    """Chạy chế độ GUI với PyQt5"""
    try:
        from DriverDetectionApp.frontend.main_window import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"❌ Lỗi import GUI: {e}")
        print("Hãy kiểm tra cấu trúc thư mục và dependencies")
    except Exception as e:
        print(f"❌ Lỗi GUI mode: {e}")


if __name__ == "__main__":
    main()
