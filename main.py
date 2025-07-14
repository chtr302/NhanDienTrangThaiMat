import cv2
from DriverDetection.driver_monitor import DriverMonitor

def main():
    monitor = DriverMonitor()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        try:
            results = monitor.process_frame(frame)
            controls = [
                "q=quit, t=tesselation, c=contours, i=iris, p=preprocessing, r=reset, s=stop alarm",
                f"T: {monitor.detection_config['tesselation']}, "
                f"C: {monitor.detection_config['contours']}, "
                f"I: {monitor.detection_config['irises']}, "
                f"P: {monitor.detection_config['use_preprocessing']}"
            ]
            
            for i, text in enumerate(controls):
                cv2.putText(results['frame'], text, 
                           (10, results['frame'].shape[0] - 40 + i*20), 
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
            monitor._DriverMonitor__stop_alarm()

    monitor._DriverMonitor__stop_alarm()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()