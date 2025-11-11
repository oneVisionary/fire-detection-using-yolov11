import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("./models/fire_detector.pt")

def detect_fire_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    total_frames = 0
    fire_frames = 0
    no_fire_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Predict on frame
        results = model(frame, verbose=False)

        fire_detected = False

        for result in results:
            if result.boxes and len(result.boxes) > 0:
                fire_detected = True

        if fire_detected:
            print("🔥 FIRE DETECTED")
            fire_frames += 1
        else:
            print("✅ NO FIRE")
            no_fire_frames += 1

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    print("\n========== SUMMARY ==========")
    print("Total Frames:", total_frames)
    print("Frames with Fire:", fire_frames)
    print("Frames with No Fire:", no_fire_frames)
    print("Fire Percentage: {:.2f}%".format((fire_frames / total_frames) * 100 if total_frames > 0 else 0))
    print("No Fire Percentage: {:.2f}%".format((no_fire_frames / total_frames) * 100 if total_frames > 0 else 0))

def main():
    detect_fire_in_video("./data/input/test1.mp4")

main()
