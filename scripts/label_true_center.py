import cv2
import sys

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def main():
    global clicked_point
    
    if len(sys.argv) < 2:
        print("Usage: python label_true_center.py path_to_image")
        return

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not open image: {image_path}")
        return

    display = img.copy()
    cv2.namedWindow("Label True Center", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Label True Center", mouse_callback)

    print("\nInstructions:")
    print("- Click where you think the true eye center should be")
    print("- Press 'r' to reset")
    print("- Press 's' to save/print the current point")
    print("- Press 'q' to quit without saving\n")

    while True:
        temp = display.copy()

        if clicked_point is not None:
            x, y = clicked_point
            cv2.drawMarker(temp, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                temp,
                f"True center: ({x}, {y})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Label True Center", temp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            clicked_point = None

        elif key == ord("s"):
            if clicked_point is not None:
                x, y = clicked_point
                print(f"Saved point: true_x={x}, true_y={y}")
                break
            else:
                print("No point selected yet.")

        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()