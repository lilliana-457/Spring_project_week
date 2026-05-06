import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import time
import pigpio
import math
import os

# --- Camera Setup ---
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(
    # Using YUV420 is much lighter on the Pi's memory
    main={"size": (320, 240), "format": "RGB888"}, # SCALED DOWN TO 320x240
    controls={
        "FrameRate": 50,
        "AeEnable": True,
        "AwbEnable": True, #False
        #"ColourGains": (1.5, 1.2)
#        "ExposureTime": 22000,
#        "AnalogueGain": 5.0
    }
)
picam2.configure(preview_config)
picam2.start()
time.sleep(0.01) 


# Folder where template images are saved and loaded from
TEMPLATE_DIR = "./orb_templates"
# The five symbols that ORB is trained to recognise.
# Each must have a saved template image in TEMPLATE_DIR to be detected.
SYMBOLS = ["BIOHAZARD", "RECYCLE", "QR_CODE", "FINGERPRINT", "BUTTON"]

def spin_360():
    print("Action: Executing 360 spin")
    # Set speed high enough to overcome friction
    set_pwm(ENL, 75) 
    set_pwm(ENR, 75)
    # Opposite directions for a zero-radius turn
    pi.write(IN1, 1); pi.write(IN2, 0) # Left Backwards
    pi.write(IN3, 0); pi.write(IN4, 1) # Right Forwards
    time.sleep(2.0) # Adjust this time until it completes exactly one circle
    stop()
def load_templates():
    """
    Scans TEMPLATE_DIR for saved PNG files matching each symbol name.
    For each found image, computes ORB keypoints and descriptors and stores
    them in the global orb_templates dict for use during detection.
    Called once at startup and again every time a new template is saved.
    """
    global loaded_templates
    loaded_templates = {}   # Clear existing templates before reloading

    if not os.path.exists(TEMPLATE_DIR):
        os.makedirs(TEMPLATE_DIR)   # Create the directory if it doesn't exist
        print(f"Directory {TEMPLATE_DIR} created.")
        return
    
    for sym in SYMBOLS:
        path = os.path.join(TEMPLATE_DIR, f"{sym}.png")   # Expected file path

        if not os.path.exists(path):
            continue   # Skip symbols that have no saved template yet

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # Load as grayscale — ORB doesn't use colour
        if img is not None:
            loaded_templates[sym] = img   # Store the loaded template image
            print(f"Successfully loaded: {sym}")

# --- Motor Pins & Pigpio Setup ---
ENL, IN1, IN2 = 13, 5, 6
ENR, IN3, IN4 = 12, 19, 26
pi = pigpio.pi()

# SCALED PID: Because the screen is half as wide, the 'error' is half as large.
# Doubling Kp and Kd keeps your steering strength exactly the same!
Kp = 1.0 # Was 0.4
Ki = 0.0
Kd = 0.8 # Was 0.4

# --- Near your other cooldown variables, add: ---
frame_count = 0
TEMPLATE_MATCH_INTERVAL = 4  # Only run the expensive template matching every N frames (
last_triangle_time = 0
TRIANGLE_COOLDOWN = 5.0  # seconds, same idea as stop_cooldown
nav_cooldown_until = 0
NAV_COOLDOWN = 2.5  # seconds
last_stop_time = 0
stop_cooldown = 5.0  # Seconds to wait before stopping for the SAME symbol again
x, y, w, h = 160, 0, 0, 0 # Default X scaled from 360 to 160
previous_error = 0
integral = 0
last_time = time.time()

base_speed = 40 #45
max_speed = 55 #80
min_speed = 0
for pin in [IN1, IN2, IN3, IN4, ENL, ENR]:
    pi.set_mode(pin, pigpio.OUTPUT)

def set_pwm(pin, duty_percent):
    pi.set_PWM_dutycycle(pin, int(255 * duty_percent / 100))

def stop():
    for pin in [IN1, IN2, IN3, IN4]: pi.write(pin, 0)
    set_pwm(ENL, 0); set_pwm(ENR, 0)

# Standard motor functions
def forward():
    pi.write(IN1, 0); pi.write(IN2, 1); pi.write(IN3, 0); pi.write(IN4, 1)
def turn_right():
    pi.write(IN1, 1); pi.write(IN2, 0); pi.write(IN3, 0); pi.write(IN4, 1)
def turn_left():
    pi.write(IN1, 0); pi.write(IN2, 1); pi.write(IN3, 1); pi.write(IN4, 0)

def detect_shape(cnt, thresh_frame):
    shape = ""
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    
    if peri == 0: return ""
    
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    # --- Solidity check: real arrows are solid, not hollow symbol edges ---
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return ""
    solidity = area / hull_area
    if solidity < 0.5:  # Too hollow — likely a symbol edge, not an arrow
        return ""

    # --- Aspect ratio: arrows are not too thin or too square ---
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 0 or h == 0: return ""
    aspect = w / h
    if aspect < 0.3 or aspect > 3.5:  # Too thin/tall or too wide — not an arrow
        return ""

    M = cv2.moments(cnt)
    if M["m00"] == 0: return ""
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if cy >= thresh_frame.shape[0] or cx >= thresh_frame.shape[1]: return ""

    # --- Centre pixel must be the target colour ---
    if thresh_frame[cy, cx] != 255:
        return ""

    circularity = (4 * math.pi * area) / (peri * peri)

    # --- Triangle: only if solidity is high and it's pointy enough ---
    # if len(approx) == 3:
    #     if solidity > 0.75 and circularity < 0.75:
    #         return "Biohazard"
    #     else:
    #         return ""  # Reject weak triangles

    # --- Arrow: must have 6-8 vertices (classic arrow polygon point count) ---
    if not (6 <= len(approx) <= 8):
        return ""

    # --- Find the tip (farthest vertex from centroid) ---
    maxdist = 0
    tip_idx = 0
    for i in range(len(approx)):
        dist = np.linalg.norm(approx[i][0] - (cx, cy))
        if dist > maxdist:
            maxdist = dist
            tip_idx = i

    # Tip must be meaningfully far from centre — rejects near-circular blobs
    if maxdist < 15:
        return ""

    tip_ang = math.atan2(cy - approx[tip_idx][0][1], approx[tip_idx][0][0] - cx) * 180 / math.pi
    arrow_ang = (tip_ang - 90) % 360

    print(f"DEBUG: Ang={arrow_ang:.1f} | Circ={circularity:.2f} | Solid={solidity:.2f} | Pts={len(approx)}")

    if 45 <= arrow_ang < 135:
        shape = "arrow (right)" if circularity <= 0.5 else "arrow (left)"
    elif 135 <= arrow_ang < 225:
        shape = "arrow (up)" if circularity <= 0.5 else "arrow (down)"
    elif 225 <= arrow_ang < 315:
        shape = "arrow (left)" if circularity <= 0.5 else "arrow (right)"
    else:
        shape = "arrow (down)" if circularity <= 0.5 else "arrow (up)"

    return shape

try:
    load_templates()
    error = 0
    while True:
        # shape = ""
        # detected_symbol = None
        current_time = time.time()
        in_nav_cooldown = current_time < nav_cooldown_until
        # 1. Capture in YUV
        yuv_frame = picam2.capture_array()
        image = yuv_frame
        
        # SCALED ROI: Old was [260:360, 0:640]. New is divided by 2.
        roi = image[130:180, 0:320] 
        kernel = np.ones((5, 5), np.uint8)
        
        thresh_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Changed from RGB to BGRA to match the camera output format
        HSVimage = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # SCALED PATCH: Old was [40:60, 300:340]. New is divided by 2.
        # patch = HSVimage[20:30, 150:170]  # centre of ROI
        # print(f"Yellow patch H={np.median(patch[:,:,0]):.1f} S={np.median(patch[:,:,1]):.1f} V={np.median(patch[:,:,2]):.1f}")
        
        #for arrows and symbols
        #redthresh_frame = cv2.inRange(thresh_frame, (115, 50, 50), (175, 255, 255))
        bluethresh_frame = cv2.inRange(thresh_frame, (100, 60, 60), (140, 255, 255))
        redthresh_frame1 = cv2.inRange(thresh_frame,(160,70,70),(180,255,255))
        redthresh_frame2=cv2.inRange(thresh_frame,(15,70,70),(25,255,255))
        redthresh_frame=cv2.bitwise_or(redthresh_frame1,redthresh_frame2)
        #greenthresh_frame=cv2.inRange(thresh_frame,(50,85,40),(90,255,255))
        greenthresh_frame = cv2.inRange(thresh_frame, (40, 60, 60), (90, 255, 255))
        #bluethresh_frame=cv2.inRange(thresh_frame,(100,80,60),(120,255,255))
        yellowthresh_frame = cv2.inRange(thresh_frame, (15, 100, 100), (99, 255, 255)) #prev 54
        # New Purple (approx 135-160)
        purplethresh_frame = cv2.inRange(thresh_frame, (125, 40, 30), (155, 255, 255))
       
        #for line detection
        Blackline=cv2.inRange(roi,(0,0,0),(100,100, 100))
        Redline1=cv2.inRange(HSVimage,(101,101,101),(200,255,255))
        Redline2=cv2.inRange(HSVimage,(0,100,100),(8,255,255))
        Redline=cv2.bitwise_or(Redline1,Redline2)
        #9, 100, 100
        Yellowline = cv2.inRange(HSVimage,(15, 60, 60), (98, 255, 255))
    
        #line erosion
        kernel_thin = np.ones((5, 5), np.uint8)
        Redline=cv2.erode(Redline,kernel,iterations=2)
        Yellowline=cv2.erode(Yellowline,kernel_thin,iterations=1)
        Blackline=cv2.erode(Blackline,kernel,iterations=2)

        #finding contours and hierarchies for line detection
        blackcontours, blackhierarchy = cv2.findContours(Blackline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        redcontours, redhierarchy = cv2.findContours(Redline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellowcontours, yellowhierarchy = cv2.findContours(Yellowline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #clean up images for arrows/symbols. Add erosion if needed !
        redthresh_frame = cv2.GaussianBlur(redthresh_frame, (5, 5), 0)
        greenthresh_frame = cv2.GaussianBlur(greenthresh_frame, (5, 5), 0)
        bluethresh_frame = cv2.GaussianBlur(bluethresh_frame, (3, 3), 0)
        yellowthresh_frame = cv2.GaussianBlur(yellowthresh_frame, (5, 5), 0)
        purplethresh_frame = cv2.GaussianBlur(purplethresh_frame, (5, 5), 0)
        #applying canny edge detection for arrows/symbols.
        rededges = redthresh_frame
        greenedges = greenthresh_frame
        blueedges = bluethresh_frame
        yellowedges = yellowthresh_frame
        purpleedges = purplethresh_frame
        # rededges = cv2.Canny(redthresh_frame, 30, 100)
        # greenedges = cv2.Canny(greenthresh_frame, 30, 100)
        # blueedges = cv2.Canny(bluethresh_frame, 30, 100)
        # yellowedges = cv2.Canny(yellowthresh_frame, 30, 100)
        # purpleedges = cv2.Canny(purplethresh_frame, 30, 100)

        #finding contours and hierarchies for arrows/symbols
        red_symbol_contours, _ = cv2.findContours(rededges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        green_symbol_contours, _ = cv2.findContours(greenedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blue_symbol_contours, _ = cv2.findContours(blueedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellow_symbol_contours, _ = cv2.findContours(yellowedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
        purple_symbol_contours, _ = cv2.findContours(purpleedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shape = ""
        detected_symbol = None
        frame_count += 1
        if not in_nav_cooldown and frame_count % TEMPLATE_MATCH_INTERVAL == 0:
            best_symbol = None
            best_score = 0
            best_loc = None
            best_size = None

            symbol_thresholds = {
                "BIOHAZARD": 0.3,
                "RECYCLE": 0.45,
                "QR_CODE": 0.45,
                "FINGERPRINT": 0.5,
                "BUTTON": 0.5
            }
            
            for name, temp_img in loaded_templates.items():
                w, h = temp_img.shape[::-1]

                res = cv2.matchTemplate(gray_frame, temp_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                threshold = symbol_thresholds.get(name, 0.5)  # Default threshold if not specified

                if max_val >= threshold and max_val > best_score:
                    best_symbol = name
                    best_score = max_val
                    best_loc = max_loc
                    best_size = (w, h)

            if best_symbol is not None:
                detected_symbol = best_symbol
                w, h = best_size
                cv2.rectangle(image, best_loc, (best_loc[0] + w, best_loc[1] + h), (255, 0, 255), 3)
                print(f"Detected {detected_symbol} | score={best_score:.2f}")

        if detected_symbol is not None:
            action_taken = False
            if current_time - last_stop_time > stop_cooldown:
                if detected_symbol in ["BIOHAZARD", "BUTTON"]:
                    print(f"Action: {detected_symbol} detected, stopping")
                    stop()
                    time.sleep(1)
                    last_stop_time = time.time()
                    nav_cooldown_until = current_time + NAV_COOLDOWN
                    action_taken = True

                elif detected_symbol == "RECYCLE":
                    print("Action: RECYCLE detected, turning 360")
                    spin_360()
                    last_stop_time = time.time()
                    nav_cooldown_until = current_time + NAV_COOLDOWN
                    action_taken = True
            
                elif detected_symbol in ["FINGERPRINT", "QR_CODE"]:
                    print("BIOMETRICS")
                    time.sleep(1)
                    last_stop_time = time.time()
                    nav_cooldown_until = current_time + NAV_COOLDOWN
                    action_taken = True
            if action_taken:
                continue
                
        
        if not in_nav_cooldown:
            color_mask = cv2.bitwise_or(redthresh_frame, greenthresh_frame)
            cleaned    = cv2.bitwise_or(color_mask, bluethresh_frame)

            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 500 or area > 25000: continue
                shape = detect_shape(cnt, cleaned)
                if shape != "": 
                    print(shape); 
                    break

            # if shape == "":
            #     for cnt in green_symbol_contours:
            #         area = cv2.contourArea(cnt)
            #         if area > 2000:
            #             shape = detect_shape(cnt, greenthresh_frame)
            #             if shape != "": print(shape); break

            # if shape == "":
            #     for cnt in blue_symbol_contours:
            #         area = cv2.contourArea(cnt)
            #         if area > 2000:
            #             shape = detect_shape(cnt, bluethresh_frame)
            #             if shape != "": print(shape); break

            # if shape == "":
            #     for cnt in yellow_symbol_contours:
            #         area = cv2.contourArea(cnt)
            #         if area > 2000:
            #             shape = detect_shape(cnt, yellowthresh_frame)
            #             if shape != "": print(shape); break

            if "left" in shape:
                set_pwm(ENL, 70); 
                set_pwm(ENR, 70)
                turn_left(); 
                time.sleep(0.5)
                print("Turning left")
                nav_cooldown_until = current_time + NAV_COOLDOWN
                previous_error = 0

            elif "right" in shape:
                set_pwm(ENL, 70); set_pwm(ENR, 70)
                turn_right(); time.sleep(0.5)
                print("Turning right")
                nav_cooldown_until = current_time + NAV_COOLDOWN
                previous_error = 0

            elif "Biohazard" in shape:
                if current_time - last_triangle_time > TRIANGLE_COOLDOWN:  # ← fixed
                    print("Action: Biohazard shape detected, stopping")
                    stop()
                    time.sleep(1)
                    last_triangle_time = time.time()
                    nav_cooldown_until = current_time + NAV_COOLDOWN

        # --- Line following + display always runs, regardless of cooldown ---
        if len(redcontours) > 0:
            c = max(redcontours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.line(image, (160, 215), (int(x + (w / 2)), 155), (255, 0, 0), 3)
            #print(f"Red contour found at X={x + (w / 2)} | Error={int(x + (w / 2)) - 160}")

        elif len(yellowcontours) > 0:
            c = max(yellowcontours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.line(image, (160, 215), (int(x + (w / 2)), 155), (0, 255, 255), 3)
            #print(f"Yellow contour found at X={x + (w / 2)} | Error={int(x + (w / 2)) - 160}")

        elif len(blackcontours) > 0:
            c = max(blackcontours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.line(image, (160, 215), (int(x + (w / 2)), 155), (0, 0, 255), 3)

        error = int(x + (w / 2)) - 160
        cv2.putText(image, str(error), (140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"R:{len(redcontours)} Y:{len(yellowcontours)} B:{len(blackcontours)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        cv2.imshow("Contour", image)
        cv2.imshow("Red roi", Redline)
        cv2.imshow("Yellow roi", Yellowline)
        cv2.imshow("Black roi", Blackline)

        # Only drive motors if no arrow/symbol action was taken this frame
        if "left" not in shape and "right" not in shape and "Biohazard" not in shape:
            current_time = time.time()
            dt = max(current_time - last_time, 0.01)
            last_time = current_time

            P = Kp * error
            integral += error * dt
            I = Ki * integral
            derivative = (error - previous_error) / dt if dt > 0 else 0
            D = Kd * derivative

            pid = P + I + D
            previous_error = error

            left_speed = max(min_speed, min(max_speed, base_speed + pid))
            right_speed = max(min_speed, min(max_speed, base_speed - pid))

            set_pwm(ENL, left_speed)
            set_pwm(ENR, right_speed)
            forward()

        if len(redcontours) == 0 and len(yellowcontours) == 0 and len(blackcontours) == 0:
            if previous_error > 0:
                set_pwm(ENL, 60); set_pwm(ENR, 60); turn_right()
            else:
                set_pwm(ENL, 60); set_pwm(ENR, 60); turn_left()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except Exception as e:
    print(f"Error: {e}")

finally:
    stop()
    picam2.stop()
    cv2.destroyAllWindows()
    pi.stop()