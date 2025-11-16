# star_mapper_cv2.py
# DETECT STARS ABOVE YOU — LIVE FROM YOUR PHONE CAMERA
# Draws real star map on screen with colors + names
# 100% offline, secure, no faces, no humans
# Uses GPS + time + skyfield + cv2
import cv2
import numpy as np
import time
import math
from skyfield.api import load, wgs84
from plyer import gps
import threading

# ———— YOUR LOCATION (SECURE) ————
your_lat, your_lon = 40.7128, -74.0060
last_gps = time.time()

def update_gps():
    global your_lat, your_lon, last_gps
    def on_loc(**kwargs):
        global your_lat, your_lon, last_gps
        your_lat, your_lon = kwargs['lat'], kwargs['lon']
        last_gps = time.time()
    gps.configure(on_location=on_loc)
    gps.start()

threading.Thread(target=update_gps, daemon=True).start()

# ———— SKYFIELD: REAL STAR CATALOG ————
ts = load.timescale()
bodies = load('de441.bsp')
earth = bodies['earth']
you = earth + wgs84.latlon(your_lat, your_lon)

# Load bright stars (Hipparcos subset)
stars = load('hip_main.dat')
star_names = {
    32349: "Sirius",      # Alpha CMa
    25336: "Canopus",     # Alpha Car
    30438: "Arcturus",    # Alpha Boo
    27989: "Vega",        # Alpha Lyr
    49669: "Capella",     # Alpha Aur
    37826: "Rigel",       # Beta Ori
    33579: "Procyon",     # Alpha CMi
    24436: "Betelgeuse",  # Alpha Ori
    54061: "Achernar",    # Alpha Eri
    68702: "Hadar",       # Beta Cen
}

# ———— CAMERA ————
cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 700)

# ———— STAR PROJECTION (AZIMUTH/ALTITUDE → SCREEN) ————
def sky_to_screen(az, alt, cx=500, cy=350, fov=90):
    # az: 0=N, 90=E, 180=S, 270=W
    # alt: 0=horizon, 90=zenith
    if alt < 0:
        return None
    scale = (fov / 180) * 300
    x = cx + scale * math.sin(math.radians(az))
    y = cy - scale * math.cos(math.radians(alt)) * math.cos(math.radians(az - 90))
    return int(x), int(y)

# ———— MAIN LOOP ————
cv2.namedWindow("STAR MAPPER — LIVE SKY", cv2.WINDOW_NORMAL)
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if time.time() - start_time > 1:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()

    overlay = frame.copy()

    # ———— GET YOUR VIEW ————
    t = ts.now()
    you = earth + wgs84.latlon(your_lat, your_lon)
    gps_age = int(time.time() - last_gps)
    status = "LIVE" if gps_age < 10 else "GPS UPDATING..."

    # ———— DRAW HORIZON & COMPASS ————
    cx, cy = 500, 350
    cv2.circle(overlay, (cx, cy), 300, (100,100,100), 2)  # FOV circle
    cv2.line(overlay, (cx, cy-300), (cx, cy+300), (50,50,50), 1)  # N-S
    cv2.line(overlay, (cx-300, cy), (cx+300, cy), (50,50,50), 1)  # E-W
    cv2.putText(overlay, "N", (cx-10, cy-310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(overlay, "S", (cx-10, cy+330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(overlay, "E", (cx+310, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(overlay, "W", (cx-330, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    # ———— PLOT VISIBLE STARS ————
    visible_stars = 0
    for star in stars:
        try:
            astrometric = you.at(t).observe(star)
            alt, az, _ = astrometric.apparent().altaz()
            if alt.degrees > 0:
                screen_pos = sky_to_screen(az.degrees, alt.degrees, cx, cy)
                if screen_pos:
                    hip = star.hip
                    name = star_names.get(hip, "")
                    mag = star.magnitude
                    brightness = max(3, int(15 - mag * 2))
                    color = (255, 255, 0) if hip in star_names else (200, 200, 255)
                    cv2.circle(overlay, screen_pos, brightness, color, -1)
                    if name:
                        cv2.putText(overlay, name, (screen_pos[0]+10, screen_pos[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 2)
                    visible_stars += 1
        except:
            continue

    # ———— DATA PANEL ————
    lines = [
        "STAR MAPPER — LIVE SKY ABOVE YOU",
        f"Time: {t.utc_strftime('%H:%M:%S')}",
        f"GPS: {your_lat:.5f}, {your_lon:.5f}",
        f"Status: {status}",
        f"Visible: {visible_stars} stars",
        f"FPS: {fps:.1f}",
        "Point phone up. See the sky."
    ]
    for i, line in enumerate(lines):
        color = (255, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(overlay, line, (20, 50 + i * 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    # ———— COMPASS ARROW (NORTH) ————
    north_x = cx + 280 * math.sin(math.radians(0))
    north_y = cy - 280 * math.cos(math.radians(0))
    cv2.arrowedLine(overlay, (cx, cy), (int(north_x), int(north_y)), (0,255,255), 3, tipLength=0.2)

    cv2.imshow("STAR MAPPER — LIVE SKY", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
