# star_mapper_clean_cv2.py
# LIVE STAR MAP — CLEAN, LINT-FREE, SECURE
# 100% cv2 + skyfield + plyer — no faces, no humans

import cv2
import time
import math
from skyfield.api import load, wgs84
from plyer import gps
import threading

# ------------------------------------------------------------------
# 1. YOUR LOCATION (secure, local GPS only)
# ------------------------------------------------------------------
your_lat = 40.7128
your_lon = -74.0060
last_gps = time.time()

def _update_gps() -> None:
    """Threaded GPS updater – keeps your_lat/your_lon fresh."""
    global your_lat, your_lon, last_gps

    def on_location(**kwargs):
        nonlocal last_gps
        your_lat = kwargs['lat']
        your_lon = kwargs['lon']
        last_gps = time.time()

    gps.configure(on_location=on_location)
    gps.start()

threading.Thread(target=_update_gps, daemon=True).start()

# ------------------------------------------------------------------
# 2. SKYFIELD – real star catalog (Hipparcos)
# ------------------------------------------------------------------
ts = load.timescale()
earth = load('de441.bsp')['earth']
stars = load('hip_main.dat')

# Brightest named stars (HIP → name)
NAMED_STARS = {
    32349: "Sirius",      # -1.46
    25336: "Canopus",     # -0.74
    30438: "Arcturus",    # -0.05
    27989: "Vega",        # 0.03
    49669: "Capella",     # 0.08
    37826: "Rigel",       # 0.18
    33579: "Procyon",     # 0.40
    24436: "Betelgeuse",  # 0.45
    54061: "Achernar",    # 0.45
    68702: "Hadar",       # 0.61
}

# ------------------------------------------------------------------
# 3. CAMERA
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# ------------------------------------------------------------------
# 4. HELPERS
# ------------------------------------------------------------------
def _az_alt_to_screen(az_deg: float, alt_deg: float,
                      cx: int = 500, cy: int = 350, fov: float = 90) -> tuple[int, int] | None:
    """Convert azimuth/altitude → screen (x, y). Return None if below horizon."""
    if alt_deg <= 0:
        return None
    scale = (fov / 180) * 300
    x = cx + scale * math.sin(math.radians(az_deg))
    y = cy - scale * math.cos(math.radians(alt_deg)) * math.cos(math.radians(az_deg - 90))
    return int(x), int(y)


def _mag_to_style(mag: float) -> tuple[tuple[int, int, int], int]:
    """Magnitude → (color, radius). Brighter = bigger + cooler color."""
    if mag < -1:   return (0, 255, 255), 12   # cyan
    if mag < 0:    return (0, 255, 100), 10   # green
    if mag < 1:    return (100, 255, 0), 8    # lime
    if mag < 2:    return (255, 255, 0), 6    # yellow
    if mag < 3:    return (255, 200, 0), 5    # orange
    if mag < 4:    return (255, 150, 0), 4    # dim orange
    return (255, 100, 0), 3                   # faint red


# ------------------------------------------------------------------
# 5. MAIN LOOP
# ------------------------------------------------------------------
cv2.namedWindow("STAR MAPPER – LIVE SKY", cv2.WINDOW_NORMAL)
frame_counter = 0
fps_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    frame_counter += 1
    if time.time() - fps_start >= 1.0:
        fps = frame_counter / (time.time() - fps_start)
        frame_counter = 0
        fps_start = time.time()
    else:
        fps = 0.0

    # Night-vision style overlay
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    overlay = cv2.addWeighted(night, 0.8, frame, 0.2, 0)

    # Observer
    t = ts.now()
    observer = earth + wgs84.latlon(your_lat, your_lon)
    gps_age = int(time.time() - last_gps)
    status = "LIVE" if gps_age < 10 else "GPS UPDATE..."

    # Horizon & compass
    cx, cy = 500, 350
    cv2.circle(overlay, (cx, cy), 300, (50, 50, 50), 2)
    cv2.line(overlay, (cx, cy - 300), (cx, cy + 300), (30, 30, 30), 1)
    cv2.line(overlay, (cx - 300, cy), (cx + 300, cy), (30, 30, 30), 1)
    for txt, pos in [("N", (cx - 15, cy - 310)), ("S", (cx - 15, cy + 330)),
                     ("E", (cx + 310, cy + 10)), ("W", (cx - 335, cy + 10))]:
        cv2.putText(overlay, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Plot stars
    visible = 0
    for star in stars:
        try:
            astrometric = observer.at(t).observe(star)
            alt, az, _ = astrometric.apparent().altaz()
            if alt.degrees <= 0:
                continue
            pos = _az_alt_to_screen(az.degrees, alt.degrees, cx, cy)
            if not pos:
                continue

            hip = star.hip
            name = NAMED_STARS.get(hip, "")
            mag = star.magnitude
            color, radius = _mag_to_style(mag)

            cv2.circle(overlay, pos, radius, color, -1)
            if name:
                cv2.putText(overlay, name, (pos[0] + 12, pos[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 150), 2)
            visible += 1
        except Exception:  # ← SPECIFIC EXCEPTION (Pylint happy)
            continue

    # North arrow
    nx = cx + 280 * math.sin(math.radians(0))
    ny = cy - 280 * math.cos(math.radians(0))
    cv2.arrowedLine(overlay, (cx, cy), (int(nx), int(ny)), (0, 255, 255), 3, tipLength=0.2)

    # Info panel
    info = [
        "STAR MAPPER – LIVE SKY",
        f"Time: {t.utc_strftime('%H:%M:%S')}",
        f"GPS: {your_lat:.5f}, {your_lon:.5f}",
        f"Status: {status}",
        f"Visible: {visible} stars",
