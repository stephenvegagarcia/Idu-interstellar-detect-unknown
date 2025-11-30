import pygame
import random
import sys
import math
import numpy as np
import logging
import threading
import time
import requests
import io
import json
from collections import deque
from skyfield.api import load, wgs84

# ==============================================================================
#  SOLAR DUAL MONITOR | QUANTUM SIMULATION + NASA SDO UPLINK
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
pygame.init()

# Try audio init
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    SOUND_AVAILABLE = True
except:
    SOUND_AVAILABLE = False

# --- LAYOUT CONFIGURATION ---
WIDTH, HEIGHT = 1600, 900 # Wider to fit 3 panels
SIDE_PANEL_W = 350
VIEW_TOTAL_W = WIDTH - SIDE_PANEL_W
VIEW_SPLIT_W = VIEW_TOTAL_W // 2 # Split the remaining space in two

# Screen Areas
RECT_3D = pygame.Rect(0, 0, VIEW_SPLIT_W, HEIGHT)
RECT_REAL = pygame.Rect(VIEW_SPLIT_W, 0, VIEW_SPLIT_W, HEIGHT)
RECT_HUD = pygame.Rect(WIDTH - SIDE_PANEL_W, 0, SIDE_PANEL_W, HEIGHT)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SOLAR DUAL MONITOR | 3D MODEL VS REALITY")

# --- COLORS ---
BLACK = (5, 5, 10)
WHITE = (255, 255, 255)
SOLAR_CORE = (255, 255, 200) 
SOLAR_SURFACE = (255, 180, 0) 
SOLAR_CORONA = (255, 50, 50) 
HUD_BG = (20, 10, 5) 
TEXT_ORANGE = (255, 150, 50)
ALERT_RED = (255, 50, 50)
AI_CYAN = (0, 255, 255)
GRID_COL = (40, 20, 20)

# --- 3D CONSTANTS ---
FOV = 600
# Center the 3D sun in the LEFT panel
PROJECTION_CENTER = (VIEW_SPLIT_W // 2, HEIGHT // 2)

# --- GLOBAL STATE ---
cam_rot_x = 0.0
cam_rot_y = 0.0
cam_zoom = 1.0
particles = [] 

current_flux_val = 1e-8 
flux_trend = "STABLE"
solar_activity_index = 0.0 
q_bell_state = 0.0

# NASA Image Container
sdo_image_surface = None
sdo_last_update = "CONNECTING..."

# --- AUDIO ENGINE ---
def generate_solar_hum(intensity):
    if not SOUND_AVAILABLE: return None
    duration = 0.5
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    
    # Pitch shifts with intensity
    base_freq = 40 + (intensity * 60) 
    
    wave = 0.5 * np.sin(2 * np.pi * base_freq * t)
    wave += 0.3 * np.sin(2 * np.pi * (base_freq * 1.02) * t)
    
    # Add deep rumble
    noise = np.random.normal(0, 0.1, n_samples)
    modulator = np.sin(2 * np.pi * 5 * t)
    wave += noise * modulator * intensity * 0.5
    
    sound_array = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((sound_array, sound_array)))

current_sound = None
sound_enabled = True
last_sound_update = 0

# ==============================================================================
#  <<< LIVE DATA THREAD (NUMBERS) >>>
# ==============================================================================
def fetch_flux_data():
    global current_flux_val, flux_trend, solar_activity_index
    URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
    
    while True:
        try:
            r = requests.get(URL, timeout=5)
            if r.status_code == 200:
                data = r.json()
                current_flux_val = data[-1].get('flux', 1e-8)
                
                # Normalize 0.0 - 1.0
                solar_activity_index = (math.log10(current_flux_val) + 9) / 5
                solar_activity_index = np.clip(solar_activity_index, 0.0, 1.0)
                
                if current_flux_val > 1e-5: flux_trend = "FLARE (M/X)"
                elif current_flux_val > 1e-6: flux_trend = "ACTIVE (C)"
                else: flux_trend = "QUIET"
        except: pass
        time.sleep(60)

threading.Thread(target=fetch_flux_data, daemon=True).start()

# ==============================================================================
#  <<< NASA IMAGE THREAD (VISUALS) >>>
# ==============================================================================
def fetch_sdo_image():
    global sdo_image_surface, sdo_last_update
    # AIA 171 = Gold/Yellow (Corona) - Best for visualizing structure
    URL_IMG = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg"
    
    while True:
        try:
            r = requests.get(URL_IMG, timeout=15)
            if r.status_code == 200:
                # Load image from RAM
                image_bytes = io.BytesIO(r.content)
                raw_img = pygame.image.load(image_bytes)
                
                # Scale to fit the Center Panel
                sdo_image_surface = pygame.transform.smoothscale(raw_img, (VIEW_SPLIT_W, HEIGHT))
                sdo_last_update = time.strftime("%H:%M UTC")
        except Exception as e:
            print(f"SDO Error: {e}")
            
        time.sleep(300) # Updates every 5 minutes

threading.Thread(target=fetch_sdo_image, daemon=True).start()

# ==============================================================================
#  <<< 3D GENERATION >>>
# ==============================================================================
def project_3d_point(x, y, z):
    xz = x * math.cos(cam_rot_y) - z * math.sin(cam_rot_y)
    zz = z * math.cos(cam_rot_y) + x * math.sin(cam_rot_y)
    yz = y * math.cos(cam_rot_x) - zz * math.sin(cam_rot_x)
    zz = zz * math.cos(cam_rot_x) + y * math.sin(cam_rot_x)
    
    dist = 600
    fz = dist + zz
    if fz <= 0: return None
    scale = (FOV * cam_zoom) / fz
    
    # Project to the LEFT PANEL Center
    px = int(xz * scale + PROJECTION_CENTER[0])
    py = int(yz * scale + PROJECTION_CENTER[1])
    return px, py, scale

def generate_sun_structure(quantum_stability):
    global particles
    particles = []
    
    # 1. Surface
    num_points = 500
    phi = math.pi * (3. - math.sqrt(5.))
    turbulence = solar_activity_index * 15
    
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i 
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        # Pulse
        r = 150 + (math.sin(pygame.time.get_ticks() * 0.002 + y*5) * turbulence)
        
        if random.random() < solar_activity_index: col = SOLAR_CORE
        elif random.random() < 0.8: col = SOLAR_SURFACE
        else: col = SOLAR_CORONA
            
        particles.append([x * r, y * r, z * r, col, 3])

    # 2. Corona
    corona_count = 300
    expansion = (1.0 - quantum_stability) * 100 
    
    for i in range(corona_count):
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, math.pi)
        r = 180 + random.uniform(0, 50 + expansion)
        
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        
        particles.append([x, y, z, SOLAR_CORONA, 1])

# ==============================================================================
#  <<< MAIN RENDER LOOP >>>
# ==============================================================================
clock = pygame.time.Clock()
running = True
mouse_down = False
last_mouse_pos = (0, 0)

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_m: sound_enabled = not sound_enabled
            if e.key == pygame.K_w: cam_zoom = min(2.0, cam_zoom + 0.1)
            if e.key == pygame.K_s: cam_zoom = max(0.5, cam_zoom - 0.1)
        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.pos[0] < VIEW_SPLIT_W: # Only rotate if clicking left screen
                mouse_down = True; last_mouse_pos = e.pos
        if e.type == pygame.MOUSEBUTTONUP: mouse_down = False
        if e.type == pygame.MOUSEMOTION and mouse_down:
            dx, dy = e.pos[0] - last_mouse_pos[0], e.pos[1] - last_mouse_pos[1]
            cam_rot_y += dx * 0.01; cam_rot_x += dy * 0.01; last_mouse_pos = e.pos

    # Logic
    t = pygame.time.get_ticks() / 1000.0
    noise = solar_activity_index * 0.5
    angle = (math.pi/2) + (math.sin(t) * noise)
    q_bell_state = max(0, 1.0 - abs(angle - (math.pi/2)) * 2)
    
    generate_sun_structure(q_bell_state)
    
    # Sound
    now = pygame.time.get_ticks()
    if now - last_sound_update > 500:
        if sound_enabled and SOUND_AVAILABLE:
            if current_sound: current_sound.stop()
            current_sound = generate_solar_hum(solar_activity_index)
            if current_sound: current_sound.play(-1)
        elif not sound_enabled and current_sound:
            current_sound.stop()
        last_sound_update = now

    screen.fill(BLACK)
    
    # --- PANEL 1: 3D SIMULATION (LEFT) ---
    pygame.draw.rect(screen, BLACK, RECT_3D)
    
    # Draw Stars in Left Panel
    for _ in range(30):
        sx = random.randint(0, VIEW_SPLIT_W)
        sy = random.randint(0, HEIGHT)
        screen.set_at((sx, sy), WHITE)

    # Draw 3D Sun
    def get_z(p): return p[2] * math.cos(cam_rot_y) + p[0] * math.sin(cam_rot_y)
    particles.sort(key=get_z, reverse=True)

    for p in particles:
        proj = project_3d_point(p[0], p[1], p[2])
        if proj:
            px, py, sc = proj
            rad = max(2, int(p[4] * sc))
            if rad > 1: pygame.draw.circle(screen, p[3], (px, py), rad)
            else: screen.set_at((px, py), p[3])
            
    # Label
    font = pygame.font.SysFont('monospace', 18, bold=True)
    screen.blit(font.render("DIGITAL TWIN (SIMULATION)", True, AI_CYAN), (20, 20))

    # --- PANEL 2: REAL NASA FEED (CENTER) ---
    pygame.draw.rect(screen, (10, 10, 15), RECT_REAL)
    
    if sdo_image_surface:
        screen.blit(sdo_image_surface, (VIEW_SPLIT_W, 0))
    else:
        load_txt = font.render("ESTABLISHING NASA UPLINK...", True, TEXT_ORANGE)
        screen.blit(load_txt, (VIEW_SPLIT_W + 50, HEIGHT // 2))
        
    screen.blit(font.render("REALITY (NASA SDO - AIA 171)", True, TEXT_ORANGE), (VIEW_SPLIT_W + 20, 20))
    screen.blit(font.render(f"UPDATED: {sdo_last_update}", True, WHITE), (VIEW_SPLIT_W + 20, 45))

    # --- PANEL 3: HUD (RIGHT) ---
    pygame.draw.rect(screen, HUD_BG, RECT_HUD)
    pygame.draw.line(screen, TEXT_ORANGE, (WIDTH - SIDE_PANEL_W, 0), (WIDTH - SIDE_PANEL_W, HEIGHT), 2)
    pygame.draw.line(screen, AI_CYAN, (VIEW_SPLIT_W, 0), (VIEW_SPLIT_W, HEIGHT), 2) # Split line
    
    x = WIDTH - SIDE_PANEL_W + 20
    y = 30
    
    screen.blit(font.render("SOLAR TELEMETRY", True, TEXT_ORANGE), (x, y))
    y += 40
    screen.blit(font.render(f"FLUX: {current_flux_val:.2e}", True, WHITE), (x, y))
    y += 25
    col = ALERT_RED if "FLARE" in flux_trend else SOLAR_SURFACE
    screen.blit(font.render(f"STATUS: {flux_trend}", True, col), (x, y))
    
    y += 60
    screen.blit(font.render("QUANTUM CONTAINMENT", True, AI_CYAN), (x, y))
    y += 30
    screen.blit(font.render(f"CORONAL LOCK: {int(q_bell_state*100)}%", True, WHITE), (x, y))
    pygame.draw.rect(screen, (50, 20, 20), (x, y+20, 250, 15))
    pygame.draw.rect(screen, AI_CYAN, (x, y+20, 250 * q_bell_state, 15))
    
    y = HEIGHT - 100
    screen.blit(font.render(f"AUDIO: {'ON' if sound_enabled else 'MUTED'} [M]", True, WHITE), (x, y))
    screen.blit(font.render("MOUSE: ROTATE LEFT VIEW", True, (150, 150, 150)), (x, y+20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
