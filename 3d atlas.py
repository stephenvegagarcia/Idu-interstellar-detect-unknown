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
import cv2
from skyfield.api import load, wgs84

# Try importing QuTiP for real quantum simulation
try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    logging.warning("QuTiP not found. Falling back to emulation.")

# ==============================================================================
#  TARGET: 3I/ATLAS | SINGLE SCREEN MODE
#  ALGORITHM: |φ⁺⟩ = 1/√2 (|00⟩ + |11⟩)
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
pygame.init()

# Audio Init
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    SOUND_AVAILABLE = True
except:
    SOUND_AVAILABLE = False

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1400, 900
SIDE_PANEL_W = 400
VIEW_W = WIDTH - SIDE_PANEL_W

# Screen Areas
RECT_3D = pygame.Rect(0, 0, VIEW_W, HEIGHT)
RECT_HUD = pygame.Rect(VIEW_W, 0, SIDE_PANEL_W, HEIGHT)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3I/ATLAS | 3D QUANTUM SIMULATOR")

# --- COLORS ---
SPACE_BLACK = (5, 5, 10)
WHITE = (255, 255, 255)
ATLAS_GREEN = (0, 255, 128) 
ION_BLUE = (50, 100, 255)   
NUCLEUS_WHITE = (220, 220, 255)
HUD_BG = (10, 15, 25) 
TEXT_CYAN = (0, 255, 255)
ALERT_RED = (255, 50, 50)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255) 

# --- GLOBAL STATE ---
cam_rot_x = 0.0
cam_rot_y = 0.0
cam_zoom = 1.0
particles = [] 

solar_flux_sim = 1.0 
q_bell_state = 0.0
q_coeff_00 = 0.0 
q_coeff_11 = 0.0 

# --- AUDIO ENGINE ---
def generate_comet_chirp(intensity):
    if not SOUND_AVAILABLE: return None
    duration = 0.2 + (random.random() * 0.2)
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    
    start_freq = 800 + (intensity * 500)
    end_freq = 100
    freqs = np.linspace(start_freq, end_freq, n_samples)
    phases = np.cumsum(freqs) / sample_rate * 2 * np.pi
    
    wave = np.sin(phases) * 0.3
    sound_array = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((sound_array, sound_array)))

current_sound = None
sound_enabled = True
last_chirp_time = 0

# ==============================================================================
#  <<< QUANTUM ENGINE (QUTIP) >>>
# ==============================================================================
def run_quantum_simulation(flux_noise):
    if not QUTIP_AVAILABLE:
        return 0.8 + (math.sin(pygame.time.get_ticks() * 0.001) * 0.1)

    try:
        # 1. Define Gates
        cnot = qt.Qobj([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dims=[[2, 2], [2, 2]])
        
        # 2. Rotation with Noise
        theta = (math.pi/2) + (flux_noise * 2.0) 
        
        # 3. Circuit
        psi_0 = qt.tensor(qt.basis(2,0), qt.basis(2,0)) 
        rot_y = (-1j * (theta/2) * qt.sigmay()).expm() 
        op_rot = qt.tensor(rot_y, qt.qeye(2))
        psi_final = cnot * op_rot * psi_0
        
        # 4. Measure Fidelity
        ideal_bell = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
        fidelity = abs(psi_final.overlap(ideal_bell))**2
        return fidelity
        
    except Exception as e:
        return 0.5

# ==============================================================================
#  <<< 3D PARTICLE ENGINE >>>
# ==============================================================================
FOV = 800
PROJECTION_CENTER = (VIEW_W // 2, HEIGHT // 2)

def project_3d_point(x, y, z):
    xz = x * math.cos(cam_rot_y) - z * math.sin(cam_rot_y)
    zz = z * math.cos(cam_rot_y) + x * math.sin(cam_rot_y)
    yz = y * math.cos(cam_rot_x) - zz * math.sin(cam_rot_x)
    zz = zz * math.cos(cam_rot_x) + y * math.sin(cam_rot_x)
    
    dist = 600
    fz = dist + zz
    if fz <= 0: return None
    scale = (FOV * cam_zoom) / fz
    
    px = int(xz * scale + PROJECTION_CENTER[0])
    py = int(yz * scale + PROJECTION_CENTER[1])
    return px, py, scale

def update_comet_physics():
    global particles
    particles = []
    
    # Amplitudes (0.707 is ideal)
    amp_00 = 0.707 * q_bell_state 
    amp_11 = 0.707 * q_bell_state 
    
    global q_coeff_00, q_coeff_11
    q_coeff_00 = amp_00
    q_coeff_11 = amp_11
    
    # 1. NUCLEUS (|00⟩)
    core_count = int(400 * amp_00)
    core_scatter = (1.0 - amp_00) * 30
    
    for _ in range(core_count):
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, math.pi)
        r = (15 + core_scatter) * (random.random()**(1/3))
        
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        
        col = NUCLEUS_WHITE if amp_00 < 0.65 else MAGENTA
        particles.append([x, y, z, col, 2])

    # 2. ION TAIL (|11⟩)
    tail_count = int(700 * amp_11)
    tail_len = 600 * amp_11
    
    for i in range(tail_count):
        z_pos = -random.uniform(20, tail_len)
        width = 5 + (abs(z_pos) * 0.1 * (1.5 - amp_11))
        
        x = random.gauss(0, width)
        y = random.gauss(0, width)
        
        # Vibration
        pulse = math.sin(pygame.time.get_ticks()*0.01 - z_pos*0.05) * 10
        y += pulse
        
        particles.append([x, y, z_pos, ION_BLUE, 1])
        
    # 3. COMA
    coma_count = 400
    for _ in range(coma_count):
        r = random.gauss(0, 50)
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, math.pi)
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        particles.append([x, y, z, ATLAS_GREEN, 1])

# ==============================================================================
#  <<< MAIN LOOP >>>
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
            if e.pos[0] < VIEW_W: mouse_down = True; last_mouse_pos = e.pos
        if e.type == pygame.MOUSEBUTTONUP: mouse_down = False
        if e.type == pygame.MOUSEMOTION and mouse_down:
            dx, dy = e.pos[0] - last_mouse_pos[0], e.pos[1] - last_mouse_pos[1]
            cam_rot_y += dx * 0.01; cam_rot_x += dy * 0.01; last_mouse_pos = e.pos

    # Logic
    t = pygame.time.get_ticks() / 1000.0
    solar_flux_sim = (math.sin(t * 0.5) * 0.2) + (random.uniform(-0.1, 0.1))
    
    q_bell_state = run_quantum_simulation(solar_flux_sim)
    update_comet_physics()
    
    # Audio
    now = pygame.time.get_ticks()
    if now - last_chirp_time > 200:
        chance = 0.05 * (1.0 + abs(solar_flux_sim))
        if random.random() < chance and sound_enabled and SOUND_AVAILABLE:
            chirp = generate_comet_chirp(1.0 + abs(solar_flux_sim))
            if chirp: chirp.play()
        last_chirp_time = now

    screen.fill(SPACE_BLACK)
    
    # --- MAIN VIEW: 3D SIMULATION ---
    pygame.draw.rect(screen, SPACE_BLACK, RECT_3D)
    
    def get_z(p): return p[2] * math.cos(cam_rot_y) + p[0] * math.sin(cam_rot_y)
    particles.sort(key=get_z, reverse=True)

    for p in particles:
        proj = project_3d_point(p[0], p[1], p[2])
        if proj:
            px, py, sc = proj
            rad = max(1, int(p[4] * sc))
            if rad > 1: pygame.draw.circle(screen, p[3], (px, py), rad)
            else: screen.set_at((px, py), p[3])
    
    font = pygame.font.SysFont('monospace', 18, bold=True)
    screen.blit(font.render("DIGITAL TWIN (QUTIP ENHANCED)", True, ATLAS_GREEN), (20, 20))

    # --- SIDE PANEL: HUD ---
    pygame.draw.rect(screen, HUD_BG, RECT_HUD)
    pygame.draw.line(screen, ATLAS_GREEN, (WIDTH - SIDE_PANEL_W, 0), (WIDTH - SIDE_PANEL_W, HEIGHT), 2)
    
    x = WIDTH - SIDE_PANEL_W + 20
    y = 30
    
    screen.blit(font.render("TARGET: 3I/ATLAS", True, ATLAS_GREEN), (x, y))
    y += 40
    screen.blit(font.render(f"NOISE INPUT: {solar_flux_sim:.3f}", True, WHITE), (x, y))
    y += 40
    
    # ALGORITHM DISPLAY
    screen.blit(font.render("ALGORITHM: BELL STATE", True, TEXT_CYAN), (x, y))
    y += 25
    screen.blit(font.render("|φ⁺⟩ = 1/√2 (|00⟩ + |11⟩)", True, MAGENTA), (x, y))
    y += 40
    
    # Amplitude Bars
    screen.blit(font.render(f"|00⟩ NUCLEUS: {q_coeff_00:.3f}", True, WHITE), (x, y))
    pygame.draw.rect(screen, (50, 50, 50), (x, y+20, 250, 10))
    pygame.draw.rect(screen, NUCLEUS_WHITE, (x, y+20, 250 * q_coeff_00, 10))
    y += 50
    
    screen.blit(font.render(f"|11⟩ TAIL: {q_coeff_11:.3f}", True, WHITE), (x, y))
    pygame.draw.rect(screen, (50, 50, 50), (x, y+20, 250, 10))
    pygame.draw.rect(screen, ION_BLUE, (x, y+20, 250 * q_coeff_11, 10))
    
    y = HEIGHT - 100
    screen.blit(font.render(f"AUDIO: {'ON' if sound_enabled else 'MUTED'} [M]", True, WHITE), (x, y))
    screen.blit(font.render("MOUSE: ROTATE 3D VIEW", True, (150, 150, 150)), (x, y+20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
