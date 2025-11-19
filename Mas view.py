import pygame
import random
import sys
import math
import numpy as np
import pandas as pd
from skimage.restoration import denoise_bilateral
import cv2
import socket
import logging
import qutip as qt 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize Pygame
pygame.init()

# --- TRANSMISSION CONSTANTS ---
NAME_BINARY_SEQUENCE = "010100110101010001000101010100000100100001000101010011100010000001001001001000000100000101001101001000000100000101001110001000000100000101001101010001010101001001001001010000110100000101001110001000000100100001010101010011010100000101001110"
BIT_COUNT = len(NAME_BINARY_SEQUENCE)
TRANSMISSION_RATE_MS = 250 

# --- Simulation Constants ---
MEASUREMENT_THRESHOLD = 0.98  
SIGNAL_DECAY_RATE = 0.985      
SHIELD_ACTIVATION_THRESHOLD = 0.8  
NOISE_SUPPRESSION_THRESHOLD = 0.3 
CORE_STABILITY_LOCK = 0.90     
MAX_ATLAS_ZOOM = 15.0 

# --- Display Setup ---
width, height = 1000, 700 
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("ENTROPY-DRIVEN ATLAS OBSERVATION WITH REAL TELESCOPE DATA")

# --- Colors (FIXED CONSOLIDATION) ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
SHIELD_COLOR = (50, 200, 255) 
NOISE_COLOR = (255, 100, 0) 
PANEL_BG = (20, 20, 50, 220)
QML_JET_COLOR = (100, 255, 150)
NUCLEUS_COLOR = (50, 50, 50) 
COMA_GREEN = (0, 200, 100)  
ION_TAIL_BLUE = (50, 100, 255) 
BIOSIGNATURE_COLOR = (0, 255, 100)
SUN_CORE_COLOR = (255, 100, 0) 
CORONA_COLOR = (255, 255, 50) 

# --- VIEWPORT DIMENSIONS ---
ATLAS_W, ATLAS_H = 350, 200 
ATLAS_X, ATLAS_Y = 10, height - ATLAS_H - 10 
SOLAR_W, SOLAR_H = 350, 200  # Added for Sun view consistency
MOON_W, MOON_H = 350, 200  # Added for Moon view
MOON_X, MOON_Y = 10, 10  # Top left for Moon view

# --- Cosmic Data (Telescope) ---
ATLAS_RA, ATLAS_DEC = 257.6167, -18.4283 
zoom_level = 1.0 
center_x, center_y = width // 2, height // 2 
solar_radius = 500 

# --- CORE FUNCTION DEFINITIONS ---

def create_star_catalog():
    """Uses real astronomical data from SIMBAD query near ATLAS coordinates, plus moons data."""
    data = {
        'ID': [
            'UCAC4 358-087680', 'LEDA 3097061', 'UCAC4 359-085101', 'TYC 6236-1354-1',
            'ATO J257.6209-18.3264', 'V* V1527 Oph', 'TYC 6236-1345-1', 'OGLE BLG-RRLYR-44685',
            'TYC 6236-1479-1', 'NOISE_SOURCE_02',  # Retained for simulation
            'Moon', 'Jupiter', 'Io', 'Europa', 'Ganymede', 'Callisto', 'Saturn', 'Titan'  # Added moons and planets data
        ],
        'RA_deg': [
            257.6733, 257.6167, 257.6842, 256.8419, 257.6305, 257.6825, 256.8245, 257.7406, 256.8054, ATLAS_RA - 10,
            207.343, 116.594, 116.594, 116.594, 116.594, 116.594, 356.099, 356.099  # Updated Moon position, approximate for others
        ],
        'Dec_deg': [
            -18.4103, -18.4147, -18.3847, -18.4240, -18.3267, -18.5122, -18.3561, -18.3506, -18.2375, ATLAS_DEC + 10,
            -15.082, 21.239, 21.239, 21.239, 21.239, 21.239, -4.1026, -4.1026
        ],
        'Magnitude': [
            13.109, 16.09, 13.275, 11.47, 13.354, 16.209, 11.37, 15.284, 10.65, 4.0,
            -12.5, -2.5, 5.0, 5.3, 4.6, 5.7, 0.7, 8.4  # Approximate magnitudes
        ],
        'Type': [
            'Star', 'Galaxy', 'Star', 'Star', 'Star', 'Star', 'Star', 'Star', 'Star', 'Noise',
            'Moon', 'Planet', 'Moon', 'Moon', 'Moon', 'Moon', 'Planet', 'Moon'
        ],
        'Unusual': [False] * 18 
    }
    catalog = pd.DataFrame(data)
    catalog.loc[0, ['ID', 'RA_deg', 'Dec_deg', 'Type', 'Magnitude']] = ['3I/ATLAS_ANCHOR', ATLAS_RA, ATLAS_DEC, 'Comet', 0.5]
    return catalog

def run_vivid_clarity_factor():
    """
    Mocks the complex output of the quantum circuit for clarity.
    """
    stable_baseline = 0.85
    random_fluctuation = random.uniform(-0.15, 0.15)
    
    clarity = stable_baseline + random_fluctuation
    
    return np.clip(clarity, 0.0, 1.0) 

def draw_sun_vivid(surface, clarity_factor, entropy, biosignature_metric):
    """Draws a vivid, simulated view of the Sun."""
    surface.fill(BLACK)
    
    local_c_x, local_c_y = SOLAR_W // 2, SOLAR_H // 2
    
    disk_radius = int(50 * (1.0 + clarity_factor * 0.1))
    
    # --- 1. Corona (Outer Glow/Flare) ---
    flare_radius = int(disk_radius + entropy * 30)
    
    for i in range(1, 3):
        alpha = int(100 * entropy) - (i * 30)
        corona_color = CORONA_COLOR + (max(0, alpha),)
        pygame.draw.circle(surface, corona_color, (local_c_x, local_c_y), flare_radius + i*5, 0)
    
    # --- 2. Photosphere (Vivid Detail) ---
    vivid_alpha = int(200 * clarity_factor)
    vivid_color = SUN_CORE_COLOR + (vivid_alpha,)
    pygame.draw.circle(surface, vivid_color, (local_c_x, local_c_y), disk_radius, 0)
    
    pygame.draw.circle(surface, WHITE, (local_c_x, local_c_y), disk_radius // 2, 0)
    
    # --- 3. Biosignature Detection (The Structure) ---
    if biosignature_metric > 0.6:
        sig_radius = int(10 + biosignature_metric * 10)
        sig_color = BIOSIGNATURE_COLOR + (255,)
        
        # Draw the square structure indicating biosignature
        pygame.draw.rect(surface, sig_color, 
                         (local_c_x + disk_radius - 5, local_c_y - sig_radius//2, sig_radius, sig_radius), 0)
        
        font_tiny = pygame.font.SysFont(None, 14)
        warning_text = font_tiny.render("STRUCT", True, BLACK)
        surface.blit(warning_text, (local_c_x + disk_radius - 5, local_c_y - 4))


    # Label the visual
    font_small = pygame.font.SysFont(None, 20)
    text = font_small.render(f"QC SOLAR VIEW | Vivid Clarity={clarity_factor:.3f}", True, BIOSIGNATURE_COLOR)
    surface.blit(text, (5, 5))
    
    return surface

def draw_atlas_close_up(surface, clarity_factor, entropy):
    """
    Draws a realistic, QML-enhanced visual of Comet 3I/ATLAS.
    """
    surface.fill(BLACK)
    pygame.draw.rect(surface, (50, 50, 50), (0, 0, ATLAS_W, ATLAS_H), 1)
    
    center_x_local, center_y_local = ATLAS_W // 2, ATLAS_H // 2
    
    clarity_level = np.clip(clarity_factor, 0.2, 1.0)
    
    # --- 1. COMA (Green Gas) ---
    coma_radius = int(25 * (1 + clarity_level * 0.5))
    
    for i in range(1, 4):
        alpha = int(100 * clarity_level) - (i * 10)
        coma_color = COMA_GREEN + (max(0, alpha),)
        pygame.draw.circle(surface, coma_color, (center_x_local, center_y_local), coma_radius - i*5, 0)

    # --- 2. NUCLEUS (Rock) ---
    nucleus_radius = 5
    pygame.draw.circle(surface, NUCLEUS_COLOR, (center_x_local, center_y_local), nucleus_radius)
    pygame.draw.circle(surface, WHITE, (center_x_local, center_y_local), nucleus_radius, 1) 
    
    # --- 3. QML-Enhanced JETS (Blue/Cyan Ion Tail) ---
    jet_length = int(40 + clarity_level * 100)
    
    if clarity_level > 0.4: 
        
        jet_color = ION_TAIL_BLUE
        
        # Jet 1: Top-Right (Angle 45 deg)
        end_point_x1 = center_x_local + int(jet_length * 0.707)
        end_point_y1 = center_y_local - int(jet_length * 0.707)
        pygame.draw.line(surface, jet_color, (center_x_local, center_y_local), (end_point_x1, end_point_y1), 2)
        
        # Jet 2: Bottom-Left (Angle 225 deg)
        end_point_x2 = center_x_local - int(jet_length * 0.707)
        end_point_y2 = center_y_local + int(jet_length * 0.707)
        pygame.draw.line(surface, jet_color, (center_x_local, center_y_local), (end_point_x2, end_point_y2), 2)
    
    # Label the visual
    font_small = pygame.font.SysFont(None, 20)
    text = font_small.render(f"ATLAS QML VIEW | Clarity={clarity_level:.3f}", True, QML_JET_COLOR)
    surface.blit(text, (5, 5))
    
    return surface

def draw_moon_close_up(surface, clarity_factor, entropy):
    """
    Draws a realistic view of the Moon with craters.
    """
    surface.fill(BLACK)
    pygame.draw.rect(surface, (50, 50, 50), (0, 0, MOON_W, MOON_H), 1)
    
    center_x_local, center_y_local = MOON_W // 2, MOON_H // 2
    
    clarity_level = np.clip(clarity_factor, 0.2, 1.0)
    
    moon_radius = int(40 * (1 + clarity_level * 0.5))
    
    moon_color = (200, 200, 200)  # Gray moon surface
    
    pygame.draw.circle(surface, moon_color, (center_x_local, center_y_local), moon_radius)
    
    # Add craters (simple dark circles)
    crater_positions = [
        (center_x_local - 20, center_y_local - 15, 10),
        (center_x_local + 15, center_y_local + 10, 8),
        (center_x_local - 10, center_y_local + 20, 12),
        (center_x_local + 25, center_y_local - 10, 7)
    ]
    
    for cx, cy, cr in crater_positions:
        pygame.draw.circle(surface, (100, 100, 100), (cx, cy), cr)
    
    # Add some noise or texture based on entropy
    if entropy > 0.2:
        for _ in range(int(entropy * 50)):
            nx = random.randint(center_x_local - moon_radius, center_x_local + moon_radius)
            ny = random.randint(center_y_local - moon_radius, center_y_local + moon_radius)
            if (nx - center_x_local)**2 + (ny - center_y_local)**2 <= moon_radius**2:
                pygame.draw.circle(surface, (150 + random.randint(-50, 50),) * 3, (nx, ny), 1)
    
    # Label the visual
    font_small = pygame.font.SysFont(None, 20)
    text = font_small.render(f"MOON VIEW | Clarity={clarity_level:.3f}", True, WHITE)
    surface.blit(text, (5, 5))
    
    return surface

def world_to_screen_coords(ra_deg, dec_deg, view_w, view_h, zoom, offset_x, offset_y):
    """Converts astronomical coordinates to screen coords relative to a viewport center."""
    center_ra, center_dec = 250, -15 
    x_rel = (ra_deg - center_ra) * 5 * zoom
    y_rel = (center_dec - dec_deg) * 5 * zoom
    x = view_w // 2 + x_rel - offset_x
    y = view_h // 2 + y_rel - offset_y
    return x, y

def draw_stars_and_threats(catalog, sensor_power, zoom, transmitting_bit, offset_x, offset_y):
    """Draws stars and the CME threat in the telescope view."""
    
    for index, star in catalog.iterrows():
        x, y = world_to_screen_coords(star['RA_deg'], star['Dec_deg'], width, height, zoom, offset_x, offset_y)
        
        if -50 < x < width + 50 and -50 < y < height + 50:
            size = max(2, int((6 - star['Magnitude'] / 2.0) * zoom))
            color = WHITE
            
            if index == ATLAS_INDEX:
                if transmitting_bit == '1':
                    flash_intensity = int(255 * sensor_power) 
                    color = (flash_intensity, 50, flash_intensity) if (pygame.time.get_ticks() // 50) % 2 == 0 else MAGENTA
                    size = max(size, 8) + int(sensor_power * 10)
                else: 
                    color = (20, 20, 20) 
            
            elif index == NOISE_INDEX:
                if sensor_power < NOISE_SUPPRESSION_THRESHOLD: 
                    if random.random() > 0.9:
                        color = NOISE_COLOR
                        size = max(size, 8)
                    else:
                        color = (20, 20, 20) 
                else:
                    color = BLACK 
                    size = 1 

            pygame.draw.circle(screen, color, (int(x), int(y)), size)
            
    cme_screen_x = cme_threat['x'] + center_x - camera_offset_x
    cme_screen_y = cme_threat['y'] + center_y - camera_offset_y
    if cme_screen_x < width + 100:
        pygame.draw.circle(screen, cme_threat['color'], (int(cme_threat['x']), int(cme_threat['y'])), 15)

def draw_quantum_shield(shield_power):
    """Draws the planetary defense shield based on power level."""
    
    if shield_power > SHIELD_ACTIVATION_THRESHOLD:
        radius = int(50 + shield_power * 150)
        alpha = int(255 * (shield_power - SHIELD_ACTIVATION_THRESHOLD) * 2)
        
        pygame.draw.circle(screen, (0, 0, 100), (center_x, center_y), 30)

        shield_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        shield_surface.fill((0, 0, 0, 0))
        
        shield_color = SHIELD_COLOR + (min(255, alpha),)
        
        pygame.draw.circle(shield_surface, shield_color, (center_x, center_y), radius, 0)
        pygame.draw.circle(shield_surface, WHITE + (255,), (center_x, center_y), radius, 2)
        
        screen.blit(shield_surface, (0, 0))


def draw_dashboard(sensor_power, current_bit, current_char_index, entropy, biosignature, clarity_factor):
    """Draws the main telemetry status panel with specialized ATLAS data, including ENTROPY."""
    
    panel_width, panel_height = 400, 200 
    panel_x, panel_y = width - panel_width - 10, 10
    
    panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel_surface.fill(PANEL_BG)
    pygame.draw.rect(panel_surface, WHITE, (0, 0, panel_width, panel_height), 1)

    font_large = pygame.font.SysFont(None, 36, bold=True)
    font_small = pygame.font.SysFont(None, 24)
    
    if biosignature > 0.6:
        title_text = font_large.render("BIOSIGNATURE DETECTED!", True, BIOSIGNATURE_COLOR)
    else:
        title_text = font_large.render("SOLAR OBSERVATION", True, YELLOW)
        
    sig_status = f"STRUCTURAL SIGNAL: {biosignature*100:.1f}%"
    sig_color = BIOSIGNATURE_COLOR if biosignature > 0.6 else RED
    
    entropy_text = f"System Entropy: {entropy*100:.1f}%"
    clarity_text = f"QC Vivid Clarity: {clarity_factor*100:.1f}%"
    
    char_index = current_char_index
    current_char_binary = NAME_BINARY_SEQUENCE[char_index * 8 : char_index * 8 + 8]
    try:
        current_char_text = chr(int(current_char_binary, 2))
    except ValueError:
        current_char_text = '?'
    binary_text = f"TX Char: {current_char_text} | Bit: {current_bit}"
    
    
    panel_surface.blit(title_text, (10, 10))
    panel_surface.blit(font_small.render(sig_status, True, sig_color), (10, 50))
    panel_surface.blit(font_small.render(clarity_text, True, MAGENTA), (10, 75))
    panel_surface.blit(font_small.render(entropy_text, True, RED), (10, 100))
    panel_surface.blit(font_small.render(binary_text, True, WHITE), (10, 130))
    
    # Concealment status
    if sensor_power >= NOISE_SUPPRESSION_THRESHOLD:
        concealment_text = "CONCEALMENT ACTIVE"
        concealment_color = SHIELD_COLOR
    else:
        concealment_text = "Background Noise Unfiltered"
        concealment_color = RED
        
    concealment_render = font_small.render(f"NOISE FILTER: {concealment_text}", True, concealment_color)
    panel_surface.blit(concealment_render, (10, 160))
    
    screen.blit(panel_surface, (panel_x, panel_y))


def auto_center_on_object(object_data, zoom, width, height):
    """Calculates the camera offset needed to center on the specified object."""
    target_x_screen, target_y_screen = width / 2, height / 2
    
    raw_x, raw_y = world_to_screen_coords(object_data['RA_deg'], object_data['Dec_deg'], width, height, zoom, 0, 0)
    
    new_offset_x = raw_x - target_x_screen
    new_offset_y = raw_y - target_y_screen
    
    return new_offset_x, new_offset_y

def apply_realism_filters(surface, entropy):
    """
    Applies realism filters (Gaussian Blur for bloom, and CV2 grain/static)
    based on system entropy.
    """
    img_array = pygame.surfarray.array3d(surface)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    blur_kernel = 3
    img_array = cv2.GaussianBlur(img_array, (blur_kernel, blur_kernel), 0)

    noise_level = int(entropy * 50) 
    
    if noise_level > 0:
        mean = 0
        sigma = noise_level 
        gauss = np.random.normal(mean, sigma, img_array.shape).astype('uint8')
        img_array = cv2.add(img_array, gauss)

    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(img_array)

def draw_antenna(screen, current_bit, sensor_power):
    """Draws a simple radio antenna (mast + parabolic dish) that pulses during transmission."""
    # Base position (left of center/Earth)
    base_x = center_x - 250
    base_y = center_y + 150  # Near the bottom
    
    # Mast (vertical tower)
    mast_start = (base_x, base_y)
    mast_end = (base_x, base_y - 200)
    pygame.draw.line(screen, WHITE, mast_start, mast_end, 5)
    
    # Dish (parabolic, represented as a circle)
    dish_center = (base_x, base_y - 250)
    dish_radius = 50
    dish_color = WHITE
    if current_bit == '1':  # Pulse effect when transmitting '1'
        pulse_intensity = int(255 * sensor_power)
        dish_color = (pulse_intensity, pulse_intensity, 255)  # Blue-ish glow
        dish_radius += int(5 * math.sin(pygame.time.get_ticks() / 200))  # Slight size pulse
    pygame.draw.circle(screen, dish_color, dish_center, dish_radius, 2)
    
    # Feed lines (simple cross in dish)
    pygame.draw.line(screen, WHITE, (dish_center[0] - 20, dish_center[1]), (dish_center[0] + 20, dish_center[1]), 2)
    pygame.draw.line(screen, WHITE, (dish_center[0], dish_center[1] - 20), (dish_center[0], dish_center[1] + 20), 2)
    
    # Label
    font_tiny = pygame.font.SysFont(None, 18)
    label = font_tiny.render("TX Antenna", True, YELLOW)
    screen.blit(label, (base_x - 40, base_y + 10))


# --- INITIALIZATION ---
star_catalog = create_star_catalog()
ATLAS_INDEX = 0
NOISE_INDEX = star_catalog[star_catalog['ID'] == 'NOISE_SOURCE_02'].index[0]  # Find index dynamically
MOON_INDEX = star_catalog[star_catalog['ID'] == 'Moon'].index[0]

# PRINT CATALOG CONTENTS
print("-------------------------------------------------------")
print("SkyWatch Catalog Dump (Real Telescope Data from SIMBAD with Moons)")
print("-------------------------------------------------------")
print(star_catalog)
print("-------------------------------------------------------")


cme_threat = {
    'x': width + 100,
    'y': height // 2 - 50,
    'speed': 3.0,
    'color': YELLOW
}
# Initialize local variables used in drawing functions
vivid_clarity_factor = 0.0
current_bit = '0' 

atlas_surface = pygame.Surface((ATLAS_W, ATLAS_H), pygame.SRCALPHA)
solar_surface = pygame.Surface((SOLAR_W, SOLAR_H), pygame.SRCALPHA)  # New for Sun view
moon_surface = pygame.Surface((MOON_W, MOON_H), pygame.SRCALPHA)  # New for Moon view
SOLAR_X, SOLAR_Y = width - ATLAS_W - 10, height - ATLAS_H - 10  # Position for Sun view

# Video saving setup (saves 30 seconds at 60 FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('atlas_simulation.mp4', fourcc, 60, (width, height))
max_frames = 1800  # 30 seconds
frame_count = 0

# --- Main loop ---
running = True
clock = pygame.time.Clock()
quantum_sensor_power = 0.0 
last_bit_time = pygame.time.get_ticks()
current_bit_index = 0
planetary_instability = 0.5 
calculated_entropy = 0.0
camera_offset_x, camera_offset_y = 0, 0 
biosignature_metric = 0.0
atlas_data = star_catalog.loc[ATLAS_INDEX]
moon_data = star_catalog.loc[MOON_INDEX]


while running and frame_count < max_frames:
    
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    # --- THREAT MOVEMENT & RECHARGE LOGIC ---
    if cme_threat['x'] > center_x:
        cme_threat['x'] -= cme_threat['speed']
    elif cme_threat['x'] <= center_x and quantum_sensor_power >= SHIELD_ACTIVATION_THRESHOLD:
        cme_threat['speed'] = -cme_threat['speed'] * 1.5 
        cme_threat['color'] = RED 
    elif cme_threat['x'] < -100:
        cme_threat['x'] = width + 100
        cme_threat['speed'] = random.uniform(2.5, 4.0)
        cme_threat['color'] = YELLOW
        
    # --- QUANTUM SENSOR / BINARY TRANSMITTER LOGIC ---
    now = pygame.time.get_ticks()
    
    if now - last_bit_time >= TRANSMISSION_RATE_MS:
        current_bit_index = (current_bit_index + 1) % BIT_COUNT
        last_bit_time = now
        
    current_bit = NAME_BINARY_SEQUENCE[current_bit_index]
    current_char_index = current_bit_index // 8
    
    if random.random() > MEASUREMENT_THRESHOLD: 
        quantum_sensor_power = 1.0
    else:
        quantum_sensor_power *= SIGNAL_DECAY_RATE
        quantum_sensor_power = max(0.0,
