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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize Pygame
pygame.init()

# --- TRANSMISSION CONSTANTS: "STEPHEN I AM AN AMERICAN HUMAN" ---
NAME_BINARY_SEQUENCE = "010100110101010001000101010100000100100001000101010011100010000001001001001000000100000101001101001000000100000101001110001000000100000101001101010001010101001001001001010000110100000101001110001000000100100001010101010011010100000101001110"
BIT_COUNT = len(NAME_BINARY_SEQUENCE)
TRANSMISSION_RATE_MS = 250 # Time per bit: 250ms (4 bits per second)

# --- Simulation Constants ---
MEASUREMENT_THRESHOLD = 0.98  # Probability threshold for Quantum Sensor PEAK
SIGNAL_DECAY_RATE = 0.985      # Rate at which the sensor power decreases
SHIELD_ACTIVATION_THRESHOLD = 0.8  # Required power to visibly activate shield

# --- Display Setup ---
width, height = 1000, 700 
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("BINARY TRANSMITTER: My Name Is Stephen, I Am An American Human")

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
SHIELD_COLOR = (50, 200, 255) 
NOISE_COLOR = (255, 100, 0) 
PANEL_BG = (20, 20, 50, 220)

# --- Cosmic Data (Telescope) ---
comet_ra_deg, comet_dec_deg, comet_mag = 257.6167, -18.4283, 8.4 
zoom_level = 1.0 
center_x, center_y = width // 2, height // 2 

def create_star_catalog(num_stars=100):
    data = {
        'ID': [f'STAR_{i:03d}' for i in range(num_stars)],
        'RA_deg': np.random.uniform(200, 300, num_stars),
        'Dec_deg': np.random.uniform(-30, 0, num_stars),
        'Magnitude': np.random.uniform(3.0, 9.0, num_stars),
        'Type': [random.choice(['Star', 'Star', 'Galaxy']) for _ in range(num_stars)],
        'Unusual': [False] * num_stars 
    }
    catalog = pd.DataFrame(data)
    catalog.loc[0, ['ID', 'RA_deg', 'Dec_deg', 'Type', 'Magnitude']] = ['Q-SENSOR_L1', comet_ra_deg, comet_dec_deg, 'Sensor', 1.0]
    catalog.loc[1, ['ID', 'RA_deg', 'Dec_deg', 'Type', 'Magnitude']] = ['NOISE_SOURCE_02', comet_ra_deg - 10, comet_dec_deg + 10, 'Noise', 4.0]
    return catalog

star_catalog = create_star_catalog()
SENSOR_INDEX = 0
NOISE_INDEX = 1

# --- Threat Data (The CME) ---
cme_threat = {
    'x': width + 100,
    'y': height // 2 - 50,
    'speed': 3.0,
    'color': YELLOW
}


# --- Drawing Functions ---

def world_to_screen_coords(ra_deg, dec_deg, view_w, view_h, zoom):
    center_ra, center_dec = 250, -15 
    x_rel = (ra_deg - center_ra) * 5 * zoom
    y_rel = (center_dec - dec_deg) * 5 * zoom
    x = view_w // 2 + x_rel
    y = view_h // 2 + y_rel
    return x, y

def draw_stars_and_threats(catalog, sensor_power, zoom, transmitting_bit):
    """Draws stars and the CME threat, modulating the Sensor Star for binary output."""
    
    # 1. Draw Stars
    for index, star in catalog.iterrows():
        x, y = world_to_screen_coords(star['RA_deg'], star['Dec_deg'], width, height, zoom)
        
        if -50 < x < width + 50 and -50 < y < height + 50:
            size = max(2, int((6 - star['Magnitude'] / 2.0) * zoom))
            color = WHITE
            
            # --- QUANTUM SENSOR / BINARY TRANSMITTER VISUALIZATION ---
            if index == SENSOR_INDEX:
                # Sensor star represents the transmitted BIT (1 or 0)
                if transmitting_bit == '1':
                    # Transmitting '1' (HIGH signal) uses MAGENTA
                    flash_intensity = int(255 * sensor_power) 
                    color = (flash_intensity, 50, flash_intensity) if (pygame.time.get_ticks() // 50) % 2 == 0 else MAGENTA
                    size = max(size, 8) + int(sensor_power * 10)
                else: 
                    # Transmitting '0' (LOW signal) is dim or dark
                    color = (20, 20, 20) 
            
            # --- REVERSE NOISE VISUALIZATION ---
            elif index == NOISE_INDEX:
                # Flashes ORANGE only when the main sensor is LOW
                if sensor_power < 0.2 and random.random() > 0.9:
                    color = NOISE_COLOR
                    size = max(size, 8)
                else:
                    color = (20, 20, 20) 

            pygame.draw.circle(screen, color, (int(x), int(y)), size)
            
    # 2. Draw CME Threat
    if cme_threat['x'] < width + 100:
        pygame.draw.circle(screen, cme_threat['color'], (int(cme_threat['x']), int(cme_threat['y'])), 15)

def draw_quantum_shield(shield_power):
    """Draws the planetary defense shield based on power level."""
    
    if shield_power > SHIELD_ACTIVATION_THRESHOLD:
        radius = int(50 + shield_power * 150)
        alpha = int(255 * (shield_power - SHIELD_ACTIVATION_THRESHOLD) * 2)
        
        # Draw the Earth/Core (implied center)
        pygame.draw.circle(screen, (0, 0, 100), (center_x, center_y), 30)

        # Draw the Shield Plasma
        shield_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        shield_surface.fill((0, 0, 0, 0))
        
        shield_color = SHIELD_COLOR + (min(255, alpha),)
        
        pygame.draw.circle(shield_surface, shield_color, (center_x, center_y), radius, 0)
        pygame.draw.circle(shield_surface, WHITE + (255,), (center_x, center_y), radius, 2)
        
        screen.blit(shield_surface, (0, 0))

def draw_dashboard(sensor_power, current_bit, current_char_index):
    """Draws the main telemetry status panel with binary data."""
    panel_width, panel_height = 400, 180
    panel_x, panel_y = width - panel_width - 10, 10
    
    panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel_surface.fill(PANEL_BG)
    pygame.draw.rect(panel_surface, WHITE, (0, 0, panel_width, panel_height), 1)

    font_large = pygame.font.SysFont(None, 36, bold=True)
    font_small = pygame.font.SysFont(None, 24)
    font_mono = pygame.font.SysFont('monospace', 22) 
    
    shield_active = sensor_power >= SHIELD_ACTIVATION_THRESHOLD
    
    if shield_active:
        title_text = font_large.render("TRANSMISSION LOCK", True, MAGENTA)
        status_line = font_small.render(f"DEFENSE POWER: {sensor_power*100:.1f}%", True, SHIELD_COLOR)
    else:
        title_text = font_large.render("TRANSMITTING", True, GREEN)
        status_line = font_small.render(f"BIT RATE: 4 bps (250ms/bit)", True, WHITE)
    
    # Calculate the full character from the current index (8 bits)
    char_index = current_char_index
    current_char_binary = NAME_BINARY_SEQUENCE[char_index * 8 : char_index * 8 + 8]
    
    try:
        current_char_text = chr(int(current_char_binary, 2))
    except ValueError:
        current_char_text = '?'
    
    # Display the binary stream sequence
    stream_display = NAME_BINARY_SEQUENCE[current_bit_index - 8: current_bit_index + 8] 
    
    binary_stream_render = font_mono.render(stream_display, True, (150, 150, 150))
    panel_surface.blit(binary_stream_render, (10, 90))

    # Highlight the current character being transmitted 
    char_text = f"Current Char: {current_char_text} ({current_char_binary})"
    char_render = font_small.render(char_text, True, YELLOW)
    panel_surface.blit(char_render, (10, 120))
    
    
    panel_surface.blit(title_text, (10, 10))
    panel_surface.blit(status_line, (10, 60))

    screen.blit(panel_surface, (panel_x, panel_y))


# --- Loop Initialization ---
running = True
clock = pygame.time.Clock()
quantum_sensor_power = 0.0 
last_bit_time = pygame.time.get_ticks()
current_bit_index = 0

while running:
    
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    # --- THREAT MOVEMENT & RECHARGE LOGIC (Unchanged) ---
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
    
    # 1. Advance the Binary Bit Index (Time-based transmission)
    if now - last_bit_time >= TRANSMISSION_RATE_MS:
        current_bit_index = (current_bit_index + 1) % BIT_COUNT
        last_bit_time = now
        
    current_bit = NAME_BINARY_SEQUENCE[current_bit_index]
    current_char_index = current_bit_index // 8
    
    # 2. Mock Quantum Peak 
    if random.random() > MEASUREMENT_THRESHOLD: 
        if quantum_sensor_power < 0.9: 
            logging.warning("--- Q-SENSOR PEAK DETECTED: CHARGING SHIELD ---")
        quantum_sensor_power = 1.0
    else:
        # Decay: Signal decays, but is used by the binary output for visualization
        quantum_sensor_power *= SIGNAL_DECAY_RATE
        quantum_sensor_power = max(0.0, quantum_sensor_power)
            
    # --- Drawing ---
    screen.fill(BLACK)
    
    draw_quantum_shield(quantum_sensor_power)

    # Pass the current bit to modulate the Sensor Star's appearance
    draw_stars_and_threats(star_catalog, quantum_sensor_power, zoom_level, current_bit)

    # Pass the current bit and character index to the dashboard for display
    draw_dashboard(quantum_sensor_power, current_bit, current_char_index)

    # Update the display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
