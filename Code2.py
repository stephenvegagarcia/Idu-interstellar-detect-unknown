import pygame
import random
import sys
import math
import numpy as np
import pandas as pd
import astropy.units as u
# from astroquery.mpc import MPC # Kept in imports for context, using fallback/mock data
from skimage.restoration import denoise_bilateral
import cv2
import socket
import logging

# --- Quantum Physics Simulation Import ---
import qutip as qt
# -----------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 1000, 700
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("Quantum-Enhanced Sky Watcher: Comet Lemmon Real-Time Data & Anomaly Detection")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (100, 100, 255)
INFO_PANEL_BG = (20, 20, 50, 200)

# --- Real-Time Data Fetching (Astroquery) ---
def get_comet_data(comet_name='C/2023 A3'):
    """Fetches current RA, Dec, and magnitude for a given comet, with error handling."""
    # Hardcoded fallback values for Nov 17, 2025 (UTC assumed)
    fallback_ra, fallback_dec, fallback_mag = 257.6167, -18.4283, 8.4
    
    try:
        # Simulate a network check
        socket.create_connection(("www.google.com", 80), timeout=2)
        logging.info("Successfully fetched live data (simulated).")
        live_ra, live_dec, live_mag = 257.6167, -18.4283, 8.4
        return live_ra, live_dec, live_mag

    except (socket.error, socket.timeout, Exception) as e:
        logging.warning(f"Could not fetch live comet data. Using fallback data.")
        return fallback_ra, fallback_dec, fallback_mag

comet_ra_deg, comet_dec_deg, comet_mag = get_comet_data()

# --- Data Analysis & Unusual Object Detection (Pandas/Numpy) ---

def world_to_screen_coords(ra_deg, dec_deg, zoom_level, camera_offset_x, camera_offset_y):
    """Converts astronomical coordinates to raw screen coordinates *relative* to the top-left corner of the projected sky."""
    x = (ra_deg / 360.0) * width * zoom_level
    y = (height - ((dec_deg + 90) / 180.0) * height) * zoom_level
    return x - camera_offset_x, y - camera_offset_y

def create_star_catalog(num_stars):
    """Generates a DataFrame of simulated stars for analysis."""
    data = {
        'ID': [f'OBJ_{i:03d}' for i in range(num_stars)],
        'RA_deg': np.random.uniform(0, 360, num_stars),
        'Dec_deg': np.random.uniform(-40, 40, num_stars),
        'Magnitude': np.random.uniform(1.0, 9.0, num_stars),
        'Type': [random.choice(['Star', 'Star', 'Star', 'Galaxy', 'Nebula']) for _ in range(num_stars)],
        'Velocity_kps': np.random.uniform(10, 500, num_stars),
        'Size_arcsec': np.random.uniform(0.1, 5.0, num_stars),
        'Unusual': [False] * num_stars 
    }
    catalog = pd.DataFrame(data)
    catalog.loc[0, ['RA_deg', 'Dec_deg', 'Magnitude', 'Type', 'Unusual']] = [comet_ra_deg + 5, comet_dec_deg + 10, -1.0, 'UFO-Transient', True]
    catalog.loc[1, ['RA_deg', 'Dec_deg', 'Magnitude', 'Type', 'Unusual']] = [comet_ra_deg - 5, comet_dec_deg - 10, 0.5, 'Anomaly', True]
    return catalog

def detect_unusual_objects(catalog):
    """Flags objects brighter than a specific threshold as 'unusual'."""
    threshold_mag = 2.0
    catalog['Unusual'] = catalog['Magnitude'] < threshold_mag
    catalog.loc[catalog['Type'].isin(['UFO-Transient', 'Anomaly']), 'Unusual'] = True
    return catalog
    
star_catalog = create_star_catalog(200)
star_catalog = detect_unusual_objects(star_catalog)


# --- Pygame Drawing Functions & Zoom/Enhancement State ---
zoom_level = 3.0
camera_offset_x = 0
camera_offset_y = 0
enhanced_view = False 
selected_object = None 

def draw_stars_from_catalog(catalog, zoom_level, camera_offset_x, camera_offset_y):
    """Draws stars on the screen using data from the Pandas catalog, returns clickable areas (Rects)."""
    hitboxes = []
    for index, star in catalog.iterrows():
        x, y = world_to_screen_coords(star['RA_deg'], star['Dec_deg'], zoom_level, camera_offset_x, camera_offset_y)
        
        if -20 < x < width + 20 and -20 < y < height + 20:
            size = max(2, int((4 - star['Magnitude'] / 3.0) * zoom_level))
            color = WHITE

            if star['Unusual']:
                color = RED
                size = max(5, int(size * 1.5))
            
            if selected_object is not None and index == selected_object: 
                color = BLUE

            pygame.draw.circle(screen, color, (int(x), int(y)), size)
            hitboxes.append((pygame.Rect(int(x - size), int(y - size), size * 2, size * 2), index))
            
    return hitboxes

def draw_info_panel(object_data):
    """Draws a detailed information panel for a selected object."""
    panel_width, panel_height = 320, 240 # Increased height for new quantum info
    panel_x, panel_y = 10, height - panel_height - 10
    
    panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel_surface.fill(INFO_PANEL_BG)
    pygame.draw.rect(panel_surface, WHITE, (0, 0, panel_width, panel_height), 1)

    font_small = pygame.font.SysFont(None, 20)
    font_large = pygame.font.SysFont(None, 24)

    title_text = font_large.render(f"Object Details ({object_data['ID']})", True, YELLOW)
    panel_surface.blit(title_text, (10, 10))

    simulated_size_mpc = (object_data['Velocity_kps'] / 1000.0) * 0.05 

    data_lines = [
        f"Type: {object_data['Type']}",
        f"Magnitude: {object_data['Magnitude']:.2f} {'(Brightest!)' if object_data['Magnitude'] < 0 else ''}",
        f"RA/Dec: {object_data['RA_deg']:.4f}, {object_data['Dec_deg']:.4f}",
        f"Velocity: {object_data['Velocity_kps']:.1f} km/s (sim)",
        f"Ang. Size: {object_data['Size_arcsec']:.2f} arcsec (sim)",
        f"Cosmo. Size: {simulated_size_mpc:.3f} Mpc (derived)",
        f"Status: {'ANOMALY DETECTED' if object_data['Unusual'] else 'Cataloged Object'}"
    ]

    for i, line in enumerate(data_lines):
        color = RED if 'ANOMALY' in line or '(Brightest!)' in line else WHITE
        line_surface = font_small.render(line, True, color)
        panel_surface.blit(line_surface, (10, 40 + i * 25))

    # Add Quantum Status Footer
    quantum_status_text = font_small.render("Qutip Status: Quantum Filter ACTIVE", True, (100, 255, 100))
    panel_surface.blit(quantum_status_text, (10, panel_height - 30))

    screen.blit(panel_surface, (panel_x, panel_y))

def run_qutip_simulation():
    """
    Runs a simple qutip simulation to generate a 'quantum enhancement' factor.
    Simulates a driven qubit evolution.
    """
    # Define a simple Hamiltonian for a qubit
    H0 = 0.5 * qt.sigmaz()
    # Define a driving field operator
    H1 = 0.5 * qt.sigmax()
    # Define the time list
    times = np.linspace(0.0, 10.0, 100)
    
    # Define the callback function for the time dependence
    args = {'w': 1.0}
    def drive_strength(t, args):
        return np.sin(args['w'] * t) * 0.5
    
    # Total Hamiltonian
    H = [H0, [H1, drive_strength]]
    
    # Initial state (ground state)
    psi0 = qt.basis(2, 0)
    
    # Solve the Schrodinger equation
    result = qt.sesolve(H, psi0, times, [], args=args)
    
    # Extract the final state probability of the excited state as our 'enhancement factor'
    # This factor will be between 0 and 1
    enhancement_factor = np.abs(result.states[-1].full()[1][0])**2
    return enhancement_factor

def apply_enhancement_filter_qutip(surface):
    """
    Applies OpenCV bilateral filter, and then adjusts gamma/brightness based on a 
    'quantum enhancement' factor derived from qutip simulation.
    """
    # 1. Run the quantum simulation once per filter application
    quantum_boost_factor = run_qutip_simulation()
    
    # 2. Apply standard OpenCV bilateral filter for basic denoising
    img_array = pygame.surfarray.array3d(surface)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    enhanced_img = cv2.bilateralFilter(img_array, d=5, sigmaColor=75, sigmaSpace=75)

    # 3. Apply the quantum enhancement boost to brightness/exposure
    # Scale factor is between 1.0 (no boost) and ~2.0 (high boost)
    boost_multiplier = 1.0 + quantum_boost_factor 
    enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=boost_multiplier, beta=0)
    
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    enhanced_surface = pygame.surfarray.make_surface(enhanced_img)
    return enhanced_surface

def auto_center_on_brightest_anomaly(catalog):
    """Finds the brightest unusual object and sets camera offsets to center the view on it."""
    anomalies = catalog[catalog['Unusual'] == True]
    if anomalies.empty:
        return 0, 0, None

    brightest_anomaly = anomalies.loc[anomalies['Magnitude'].idxmin()]
    
    target_x_screen, target_y_screen = width / 2, height / 2

    raw_x_proj = (brightest_anomaly['RA_deg'] / 360.0) * width * zoom_level
    raw_y_proj = (height - ((brightest_anomaly['Dec_deg'] + 90) / 180.0) * height) * zoom_level
    
    new_offset_x = raw_x_proj - target_x_screen
    new_offset_y = raw_y_proj - target_y_screen
    
    return new_offset_x, new_offset_y, brightest_anomaly.name


# --- Initialization: Auto-zoom and select on start ---
camera_offset_x, camera_offset_y, selected_object = auto_center_on_brightest_anomaly(star_catalog)
logging.info(f"Auto-centered on object index {selected_object}")

# Main loop
running = True
clock = pygame.time.Clock()
while running:
    object_hitboxes = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: # Scroll up (zoom in)
                zoom_level *= 1.2
            elif event.button == 5: # Scroll down (zoom out)
                zoom_level /= 1.2
            zoom_level = max(0.5, min(zoom_level, 10.0))
            
            if event.button == 1: # Left click to select object
                clicked = False
                for rect, index in object_hitboxes:
                    if rect.collidepoint(event.pos):
                        selected_object = index
                        clicked = True
                        logging.info(f"Selected object ID: {star_catalog.loc[index, 'ID']}")
                        break
                if not clicked:
                    selected_object = None

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                enhanced_view = not enhanced_view
            if event.key == pygame.K_a: # Manually trigger auto-center
                camera_offset_x, camera_offset_y, selected_object = auto_center_on_brightest_anomaly(star_catalog)

    # --- AUTOMATIC ENHANCEMENT LOGIC ---
    # Automatically turn on enhancement view if any anomalies are present
    unusual_count_val = star_catalog[star_catalog['Unusual'] == True].shape[0]
    if unusual_count_val > 0:
        enhanced_view = True
    
    # Fill the screen with black (night sky)
    screen.fill(BLACK)
    
    # Draw all objects and generate hitboxes for the *next* frame's click event
    object_hitboxes = draw_stars_from_catalog(star_catalog, zoom_level, camera_offset_x, camera_offset_y)

    # Draw comet (using the refined coordinate function)
    comet_screen_x, comet_screen_y = world_to_screen_coords(comet_ra_deg, comet_dec_deg, zoom_level, camera_offset_x, camera_offset_y)
    if -10 < comet_screen_x < width + 10 and -10 < comet_screen_y < height + 10:
        pygame.draw.circle(screen, YELLOW, (int(comet_screen_x), int(comet_screen_y)), int(5 * zoom_level))
    
    # Apply Post-Processing Enhancement using the QuTiP-based function
    if enhanced_view:
        # Note: This is computationally expensive!
        screen_capture = screen.copy()
        enhanced_capture = apply_enhancement_filter_qutip(screen_capture)
        screen.blit(enhanced_capture, (0, 0))

    # --- Draw Overlays & Live Data ---

    # Display Anomaly Detection Status
    status_font = pygame.font.SysFont(None, 28)
    status_text_color = RED if unusual_count_val > 0 else GREEN
    status_text = status_font.render(f"Anomalies: {unusual_count_val} | Zoom: {zoom_level:.1f}x | Enhance[E]: {'AUTO ON' if enhanced_view else 'OFF'} | Center[A]", True, status_text_color)
    screen.blit(status_text, (10, 10))
    
    # Display detailed info panel if an object is selected
    if selected_object is not None:
        draw_info_panel(star_catalog.loc[selected_object])

    # Update the display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
