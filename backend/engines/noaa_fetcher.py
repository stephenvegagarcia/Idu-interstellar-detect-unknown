"""
NOAA Space Weather Prediction Center Fetcher
Real-time Solar X-Ray Flux Data
"""

import requests
import math
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class NOAAFetcher:
    def __init__(self):
        self.current_flux = 1e-8
        self.flux_trend = "QUIET"
        self.activity_index = 0.0
    
    def get_latest_flux(self) -> dict:
        """
        Fetch latest X-ray flux data from NOAA SWPC
        
        Returns:
            dict with flux_value, flux_trend, and activity_index
        """
        
        try:
            url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    # Get latest flux value
                    latest_flux = data[-1].get('flux', 1e-8)
                    self.current_flux = float(latest_flux)
                    
                    # Normalize to 0.0 - 1.0 scale
                    # X-ray flux range: 1e-9 to 1e-2 (in watts/m²)
                    # log10(1e-9) = -9, log10(1e-2) = -2
                    activity = (math.log10(self.current_flux) + 9) / 7
                    self.activity_index = np.clip(activity, 0.0, 1.0)
                    
                    # Classify flux trend
                    if self.current_flux > 1e-4:
                        self.flux_trend = "FLARE (X)"
                    elif self.current_flux > 1e-5:
                        self.flux_trend = "FLARE (M)"
                    elif self.current_flux > 1e-6:
                        self.flux_trend = "ACTIVE (C)"
                    elif self.current_flux > 1e-7:
                        self.flux_trend = "ACTIVE (B)"
                    else:
                        self.flux_trend = "QUIET"
                    
                    logger.info(f"✓ NOAA Flux: {self.current_flux:.2e} ({self.flux_trend})")
            
        except requests.exceptions.Timeout:
            logger.error("NOAA request timeout")
        except Exception as e:
            logger.error(f"NOAA fetch error: {e}")
        
        return {
            "flux_value": self.current_flux,
            "flux_trend": self.flux_trend,
            "activity_index": self.activity_index,
            "corona_lock": 1.0 - self.activity_index,  # Inverse of activity
        }
    
    def get_classification(self, flux: float) -> str:
        """
        Classify X-ray flux level
        
        Args:
            flux: Flux value in watts/m²
        
        Returns:
            Classification string
        """
        
        if flux > 1e-4:
            return "X (Extreme)"
        elif flux > 1e-5:
            return "M (Major)"
        elif flux > 1e-6:
            return "C (Moderate)"
        elif flux > 1e-7:
            return "B (Minor)"
        else:
            return "A (Quiet)"
    
    def get_historical_trends(self, hours: int = 6) -> dict:
        """Get historical flux trends"""
        
        try:
            url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract time series data
                timestamps = [entry.get('time_tag') for entry in data]
                fluxes = [float(entry.get('flux', 1e-8)) for entry in data]
                
                return {
                    "timestamps": timestamps,
                    "fluxes": fluxes,
                    "current": self.current_flux,
                    "trend": self.flux_trend,
                }
        
        except Exception as e:
            logger.error(f"Error fetching historical trends: {e}")
            return None
