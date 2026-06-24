"""
NASA SDO Uplink Manager
Fetches live solar imagery from NASA
"""

import requests
import logging
from datetime import datetime
import io
from PIL import Image
import base64

logger = logging.getLogger(__name__)

class NASAUplink:
    def __init__(self):
        self.last_image_url = None
        self.last_update = None
        self.cache = {}
    
    def get_latest_aia171(self) -> str:
        """
        Fetch latest NASA SDO AIA 171 Angstrom image (Corona in gold)
        
        Returns:
            URL string to the latest image
        """
        
        try:
            # NASA SDO latest image endpoint
            url = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg"
            
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                self.last_image_url = url
                self.last_update = datetime.now().isoformat()
                logger.info(f"✓ NASA SDO image fetched: {url}")
                return url
            else:
                logger.warning(f"NASA SDO returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("NASA SDO request timeout")
            return None
        except Exception as e:
            logger.error(f"NASA SDO fetch error: {e}")
            return None
    
    def get_aia_wavelengths(self) -> dict:
        """Get available AIA wavelength images"""
        
        wavelengths = {
            "94": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0094.jpg",   # Hot Corona
            "131": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0131.jpg",  # Flare
            "171": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg",  # Corona (Default)
            "193": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg",  # Corona
            "211": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0211.jpg",  # Quiet Corona
            "304": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0304.jpg",  # Chromosphere
            "335": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0335.jpg",  # Hot Corona
            "1600": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_1600.jpg", # UV continuum
            "1700": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_1700.jpg", # UV continuum
        }
        
        return wavelengths
    
    def get_latest_hmi(self) -> str:
        """
        Fetch latest NASA SDO HMI (Helioseismic & Magnetic Imager) image
        Shows magnetic field
        
        Returns:
            URL to latest HMI image
        """
        
        try:
            url = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_hmi_ic_fits.jpg"
            
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                logger.info(f"✓ NASA SDO HMI image fetched")
                return url
            else:
                return None
                
        except Exception as e:
            logger.error(f"NASA SDO HMI fetch error: {e}")
            return None
    
    def get_metadata(self) -> dict:
        """Get metadata about the latest fetch"""
        
        return {
            "last_image_url": self.last_image_url,
            "last_update": self.last_update,
            "source": "NASA SDO (Solar Dynamics Observatory)",
            "instrument": "AIA 171 Angstrom",
        }
