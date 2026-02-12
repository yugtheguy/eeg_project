"""
EEG Signal Acquisition Module.

Handles serial communication with Arduino, including:
- Automatic port detection
- Robust reconnection on failures
- Non-blocking data acquisition
- Data validation and parsing
"""

import serial
import serial.tools.list_ports
import logging
import time
from typing import Optional, List
from collections import deque
import numpy as np

from config import SerialConfig, get_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerialAcquisition:
    """
    Handles real-time EEG signal acquisition from Arduino via serial port.
    
    Implements automatic reconnection, port detection, and robust error handling.
    """
    
    def __init__(self, serial_config: Optional[SerialConfig] = None):
        """
        Initialize serial acquisition interface.
        
        Args:
            serial_config: Serial configuration object. If None, uses global config.
        """
        if serial_config is None:
            serial_config = get_config().serial
        
        self.config = serial_config
        self.serial_port: Optional[serial.Serial] = None
        self.is_connected: bool = False
        self.reconnect_attempts: int = 0
        self.last_reconnect_time: float = 0.0
        
        # Data buffers for two channels (left and right hemisphere)
        self.buffer_left: deque = deque(maxlen=10000)
        self.buffer_right: deque = deque(maxlen=10000)
        
        # Statistics
        self.packets_received: int = 0
        self.packets_corrupted: int = 0
        self.last_packet_time: float = time.time()
        
    def detect_arduino_port(self) -> Optional[str]:
        """
        Auto-detect Arduino serial port.
        
        Returns:
            str: Port name if found, None otherwise
        """
        logger.info("Scanning for Arduino ports...")
        ports = serial.tools.list_ports.comports()
        
        # Look for common Arduino identifiers
        arduino_keywords = ['arduino', 'ch340', 'usb serial', 'atmega']
        
        for port in ports:
            port_description = (port.description + " " + port.manufacturer).lower()
            logger.debug(f"Found port: {port.device} - {port.description}")
            
            for keyword in arduino_keywords:
                if keyword in port_description:
                    logger.info(f"Arduino detected on {port.device}")
                    return port.device
        
        # If no Arduino found, return first available port as fallback
        if ports:
            logger.warning(f"No Arduino found, using first available port: {ports[0].device}")
            return ports[0].device
        
        logger.error("No serial ports found")
        return None
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Establish serial connection to Arduino.
        
        Args:
            port: Serial port name. If None, auto-detects Arduino port.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Determine port to use
            if port is None:
                port = self.config.port
            
            # Auto-detect if port is None or "AUTO"
            if port is None or port.upper() == "AUTO":
                port = self.detect_arduino_port()
                if port is None:
                    return False
            
            # Close existing connection if any
            if self.serial_port is not None and self.serial_port.is_open:
                self.serial_port.close()
            
            # Open serial connection
            logger.info(f"Connecting to {port} at {self.config.baudrate} baud...")
            self.serial_port = serial.Serial(
                port=port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for Arduino to reset (DTR toggle causes reset)
            time.sleep(2.0)
            
            # Flush any initial garbage data
            self.serial_port.reset_input_buffer()
            
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info(f"Successfully connected to {port}")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect to {port}: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Safely close serial connection."""
        if self.serial_port is not None and self.serial_port.is_open:
            try:
                self.serial_port.close()
                logger.info("Serial connection closed")
            except Exception as e:
                logger.error(f"Error closing serial connection: {e}")
        
        self.is_connected = False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Arduino with exponential backoff.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        current_time = time.time()
        
        # Check if enough time has passed since last reconnect attempt
        if current_time - self.last_reconnect_time < self.config.reconnect_delay:
            return False
        
        self.last_reconnect_time = current_time
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.config.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.config.max_reconnect_attempts}) exceeded")
            return False
        
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.config.max_reconnect_attempts}")
        
        # Try to reconnect
        return self.connect()
    
    def parse_data_packet(self, line: str) -> Optional[tuple[float, float, float]]:
        """
        Parse incoming data packet from Arduino.
        
        Expected format: "timestamp,left_channel,right_channel\n"
        Example: "1234.567,512.3,498.1\n"
        
        Args:
            line: Raw line from serial port
        
        Returns:
            tuple: (timestamp, left_value, right_value) or None if parsing fails
        """
        try:
            # Remove whitespace and split by comma
            parts = line.strip().split(',')
            
            if len(parts) != 3:
                self.packets_corrupted += 1
                logger.debug(f"Invalid packet format: expected 3 fields, got {len(parts)}")
                return None
            
            # Parse values
            timestamp = float(parts[0])
            left_value = float(parts[1])
            right_value = float(parts[2])
            
            # Basic sanity check (ADC values typically 0-1023 for 10-bit, 0-4095 for 12-bit)
            if not (0 <= left_value <= 5000) or not (0 <= right_value <= 5000):
                self.packets_corrupted += 1
                logger.debug(f"Values out of valid range: left={left_value}, right={right_value}")
                return None
            
            return (timestamp, left_value, right_value)
            
        except (ValueError, IndexError) as e:
            self.packets_corrupted += 1
            logger.debug(f"Error parsing packet: {e} | Line: {line[:50]}")
            return None
    
    def read_sample(self) -> Optional[tuple[float, float, float]]:
        """
        Read one sample from serial port (non-blocking).
        
        Returns:
            tuple: (timestamp, left_channel, right_channel) or None if no valid data
        """
        if not self.is_connected:
            return None
        
        try:
            # Check if data is available
            if self.serial_port.in_waiting > 0:
                # Read one line
                line = self.serial_port.readline().decode('utf-8', errors='ignore')
                
                # Parse the data
                parsed_data = self.parse_data_packet(line)
                
                if parsed_data is not None:
                    timestamp, left_val, right_val = parsed_data
                    
                    # Add to buffers
                    self.buffer_left.append(left_val)
                    self.buffer_right.append(right_val)
                    
                    # Update statistics
                    self.packets_received += 1
                    self.last_packet_time = time.time()
                    
                    return parsed_data
            
            return None
            
        except serial.SerialException as e:
            logger.error(f"Serial exception during read: {e}")
            self.is_connected = False
            return None
        except Exception as e:
            logger.error(f"Unexpected error during read: {e}")
            return None
    
    def read_batch(self, max_samples: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read multiple samples in a batch (non-blocking).
        
        Args:
            max_samples: Maximum number of samples to read
        
        Returns:
            tuple: (timestamps, left_channel_array, right_channel_array)
        """
        timestamps = []
        left_values = []
        right_values = []
        
        samples_read = 0
        
        while samples_read < max_samples:
            data = self.read_sample()
            
            if data is None:
                break
            
            timestamp, left_val, right_val = data
            timestamps.append(timestamp)
            left_values.append(left_val)
            right_values.append(right_val)
            samples_read += 1
        
        return (
            np.array(timestamps),
            np.array(left_values),
            np.array(right_values)
        )
    
    def get_buffer_data(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get recent data from buffers.
        
        Args:
            num_samples: Number of recent samples to retrieve
        
        Returns:
            tuple: (left_channel_array, right_channel_array)
        """
        left_data = np.array(list(self.buffer_left)[-num_samples:])
        right_data = np.array(list(self.buffer_right)[-num_samples:])
        
        return left_data, right_data
    
    def get_statistics(self) -> dict:
        """
        Get acquisition statistics.
        
        Returns:
            dict: Statistics including packet counts, buffer sizes, etc.
        """
        corruption_rate = 0.0
        if self.packets_received > 0:
            total_packets = self.packets_received + self.packets_corrupted
            corruption_rate = self.packets_corrupted / total_packets * 100
        
        time_since_last_packet = time.time() - self.last_packet_time
        
        return {
            'connected': self.is_connected,
            'packets_received': self.packets_received,
            'packets_corrupted': self.packets_corrupted,
            'corruption_rate_percent': corruption_rate,
            'buffer_left_size': len(self.buffer_left),
            'buffer_right_size': len(self.buffer_right),
            'time_since_last_packet': time_since_last_packet,
            'reconnect_attempts': self.reconnect_attempts
        }
    
    def clear_buffers(self) -> None:
        """Clear all internal data buffers."""
        self.buffer_left.clear()
        self.buffer_right.clear()
        logger.info("Buffers cleared")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
