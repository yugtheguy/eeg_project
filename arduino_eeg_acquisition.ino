/*
 * ============================================================================
 * EEG Alpha Band Monitor - Arduino Data Acquisition
 * ============================================================================
 * 
 * PROJECT: Real-Time EEG Alpha Band Monitoring System for USCAPES
 * PURPOSE: Acquire EEG signals at 250 Hz and transmit to computer via serial
 * DATE: February 2026
 * 
 * HARDWARE REQUIREMENTS:
 * - Arduino Board (Uno/Nano/Mega with 10-bit ADC)
 * - EEG Amplifier (output: 0-5V, gain: 1000-5000x)
 * - EEG Electrodes (Ag/AgCl surface electrodes)
 * - USB Cable for data transmission
 * 
 * CONNECTIONS:
 * - EEG Amplifier LEFT Channel  → Arduino A0
 * - EEG Amplifier RIGHT Channel → Arduino A1 (optional, currently dummy)
 * - EEG Amplifier Ground        → Arduino GND
 * - Arduino USB                 → Computer
 * 
 * OUTPUT FORMAT:
 * CSV format: timestamp,left_value,right_value
 * Example: 1234567,512,0
 * 
 * TECHNICAL SPECS:
 * - Sampling Rate: 250 Hz (one sample every 4000 microseconds)
 * - ADC Resolution: 10-bit (values 0-1023)
 * - Voltage Range: 0-5V (512 = 2.5V midpoint)
 * - Serial Baudrate: 115200 bps
 * - Timing Method: Precise microsecond timing with micros()
 * 
 * NOTES:
 * - This code uses non-blocking timing for stable sampling rate
 * - The RIGHT channel currently sends dummy data (0)
 * - Modify line 59 to read actual right channel: analogRead(A1)
 * - Ensure EEG amplifier output is within 0-5V range
 * 
 * ============================================================================
 */

// ============================================================================
// CONFIGURATION PARAMETERS
// ============================================================================

const int eegPin = A0;              // Analog input pin for EEG signal
const int sampleRate = 250;         // Sampling rate in Hz (samples per second)
const unsigned long interval = 1000000UL / sampleRate;  // 4000 microseconds

// Timing variables
unsigned long lastMicros = 0;

// ============================================================================
// SETUP - Runs once at startup
// ============================================================================

void setup() {
  // Initialize serial communication at 115200 baud
  // This allows fast data transmission to Python software
  Serial.begin(115200);
  
  // Configure analog input pin
  // Arduino Uno/Nano: 10-bit ADC (0-1023 for 0-5V)
  pinMode(eegPin, INPUT);
  
  // Optional: Set analog reference to default (5V)
  // analogReference(DEFAULT);
  
  // Brief startup delay (optional)
  // delay(100);
}

// ============================================================================
// MAIN LOOP - Runs continuously
// ============================================================================

void loop() {
  // Get current time in microseconds
  unsigned long currentMicros = micros();
  
  // Check if it's time for next sample (every 4000 microseconds = 250 Hz)
  if (currentMicros - lastMicros >= interval) {
    // Update last sample time
    lastMicros = currentMicros;
    
    // Read EEG signal from analog pin A0
    // Returns value 0-1023 (10-bit ADC)
    // 0 = 0V, 512 = 2.5V, 1023 = 5V
    int eeg = analogRead(eegPin);
    
    // Get precise timestamp in microseconds
    unsigned long timestamp = micros();
    
    // Transmit data in CSV format: timestamp,left,right
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(eeg);
    Serial.print(",");
    Serial.println(0);  // Dummy right channel (change to analogRead(A1) for dual-channel)
  }
}

// ============================================================================
// USAGE INSTRUCTIONS
// ============================================================================
/*
 
 1. UPLOAD TO ARDUINO:
    - Connect Arduino to computer via USB
    - Open Arduino IDE
    - Select correct board: Tools → Board → Arduino Uno (or your model)
    - Select correct port: Tools → Port → COM# (Windows) or /dev/tty* (Mac/Linux)
    - Click Upload button (→)
 
 2. VERIFY OPERATION:
    - Open Tools → Serial Monitor
    - Set baud rate to 115200
    - You should see data streaming:
      1234567,512,0
      1238567,515,0
      1242567,510,0
      ...
 
 3. CONNECT EEG HARDWARE:
    - Connect EEG amplifier output to A0
    - Connect amplifier ground to Arduino GND
    - Ensure amplifier output is 0-5V range
 
 4. RUN PYTHON SOFTWARE:
    - Close Serial Monitor (Python needs the serial port)
    - Run: python eeg_alpha_monitor.py
    - Or run: python main.py for full system
 
 5. TROUBLESHOOTING:
    - No output? Check baud rate is 115200
    - Constant 0 or 1023? Check amplifier connection
    - Erratic values? Check electrode contact
    - Python can't connect? Close Arduino Serial Monitor first
 
 ============================================================================
 */

// ============================================================================
// ADVANCED MODIFICATIONS
// ============================================================================
/*
 
 FOR DUAL-CHANNEL MODE:
 - Change line 59 from:
   Serial.println(0);
   
   To:
   Serial.println(analogRead(A1));
 
 FOR HIGHER SAMPLING RATE (500 Hz):
 - Change line 38 to:
   const int sampleRate = 500;
 
 FOR DIFFERENT VOLTAGE REFERENCE (3.3V):
 - Add to setup():
   analogReference(EXTERNAL);  // Use AREF pin for external reference
 
 FOR DATA FILTERING ON ARDUINO:
 - Add simple moving average:
   
   const int numReadings = 10;
   int readings[numReadings];
   int readIndex = 0;
   int total = 0;
   
   // In loop():
   total = total - readings[readIndex];
   readings[readIndex] = analogRead(eegPin);
   total = total + readings[readIndex];
   readIndex = (readIndex + 1) % numReadings;
   int eeg = total / numReadings;  // Averaged value
 
 ============================================================================
 */
