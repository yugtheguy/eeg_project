/*
 * ============================================================================
 * EEG Alpha Band Monitor - Arduino Data Acquisition (MAC COMPATIBLE)
 * ============================================================================
 * 
 * This version includes Mac-specific fixes:
 * - Startup delay for Serial connection stability
 * - Startup messages to confirm code is running
 * - LED indicator for visual feedback
 * - More robust serial communication
 * 
 * ============================================================================
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
const int eegPin = A0;              // EEG signal input pin
const int ledPin = 13;              // Built-in LED for status indication
const int sampleRate = 250;         // Sampling rate in Hz
const unsigned long interval = 1000000UL / sampleRate;  // 4000 microseconds

// Timing variables
unsigned long lastMicros = 0;
unsigned long startTime = 0;
int sampleCount = 0;

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  // Initialize serial at 115200 baud
  Serial.begin(115200);
  
  // Setup pins
  pinMode(eegPin, INPUT);
  pinMode(ledPin, OUTPUT);
  
  // MAC FIX: Wait for serial connection to stabilize
  // Mac needs 2-3 seconds for proper serial initialization
  delay(2000);
  
  // Blink LED rapidly to show startup
  for (int i = 0; i < 5; i++) {
    digitalWrite(ledPin, HIGH);
    delay(100);
    digitalWrite(ledPin, LOW);
    delay(100);
  }
  
  // Send startup messages
  Serial.println("========================================");
  Serial.println("EEG ACQUISITION SYSTEM - MAC VERSION");
  Serial.println("========================================");
  Serial.println("Status: READY");
  Serial.println("Baud Rate: 115200");
  Serial.println("Sample Rate: 250 Hz");
  Serial.println("Format: timestamp,left,right");
  Serial.println("========================================");
  Serial.println("");
  
  // Brief pause before data streaming
  delay(1000);
  
  // Turn on LED to show active data acquisition
  digitalWrite(ledPin, HIGH);
  
  // Record start time
  startTime = millis();
  
  Serial.println("Starting data transmission NOW...");
  Serial.println("");
}

// ============================================================================
// MAIN LOOP
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
    int eeg = analogRead(eegPin);
    
    // Get precise timestamp in microseconds
    unsigned long timestamp = micros();
    
    // Transmit data in CSV format: timestamp,left,right
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(eeg);
    Serial.print(",");
    Serial.println(0);  // Dummy right channel
    
    // Increment sample counter
    sampleCount++;
    
    // Every 250 samples (1 second), send status message
    if (sampleCount % 250 == 0) {
      // Blink LED briefly to show activity
      digitalWrite(ledPin, LOW);
      delay(1);
      digitalWrite(ledPin, HIGH);
    }
    
    // Every 5000 samples (20 seconds), send heartbeat
    if (sampleCount % 5000 == 0) {
      unsigned long elapsed = (millis() - startTime) / 1000;
      Serial.print("# STATUS: ");
      Serial.print(sampleCount);
      Serial.print(" samples, ");
      Serial.print(elapsed);
      Serial.println(" seconds running");
    }
  }
}

/*
 * ============================================================================
 * USAGE INSTRUCTIONS FOR MAC
 * ============================================================================
 * 
 * 1. UPLOAD THIS SKETCH:
 *    - Connect Arduino via USB
 *    - Open Arduino IDE
 *    - Tools → Board → Select your Arduino model
 *    - Tools → Port → Select /dev/cu.usbserial* or /dev/cu.usbmodem*
 *    - Click Upload (→)
 * 
 * 2. VERIFY IT'S WORKING:
 *    - Watch for rapid LED blinking during startup (5 blinks)
 *    - LED stays ON during data acquisition
 *    - LED blinks briefly once per second (shows it's alive)
 * 
 * 3. CHECK SERIAL OUTPUT:
 *    - Tools → Serial Monitor
 *    - Set baud to 115200 baud
 *    - You should see:
 *      ========================================
 *      EEG ACQUISITION SYSTEM - MAC VERSION
 *      ========================================
 *      ...
 *      Starting data transmission NOW...
 *      
 *      1234567,512,0
 *      1238567,515,0
 *      1242567,510,0
 *      ...
 * 
 * 4. TROUBLESHOOTING:
 *    - LED not blinking? → Code not uploaded, try re-uploading
 *    - Serial Monitor blank? → Wrong port selected OR wrong baud rate
 *    - Serial Monitor says "disconnected"? → Close and reopen Serial Monitor
 *    - On Mac, you may need to close and reopen Serial Monitor after upload
 * 
 * 5. RUN PYTHON SOFTWARE:
 *    - IMPORTANT: Close Serial Monitor first!
 *    - Run: python eeg_alpha_monitor_mac.py
 *    - OR: python check_port_mac.py (to test connection)
 * 
 * ============================================================================
 */
