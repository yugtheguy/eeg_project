/*
 * ============================================================================
 * RESEARCH-GRADE EEG ACQUISITION WITH HARDWARE SYNC
 * ============================================================================
 * 
 * CRITICAL ADDITION: Hardware synchronization for AAD experiments
 * 
 * SYNCHRONIZATION STRATEGY:
 * - Digital trigger input (pin 2) marks audio stimulus events
 * - Rising edge on trigger = audio onset timestamp
 * - Timestamps relative to first trigger (not boot time)
 * - Embedded sync markers in data stream
 * 
 * HARDWARE CONNECTIONS:
 * - EEG Amplifier LEFT  → Arduino A0
 * - EEG Amplifier RIGHT → Arduino A1 (optional)
 * - EEG Amplifier GND   → Arduino GND
 * - Audio trigger OUT   → Arduino Pin 2 (digital input)
 * - Audio trigger GND   → Arduino GND (CRITICAL - common ground!)
 * 
 * TRIGGER SOURCE OPTIONS:
 * 
 * Option 1: Computer audio out → simple circuit → Arduino pin 2
 *   - Use voltage divider: Audio jack tip (0-1V) → divide → 5V logic
 *   - Resistors: 10kΩ + 2.2kΩ divider
 *   - Coupling capacitor: 10µF to remove DC
 *   - Comparator: LM393 to convert to clean digital pulse
 * 
 * Option 2: Python software trigger via second Arduino pin
 *   - Use Python to set DTR/RTS pin high when audio plays
 *   - Arduino reads serial line "TRIG\n" command
 * 
 * Option 3: Audio-to-trigger converter board
 *   - Use commercial TTL trigger box
 *   - Connect trigger output to pin 2
 * 
 * OUTPUT FORMAT:
 * - Data: timestamp_us,left_adc,right_adc,trigger_state
 * - Trigger marker: SYNC,timestamp_us,trigger_count
 * 
 * ============================================================================
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
const int eegLeftPin = A0;          // EEG left channel
const int eegRightPin = A1;         // EEG right channel (optional)
const int triggerPin = 2;           // Hardware trigger input (digital)
const int ledPin = 13;              // Status LED

const int sampleRate = 250;         // Hz
const unsigned long interval = 1000000UL / sampleRate;  // 4000 microseconds

// Timing variables
unsigned long lastMicros = 0;
unsigned long firstTriggerTime = 0; // Timestamp of first trigger
bool firstTriggerReceived = false;

// Trigger detection
volatile bool triggerFlag = false;
volatile unsigned long triggerTime = 0;
volatile int triggerCount = 0;

// ============================================================================
// INTERRUPT SERVICE ROUTINE - Trigger Detection
// ============================================================================
void triggerISR() {
  // Record trigger time immediately
  unsigned long currentTime = micros();
  
  // First trigger establishes time zero
  if (!firstTriggerReceived) {
    firstTriggerTime = currentTime;
    firstTriggerReceived = true;
  }
  
  triggerTime = currentTime;
  triggerFlag = true;
  triggerCount++;
  
  // Blink LED briefly
  digitalWrite(ledPin, HIGH);
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  
  // Configure pins
  pinMode(eegLeftPin, INPUT);
  pinMode(eegRightPin, INPUT);
  pinMode(triggerPin, INPUT_PULLUP);  // Use internal pullup
  pinMode(ledPin, OUTPUT);
  
  // Attach interrupt for trigger detection
  // RISING edge = trigger goes from LOW to HIGH
  // For active-low triggers, use FALLING instead
  attachInterrupt(digitalPinToInterrupt(triggerPin), triggerISR, RISING);
  
  // Mac compatibility delay
  delay(2000);
  
  // Startup blinks
  for (int i = 0; i < 3; i++) {
    digitalWrite(ledPin, HIGH);
    delay(200);
    digitalWrite(ledPin, LOW);
    delay(200);
  }
  
  // Send header
  Serial.println("========================================");
  Serial.println("RESEARCH-GRADE EEG WITH HARDWARE SYNC");
  Serial.println("========================================");
  Serial.println("Sampling: 250 Hz");
  Serial.println("Trigger: Pin 2 (RISING edge)");
  Serial.println("Format: timestamp_us,left,right,trigger");
  Serial.println("Sync markers: SYNC,timestamp_us,count");
  Serial.println("========================================");
  Serial.println("");
  
  // Wait for first trigger before starting acquisition
  Serial.println("⏳ WAITING FOR FIRST TRIGGER...");
  Serial.println("   (Start audio stimulus to begin recording)");
  Serial.println("");
  
  while (!firstTriggerReceived) {
    delay(10);
  }
  
  // First trigger received!
  Serial.println("✓ FIRST TRIGGER RECEIVED!");
  Serial.print("   Time zero established at: ");
  Serial.println(firstTriggerTime);
  Serial.println("   Starting synchronized acquisition...");
  Serial.println("");
  
  digitalWrite(ledPin, HIGH);
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
  // Check for trigger events
  if (triggerFlag) {
    noInterrupts();  // Atomic read
    unsigned long tTime = triggerTime;
    int tCount = triggerCount;
    triggerFlag = false;
    interrupts();
    
    // Send sync marker
    Serial.print("SYNC,");
    Serial.print(tTime - firstTriggerTime);  // Relative to time zero
    Serial.print(",");
    Serial.println(tCount);
    
    // Brief LED blink
    delay(50);
    digitalWrite(ledPin, LOW);
    delay(50);
    digitalWrite(ledPin, HIGH);
  }
  
  // Regular EEG sampling
  unsigned long currentMicros = micros();
  
  if (currentMicros - lastMicros >= interval) {
    lastMicros = currentMicros;
    
    // Read EEG channels
    int leftValue = analogRead(eegLeftPin);
    int rightValue = analogRead(eegRightPin);
    
    // Read trigger state (for verification)
    int triggerState = digitalRead(triggerPin);
    
    // Calculate timestamp relative to first trigger
    unsigned long relativeTime = currentMicros - firstTriggerTime;
    
    // Send data: timestamp,left,right,trigger_state
    Serial.print(relativeTime);
    Serial.print(",");
    Serial.print(leftValue);
    Serial.print(",");
    Serial.print(rightValue);
    Serial.print(",");
    Serial.println(triggerState);
  }
}

/*
 * ============================================================================
 * HARDWARE TRIGGER CIRCUIT (Option 1: Audio-derived trigger)
 * ============================================================================
 * 
 * Simple audio-to-trigger converter:
 * 
 *                  Audio Jack (3.5mm)
 *                       |
 *                       | (Tip = Left channel)
 *                       |
 *                      [C1]  10µF coupling capacitor
 *                       |
 *         R1            |
 *  Audio ----/\/\/------+------/\/\/---- GND
 *         10kΩ          |         2.2kΩ
 *                       |  R2
 *                       |
 *                       +----> To comparator or Arduino pin
 * 
 * Better solution with comparator (clean digital pulse):
 * 
 *  Audio ---[10µF]---+---[10kΩ]--- GND
 *                    |
 *                    +---> LM393 comparator IN+
 *                              |
 *                           Reference voltage (2.5V divider)
 *                              |
 *                           LM393 OUT ----> Arduino Pin 2
 * 
 * Threshold adjustment:
 * - Use 10kΩ potentiometer for reference voltage
 * - Adjust until trigger fires reliably at audio onset
 * - Test with oscilloscope if available
 * 
 * ============================================================================
 * VALIDATION CHECKLIST
 * ============================================================================
 * 
 * Before experiments:
 * 
 * 1. Test trigger timing precision:
 *    - Play 1 Hz click train
 *    - Verify SYNC markers appear every 1000 ms ± 1 ms
 * 
 * 2. Test trigger reliability:
 *    - Play 100 triggers
 *    - Verify 100 SYNC markers received (no drops)
 * 
 * 3. Measure trigger latency:
 *    - Use oscilloscope on audio out + trigger pin
 *    - Measure delay: should be < 100 µs
 * 
 * 4. Test common ground:
 *    - Audio GND must connect to Arduino GND
 *    - Otherwise, floating trigger causes false triggers
 * 
 * 5. Document in paper:
 *    - "EEG and audio synchronized via hardware trigger"
 *    - "Trigger latency: X µs (measured with oscilloscope)"
 *    - "Zero dropped triggers across N hours of recording"
 * 
 * ============================================================================
 * CLOCK DRIFT HANDLING
 * ============================================================================
 * 
 * Even with hardware triggers, clocks still drift.
 * 
 * Solution: Periodic re-sync markers
 * 
 * Add to your audio stimulus:
 * - Trigger pulse every 30 seconds
 * - Python logs: audio_time when trigger sent
 * - Arduino logs: SYNC,arduino_time,count
 * 
 * Offline analysis:
 * 1. Build mapping: audio_time[i] ↔ arduino_time[i]
 * 2. Fit linear drift model: arduino_time = α + β * audio_time
 * 3. Correct all EEG timestamps using this model
 * 4. Report β (clock ratio) and residual error in paper
 * 
 * Expected results:
 * - β ≈ 1.0001 or 0.9999 (typical drift)
 * - Residual error < 1 ms after correction
 * 
 * ============================================================================
 */
