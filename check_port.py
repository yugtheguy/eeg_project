"""Quick script to check COM port availability."""
import serial.tools.list_ports
import serial

print("Available COM ports:")
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"\n{port.device}")
    print(f"  Description: {port.description}")
    print(f"  Hardware ID: {port.hwid}")
    
    # Try to open it
    try:
        test_serial = serial.Serial(port.device, 9600, timeout=0.5)
        test_serial.close()
        print(f"  Status: ✅ Available")
    except serial.SerialException as e:
        print(f"  Status: ❌ In use or blocked")
        print(f"  Error: {e}")

print("\n" + "="*70)
print("If COM7 shows 'In use or blocked':")
print("  1. Close Arduino IDE Serial Monitor")
print("  2. Close any other serial programs")
print("  3. Unplug and replug the Arduino USB cable")
print("  4. Try running this script again")
