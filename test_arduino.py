"""
Arduino Connection Tester.

Helps diagnose serial connection issues with Arduino.
"""

import serial
import serial.tools.list_ports
import time
import sys


def list_available_ports():
    """List all available COM ports."""
    print("\n" + "="*70)
    print("AVAILABLE SERIAL PORTS")
    print("="*70)
    
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports detected!")
        print("\nTroubleshooting:")
        print("  1. Check if Arduino is connected via USB")
        print("  2. Try a different USB cable")
        print("  3. Check Device Manager (Windows) or dmesg (Linux)")
        return None
    
    port_list = []
    for i, port in enumerate(ports, 1):
        print(f"\n{i}. {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Hardware ID: {port.hwid}")
        if port.manufacturer:
            print(f"   Manufacturer: {port.manufacturer}")
        
        # Check if it looks like an Arduino
        port_info = (port.description + " " + (port.manufacturer or "")).lower()
        if any(keyword in port_info for keyword in ['arduino', 'ch340', 'usb serial']):
            print(f"   ⭐ Likely Arduino!")
        
        port_list.append(port.device)
    
    print("\n" + "="*70)
    return port_list


def test_connection(port, baudrate=115200):
    """Test connection to a specific port."""
    print(f"\n{'='*70}")
    print(f"TESTING CONNECTION: {port} @ {baudrate} baud")
    print("="*70)
    
    try:
        # Open serial port
        print(f"Opening {port}...")
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=2.0,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        
        print("✅ Port opened successfully")
        
        # Wait for Arduino reset
        print("Waiting for Arduino to reset (2 seconds)...")
        time.sleep(2)
        
        # Flush initial data
        ser.reset_input_buffer()
        print("Input buffer flushed")
        
        # Try to read some lines
        print("\n📡 Reading data (10 seconds)...")
        print("-" * 70)
        
        start_time = time.time()
        line_count = 0
        valid_count = 0
        
        while time.time() - start_time < 10:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        line_count += 1
                        
                        # Try to parse
                        parts = line.split(',')
                        
                        if len(parts) == 3:
                            try:
                                timestamp = float(parts[0])
                                left = float(parts[1])
                                right = float(parts[2])
                                
                                if line_count <= 5:  # Show first 5 lines
                                    print(f"✅ Line {line_count}: {line}")
                                    print(f"   Parsed: t={timestamp:.2f}, L={left:.2f}, R={right:.2f}")
                                
                                valid_count += 1
                                
                            except ValueError:
                                if line_count <= 5:
                                    print(f"⚠️  Line {line_count}: {line}")
                                    print(f"   Could not parse as floats")
                        else:
                            if line_count <= 5:
                                print(f"⚠️  Line {line_count}: {line}")
                                print(f"   Expected 3 fields, got {len(parts)}")
                except Exception as e:
                    print(f"⚠️  Error reading line: {e}")
            else:
                time.sleep(0.01)
        
        print("-" * 70)
        print(f"\n📊 RESULTS:")
        print(f"   Total lines received: {line_count}")
        print(f"   Valid data lines: {valid_count}")
        
        if valid_count > 0:
            success_rate = (valid_count / line_count) * 100
            print(f"   Success rate: {success_rate:.1f}%")
            
            if success_rate >= 90:
                print("\n✅ CONNECTION EXCELLENT - Ready to use!")
            elif success_rate >= 70:
                print("\n⚠️  CONNECTION GOOD - Some data loss detected")
            else:
                print("\n❌ CONNECTION POOR - Check Arduino code and wiring")
        else:
            print("\n❌ NO VALID DATA - Check Arduino code format")
            print("\nExpected format: timestamp,left_channel,right_channel")
            print("Example: 1234.567,512.3,498.1")
        
        # Close port
        ser.close()
        print(f"\n✅ Port closed")
        
        return valid_count > 0
        
    except serial.SerialException as e:
        print(f"❌ Serial error: {e}")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_baudrates(port):
    """Test multiple baud rates."""
    print(f"\n{'='*70}")
    print(f"TESTING MULTIPLE BAUD RATES: {port}")
    print("="*70)
    
    baudrates = [9600, 19200, 38400, 57600, 115200, 230400]
    
    print("\nThis will test each baud rate for 5 seconds...")
    print("Press Ctrl+C to skip\n")
    
    results = {}
    
    for baud in baudrates:
        print(f"\n--- Testing {baud} baud ---")
        
        try:
            ser = serial.Serial(port=port, baudrate=baud, timeout=1.0)
            time.sleep(2)  # Arduino reset
            ser.reset_input_buffer()
            
            line_count = 0
            start_time = time.time()
            
            while time.time() - start_time < 5:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        line_count += 1
            
            ser.close()
            
            results[baud] = line_count
            print(f"   Received {line_count} lines")
            
        except KeyboardInterrupt:
            print("\n⏭️  Skipped")
            break
        except Exception as e:
            print(f"   Error: {e}")
            results[baud] = 0
    
    # Show results
    if results:
        print(f"\n{'='*70}")
        print("BAUD RATE TEST RESULTS")
        print("="*70)
        
        best_baud = max(results.items(), key=lambda x: x[1])
        
        for baud, count in sorted(results.items()):
            marker = " ⭐" if baud == best_baud[0] else ""
            print(f"  {baud:>7} baud: {count:>4} lines{marker}")
        
        print(f"\n✅ Recommended baud rate: {best_baud[0]} (received {best_baud[1]} lines)")


def interactive_test():
    """Interactive testing mode."""
    print("\n" + "="*70)
    print("INTERACTIVE ARDUINO CONNECTION TESTER")
    print("="*70)
    
    # Step 1: List ports
    ports = list_available_ports()
    
    if not ports:
        return
    
    # Step 2: Select port
    print("\nSelect a port to test:")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port}")
    
    try:
        choice = input("\nEnter port number (or type port name directly): ").strip()
        
        if choice.isdigit():
            port_idx = int(choice) - 1
            if 0 <= port_idx < len(ports):
                selected_port = ports[port_idx]
            else:
                print("Invalid selection")
                return
        else:
            selected_port = choice
        
        # Step 3: Test connection
        print("\nTest options:")
        print("  1. Quick test (default baud rate: 115200)")
        print("  2. Test multiple baud rates")
        print("  3. Custom baud rate")
        
        test_choice = input("\nSelect test (1-3) [default: 1]: ").strip() or "1"
        
        if test_choice == "1":
            test_connection(selected_port, 115200)
        
        elif test_choice == "2":
            test_baudrates(selected_port)
        
        elif test_choice == "3":
            baud = int(input("Enter baud rate: "))
            test_connection(selected_port, baud)
        
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def quick_test(port=None, baudrate=115200):
    """Quick test mode for command line usage."""
    if port:
        test_connection(port, baudrate)
    else:
        ports = list_available_ports()
        if ports:
            print(f"\n🔍 Auto-testing first port: {ports[0]}")
            test_connection(ports[0], baudrate)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Command line mode
        port = sys.argv[1]
        baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
        quick_test(port, baudrate)
    else:
        # Interactive mode
        interactive_test()


if __name__ == "__main__":
    main()
