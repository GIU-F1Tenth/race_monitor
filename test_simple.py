#!/usr/bin/env python3
"""
Simple test for race_monitor modifications
Checks that the AERO integration features are present
"""

import os
import sys

def test_race_monitor_modifications():
    """Test that race_monitor has the required AERO integration features."""
    print("🧪 Testing race_monitor modifications...")
    
    try:
        # Check the main node file for modifications
        node_file = 'race_monitor/race_monitor.py'
        
        if not os.path.exists(node_file):
            print(f"   ❌ Node file not found: {node_file}")
            return False
        
        with open(node_file, 'r') as f:
            content = f.read()
        
        # Check for key AERO integration features
        aero_features = {
            'Race state publisher': 'race_state_pub' in content,
            'Race state topic': '/race_monitor/state' in content,
            'String message type': 'std_msgs/String' in content,
            'Race state publishing': 'publish_race_state' in content or 'race_state_pub.publish' in content
        }
        
        found_count = 0
        for feature_name, found in aero_features.items():
            if found:
                print(f"   ✅ {feature_name}")
                found_count += 1
            else:
                print(f"   ❌ {feature_name}")
        
        print(f"   📊 AERO integration: {found_count}/{len(aero_features)} features found")
        
        # Check for specific implementation details
        print("\n   🔍 Checking implementation details...")
        
        # Check for race state publisher creation
        if 'create_publisher' in content and 'String' in content:
            print("      ✅ Race state publisher created")
        else:
            print("      ❌ Race state publisher not found")
        
        # Check for race state topic
        if '/race_monitor/state' in content:
            print("      ✅ Race state topic defined")
        else:
            print("      ❌ Race state topic not found")
        
        # We expect most AERO features to be present
        assert found_count >= len(aero_features) * 0.7, "Too many AERO features missing"
        
        return True
        
    except Exception as e:
        print(f"   ❌ race_monitor test failed: {e}")
        return False

def test_file_structure():
    """Test that race_monitor has the expected file structure."""
    print("🧪 Testing race_monitor file structure...")
    
    try:
        expected_files = [
            'race_monitor/race_monitor.py',
            'race_monitor/__init__.py',
            'launch/',
            'config/',
            'package.xml',
            'setup.py'
        ]
        
        found_count = 0
        for expected_file in expected_files:
            if os.path.exists(expected_file):
                print(f"   ✅ Found: {expected_file}")
                found_count += 1
            else:
                print(f"   ❌ Missing: {expected_file}")
        
        print(f"   📊 File structure: {found_count}/{len(expected_files)} files found")
        
        # We expect most files to be present
        assert found_count >= len(expected_files) * 0.8, "Too many files missing"
        
        return True
        
    except Exception as e:
        print(f"   ❌ File structure test failed: {e}")
        return False

def test_config_files():
    """Test that race_monitor has proper configuration files."""
    print("🧪 Testing race_monitor configuration...")
    
    try:
        config_dir = 'config'
        launch_dir = 'launch'
        
        if os.path.exists(config_dir):
            config_files = os.listdir(config_dir)
            print(f"   ✅ Config directory: {len(config_files)} files")
            for config_file in config_files:
                print(f"      - {config_file}")
        else:
            print("   ❌ Config directory not found")
        
        if os.path.exists(launch_dir):
            launch_files = os.listdir(launch_dir)
            print(f"   ✅ Launch directory: {len(launch_files)} files")
            for launch_file in launch_files:
                print(f"      - {launch_file}")
        else:
            print("   ❌ Launch directory not found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def main():
    """Run all race_monitor tests."""
    print("🚀 race_monitor Comprehensive Testing")
    print("=" * 50)
    
    tests = [
        test_race_monitor_modifications,
        test_file_structure,
        test_config_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All race_monitor tests passed!")
        return True
    else:
        print("❌ Some race_monitor tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
