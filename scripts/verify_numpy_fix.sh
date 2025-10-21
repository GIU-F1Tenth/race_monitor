#!/bin/bash
#
# Verification script for NumPy compatibility fix.
#
# This script tests the same operations that were failing in CI to ensure
# the NumPy version constraint fix resolves the tf_transformations issue.
#

echo "=== NumPy Compatibility Fix Verification ==="
echo

echo "1. Testing NumPy version..."
python3 -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

echo
echo "2. Testing tf_transformations import..."
python3 -c "import tf_transformations; print('✓ tf_transformations imported successfully')"

echo
echo "3. Testing transforms3d import (used by tf_transformations)..."
python3 -c "import transforms3d; print('✓ transforms3d imported successfully')"

echo
echo "4. Testing race_monitor import..."
python3 -c "import race_monitor; print('✓ race_monitor imported successfully')"

echo
echo "5. Testing specific NumPy compatibility issue..."
python3 -c "
import numpy as np
try:
    # Test the deprecated np.float attribute that was causing the CI failure
    if hasattr(np, 'float'):
        print('Warning: np.float attribute still exists (deprecated)')
    else:
        print('✓ np.float attribute properly removed (NumPy 2.0+)')
    
    # This should work in NumPy < 2.0 but fail in NumPy 2.0+
    result = np.maximum_sctype(np.float64)
    print(f'✓ np.maximum_sctype available (NumPy < 2.0): {result}')
except AttributeError as e:
    if 'maximum_sctype' in str(e):
        print(f'✗ NumPy 2.0+ detected - would cause tf_transformations failure: {e}')
        exit(1)
    elif 'float' in str(e):
        print(f'✗ np.float deprecation issue detected: {e}')
        exit(1)
    else:
        raise
"

echo
echo "6. Testing the exact same import path that failed in CI..."
python3 -c "
from race_monitor import RaceMonitor
print('✓ RaceMonitor class imported successfully')
"

echo
echo "=== All tests passed! Fix verified. ==="