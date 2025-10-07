#!/bin/bash
echo "Testing Race Monitor Backend API"
echo "================================"

echo "1. Health Check:"
curl -s http://localhost:9003/api/health || echo "Failed"

echo -e "\n\n2. Data Summary:"
curl -s http://localhost:9003/api/data/summary || echo "Failed"

echo -e "\n\n3. Port Info:"
curl -s http://localhost:9003/api/info/ports || echo "Failed"

echo -e "\n\nTest complete!"