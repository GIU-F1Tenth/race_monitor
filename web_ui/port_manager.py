#!/usr/bin/env python3
"""
Port Manager for Race Monitor Web UI
Handles dynamic port allocation and service discovery
"""

import socket
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

class PortManager:
    def __init__(self, config_file: str = ".ports.json"):
        self.config_file = Path(__file__).parent / config_file
        self.base_backend_port = 9000  # Much higher port range
        self.base_frontend_port = 4000
        
    def find_free_port(self, start_port: int, max_attempts: int = 100) -> int:
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_free(port):
                return port
        raise RuntimeError(f"Could not find free port starting from {start_port}")
    
    def is_port_free(self, port: int) -> bool:
        """Check if a port is free"""
        # Use a more robust method to check port availability
        import subprocess
        try:
            # Check if port is in use with netstat
            result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
            port_pattern = f':{port} '
            if port_pattern in result.stdout:
                return False
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fallback to socket method if netstat not available
            pass
        
        # Double-check with socket binding
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                s.close()
                return True
            except OSError:
                return False
    
    def allocate_ports(self) -> Dict[str, int]:
        """Allocate free ports for backend and frontend"""
        backend_port = self.find_free_port(self.base_backend_port)
        frontend_port = self.find_free_port(self.base_frontend_port)
        
        ports = {
            "backend": backend_port,
            "frontend": frontend_port,
            "allocated_at": int(time.time())
        }
        
        # Save to config file
        with open(self.config_file, 'w') as f:
            json.dump(ports, f, indent=2)
        
        return ports
    
    def get_ports(self) -> Optional[Dict[str, int]]:
        """Get currently allocated ports"""
        if not self.config_file.exists():
            return None
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def cleanup(self):
        """Clean up port configuration"""
        if self.config_file.exists():
            self.config_file.unlink()

if __name__ == "__main__":
    import sys
    
    manager = PortManager()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "allocate":
            ports = manager.allocate_ports()
            print(f"Backend: {ports['backend']}")
            print(f"Frontend: {ports['frontend']}")
        elif sys.argv[1] == "get":
            ports = manager.get_ports()
            if ports:
                print(f"Backend: {ports['backend']}")
                print(f"Frontend: {ports['frontend']}")
            else:
                print("No ports allocated")
        elif sys.argv[1] == "cleanup":
            manager.cleanup()
            print("Ports cleaned up")
    else:
        ports = manager.allocate_ports()
        print(json.dumps(ports, indent=2))