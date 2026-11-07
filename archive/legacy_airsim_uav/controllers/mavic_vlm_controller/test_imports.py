import sys
import os
sys.path.append(r"c:\Users\Kruti\Desktop\PROG\UAV\controllers\mavic_vlm_controller")
from unittest.mock import MagicMock
sys.modules['controller'] = MagicMock()
import mavic_vlm_controller
print("Imports successful!")
