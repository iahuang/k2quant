import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "bin"))

import vptq
print(vptq.add(1, 2))