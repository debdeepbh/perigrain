import sys
import os
# previous directory of the current script
prev_dir = os.path.dirname(sys.path[0])
print(prev_dir)
# sys.path.append(prev_dir)
# sys.path.append(os.path.join(prev_dir,'bar'))
cwd = os.getcwd()
print(cwd)
