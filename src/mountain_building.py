import numpy as np
from utils import find_project_path


maps2 = np.array([['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG'],
                ['WG', '3^', '  ', '  ', 'WG', 'WG', 'WG'],
                ['WG', '  ', 'WG', 'WG', '  ', '  ', 'WG'],
                ['WG', '  ', '  ', '  ', '  ', '  ', 'WG'],
                ['WG', 'WG', 'WG', '  ', 'WG', '  ', 'WG'],
                ['WG', 'WG', 'WG', '  ', '  ', 'GG', 'WG'],
                ['WG', 'WG', 'WG', 'WG', 'WG', 'WG', 'WG']])
r,c  = maps2.shape
obstables = []
for i in range(r):
    for j in range(c):
        if maps2[i][j]=='WG' and i*j!=0 and i!=6 and j!=6:
            obstables.append((i,j))
print(obstables)

def get_mountain_yaml(obstables):
    path = find_project_path()
    mountains = ''
    i = 0
    for (r,c) in obstables:
        i+=1
      
        x = 2*c-6
        y = -2*r+6
        path = find_project_path()
        mountains+= f'''
- add_model:
    name: mountain{i}
    file: file://{path}/objects/mountain_OBJ.sdf
- add_weld:
    parent: world
    child: mountain{i}::mountain_OBJ
    X_PC:
        translation: [{x}, {y}, 0]       
        '''
    return mountains
# print(get_mountain_yaml(obstables))