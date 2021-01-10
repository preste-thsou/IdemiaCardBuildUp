# ALl problem variables
import numpy as np
import random

"""
General Variables:

list_u : authorized thickness of PC layers
0. 50 μm
1. 75 μm
2. 100 μm
3. 125 μm
4. 150 μm
5. 175 μm
6. 200 μm
7. 225 μm
8. 250 μm

type_pc : caracterize position of layer : overlay or inside
0. inside
1. overlay

type_lc : types of reactivity of the layer of PC
0. Laser reactive
1. Laser non reactive



list_type_e  : type of security elements
0. offset ink
1. silkscreen ink
2. Lasink matrix
3. CLI
4. SLI
5. DOVID
6. contactless inlay + chip
7. contactless dual inlay
8. contact chip
9. dual chip
10. classical B&W engravable areas
11. Tactile laser engravable areas
12. DP extrudable area



C_e_lc_pc_u : contains 1 if element e can be placed on layer of type
pc (from list_pc) of laser reactivity lc (from list_lc) of thickness u (from list_u) , 0 otherwise

Position matrix IP_e, indicates the initial position of element e in the x, y axis (x, y, width,
height). It is set to 0, 0, width_e, height_e if the position can vary, and to −1, −1, −1, −1 if
not applicable for this element e. Width and eight includes the positional tolerance of
element e. Position x, y represent the left bottom corner of the element. It contains integers,
ranging from 0 to res x width_doc and res x length_doc where res is the minimal
resolution step to consider for the model.

Min_x, min_y, max_x and max_y define the usable area on the card (origin 0, 0 is the left
bottom corner).

Min_ISO_u, max_ISO_u and factor_comp are used to compute the total
thickness and the lower and upper boundaries of possible thicknesses

Exclusion zones of specific element e for some other elements e’ etc. :
• the radius r(e) (integer) of exclusion zone around the element e. If 0, the exclusion is
exactly the area where e is positioned.
• the direction dir(e) in axis z for exclusion zone: 0 : from center to external, 1: from
external to center, 2 bidirectional

by convention, if IP_e = -1, -1, -1, -1 and Ex_ee’ = 1, then it means that e and e’ cannot be on the
same layer l

Exclusion matrix of elements Ex_ee’ contains 1 if e’ cannot be present in the exclusion zone
of e and 0 otherwise. Ex_ee’ is not necessarily symmetric

Problem Specific variabes:

Max nb of layers : max_nl
Nb of required security elements: max_e

Required list of security elements list_e : contains the list of types of elements e (as in
type_e). Ex, 0 is the first element of the list_e and list_e[0] indicates its type.
Note : len( list_e) = max_e


Decision variables
Nb of layers : nl
Centre of the card / DP c = nl/2 (possibly a float value)


Thickness u(l) (within list_u )
Type of PC : pc(l) (within list_PC)
Ranking / position p(l) in Z axis (0 being the bottom layer (back of the card), from 0 to
nl − 1
P_e : if IP_e == -1, -1, -1, -1, P_e = IP_e, otherwise contains x,y, width and
length, defining the bounding box of e
AL(e) = l with l ranging from 0 to nl – 1. Initialized to nl.

total_u : total thickness of the doc
half1_u : thickness of the first half of the doc (not including compression factor, used for symmetry)
half2_u : thickness of the second half ot the doc (not including compression factor, usef for symmetry)
"""




####################  First Ex    ########
SETTINGS = {
    'min_x' : 10, #res = 0.1 mm
    'max_x' : 855, #856 mm - 1 mm
    'min_y' : 10,
    'max_y' : 539, # 539.8 mm - 1 mm
    'type_pc' : 2,
    'laser_pc' : 2,
    'list_u' : np.array([
        50,75,100,125,150,200,250,275,350
    ]),
    'list_e' : np.array([
        0, #offsetink
        1, #silkscreenink
        2, #lasinkmatrix
        3, #CLI
        5, #DOVID
        5, #DOVID
        8 #contactchip
    ]),

    'len_e': 7,
    'list_type_e' : 13,

    'C_e_pc_lc_u': np.array([
        [ #element 0 of list__type_e: offset ink
            [ # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0], #laser reactive
                [0, 0, 0, 1, 1, 1, 1, 1, 1]  #laser non reactive
            ],
            [ #pc type overlay
                [0, 0, 0, 0, 0, 0, 0, 0, 0], #laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0] #laser non reactive
            ]
        ],
        [  # element 1 of list_type_e : silkscreen ink
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 1, 1, 1, 1, 1, 1]  # laser non reactive
            ],
            [  # pc type overlay
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 2 of list_type_e : lasinkmatrix
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 1, 1, 1, 1, 1, 1]  # laser non reactive
            ],
            [  # pc type overlay
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 3 of list_type_e: CLI
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 4 of list_type_e: SLI
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 5 of list_type_e: DOVID
            [  # pc type inside
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [1, 1, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [1, 1, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 6 of list_type_e :contactless inlay + chip
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 1, 1, 1, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 7 of list_type_e :contactless dual_inlay
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 1, 1, 1, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 8 of list_type_e: contactchip
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [1, 1, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 9 of list_type_e: dual chip
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [1, 1, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 10 of list_type_e: classical B&W engravable area
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [1, 1, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 11 of list_type_e: tactile B&W engravable area
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
        [  # element 12 of list_type_e: DP extrudable area
            [  # pc type inside
                [0, 0, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [0, 0, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ],
            [  # pc type overlay
                [1, 1, 0, 0, 0, 0, 0, 0, 0],  # laser reactive
                [1, 1, 0, 0, 0, 0, 0, 0, 0]  # laser non reactive
            ]
        ],
    ]),
    'IP_e' : np.array([
        [-1, -1, -1, -1], #offset ink
        [-1, -1, -1, -1], #silckscreen ink
        [570, 100, 250, 300 ], #lasink matrix fixed position (verso of the card)
        [0, 0, 100, 150], #CLI
        [0, 0, 150, 150], #DOVID
        [0, 0, 150, 150], #DOVID
        [80,255,130, 120], #contactchip
    ]),
    'min_ISO_u' : 730,
    'max_ISO_u' : 790,
    'factor_comp_min' : 0.8,
    'factor_comp_max' : 0.9,
    'r_e': np.array([ #incompatibility radius of list_type_e
        0, #offset
        0, #silkscreen
        10, #Lasink
        10, #CLI
        10, #SLI
        10, #DOVID
        10, #contactless inlay + chip
        10, #contactless dual inlay
        10, #dual chip
        10, #contacthip
        0, #classical B&W engravable areas
        0, #tactile laser engravable area
        10, #DP extrudable area
    ]),
    'dir_e': np.array([ #incompatibilty directions of list_type_e
        0, #offset
        0, #silkscreen
        0, #lasink
        0, #CLI
        0, #SLI
        0, #DOVID
        2, #contactless inlay + chip
        2, #contactless dual inlay
        2, #dual chip
        2, #contacthip
        1, #classical B&W engravable areas
        1, #tactile laser engravable area
        1 #DP extrudable area
    ]),
    'EX_ee': np.array([
        [0,0,1,1,1,1,1,1,1,1,1,1,1], #offset
        [0,0,1,1,1,1,1,1,1,1,1,1,1], #silkscreen
        [0,0,1,1,1,1,1,1,1,1,1,1,1], #lasink
        [0,0,1,1,1,1,1,1,1,1,1,1,1], #CLI
        [0,0,1,1,1,1,1,1,1,1,1,1,1], #SLI
        [0,0,0,1,1,1,1,1,1,1,1,1,1], #DOVID
        [0,0,0,1,1,0,1,1,1,1,1,1,1], #ctl inlay + chip
        [0,0,0,1,1,0,1,1,0,1,1,1,1], #ctl dual inlay
        [0,0,0,1,1,1,1,0,1,1,1,1,1], #dual chip
        [0,0,1,1,1,1,1,1,1,1,1,1,1], #contact chip
        [0,0,0,1,1,0,1,1,1,1,1,1,1], #classical B&W engravable areas
        [0,0,0,1,1,0,1,1,1,1,1,1,1], #tactile laser engravable areas
        [0,0,0,1,1,1,1,1,1,1,1,1,1]
    ]),
    'max_nl' : 10
}

################################################################################################################
################################################################################################################

#Decision variables"

PLAN = {
    'nl'  : 0, #0 is an init value
    'c' : 0.0, #0 is an init value
    #                           0 = u                   1= lc                  2= nl             3 = type
    #properties of layer l : thickness (init at 0), laser reactivity, position (final rank), type (overlay or inner, initialized as inner)
    'l' : np.array([
        [0, SETTINGS['laser_pc'], SETTINGS['max_nl'], 0] for i in range(SETTINGS['max_nl'])]),
    'P_e': np.array(SETTINGS['IP_e']),
    'AL_e': np.zeros((SETTINGS['len_e'])).astype(np.int) + SETTINGS['max_nl'],
    'total_u_min' :  0, #0 is an init value
    'total_u_max' : 0, #0 is an init value
    'half1_u' : 0, #0 is an init value
    'half2_u' : 0, #0 is an init value
    'interaction' :0
    }
