import numpy as np

class CoordTransforms():

    def __init__(self):
        """
        Variable Notation:
            - v__x: vector expressed in "x" frame
            - q_x_y: quaternion of "x" frame with relative to "y" frame
            - p_x_y__z: position of "x" frame relative to "y" frame expressed in "z" coordinates
            - v_x_y__z: velocity of "x" frame with relative to "y" frame expressed in "z" coordinates
            - R_x2y: rotation matrix that maps vector represented in frame "x" to representation in frame "y" (right-multiply column vec)
    
        Frame Subscripts:
            - m = marker frame (x-right, y-up, z-out when looking at marker)
            - dc = downward-facing camera (if expressed in the body frame)
            - fc = forward-facing camera
            - bu = body up frame (x-forward, y-left, z-up, similar to ENU)
            - bd = body down frame (x-forward, y-right, z-down, similar to NED)
            - lenu = local East-North-Up world frame ("local" implies that it may not be aligned with east and north, but z is up)
            - lned = local North-East-Down world frame ("local" implies that it may not be aligned with north and east, but z is down)
        Rotation matrix:
            R = np.array([[       3x3     0.0]
                          [    rotation   0.0]
                          [     matrix    0.0]
                          [0.0, 0.0, 0.0, 0.0]])
            
            [[ x']      [[       3x3     0.0]  [[ x ]
             [ y']  =    [    rotation   0.0]   [ y ]
             [ z']       [     matrix    0.0]   [ z ]
             [0.0]]      [0.0, 0.0, 0.0, 0.0]]  [0.0]]
        """
        
        # Reference frames
        self.COORDINATE_FRAMES = {'lenu','lned','bu','bd','dc','fc'}
    
        self.WORLD_FRAMES = {'lenu', 'lned'}
    
        self.BODY_FRAMES = {'bu', 'bd', 'dc', 'fc'}
    
        self.STATIC_TRANSFORMS = {'R_lenu2lenu',
                                  'R_lenu2lned',
    
                                  'R_lned2lenu',
                                  'R_lned2lned', 
          
                                  'R_bu2bu', 
                                  'R_bu2bd',
                                  'R_bu2dc',
                                  'R_bu2fc',
          
                                  'R_bd2bu',
                                  'R_bd2bd',
                                  'R_bd2dc',
                                  'R_bd2fc',
          
                                  'R_dc2bu',
                                  'R_dc2bd',
                                  'R_dc2dc',
                                  'R_dc2fc',
    
                                  'R_fc2bu',
                                  'R_fc2bd',
                                  'R_fc2dc',
                                  'R_fc2fc'
                                  }
       
        # Define the transformation matrix from downward camera to body down frame
        self.R_dc2bd = np.array([
            [0.0, -1.0, 0.0, 0.0], # bd.x = -dc.y
            [1.0, 0.0, 0.0, 0.0],  # bd.y = dc.x
            [0.0, 0.0, 1.0, 0.0],  # bd.z = dc.z
            [0.0, 0.0, 0.0, 1.0]   # Fix: Last element should be 1.0, not 0.0
        ])
        
        # Define the identity matrices for same-frame transformations
        self.R_lenu2lenu = np.eye(4)
        self.R_lned2lned = np.eye(4) 
        self.R_bu2bu = np.eye(4)
        self.R_bd2bd = np.eye(4)
        self.R_dc2dc = np.eye(4)
        self.R_fc2fc = np.eye(4)
        
        # Define the transformation matrix from body down to downward camera frame
        self.R_bd2dc = self.R_dc2bd.T
    
    def static_transform(self, v__fin, fin, fout):
        """
        Given a vector expressed in frame fin, returns the same vector expressed in fout.
            
            Args:
                - v__fin: 3D vector, (x, y, z), represented in fin coordinates 
                - fin: string describing input coordinate frame 
                - fout: string describing output coordinate frame 
        
            Returns
                - v__fout: a vector, (x, y, z) represent in fout coordinates
        """
        # Check if fin is a valid coordinate frame
        if fin not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fin))

        # Check if fout is a valid coordinate frame
        if fout not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fout))
        
        # Check for a static transformation exists between the two frames
        R_str = 'R_{}2{}'.format(fin, fout)
        if R_str not in self.STATIC_TRANSFORMS:
            raise AttributeError('No static transform exists from {} to {}.'.format(fin, fout))
        
        # v4__'' are 4x1 np.array representation of the vector v__''
        # Create a 4x1 np.array representation of v__fin for matrix multiplication
        v4__fin = np.array([[v__fin[0]],
                            [v__fin[1]],
                            [v__fin[2]],
                            [     0.0]])

        # Get rotation matrix
        R_fin2fout = getattr(self, R_str)

        # Perform transformation from v__fin to v__fout
        v4__fout = np.dot(R_fin2fout, v4__fin)
        
        return (v4__fout[0,0], v4__fout[1,0], v4__fout[2,0])