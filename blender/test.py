import bpy
import mathutils
import time
import numpy as np

class Bone:
    def __init__(self, arma, bone_name):
        self.bone = arma.pose.bones[bone_name]
        self.bone.rotation_mode = 'QUATERNION'
        self.parent_bone = self.bone.parent
    
    def world_to_local(self, world_coords):
        # Convert the world coordinates to the local space of the parent bone (if it exists)
        if self.parent_bone:
            try:
                parent_matrix_inv = self.parent_bone.matrix.inverted()
                local_coords = parent_matrix_inv @ world_coords
                return local_coords
            except Exception as e:
                print(f"Matrix inversion failed: {e}")
                return world_coords  # Fallback to world coords if transformation fails
        return world_coords  # No parent, return world coords as-is

    def rotate_to_target(self, target_vector_world):
        # Get the bone's current position (head) and orientation
        bone_head_world = self.bone.head
        
        # Calculate the direction vector from the bone's tail (head) to the target in world space
        target_vector_world = mathutils.Vector(target_vector_world)
        direction_vector_world = target_vector_world - bone_head_world
        
        # Convert the direction vector from world space to local space of the bone
        direction_vector_local = self.bone.matrix.inverted() @ direction_vector_world
        
        # The bone's forward direction is along the local y-axis (0, 1, 0)
        bone_forward_local = mathutils.Vector((0, 1, 0))
        
        # Normalize the direction vector
        if direction_vector_local.length > 0:
            direction_vector_local.normalize()
        else:
            print("Warning: target vector has zero length, cannot normalize")
            return

        # Calculate the quaternion rotation to align the forward vector with the target vector
        rotation_quat = bone_forward_local.rotation_difference(direction_vector_local)

        # Apply the rotation to the bone
        self.bone.rotation_quaternion = rotation_quat

        # Update the scene to apply changes
        bpy.context.view_layer.update()
        

if __name__ == '__main__':
    a = bpy.context.object
    b = Bone(a, 'Bone')
    b.bone.rotation_quaternion.identity()
    bpy.context.view_layer.update()
    time.sleep(0.5)
    target_vec = np.array([1, 1, 1])
    b.rotate_to_target(target_vec)
