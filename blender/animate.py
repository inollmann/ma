import bpy
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion
import pickle
import json
import os
import numpy as np
import subprocess


def copy2clip(txt):
    cmd = f'echo|set /p={txt.strip()}|clip'
    return subprocess.check_call(cmd, shell=True)


class Bone:
    def __init__(self, arma, bone_name):
        self.armature = arma
        self.bone = arma.pose.bones[bone_name]
        self.bone.rotation_mode = 'QUATERNION'
        self.parent_bone = self.bone.parent


    def world_to_local(self, world_coords):
        """
        Convert world coordinates to local coordinates relative to this bone.
        :param world_coords: A Vector (or tuple/list of coordinates) in world space.
        :return: A Vector in local space relative to this bone.
        """
        # Use matrix_basis combined with parent matrix inverse to get the correct world matrix
        bone_matrix_world = self.bone.matrix_basis @ self.bone.matrix_parent_inverse
        bone_matrix_world_inverted = bone_matrix_world.inverted()  # Invert the world matrix to go from world to local
        
        # Convert the world coordinates to a Vector if needed
        if not isinstance(world_coords, mathutils.Vector):
            world_coords = Vector(world_coords)
        
        # Transform world coordinates to local space
        local_coords = bone_matrix_world_inverted @ world_coords
        
        return local_coords


class Armature:
    
    def __init__(self):
        self.armature = bpy.data.objects['RIG-Snow']
        self.bones = {
             0: {'bone': Bone(self.armature, 'FK-Neck'),                   'in_array': ['pose',        0]}, 
             1: {'bone': Bone(self.armature, 'FK-Shoulder.L'),             'in_array': ['pose',        5]}, 
             2: {'bone': Bone(self.armature, 'FK-Shoulder.R'),             'in_array': ['pose',        2]}, 
             3: {'bone': Bone(self.armature, 'FK-UpperArm.L'),             'in_array': ['pose',        6]}, 
             4: {'bone': Bone(self.armature, 'FK-UpperArm.R'),             'in_array': ['pose',        3]}, 
             5: {'bone': Bone(self.armature, 'FK-Forearm.L'),              'in_array': ['pose',        7]}, 
             6: {'bone': Bone(self.armature, 'FK-Forearm.R'),              'in_array': ['pose',        4]},
             7: {'bone': Bone(self.armature, 'FK-Finger_Index_Carpal.L'),  'in_array': ['hand_left',   5]}, 
             8: {'bone': Bone(self.armature, 'FK-Finger_Index_Carpal.R'),  'in_array': ['hand_right',  5]}, 
             9: {'bone': Bone(self.armature, 'FK-Finger_Middle_Carpal.L'), 'in_array': ['hand_left',   9]},
            10: {'bone': Bone(self.armature, 'FK-Finger_Middle_Carpal.R'), 'in_array': ['hand_right',  9]},
            11: {'bone': Bone(self.armature, 'FK-Finger_Ring_Carpal.L'),   'in_array': ['hand_left',  13]}, 
            12: {'bone': Bone(self.armature, 'FK-Finger_Ring_Carpal.R'),   'in_array': ['hand_right', 13]}, 
            13: {'bone': Bone(self.armature, 'FK-Finger_Pinky_Carpal.L'),  'in_array': ['hand_left',  17]}, 
            14: {'bone': Bone(self.armature, 'FK-Finger_Pinky_Carpal.R'),  'in_array': ['hand_right', 17]}, 
            15: {'bone': Bone(self.armature, 'FK-Finger_Thumb1.L'),        'in_array': ['hand_left',   2]}, 
            16: {'bone': Bone(self.armature, 'FK-Finger_Thumb1.R'),        'in_array': ['hand_right',  2]}, 
            17: {'bone': Bone(self.armature, 'FK-Finger_Index1.L'),        'in_array': ['hand_left',   6]}, 
            18: {'bone': Bone(self.armature, 'FK-Finger_Index1.R'),        'in_array': ['hand_right',  6]}, 
            19: {'bone': Bone(self.armature, 'FK-Finger_Middle1.L'),       'in_array': ['hand_left',  10]}, 
            20: {'bone': Bone(self.armature, 'FK-Finger_Middle1.R'),       'in_array': ['hand_right', 10]}, 
            21: {'bone': Bone(self.armature, 'FK-Finger_Ring1.L'),         'in_array': ['hand_left',  14]}, 
            22: {'bone': Bone(self.armature, 'FK-Finger_Ring1.R'),         'in_array': ['hand_right', 14]}, 
            23: {'bone': Bone(self.armature, 'FK-Finger_Pinky1.L'),        'in_array': ['hand_left',  18]}, 
            24: {'bone': Bone(self.armature, 'FK-Finger_Pinky1.R'),        'in_array': ['hand_right', 18]}, 
            25: {'bone': Bone(self.armature, 'FK-Finger_Thumb2.L'),        'in_array': ['hand_left',   3]}, 
            26: {'bone': Bone(self.armature, 'FK-Finger_Thumb2.R'),        'in_array': ['hand_right',  3]}, 
            27: {'bone': Bone(self.armature, 'FK-Finger_Index2.L'),        'in_array': ['hand_left',   7]}, 
            28: {'bone': Bone(self.armature, 'FK-Finger_Index2.R'),        'in_array': ['hand_right',  7]}, 
            29: {'bone': Bone(self.armature, 'FK-Finger_Middle2.L'),       'in_array': ['hand_left',  11]}, 
            30: {'bone': Bone(self.armature, 'FK-Finger_Middle2.R'),       'in_array': ['hand_right', 11]}, 
            31: {'bone': Bone(self.armature, 'FK-Finger_Ring2.L'),         'in_array': ['hand_left',  15]}, 
            32: {'bone': Bone(self.armature, 'FK-Finger_Ring2.R'),         'in_array': ['hand_right', 15]}, 
            33: {'bone': Bone(self.armature, 'FK-Finger_Pinky2.L'),        'in_array': ['hand_left',  19]}, 
            34: {'bone': Bone(self.armature, 'FK-Finger_Pinky2.R'),        'in_array': ['hand_right', 19]}, 
            35: {'bone': Bone(self.armature, 'FK-Finger_Thumb3.L'),        'in_array': ['hand_left',   4]}, 
            36: {'bone': Bone(self.armature, 'FK-Finger_Thumb3.R'),        'in_array': ['hand_right',  4]}, 
            37: {'bone': Bone(self.armature, 'FK-Finger_Index3.L'),        'in_array': ['hand_left',   8]}, 
            38: {'bone': Bone(self.armature, 'FK-Finger_Index3.R'),        'in_array': ['hand_right',  8]}, 
            39: {'bone': Bone(self.armature, 'FK-Finger_Middle3.L'),       'in_array': ['hand_left',  12]}, 
            40: {'bone': Bone(self.armature, 'FK-Finger_Middle3.R'),       'in_array': ['hand_right', 12]}, 
            41: {'bone': Bone(self.armature, 'FK-Finger_Ring3.L'),         'in_array': ['hand_left',  16]}, 
            42: {'bone': Bone(self.armature, 'FK-Finger_Ring3.R'),         'in_array': ['hand_right', 16]}, 
            43: {'bone': Bone(self.armature, 'FK-Finger_Pinky3.L'),        'in_array': ['hand_left',  20]}, 
            44: {'bone': Bone(self.armature, 'FK-Finger_Pinky3.R'),        'in_array': ['hand_right', 20]}}    
    
    
    def control_wrists(self, gest):
        keyframe = 0
        #keyframe_interval = 1
        dbones = bpy.data.armatures["Data_RIG-Snow.003"].bones
        
        wrist_l = bpy.data.objects["RIG-Snow"].pose.bones["IK-MSTR-Wrist.L"]
        wrist_r = bpy.data.objects["RIG-Snow"].pose.bones["IK-MSTR-Wrist.R"]
        
        wrist_l.rotation_mode = 'QUATERNION'
        wrist_r.rotation_mode = 'QUATERNION'
        
        wrist_l.keyframe_insert(data_path="location", frame=keyframe)
        wrist_r.keyframe_insert(data_path="location", frame=keyframe)
        wrist_l.keyframe_insert(data_path="rotation_quaternion", frame=keyframe)
        wrist_r.keyframe_insert(data_path="rotation_quaternion", frame=keyframe)
        bpy.context.view_layer.update()
        
        btwn_shoulders = (dbones["FK-UpperArm.L"].head_local + dbones["FK-UpperArm.R"].head_local) / 2
        wto_l = btwn_shoulders - dbones["IK-MSTR-Wrist.L"].head_local
        wto_r = btwn_shoulders - dbones["IK-MSTR-Wrist.R"].head_local
        #print("to origin l:", wto_l, "to origin r:", wto_r)
        stretch_x, stretch_y, stretch_z = 0.22, 0.35, 0.45
        #print("stretch:", stretch_x, stretch_y, stretch_z)
        
        animation_length = 35
        
        #gest[5]['coords'], gest[6]['coords'] = [], []
        #for i in range(13):
        #    gest[5]['coords'].append([math.sin(math.pi/6*i) * 0.3 + 0.5, -math.sin(math.pi/3*i) * 0.25 - 0.5, math.cos(math.pi/6*i)])
        #    gest[6]['coords'].append([-math.sin(math.pi/6*i) * 0.3 - 0.5, -math.sin(math.pi/3*i) * 0.25 - 0.5, math.cos(math.pi/6*i)])
            
        num_frames = len(gest[5]['coords'])
        keyframe_interval = 35 / num_frames
            
        for wl, wr, mkl, mkr, ikl, ikr, pkl, pkr in zip(
                    gest[5]['coords'], gest[6]['coords'], 
                    gest[19]['coords'], gest[20]['coords'], 
                    gest[7]['coords'], gest[8]['coords'], 
                    gest[13]['coords'], gest[14]['coords']):
            keyframe += keyframe_interval
            #if keyframe % (keyframe_interval * 1) != 0:
                #continue
            #print("l original:", l)
            #print("r original:", r)
            target_y_l, target_y_r = Vector(mkl) - Vector(wl), Vector(mkr) - Vector(wr)
            target_z_l, target_z_r = Vector(pkl) - Vector(ikl), Vector(pkr) - Vector(ikr)
            
            wl = Vector(( wl[2] * stretch_z,  wl[0] * stretch_x, wl[1] * stretch_y)) + Vector(( wto_l[2],  wto_l[0], wto_l[1] - 0.3))
            wr = Vector((-wr[2] * stretch_z, -wr[0] * stretch_x, wr[1] * stretch_y)) + Vector((-wto_r[2], -wto_r[0], wto_r[1] - 0.3))
            #print("l transformed:", l)
            #print("r transformed:", r)
            
            wrist_l.location = wl
            wrist_r.location = wr
            
            wrist_l.keyframe_insert(data_path="location", frame=int(keyframe))
            wrist_r.keyframe_insert(data_path="location", frame=int(keyframe))
            
            # target_y_l = dbones["IK-MSTR-Wrist.L"].matrix_local.to_3x3().transposed() @ target_y_l
            target_y_l = Vector(( target_y_l[2],  target_y_l[0], target_y_l[1])).normalized()
            target_y_r = Vector((-target_y_r[2], -target_y_r[0], target_y_r[1])).normalized()
            target_z_l = Vector(( target_z_l[2],  target_z_l[0], target_z_l[1])).normalized()
            target_z_r = Vector((-target_z_r[2], -target_z_r[0], target_z_r[1])).normalized()

            target_x_l = target_y_l.cross(target_z_l).normalized()
            target_x_r = target_y_r.cross(target_z_r).normalized()
            target_z_l = target_x_l.cross(target_y_l).normalized()
            target_z_r = target_x_r.cross(target_y_r).normalized()
            
            rotmat_l = Matrix((target_x_l, target_y_l, target_z_l)).transposed()
            rotmat_r = Matrix((target_x_r, target_y_r, target_z_r)).transposed()
            q1_l = rotmat_l.to_quaternion()
            q1_r = rotmat_r.to_quaternion()
            
            wrist_l.rotation_quaternion = q1_l
            wrist_r.rotation_quaternion = q1_r
            
            wrist_l.keyframe_insert(data_path="rotation_quaternion", frame=int(keyframe))
            wrist_r.keyframe_insert(data_path="rotation_quaternion", frame=int(keyframe))
    
    
    def control_fingers(self, gest):
        keyframe = 0
        animation_length = 35
        num_frames = len(gest[0]['coords'])
        keyframe_interval = 35 / num_frames
        
        for frame in range(num_frames):
            keyframe += keyframe_interval
            
            # Fingers
            for idx in range(17, 45):
                bone = gest[idx]
                name = bone['bone']
                dbone = bpy.data.armatures["Data_RIG-Snow.003"].bones[name]
                pbone = bpy.data.objects["RIG-Snow"].pose.bones[name]
                
                pbone.rotation_mode = 'AXIS_ANGLE'
                if frame == 0:
                    pbone.keyframe_insert(data_path="rotation_axis_angle", frame=0)
                
                coords = Vector(bone['coords'][frame])
                parent_bone = gest[bone['parent']]
                parent_coords = Vector(parent_bone['coords'][frame])
                grandparent_coords = Vector(gest[parent_bone['parent']]['coords'][frame])
                
                v0 = parent_coords - grandparent_coords
                v1 = coords - parent_coords
                
                angle = v0.angle(v1)
                rotax = dbone.x_axis
                pbone.rotation_mode = 'AXIS_ANGLE'
                pbone.rotation_axis_angle = (angle, *rotax)
                pbone.keyframe_insert(data_path="rotation_axis_angle", frame=int(keyframe))
            
            # Thumb L
            for idx, name, r in zip([15, 16], ["FK-Finger_Thumb1.L", "FK-Finger_Thumb1.R"], [(1, 1, 0.3), (-1, 1, 0.3)]):
                bone = gest[idx]
                dbone = bpy.data.armatures["Data_RIG-Snow.003"].bones[name]
                pbone = bpy.data.objects["RIG-Snow"].pose.bones[name]
                pbone.rotation_mode = 'AXIS_ANGLE'
                if frame == 0:
                    pbone.keyframe_insert(data_path="rotation_axis_angle", frame=0)
                coords = Vector(bone['coords'][frame])
                parent_coords = Vector(gest[bone['parent']]['coords'][frame])
                rotax = Vector(r).normalized()
                v = dbone.matrix_local.to_3x3().transposed() @ (coords - parent_coords)
                #v = v * Vector((1, 1, 1))
                y = dbone.y_axis
                v_proj = v - rotax * v.dot(rotax)
                y_proj = y - rotax * y.dot(rotax)
                v_proj.normalize()
                y_proj.normalize()
                angle = y_proj.angle(v_proj)
                angle = max(min(angle, math.radians(50)), math.radians(-40))
                if idx == 16:
                    angle = angle
                pbone.rotation_mode = 'AXIS_ANGLE'
                pbone.rotation_axis_angle = (angle, *rotax)
                pbone.keyframe_insert(data_path="rotation_axis_angle", frame=int(keyframe))
            
    
    def move(self, gest):
        self.control_wrists(gest)
        self.control_fingers(gest)

            
    def reset_pose(self):
        bpy.ops.object.mode_set(mode='POSE')
        for bone_data in self.bones.values():
            bone_data['bone'].bone.matrix_basis.identity()
        bpy.data.objects["RIG-Snow"].pose.bones["IK-MSTR-Wrist.L"].location = (0, 0, 0)
        bpy.data.objects["RIG-Snow"].pose.bones["IK-MSTR-Wrist.R"].location = (0, 0, 0)
        bpy.data.objects["RIG-Snow"].pose.bones["IK-MSTR-Wrist.L"].rotation_quaternion = (1, 0, 0, 0)
        bpy.data.objects["RIG-Snow"].pose.bones["IK-MSTR-Wrist.R"].rotation_quaternion = (1, 0, 0, 0)
        bpy.context.view_layer.update()
    
    
    def clear_all_keyframes(self):
        if self.armature.animation_data:  # Check if the armature has any animation data
            action = self.armature.animation_data.action
            if action:  # Check if there's an action (which holds keyframes)
                # Remove all fcurves (which store the keyframes) from the action
                self.armature.animation_data_clear()


if __name__ == '__main__':
    reset = 0
    
    arma = Armature()
    if reset:
        arma.reset_pose()
    else:
        arma.reset_pose()
        arma.clear_all_keyframes()
        print('Loading...')
        dirname = os.path.dirname(__file__)
        dirname = dirname[:-17]
        print(dirname)
        with open(os.path.join(dirname, 'vocab\\allein.json'), 'r') as f:
            j = json.load(f)
        gest = j['coords']
        rid = j['id']
        print(rid)
        copy2clip(rid)
        #print(gest[7])
        arma.move(gest)

            
    
#hand_left_e2b = {
#    2: 'FK-Finger_Thumb1.L',
#    3: 'FK-Finger_Thumb2.L',
#    4: 'FK-Finger_Thumb3.L',
#    5: 'FK-Finger_Index_Carpal.L',
#    6: 'FK-Finger_Index1.L',
#    7: 'FK-Finger_Index2.L',
#    8: 'FK-Finger_Index3.L',
#    9: 'FK-Finger_Middle_Carpal.L',
#    10: 'FK-Finger_Middle1.L',
#    11: 'FK-Finger_Middle2.L',
#    12: 'FK-Finger_Middle3.L',
#    13: 'FK-Finger_Ring_Carpal.L',
#    14: 'FK-Finger_Ring1.L',
#    15: 'FK-Finger_Ring2.L',
#    16: 'FK-Finger_Ring3.L',
#    17: 'FK-Finger_Pinky_Carpal.L',
#    18: 'FK-Finger_Pinky1.L',
#    19: 'FK-Finger_Pinky2.L',
#    20: 'FK-Finger_Pinky3.L',
#}

#hand_right_e2b = {
#    2: 'FK-Finger_Thumb1.R',
#    3: 'FK-Finger_Thumb2.R',
#    4: 'FK-Finger_Thumb3.R',
#    5: 'FK-Finger_Index_Carpal.R',
#    6: 'FK-Finger_Index1.R',
#    7: 'FK-Finger_Index2.R',
#    8: 'FK-Finger_Index3.R',
#    9: 'FK-Finger_Middle_Carpal.R',
#    10: 'FK-Finger_Middle1.R',
#    11: 'FK-Finger_Middle2.R',
#    12: 'FK-Finger_Middle3.R',
#    13: 'FK-Finger_Ring_Carpal.R',
#    14: 'FK-Finger_Ring1.R',
#    15: 'FK-Finger_Ring2.R',
#    16: 'FK-Finger_Ring3.R',
#    17: 'FK-Finger_Pinky_Carpal.R',
#    18: 'FK-Finger_Pinky1.R',
#    19: 'FK-Finger_Pinky2.R',
#    20: 'FK-Finger_Pinky3.R',
#}

#        

#pose_e2b = {
#    0: 'FK-Neck',
#    2: 'FK-Shoulder.R',
#    3: 'FK-UpperArm.R',
#    4: 'FK-Forearm.R',
#    5: 'FK-Shoulder.L',
#    6: 'FK-UpperArm.L',
#    7: 'FK-Forearm.L'
#}

#endpoint2bone = {
#    'pose': pose_e2b, 
#    'face': None, 
#    'hand_left': hand_left_e2b, 
#    'hand_right': hand_right_e2b
#}

#armature = bpy.context.object
#example_bone = armature.pose.bones['FK-Neck']
#example_bone.rotation_mode = 'QUATERNION'
#example_bone.rotation_quaternion = (0.7071, 0.0, 0.7071, 0.0)
#bpy.context.view_layer.update()


