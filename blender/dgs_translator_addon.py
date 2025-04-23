import bpy
import os
import subprocess
from collections import deque
import blf
import bgl

bl_info = {
    "name": "DGS Translator Add-on",
    "blender": (3, 0, 0),
    "category": "3D View",
}

watcher = None
action_queue = deque()
recording_process = None
subtitles_text = ""
draw_handler = None


class FileWatcher:
    def __init__(self, filepath):
        self.filepath = filepath
        self.last_modified_time = os.path.getmtime(filepath)

    def has_changed(self):
        current_modified_time = os.path.getmtime(self.filepath)
        if current_modified_time != self.last_modified_time:
            self.last_modified_time = current_modified_time
            return True
        return False


def main():
    global watcher
    playback_speed = 1
    
    if watcher is None:
        dir = os.path.split(bpy.context.scene.recording_script_path)[0]
        watcher = FileWatcher(os.path.join(dir, "blender/NNoutput.txt"))

    # Check for file change
    if watcher.has_changed():
        add_actions_to_queue(playback_speed)

    return 1.0  # Run every second


def add_actions_to_queue(speed_factor):
    out = read_nnout()
    if not out:
        return
    action_names = list(filter(None, out))
    action_queue.append((action_names, speed_factor))  # Add the new sequence to the queue
    
    # Play action if no other is playing
    if not bpy.app.handlers.frame_change_pre:
        play_next_action()


def play_next_action():
    if action_queue:
        action_names, speed_factor = action_queue.popleft()  # Get the next sequence in the queue
        play_actions(action_names, speed_factor)
    else:
        bpy.app.handlers.frame_change_pre.clear()


def play_actions(action_names, speed_factor):
    play_actions.remaining_actions = deque(action_names)
    play_actions.speed_factor = speed_factor
    play_single_action()  # Start the first action in sequence


def play_single_action():
    if play_actions.remaining_actions:
        action_name = play_actions.remaining_actions.popleft()
        play(action_name, play_actions.speed_factor)
    else:
        play_next_action()


def play(action_name, speed_factor=1):
    print(f"Playing: {action_name}")
    action = bpy.data.actions.get(action_name)
    if action:
        speed_up(action, speed_factor)
        bpy.data.objects['RIG-Snow'].animation_data.action = action
        bpy.context.scene.frame_current = bpy.context.scene.frame_start
        bpy.ops.screen.animation_play()

        # Sequence end detection handler
        def stop_playback(scene):
            if scene.frame_current == scene.frame_end:
                bpy.ops.screen.animation_play()  # Stop playback
                play_single_action()  # Next action

        # Clear existing handlers and attach a new one for this sequence
        bpy.app.handlers.frame_change_pre.clear()
        bpy.app.handlers.frame_change_pre.append(stop_playback)
    else:
        print(f"Action '{action_name}' not found!")


def speed_up(action, factor):
    interval = 2 / factor
    for fcurve in action.fcurves:
        for i, keyframe in enumerate(fcurve.keyframe_points):
            keyframe.co.x = i * interval
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = int((i + 1) * interval)


def read_nnout():
    dir = os.path.split(bpy.context.scene.recording_script_path)[0]
    nnout_file = os.path.join(dir, "blender/NNoutput.txt")
    with open(nnout_file, "r", encoding="utf-8") as f:
        contents = f.read().splitlines()
        draw_subtitles(contents[0])  # Update subtitles
        if len(contents) > 1:
            return contents[1:]
        else:
            return ["Snow_mouth_default"]


def draw_subtitles(text):
    global subtitles_text, draw_handler
    subtitles_text = text

    # remove existing draw handler
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')

    # register new draw handler
    draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (None, None), 'WINDOW', 'POST_PIXEL')


def draw_callback(self, context):
    font_id = 0
    blf.position(font_id, 80, 40, 0)
    blf.size(font_id, 30)
    blf.color(font_id, 1, 0, 0, 1)
    blf.draw(font_id, subtitles_text)


def toggle_recording():
    global recording_process
    if recording_process is None:
        # Start recording
        venv_python = bpy.context.scene.recording_venv_python
        script_path = bpy.context.scene.recording_script_path
        mic_idx = bpy.context.scene.mic_id
        recording_process = subprocess.Popen([venv_python, script_path, "--mic_idx", mic_idx])
        print("Loading...")
    else:
        # Stop recording
        recording_process.terminate()
        recording_process = None
        remove_subtitles()
        print("Recording stopped.")
        

def remove_subtitles():
    global draw_handler
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
        draw_handler = None 


class SettingsPanel(bpy.types.Panel):
    bl_label = "Settings"
    bl_idname = "VIEW3D_PT_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'DGS'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "recording_venv_python")
        layout.prop(scene, "recording_script_path")


class DgsTranslatorPanel(bpy.types.Panel):
    bl_label = "DGS Translator"
    bl_idname = "VIEW3D_PT_dgs_translator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'DGS'

    def draw(self, context):
        layout = self.layout
        
        layout.prop(context.scene, "mic_id")
        
        row = layout.row()
        row.operator("recording.start_main", text="Start Translator")
        
        row = layout.row()
        if recording_process is None:
            row.operator("recording.toggle", text="Start Recording")
        else:
            row.operator("recording.toggle", text="Stop Recording")


class StartMainProcessOperator(bpy.types.Operator):
    bl_idname = "recording.start_main"
    bl_label = "Start Translator"

    def execute(self, context):
        bpy.app.timers.register(main, first_interval=1.0)
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        print("Translator active.")
        return {'FINISHED'}


class ToggleRecordingOperator(bpy.types.Operator):
    bl_idname = "recording.toggle"
    bl_label = "Start Recording"

    def execute(self, context):
        toggle_recording()
        return {'FINISHED'}


def register():
    bpy.utils.register_class(SettingsPanel)
    bpy.utils.register_class(DgsTranslatorPanel)
    bpy.utils.register_class(StartMainProcessOperator)
    bpy.utils.register_class(ToggleRecordingOperator)
    
    bpy.types.Scene.recording_venv_python = bpy.props.StringProperty(
        name="Python Executable",
        default=r"C:\path\to\your\venv\Scripts\python.exe")
    
    bpy.types.Scene.recording_script_path = bpy.props.StringProperty(
        name="Script Path",
        default=r"D:\path\to\your\script.py")
    
    bpy.types.Scene.mic_id = bpy.props.StringProperty(
        name="Use Microphone",
        default="1")


def unregister():
    bpy.utils.unregister_class(SettingsPanel)
    bpy.utils.unregister_class(DgsTranslatorPanel)
    bpy.utils.unregister_class(StartMainProcessOperator)
    bpy.utils.unregister_class(ToggleRecordingOperator)
    
    del bpy.types.Scene.recording_venv_python
    del bpy.types.Scene.recording_script_path
    del bpy.types.Scene.mic_id


if __name__ == "__main__":
    register()



