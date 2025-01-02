from ai2thor.controller import Controller
from typing import Dict, Any, Optional, Callable
from enum import Enum
from pynput import keyboard
import threading
import time

class Action(Enum):
    MOVE_FORWARD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    FACE_NORTH = "Teleport,rotation=180"
    FACE_SOUTH = "Teleport,rotation=0"
    FACE_EAST = "Teleport,rotation=90"
    FACE_WEST = "Teleport,rotation=270"
    PASS = "Pass"

class ThorController:
    def __init__(self, scene: str = "FloorPlan_Train1_1", headless: bool = False):
        self.scene = scene
        self.headless = headless
        self.current_key = None
        self.is_running = False
        self.control_thread = None
        self.keyboard_listener = None
        self.custom_control_function = None
        self.key_mappings = {
            'shift': Action.MOVE_FORWARD,
            'ctrl': Action.MOVE_BACK,
            'up': Action.FACE_NORTH,
            'down': Action.FACE_SOUTH,
            'left': Action.FACE_WEST,
            'right': Action.FACE_EAST
        }
        self.init_controller()

    def init_controller(self):
        try:
            self.controller = Controller(
                agentMode="locobot",
                scene=self.scene,
                visibilityDistance=1.5,
                headless=self.headless,
                renderInstanceSegmentation=False,
                renderDepthImage=False,
                width=400,
                height=300,
                gridSize=0.25,
                rotateStepDegrees=90,
                quality="Very Low",
                createWindow=True
            )
            print(f"Controller initialized for scene: {self.scene}")
            first_event = self.controller.step(action="Pass")
            if not first_event.metadata['lastActionSuccess']:
                print(f"Initial action failed: {first_event.metadata['errorMessage']}")
        except Exception as e:
            print(f"Controller initialization error: {str(e)}")
            raise

    def _on_press(self, key):
        try:
            if key == keyboard.Key.shift_l:
                self.current_key = 'shift'
            elif key == keyboard.Key.ctrl_l:
                self.current_key = 'ctrl'
            elif key == keyboard.Key.up:
                self.current_key = 'up'
            elif key == keyboard.Key.down:
                self.current_key = 'down'
            elif key == keyboard.Key.left:
                self.current_key = 'left'
            elif key == keyboard.Key.right:
                self.current_key = 'right'
        except AttributeError:
            self.current_key = None

    def _on_release(self, key):
        self.current_key = None
        if key == keyboard.Key.esc:
            self.stop()
            return False

    def _handle_manual_controls(self):
        if not self.controller:
            return None
            
        if self.current_key in self.key_mappings:
            action = self.key_mappings[self.current_key]
            if action in [Action.FACE_NORTH, Action.FACE_SOUTH, Action.FACE_EAST, Action.FACE_WEST]:
                rotation = int(action.value.split('rotation=')[1])
                current_pos = self.controller.last_event.metadata['agent']['position']
                return self.controller.step(
                    action="Teleport",
                    position=current_pos,
                    rotation={"y": rotation}
                )
            return self.controller.step(action=action.value)
        return self.controller.step(action=Action.PASS.value)

    def _control_loop(self):
        while self.is_running and self.controller:
            try:
                if self.custom_control_function:
                    event = self.custom_control_function(self.controller)
                    if not event.metadata['lastActionSuccess']:
                        print(f"Action failed: {event.metadata['errorMessage']}")
                else:
                    event = self._handle_manual_controls()
                    if event and not event.metadata['lastActionSuccess']:
                        print(f"Manual action failed: {event.metadata['errorMessage']}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Control loop error: {str(e)}")
                time.sleep(1.0)  # Wait before retrying
                continue

    def start(self, control_function: Optional[Callable] = None):
        if self.is_running:
            return

        self.custom_control_function = control_function
        self.is_running = True
        
        if not control_function:
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self.keyboard_listener.start()
        
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def stop(self):
        if self.control_thread and self.control_thread.is_alive():
            self.is_running = False
            self.control_thread.join(timeout=1.0)
            
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            
        if self.controller:
            try:
                self.controller.stop()
            except Exception as e:
                print(f"Error stopping controller: {str(e)}")
            finally:
                self.controller = None

    def execute_action(self, action: Action) -> Dict[str, Any]:
        if not self.controller:
            return {}
        return self.controller.step(action=action.value)

    def get_current_state(self) -> Dict[str, Any]:
        if not self.controller:
            return {}
        return self.controller.last_event