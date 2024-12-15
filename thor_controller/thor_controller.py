# thor_controller.py
from ai2thor.controller import Controller
from pynput import keyboard
import threading
import time
from typing import Optional, Callable, Dict, Any
from enum import Enum

class Action(Enum):
    MOVE_FORWARD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    PASS = "Pass"

class ThorController:
    def __init__(self, scene: str = "FloorPlan_Train1_1", headless: bool = False):
        self.controller = Controller(
            agentMode="locobot",
            scene=scene,
            headless=headless,
        )
        self.current_key = None
        self.is_running = False
        self.control_thread = None
        self.keyboard_listener = None
        self.custom_control_function = None
        
        self.key_mappings = {
            'w': Action.MOVE_FORWARD,
            's': Action.MOVE_BACK,
            'a': Action.ROTATE_LEFT,
            'd': Action.ROTATE_RIGHT
        }

    def _on_press(self, key):
        try:
            self.current_key = key.char
        except AttributeError:
            self.current_key = None

    def _on_release(self, key):
        self.current_key = None
        if key == keyboard.Key.esc:
            self.stop()
            return False

    def _handle_manual_controls(self) -> Optional[Dict[str, Any]]:
        if self.current_key in self.key_mappings:
            action = self.key_mappings[self.current_key]
            return self.controller.step(action=action.value)
        return self.controller.step(action=Action.PASS.value)

    def _control_loop(self):
        while self.is_running:
            if self.custom_control_function:
                event = self.custom_control_function(self.controller)
            else:
                event = self._handle_manual_controls()
            time.sleep(0.1)

    def start(self, control_function: Optional[Callable] = None):
        """
        Start the controller with either manual or automatic controls.
        
        Args:
            control_function: Optional function for automatic control.
                            If None, uses manual keyboard controls.
                            Function should take a controller instance and return an event.
        """
        if self.is_running:
            return

        self.is_running = True
        self.custom_control_function = control_function

        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()

        if not control_function:
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self.keyboard_listener.start()

    def stop(self):
        """Stop the controller and clean up resources."""
        self.is_running = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            
        if self.control_thread:
            self.control_thread.join()
            
        self.controller.stop()

    def execute_action(self, action: Action) -> Dict[str, Any]:
        """
        Execute a single action directly.
        
        Args:
            action: Action enum member to execute
            
        Returns:
            Event dictionary from the controller
        """
        return self.controller.step(action=action.value)

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        return self.controller.last_event