from ai2thor.controller import Controller
from typing import Dict, Any, Optional, Callable
from enum import Enum
from pynput import keyboard
import threading
import time
import logging

logger = logging.getLogger('pathfinder')

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
        self.controller = None
        self.controller_lock = threading.Lock()
        self.key_mappings = {
            'shift': Action.MOVE_FORWARD,
            'ctrl': Action.MOVE_BACK,
            'up': Action.FACE_NORTH,
            'down': Action.FACE_SOUTH,
            'left': Action.FACE_WEST,
            'right': Action.FACE_EAST
        }
        
    def init_controller(self):
        try:
            logger.info(f"Initializing Thor controller for scene: {self.scene}")
            with self.controller_lock:
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
            logger.info(f"Controller initialized for scene: {self.scene}")
            
            # Test with a simple Pass action
            first_event = self.controller.step(action="Pass")
            logger.info(f"Initialize return: {first_event.metadata}")
            if not first_event.metadata.get('lastActionSuccess', True):
                logger.warning(f"Initial action failed: {first_event.metadata.get('errorMessage', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Controller initialization error: {str(e)}")
            self.controller = None
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
        if not self.is_running:
            return None
            
        with self.controller_lock:
            if not self.controller:
                return None
                
        if self.current_key in self.key_mappings:
            action = self.key_mappings[self.current_key]
            if action in [Action.FACE_NORTH, Action.FACE_SOUTH, Action.FACE_EAST, Action.FACE_WEST]:
                rotation = int(action.value.split('rotation=')[1])
                try:
                    with self.controller_lock:
                        if not self.controller:
                            return None
                        current_pos = self.controller.last_event.metadata['agent']['position']
                        return self.controller.step(
                            action="Teleport",
                            position=current_pos,
                            rotation={"y": rotation}
                        )
                except Exception as e:
                    logger.error(f"Error during rotation: {str(e)}")
                    return None
            try:
                with self.controller_lock:
                    if not self.controller:
                        return None
                    return self.controller.step(action=action.value)
            except Exception as e:
                logger.error(f"Error executing action {action.value}: {str(e)}")
                return None
        
        try:
            with self.controller_lock:
                if not self.controller:
                    return None
                return self.controller.step(action=Action.PASS.value)
        except Exception as e:
            logger.error(f"Error executing Pass action: {str(e)}")
            return None

    def _control_loop(self):
        retry_count = 0
        max_retries = 3
        
        while self.is_running:
            try:
                if not self.is_running:
                    break
                    
                if self.custom_control_function:
                    try:
                        with self.controller_lock:
                            if not self.controller:
                                break
                                
                            event = self.custom_control_function(self.controller)
                            if event and not event.metadata['lastActionSuccess']:
                                logger.warning(f"Action failed: {event.metadata.get('errorMessage', 'Unknown error')}")
                                
                    except Exception as e:
                        logger.error(f"Custom control function error: {str(e)}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Max retries reached, stopping controller")
                            self.stop()
                            break
                else:
                    event = self._handle_manual_controls()
                    if event and not event.metadata.get('lastActionSuccess', True):
                        logger.warning(f"Manual action failed: {event.metadata.get('errorMessage', 'Unknown error')}")
                        
                # Reset retry count on successful execution
                retry_count = 0
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Control loop error: {str(e)}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error(f"Max retries reached, stopping controller")
                    self.stop()
                    break
                    
                time.sleep(1.0)

    def start(self, control_function: Optional[Callable] = None):
        if self.is_running:
            logger.warning("Controller already running")
            return
            
        if not self.controller:
            try:
                self.init_controller()
            except Exception as e:
                logger.error(f"Failed to initialize controller: {str(e)}")
                return
            
        if not self.controller:
            logger.error("Failed to initialize controller")
            return

        self.custom_control_function = control_function
        self.is_running = True
        
        if not control_function:
            try:
                self.keyboard_listener = keyboard.Listener(
                    on_press=self._on_press,
                    on_release=self._on_release
                )
                self.keyboard_listener.start()
            except Exception as e:
                logger.error(f"Error starting keyboard listener: {str(e)}")
        
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        logger.info(f"Thor controller started for scene: {self.scene}")

    def stop(self):
        logger.info("Stopping Thor controller")
        self.is_running = False
        
        if self.control_thread and self.control_thread.is_alive():
            try:
                self.control_thread.join(timeout=1.0)
            except Exception as e:
                logger.error(f"Error stopping control thread: {str(e)}")
            
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except Exception as e:
                logger.error(f"Error stopping keyboard listener: {str(e)}")
            
        with self.controller_lock:
            if self.controller:
                try:
                    self.controller.stop()
                    logger.info("Controller stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping controller: {str(e)}")
                finally:
                    self.controller = None

    def execute_action(self, action: Action) -> Dict[str, Any]:
        with self.controller_lock:
            if not self.controller or not self.is_running:
                logger.warning("Cannot execute action: controller not initialized or not running")
                return {}
                
            try:
                return self.controller.step(action=action.value)
            except Exception as e:
                logger.error(f"Error executing action {action.value}: {str(e)}")
                return {}

    def get_current_state(self) -> Dict[str, Any]:
        with self.controller_lock:
            if not self.controller or not self.is_running:
                logger.warning("Cannot get current state: controller not initialized or not running")
                return {}
                
            try:
                return self.controller.last_event
            except Exception as e:
                logger.error(f"Error getting current state: {str(e)}")
                return {}