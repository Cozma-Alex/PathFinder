import time

from thor_controller.thor_controller import Action, ThorController

def main():
    # Example 1: Manual Control
    print("Example 1: Manual Control")
    controller = ThorController(headless=False)
    print("Use WASD to move, ESC to quit")
    controller.start()  # This will use keyboard controls
    
    # Keep main thread alive
    try:
        while controller.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        controller.stop()

def automatic_control_example():
    # Example 2: Automatic Control
    print("\nExample 2: Automatic Control")
    controller = ThorController(headless=False)
    
    # Define a custom control function
    def auto_control(thor_controller):
        # Example: Move in a square pattern
        actions = [
            Action.MOVE_FORWARD,
            Action.ROTATE_RIGHT,
            Action.MOVE_FORWARD,
            Action.ROTATE_RIGHT,
            Action.MOVE_FORWARD,
            Action.ROTATE_RIGHT,
            Action.MOVE_FORWARD,
        ]
        
        for action in actions:
            event = thor_controller.step(action=action.value)
            time.sleep(1)
        
        return event

    # Start with automatic control
    controller.start(control_function=auto_control)
    
    # Wait for pattern to complete
    time.sleep(10)
    controller.stop()

def direct_control_example():
    # Example 3: Direct Control
    print("\nExample 3: Direct Control")
    controller = ThorController(headless=False)
    
    # Execute actions directly
    controller.execute_action(Action.MOVE_FORWARD)
    time.sleep(1)
    controller.execute_action(Action.ROTATE_RIGHT)
    time.sleep(1)
    controller.execute_action(Action.MOVE_FORWARD)
    
    controller.stop()

if __name__ == "__main__":
    # main()
    # Uncomment to run other examples:
    automatic_control_example()
    # direct_control_example()