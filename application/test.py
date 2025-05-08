import time
from thor_controller.thor_controller import Action, ThorController


def main():
    print("Example 1: Manual Control")
    controller = ThorController(headless=False)
    print("Use WASD to move, ESC to quit")
    controller.start()

    try:
        while controller.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        controller.stop()


def automatic_control_example():
    print("\nExample 2: Automatic Control")
    controller = ThorController(headless=False)

    def auto_control(thor_controller):
        actions = [
            Action.MOVE_FORWARD,
            Action.FACE_EAST,
            Action.MOVE_FORWARD,
            Action.FACE_SOUTH,
            Action.MOVE_FORWARD,
            Action.FACE_WEST,
            Action.MOVE_FORWARD,
        ]

        for action in actions:
            event = thor_controller.step(action=action.value)
            time.sleep(1)

        return event

    controller.start(control_function=auto_control)

    time.sleep(10)
    controller.stop()


def direct_control_example():
    print("\nExample 3: Direct Control")
    controller = ThorController(headless=False)

    controller.execute_action(Action.MOVE_FORWARD)
    time.sleep(1)
    controller.execute_action(Action.FACE_EAST)
    time.sleep(1)
    controller.execute_action(Action.MOVE_FORWARD)

    controller.stop()


if __name__ == "__main__":
    automatic_control_example()
