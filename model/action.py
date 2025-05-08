from enum import Enum


class Action(Enum):
    MOVE_FORWARD = "MoveAhead"
    MOVE_BACK = "MoveBack"
    FACE_NORTH = "Teleport,rotation=180"
    FACE_SOUTH = "Teleport,rotation=0"
    FACE_EAST = "Teleport,rotation=90"
    FACE_WEST = "Teleport,rotation=270"
    PASS = "Pass"
