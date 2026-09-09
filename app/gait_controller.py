class GaitController:

    def __init__(self):
        self.current_command = "stop"

    def set_command(self, command):
        allowed = ["forward", "backward", "left", "right", "stop"]

        if command not in allowed:
            raise ValueError("Invalid command")

        self.current_command = command
        print("GAIT:", command)

    def stop(self):
        self.current_command = "stop"
        print("GAIT: stop")


if __name__ == "__main__":
    gait = GaitController()

    gait.set_command("forward")
    gait.set_command("left")
    gait.set_command("right")
    gait.set_command("backward")
    gait.stop()

    print("GAIT CONTROLLER OK")
