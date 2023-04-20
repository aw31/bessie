import signal


class Timeout:
    """Timeout class using SIGALRM"""

    class Timeout(Exception):
        pass

    def __init__(self, seconds=1, error_message="Call timed out!"):
        self._seconds = seconds
        self._error_message = error_message

    def __enter__(self):
        def signal_handler(signum, frame):
            raise Timeout.Timeout(self._error_message)

        # Set the signal handler and alarm
        self._old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self._seconds)

    def __exit__(self, type, value, traceback):
        # Reset the alarm
        signal.alarm(0)

        # Reset the signal handler to its previous value
        signal.signal(signal.SIGALRM, self._old_handler)
