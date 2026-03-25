import logging
from collections import deque
from typing import Optional


class ImGuiHandler(logging.Handler):
    def __init__(self, level: Optional[int] = logging.INFO, max_messages: Optional[int] = 128):
        self.buffer = deque(maxlen=max_messages)
        self.level = level
        self.filters = []
        self.lock = None
        self.formatter = None

    def emit(self, record):
        log_entry = self.format(record)
        self.buffer.append(log_entry)
