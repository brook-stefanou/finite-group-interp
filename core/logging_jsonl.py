import json
from typing import Any
from datetime import datetime, timezone
from pathlib import Path


class JSONLLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._file = open(log_path, "a")

    def log(self, event: dict[str, Any]) -> None:
        event["_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
