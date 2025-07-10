from typing import Dict, Optional


class Recorder(object):
    def __init__(self):
        self.metrics = {}
        self.batch_metrics = {}

    def record(self, metrics: Dict[str, float], scope: Optional[str] = None):
        for name, value in metrics.items():
            name = f'{scope}/{name}' if scope else name

            if name not in self.batch_metrics:
                self.batch_metrics[name] = []
            self.batch_metrics[name].append(value)

    def stamp(self, step: int = 0, scope: Optional[str] = None):
        stamped_names = []
        for name, values in self.batch_metrics.items():
            if scope is None or name.startswith(f'{scope}/'):
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append((step, sum(values) / len(values)))
                stamped_names.append(name)

        for name in stamped_names:
            del self.batch_metrics[name]

    def format(self, fstring: str) -> str:
        class SafeFormatter(dict):
            def __missing__(self, key):
                return "N/A"

        formatted_metrics = {}
        for k, v in self.metrics.items():
            formatted_metrics[k.replace('/', '_')] = float(v[-1][1])
        
        return fstring.format_map(SafeFormatter(formatted_metrics))
