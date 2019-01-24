import json
from collections import defaultdict
from abc import ABC, abstractmethod


class Logger(ABC):
	def __init__(self):
		self._log = defaultdict(float)

	@abstractmethod
	def log(self, name, value):
		raise NotImplementedError()

	@abstractmethod
	def dump(self):
		raise NotImplementedError()

	@abstractmethod
	def close(self):
		raise NotImplementedError()


class LoggerWrapper(Logger):
	def __init__(self, loggers=None):
		super().__init__()
		self._loggers = loggers if loggers is not None else list()

	def append(self, logger):
		self._loggers.append(logger)

	def log(self, name, value):
		for logger in self._loggers:
			logger.log(name, value)

	def dump(self):
		for logger in self._loggers:
			logger.dump()

	def close(self):
		for logger in self._loggers:
			logger.close()


class ConsoleLogger(Logger):
	def log(self, name, value):
		self._log[name] = value

	def dump(self):
		key_space = max(len(k) for k in self._log.keys())
		val_space = max(len(str(v)) for v in self._log.values())
		dashes = '-' * (key_space + val_space + 7)
		print(dashes)
		for (key, v) in sorted(self._log.items()):
			val = str(v)
			print('| %s%s | %s%s |' % (key, ' ' * (key_space - len(key)), val, ' ' * (val_space - len(val))))
		print(dashes)
		self._log.clear()

	def close(self):
		pass


class JSONLogger(Logger):
	def __init__(self, fn):
		super().__init__()
		self.file = open(fn, 'wt')

	def log(self, name, value):
		self._log[name] = value

	def dump(self):
		self.file.write(json.dumps(self._log) + '\n')
		self.file.flush()
		self._log.clear()

	def close(self):
		self.file.close()
