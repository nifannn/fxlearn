# -*- coding: UTF-8 -*-

import logging


class Logger:
        def __init__(self, log_fp=None, name=None, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"):
                self._fp = log_fp
                self._name = name
                self._fmt = fmt
                self._datefmt = datefmt
                self._create_logger()
                self._set_logger()
        
        def _create_logger(self):
                if self._name is None:
                        self._logger = logging.getLogger()
                else:
                        self._logger = logging.getLogger(self._name)

        def _set_logger(self):
                self._logger.setLevel(logging.INFO)
                if self._fp is None:
                        handler = logging.StreamHandler()
                else:
                        handler = logging.FileHandler(self._fp)
                formatter = logging.Formatter(fmt=self._fmt, datefmt=self._datefmt)
                handler.setFormatter(formatter)
                if not self._logger.hasHandlers():
                        self._logger.addHandler(handler)

        def info(self, msg):
                self._logger.info(msg)

        def warn(self, msg):
                self._logger.warning(msg)
