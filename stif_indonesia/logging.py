import logging
import colorlog


class CustomLogger:
    FORMAT = '[%(asctime)s | PID: %(process)d | %(name)s - %(levelname)-8s] %(message)s'
    dict_level = {
        'debug' : logging.DEBUG,
        'info' : logging.INFO,
        'warning' : logging.WARNING,
        'error' : logging.ERROR,
        'critical' : logging.CRITICAL
    }

    def create_logger(self, log_name = __name__, log_file = None, low_level = 'info',
                      datefmt='%d-%b-%y %H:%M:%S', alay=False):
        used_format = self.FORMAT
        if alay:
            logger = colorlog.getLogger(log_name)
            bold_seq = '\033[1m'
            c_format = (
                f'{bold_seq}'
                '%(log_color)s'
                f'{used_format}'
            )
            c_handler = colorlog.StreamHandler()
            c_handler.setFormatter(colorlog.ColoredFormatter(c_format, datefmt=datefmt))
            logger.addHandler(c_handler)
            logger.setLevel(self.dict_level[low_level])

        else:
            # Handlers
            logger = logging.getLogger(log_name)
            c_handler = logging.StreamHandler()
            c_format = logging.Formatter(used_format, datefmt=datefmt)
            c_handler.setFormatter(c_format)
            c_handler.setLevel(self.dict_level[low_level])
            logger.addHandler(c_handler)
            logger.setLevel(self.dict_level[low_level])

        if log_file is not None:
            f_handler = logging.FileHandler(log_file)
            f_format = logging.Formatter(used_format, datefmt=datefmt)
            f_handler.setFormatter(f_format)
            f_handler.setLevel(self.dict_level[low_level])
            logger.addHandler(f_handler)

        return logger
