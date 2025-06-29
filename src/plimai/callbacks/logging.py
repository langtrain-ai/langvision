class LoggingCallback:
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            self.f = open(log_file, 'a')
        else:
            self.f = None

    def log(self, message):
        print(message)
        if self.f:
            self.f.write(message + '\n')
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close() 