CALLBACKS = {}

def register_callback(name):
    def decorator(cls):
        CALLBACKS[name] = cls
        return cls
    return decorator

def get_callback(name):
    return CALLBACKS[name] 