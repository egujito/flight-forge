class Parachute:
    def __init__(self, name, cd_s, lag, trigger):
        self.cd_s = cd_s
        self.lag = lag
        self.name = name
        self.trigger = trigger
        self.active = False
        self.logged = False
        self.deploy_t = None
