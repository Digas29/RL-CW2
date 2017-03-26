class Turn:
    NOOP = 0        # NOOP
    LEFT = 1  # LEFT
    RIGHT = 2       # RIGHT

    @staticmethod
    def toString(a):
        return {
            0: 'NOOP',
            1: 'LEFT',
            2: 'RIGHT'}[a]
