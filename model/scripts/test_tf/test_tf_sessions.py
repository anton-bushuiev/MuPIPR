import tensorflow as tf
import collections

class TrackedSession(tf.Session):
    _sessions = collections.defaultdict(list)
    def __init__(self, target='', graph=None, config=None):
        super(tf.Session, self).__init__(target=target, graph=graph, config=config)
        TrackedSession._sessions[self.graph].append(self)
    def close(self):
        super(tf.Session, self).close()
        TrackedSession._sessions[self.graph].remove(self)
    @classmethod
    def get_open_sessions(cls, g=None):
        g = g or tf.get_default_graph()
        return list(cls._sessions[g])

print(TrackedSession.get_open_sessions())
