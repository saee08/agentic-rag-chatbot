# MCP Protocol message class
class MCPMessage:
    def __init__(self, sender, receiver, msg_type, trace_id, payload):
        self.sender = sender
        self.receiver = receiver
        self.type = msg_type
        self.trace_id = trace_id
        self.payload = payload

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "trace_id": self.trace_id,
            "payload": self.payload
        }
