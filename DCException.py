from werkzeug.exceptions import HTTPException

class DCException(Exception):
    statuscode = 400

    def __init__(self, message, statuscode=None, payload=None):
        super().__init__()
        self.message = message
        if statuscode is not None:
            self.statuscode = statuscode
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['statuscode'] = self.statuscode
        return rv