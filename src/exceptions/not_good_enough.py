class NotGoodEnoughException(Exception):
    def __init__(self, message="The image provided is not good enough.", details=None):
        super().__init__(message)
        self.details = details

    def __str__(self):
        if self.details:
            return f"{super().__str__()} (Details: {self.details})"
        return super().__str__()
