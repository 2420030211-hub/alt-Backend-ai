class AIServiceException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class ModelNotLoadedException(AIServiceException):
    def __init__(self, message: str = "AI Model is not loaded or checkpoint is missing"):
        super().__init__(message=message, status_code=503)

class InvalidInputException(AIServiceException):
    def __init__(self, message: str = "Invalid input supplied to AI inference"):
        super().__init__(message=message, status_code=400)
