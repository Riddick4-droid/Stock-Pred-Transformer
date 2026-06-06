class StockTransformerException(Exception):
    """
    Base exception for all errors raised within the project.

    Parameters
    ----------
    message : str
        Human‑readable description of the error.
    error_detail : str, optional
        Additional technical detail (e.g., system error message).
    """
    def __init__(self, message:str, error_detail:str=""):
        self.message = message
        self.error_detail = error_detail
        super().__init__()
    def __str__(self)->str:
        if self.error_detail:
            return f"{self.message}\n Detail: {self.error_detail}"
        return self.message