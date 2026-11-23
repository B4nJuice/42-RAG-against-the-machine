from src.errors.rag_error import RagError


class UnsupportedDeviceError(RagError):
    def __init__(self, device: str) -> None:
        super().__init__(f"Unsupported device: {device}")
