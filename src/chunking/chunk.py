class Chunk:
    def __init__(
                self,
                file_path: str,
                content: str,
                start_index: int,
                end_index: int
            ):
        self.file_path: str = file_path
        self.content: str = content
        self.start_index: int = start_index
        self.end_index: int = end_index