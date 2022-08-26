class Prompt:
  text: str
  seed: int

  def __init__(self, text: str, seed: int = 1):
    self.text = text
    self.seed = seed
