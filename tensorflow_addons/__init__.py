from .seq2seq import hardmax  # noqa: F401
__all__ = ["seq2seq", "hardmax"]

class _Seq2SeqModule:
    def __getattr__(self, name):
        if name == "hardmax":
            from .seq2seq import hardmax
            return hardmax
        raise AttributeError(name)

seq2seq = _Seq2SeqModule()
