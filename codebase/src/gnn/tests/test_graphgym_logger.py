from unittest.mock import MagicMock


def _safe_logger_basic(logger, round_digits: int = 4):
    """Mirror of loader_graphgym.custom_logger_basic zero-batch guard."""
    if logger._size_current == 0:
        return {
            "loss": 0.0,
            "lr": round(logger._lr, round_digits),
            "params": logger._params,
            "time_iter": 0.0,
        }
    return {
        "loss": round(logger._loss / logger._size_current, round_digits),
        "lr": round(logger._lr, round_digits),
        "params": logger._params,
        "time_iter": 0.0,
    }


def _validation_epoch_end(callback, trainer, pl_module):
    """Mirror of loader_graphgym.custom_on_validation_epoch_end."""
    if callback.val_logger._size_current > 0:
        callback.val_logger.write_epoch(trainer.current_epoch)


class _FakeLogger:
    def __init__(self):
        self._size_current = 0
        self._loss = 0
        self._lr = 0
        self._params = 0
        self.epochs_written = []

    def write_epoch(self, epoch):
        self.epochs_written.append(epoch)


def test_safe_logger_basic_handles_empty_epoch():
    logger = _FakeLogger()
    stats = _safe_logger_basic(logger)
    assert stats["loss"] == 0.0
    assert stats["time_iter"] == 0.0


def test_validation_epoch_end_skips_empty_val_logger():
    callback = MagicMock()
    callback.val_logger = _FakeLogger()
    callback.test_logger = _FakeLogger()
    trainer = MagicMock(current_epoch=3)

    _validation_epoch_end(callback, trainer, pl_module=None)

    assert callback.val_logger.epochs_written == []
    assert callback.test_logger.epochs_written == []


def test_validation_epoch_end_writes_only_non_empty_val_logger():
    callback = MagicMock()
    callback.val_logger = _FakeLogger()
    callback.val_logger._size_current = 8
    callback.test_logger = _FakeLogger()
    trainer = MagicMock(current_epoch=3)

    _validation_epoch_end(callback, trainer, pl_module=None)

    assert callback.val_logger.epochs_written == [3]
    assert callback.test_logger.epochs_written == []
