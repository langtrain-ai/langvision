"""langvision._trl_compat works with both old and current TRL argument names."""
import dataclasses

from langvision._trl_compat import make_config, make_trainer


@dataclasses.dataclass
class NewConfig:
    output_dir: str
    max_length: int = 1
    bf16: bool = False


@dataclasses.dataclass
class OldConfig:
    output_dir: str
    max_seq_length: int = 1


class NewTrainer:
    def __init__(self, model=None, processing_class=None):
        self.processing_class = processing_class


class OldTrainer:
    def __init__(self, model=None, tokenizer=None):
        self.tokenizer = tokenizer


def test_config_renames_and_drops_unknown_fields():
    new = make_config(NewConfig, output_dir="o", max_seq_length=2048, bf16=True, not_a_field=1)
    assert new.max_length == 2048 and new.bf16
    assert make_config(OldConfig, output_dir="o", max_seq_length=99).max_seq_length == 99


def test_trainer_gets_tokenizer_under_the_right_name():
    assert make_trainer(NewTrainer, "tok", model=1).processing_class == "tok"
    assert make_trainer(OldTrainer, "tok", model=1).tokenizer == "tok"
