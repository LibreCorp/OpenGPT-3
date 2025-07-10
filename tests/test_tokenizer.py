import os


def test_train_tokenizer(tmp_path, monkeypatch):
    data = "hello world\nhello test\n"
    input_file = tmp_path / "corpus.txt"
    input_file.write_text(data, encoding='utf-8')

    output_dir = tmp_path / "tokenizer"

    from opengpt3.tokenizer_train import train_tokenizer
    import argparse
    args = argparse.Namespace(
        files=[str(input_file)],
        output_dir=str(output_dir),
        vocab_size=50,
        min_frequency=1,
        special_tokens='',
    )
    train_tokenizer(args)

    files = os.listdir(output_dir)
    assert 'vocab.json' in files
    assert 'merges.txt' in files