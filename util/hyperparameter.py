import argparse


def get_hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='nama_pembimbing', help='Target Column')
    # parser.add_argument("--root", type=str, default='prodi', help='Target Root Column')
    # parser.add_argument("--node", type=str, default='nama_pembimbing', help='Target Node Column')
    parser.add_argument("--dataset", type=str, default='init_data_repo_jtik.json', help='Dataset Path')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch Size')
    parser.add_argument("--bert_model", type=str, default="indolem/indobert-base-uncased", help='BERT Model')
    parser.add_argument("--seed", type=int, default=42, help='Random Seed')
    parser.add_argument("--max_epochs", type=int, default=30, help='Number of Epochs')
    parser.add_argument("--lr", type=float, default=2e-5, help='Learning Rate')
    parser.add_argument("--dropout", type=float, default=0.1, help='Dropout')
    parser.add_argument("--patience", type=int, default=3, help='Patience')
    parser.add_argument("--num_bert_states", type=int, default=4, help='Number of BERT Last States')
    parser.add_argument("--max_length", type=int, default=360, help='Max Length')
    parser.add_argument("--in_channels", type=int, default=4, help='CNN In Channels')
    parser.add_argument("--out_channels", type=int, default=32, help='CNN Out Channels')
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5], help='CNN Kernel')
    parser.add_argument("--test_size", type=float, default=0.2, help='Percentage of Test Set')
    parser.add_argument("--valid_size", type=float, default=0.1, help='Percentage of Validation Set')
    config = vars(parser.parse_args())

    return config
