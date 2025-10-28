from flautim.pytorch.Dataset import Dataset
from st_gnca.dataloader import database


class SplitDataset(Dataset):
    def __init__(self, data, **kwargs):
        super(SplitDataset, self).__init__(name = "PEMS", **kwargs)
        self.data = data

    def train(self) -> Dataset:
        return self

    def validation(self) -> Dataset:
        return self

    def test(self) -> Dataset:
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class PEMSDataset(Dataset):

    def __init__(self, **kwargs):
        super(PEMSDataset, self).__init__(name = "PEMS", **kwargs)

        self.db = database.DataBase(**kwargs)

        batch_size = kwargs.get("batch_size", 32)
        sequence_len = kwargs.get("sequence_len", 10)
        train_split = kwargs.get("train_split", .7)
        val_split = kwargs.get("val_split", .1)

        self.bb = database.BatchBuilder(self.db, batch_size, sequence_len, train_split, val_split)
        

    def train(self) -> Dataset:
        return SplitDataset(self.bb._create_sequences(self.bb.train_data))

    def validation(self) -> Dataset:
        return SplitDataset(self.bb._create_sequences(self.bb.val_data))

    def test(self) -> Dataset:
        return SplitDataset(self.bb._create_sequences(self.bb.test_data))

    def __len__(self):
        return self.bb.num_samples

    def __getitem__(self, idx):
        return None
