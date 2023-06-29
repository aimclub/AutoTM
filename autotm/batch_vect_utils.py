import os
from artm.batches_utils import BatchVectorizer, Batch
import glob
import random


class SampleBatchVectorizer(BatchVectorizer):
    def __init__(self, sample_size=10, **kwargs):
        self.sample_size = sample_size
        super().__init__(**kwargs)

    def _parse_batches(self, data_weight, batches):
        if self._process_in_memory:
            self._model.master.import_batches(batches)
            self._batches_list = [batch.id for batch in batches]
            return

        data_paths, data_weights, target_folders = self._populate_data(
            data_weight, True
        )
        for data_p, data_w, target_f in zip(data_paths, data_weights, target_folders):
            if batches is None:
                batch_filenames = glob.glob(os.path.join(data_p, "*.batch"))
                if len(batch_filenames) < self.sample_size:
                    self.sample_size = len(batch_filenames)
                batch_filenames = random.sample(batch_filenames, self.sample_size)
                self._batches_list += [Batch(filename) for filename in batch_filenames]

                if len(self._batches_list) < 1:
                    raise RuntimeError("No batches were found")

                self._weights += [data_w for i in range(len(batch_filenames))]
            else:
                self._batches_list += [
                    Batch(os.path.join(data_p, batch)) for batch in batches
                ]
                self._weights += [data_w for i in range(len(batches))]
