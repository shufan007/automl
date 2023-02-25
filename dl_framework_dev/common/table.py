
import pyarrow.dataset as ds
import os

class TableReader:
    """Read table data from files in Parquet, ORC, or CSV format."""

    def __init__(self, path, format=None):
        """
        Args:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "parquet", "orc", or
                "csv" (case-insensitive).
        """
        if format:
            format = format.lower()
            assert format in ('parquet', 'orc', 'csv')
            self.dataset = ds.dataset(path, format=format)
        else:
            if os.path.splitext(path)[-1] == ".csv":
                self.dataset = ds.dataset(path, format='csv')
            else:
                self.dataset = ds.dataset(path)

    def read(self, in_memory=True):
        """
        Args:
            in_memory (bool): Whether to copy the data in-memory
        Returns:
           If keep in memory, return pandas.DataFrame, otherwise raise not implemented error
        """
        if in_memory:
            return self.dataset.to_table().to_pandas()
        else:
            raise NotImplementedError
