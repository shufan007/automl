# Copyright (c) DiDi Group. All rights reserved.
import os
import socket

import pyarrow.dataset as ds

from autotabular.common.logging import get_logger
from autotabular.common.utils import timeit

logger = get_logger()


class TableReader(object):
    """Read table data from files in Parquet, ORC, or CSV format, or from hive table. """

    def __init__(self, format):
        """
        Args:
            format (str): The format of the data file. One of "parquet", "orc", or
                "csv" (case-insensitive), 'table' or 'array-table'.
        """
        self.format = format.lower()

    def __call__(self, path):
        """
           Args:
               path (str): data file path

           Returns:
              dataframe
        """
        pass


class HdfsTableReader(TableReader):
    """Read table data from files in Parquet, ORC, or CSV format."""

    def __init__(self, format):
        """
        Args:
            format (str): The format of the data file. One of "parquet", "orc", or
                "csv" (case-insensitive).
        """
        super().__init__(format)

        assert self.format in ('parquet', 'orc', 'csv')

    @timeit
    def __call__(self, path):
        """
           Args:
               path (str): data file path

           Returns:
              pyarrow.dataset.Dataset: dataset in pyarrow format
        """
        dataset = ds.dataset(path, format=self.format).to_table().to_pandas()
        return dataset


class HiveTableReader(TableReader):
    """Read table data from hive table."""

    def __init__(self, format,
                 feature_cols: list = None,
                 label_cols: list = None,
                 cond_str: str = None):
        """
        Args:
            format (str): The format of the data file. One of 'table' or 'array-table'.
            feature_cols: Feature column names
            label_cols: Label column names
            cond_str: condition string of sql
        """
        super().__init__(format)
        assert self.format in ('table', 'array-table')

        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.cond_str = cond_str
        self.spark = self.create_spark_session()
        logger.info("Spark session is initialized successfully ")

    def __call__(self, path):
        """
           Args:
               path (str): data file path

           Returns:
              dataframe
        """
        dataset = self.read_table(table_name=path,
                                  feature_cols=self.feature_cols,
                                  label_cols=self.label_cols,
                                  cond_str=self.cond_str)
        return dataset

    @staticmethod
    def create_spark_session():
        """
         create spark session
         required environ vars: HADOOP_USER_NAME, HADOOP_USER_PASSWORD, SPARK_HOME
        :return: spark session
        """
        # pyspark connect
        os.environ["PYSPARK_PYTHON"] = os.path.join(os.getenv('SPARK_HOME'), "bin/python")
        import findspark
        findspark.init()  # get spark env, should be located before 'import pyspark'
        from pyspark.sql import SparkSession
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        logger.info(f"hostname:{hostname}, ip:{ip}")

        tmp_dir = '/tmp/spark'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        spark = (
            SparkSession.builder.appName("automl_pyspark_read_task")
            .config('spark.local.dir', tmp_dir)
            .config('spark.master', 'local[*]')
            .config("spark.driver.host", ip)
            .config("spark.driver.memory", "1G")
            .config("spark.hadoop.mapred.max.split.size", 32000000)
            .config("spark.sql.execution.arrow.enabled", True)
            .config('spark.sql.execution.arrow.fallback.enabled', True)
            .enableHiveSupport()
            .getOrCreate()
        )

        return spark

    @timeit
    def get_table_path(self, table_name):
        """
        get hdfs path of the table

        Args:
            table_name: Input table name

        Returns:
            table_path: hdfs path of the table
            data_format: data format
        """
        sql_str = "describe formatted {}".format(table_name)
        schema = self.spark.sql(sql_str).toPandas()
        get_schema_item = lambda col_name: schema[schema["col_name"] == col_name]["data_type"].iloc[0]
        table_path = get_schema_item("Location")
        data_format = get_schema_item("OutputFormat").split(".")[-2]

        return table_path, data_format

    @timeit
    def read_table(self, table_name,
                   feature_cols: list = None,
                   label_cols: list = None,
                   cond_str: str = None):
        """
        Read data from hive table

        Args:
            table_name: Input table name
            feature_cols: Input feature column names
            label_cols: Input label column names
            cond_str: condition string of sql

        Returns:
            Pandas dataframe
        """
        fields_str = '*'
        if label_cols is not None:
            if (feature_cols is not None) and len(feature_cols) > 0:
                fields = feature_cols + label_cols
                fields_str = ','.join(fields)

        sql_str = "select {} from {}".format(fields_str, table_name)
        if cond_str is not None:
            if ('limit' in cond_str.lower()) or ('where' in cond_str.lower()):
                sql_str += cond_str
            else:
                sql_str += " where {}".format(cond_str)

        dataset = self.spark.sql(sql_str).toPandas()
        return dataset
