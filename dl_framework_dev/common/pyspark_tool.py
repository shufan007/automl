import os
import numpy as np
import socket
from .table import TableReader
from .tools import get_logger

logger = get_logger()


def create_spark_session(mode='local'):
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
    # archives = 'hdfs://DClusterNmg2/user/prod_alita/dml/miniconda3.tgz#miniconda3'

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    logger.info(f"hostname:{hostname}, ip:{ip}")

    tmp_dir = '/tmp/spark'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if mode == 'local':
        spark = (
            SparkSession.builder.appName("automl_pyspark_read_task")
            .config('spark.local.dir', tmp_dir)
            .config('spark.master', 'local[*]')
            # .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", ip)
            .enableHiveSupport()
            .getOrCreate()
        )
    else:
        cores = 200
        spark = (
            SparkSession.builder.appName("automl_pyspark_read_task")
            #.config("spark.driver.bindAddress", "127.0.0.1")
            .config('spark.master', 'yarn')
            .config("spark.driver.host", ip)
            .config("spark.driver.memory", "1G")
            .config("spark.executor.memory", "1G")
            .config("spark.executor.cores", "2")
            .config("spark.cores.max", "{cores}".format(cores=cores))
            .config("spark.sql.files.maxPartitionBytes", "16777216")
            .config("spark.default.parallelism", "{cores}".format(cores=cores))
            .config("spark.sql.shuffle.partitions", "{cores}".format(cores=cores))
            #.config("spark.yarn.dist.archives", archives)
            #.config("spark.pyspark.driver.python", "/home/luban/miniconda3/bin/python3")
            #.config("spark.pyspark.python", "./miniconda3/miniconda3/bin/python3")
            .config("spark.executor.instances",
                    "{ins}".format(ins=min(100, cores // 4)))  # executor number should be between 2 and 100
            .config("spark.yarn.queue", "root.data_alg_strategy_alita_prod")
            .config("spark.pyspark.python", os.environ["PYSPARK_PYTHON"])
            .enableHiveSupport()
            .getOrCreate()
        )

    return spark


def load_data(table_name, data_format='table',
              label_col='label_col', features_col: list = None,
              cond_str: str = None):

    logger.info("creat spark session...")
    try:
        spark = create_spark_session(mode='local')
    except Exception as e:
        logger.warning("except occurred when create spark session with local mode:", e)
        spark = create_spark_session(mode='remote')

    logger.info("spark table reading...")
    if isinstance(label_col, list):
        label_col = label_col[0]

    fields_str = '*'
    if (features_col is not None) and len(features_col) > 0:
        fields = features_col + [label_col]
        fields_str = ','.join(fields)
    sql_str = "select {} from {}".format(fields_str, table_name)

    if cond_str is not None:
        if ('limit' in cond_str.lower()) or ('where' in cond_str.lower()):
            sql_str += cond_str
        else:
            sql_str += " where {}".format(cond_str)

    df = spark.sql(sql_str)
    # df = spark.read.table(table_name)
    logger.info(f"dtypes of spark DataFrame: {df.dtypes}")

    if data_format == 'table':
        data = df.toPandas()
    elif data_format == 'array-table':
        X = np.array(df.select(features_col[0]).collect())
        y = np.array(df.select(label_col).collect())
        X = np.squeeze(X, axis=1)
        y = np.squeeze(y, axis=1)
        data = [X, y]

    return data


def table_data_load(data_path, data_format='table',
                    label_col='label_col', features_col=None,
                    cond_str=None):
    if data_format == 'textfile':
        data = TableReader(data_path).read()
    else:
        data = load_data(data_path, data_format=data_format,
                         label_col=label_col, features_col=features_col,
                         cond_str=cond_str)
    return data
