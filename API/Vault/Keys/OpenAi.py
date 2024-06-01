import os
import json
import sys
from Configs.ConfigReader import AppConfiguration
from Vault.Client import Vaultclient, VaultKVOption

import logging
logger = logging.getLogger()

class Llm:

    @staticmethod
    def get_kv():

        if os.getenv('SQL-SERVER-HOST'):
            pass
        else:
            sidecar = AppConfiguration.get_value("hc-sidecar", "container")
            useractivities = Vaultclient()

            match sidecar.lower():
                case 'true':
                    secret = useractivities.get_secret(VaultKVOption.sidecar.value)
                    sql_server_host = json.loads(secret)["sql-server.json"]["data"]["host"]
                    sql_server_dbname = json.loads(secret)["sql-server.json"]["data"]["dbname"]
                    sql_server_username = json.loads(secret)["sql-server.json"]["data"]["username"]
                    sql_server_password = json.loads(secret)["sql-server.json"]["data"]["password"]
                case 'false':
                    secret = useractivities.get_secret(VaultKVOption.sqlsever.value)
                    sql_server_host = json.loads(secret)["data"]['data']["host"]
                    sql_server_username = json.loads(secret)["data"]['data']["username"]
                    sql_server_password = json.loads(secret)["data"]['data']["password"]
                case _:
                    logger.error("please configure vault settings with option of api or sidecar interface.")
                    raise Exception("invalid option selected in config.ini {sidecar = false/true}.")
            try:
                os.environ['SQL-SERVER-HOST'] = sql_server_host
            except KeyError:
                logger.error('[error]: `SQL-SERVER-HOST` environment variable required')
                sys.exit(1)
            try:
                os.environ['SQL-SERVER-USERNAME'] = sql_server_password
            except KeyError:
                logger.error('[error]: `SQL-SERVER-USERNAME` environment variable required')
                sys.exit(1)
            try:
                os.environ['SQL-SERVER-PASSWORD'] = sql_server_username
            except KeyError:
                logger.error('[error]: `SQL-SERVER-PASSWORD` environment variable required')
                sys.exit(1)
            try:
                os.environ['SQL-SERVER-DBCONNECTION'] = f"mssql+pymssql://{os.environ['SQL-SERVER-USERNAME']}:{os.environ['SQL-SERVER-PASSWORD']}@{os.environ['SQL-SERVER-HOST']}/{os.environ['SQL-SERVER-PASSWORD']}"
                os.environ['SQL-SERVER-PYODBC-CONNECTION'] = f"mssql+pyodbc://{os.environ['SQL-SERVER-USERNAME']}:{os.environ['SQL-SERVER-PASSWORD']}@{os.environ['SQL-SERVER-HOST']}/{os.environ['SQL-SERVER-PASSWORD']}?driver=ODBC+Driver+17+for+SQL+Server"
            except KeyError:
                logger.error('[error]: `SQL-SERVER-DBCONNECTION` environment variable required')
                sys.exit(1)