# Vault KV options enumeration
from enum import Enum


class VaultKVOption(Enum):
    ConfluenceDb = "confluence-db"
    OpenAi = "openai"
    PGVector = "pgvector"
    sqlsever = "sqlserver"
    sidecar = "container"
