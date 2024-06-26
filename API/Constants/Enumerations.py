# Vault KV options enumeration
from enum import Enum


class VaultKVOption(Enum):
    OpenAi = 'openai'
    PGVector = 'pgvector'
    sqlsever = 'sqlserver'
    sidecar = 'container'