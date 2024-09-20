from Routers import (ProcessorRouter, AgentsRouter, EssayWriterRouter, FineTunningRouter,
                     HuggingFaceRouter, JwtAuthenticationRouter,
                     LangGraphRouter, LlamaIndexRouter, OpenAiRouter,
                     QuerySqlServerRouter, SpeechAndTextRouter)

v1_routers = [
    AgentsRouter.router,
    EssayWriterRouter.router,
    FineTunningRouter.router,
    HuggingFaceRouter.router,
    LangGraphRouter.router,
    OpenAiRouter.router,
    QuerySqlServerRouter.router,
    SpeechAndTextRouter.router,
    JwtAuthenticationRouter.router,
    LlamaIndexRouter.router,
    ProcessorRouter.router,
]

routers = {"v1:": v1_routers}
